__version__ = "2023.06.30-1"

import numpy, pandas, pymc#, yt, pyvista
import ROOT
import uproot

ROOT.gInterpreter.ProcessLine(".L compton.C")

def fom_costheta(interaction):
    return 1.0e0-0.511*interaction.E_s/((interaction.E_a)*interaction.E_t)

def fom_lambda(interaction):

    norm = numpy.sqrt(
        (interaction.X_s-interaction.X_a)**2 +
        (interaction.Y_s-interaction.Y_a)**2 +
        (interaction.Z_s-interaction.Z_a)**2
    )
    n_x, n_y, n_z = [
        (interaction.X_s-interaction.X_a)/norm,
        (interaction.Y_s-interaction.Y_a)/norm,
        (interaction.Z_s-interaction.Z_a)/norm
    ]
    
    return numpy.abs(
        (n_x*interaction.X_s+n_y*interaction.Y_s+n_z*interaction.Z_s)**2\
        -(\
            1\
            +(0.511/interaction.E_t)\
            -(0.511/interaction.E_a)\
        )**2\
        *(interaction.X_s**2+interaction.Y_s**2+interaction.Z_s**2)
    )

def fom_arm(interaction, dist):
    
    norm = numpy.sqrt(
        (interaction.X_s-interaction.X_a)**2 +
        (interaction.Y_s-interaction.Y_a)**2 +
        (interaction.Z_s-interaction.Z_a)**2
    )
    rss_x, rss_y, rss_z = [
        (interaction.X_s-0),
        (interaction.Y_s-0),
        (interaction.Z_s-dist)
    ]
    rsa_x, rsa_y, rsa_z = [
        (interaction.X_a-interaction.X_s),
        (interaction.Y_a-interaction.Y_s),
        (interaction.Z_a-interaction.Z_s)
    ]
    ScalarProd=rss_x*rsa_x+rss_y*rsa_y+rss_z*rsa_z
    ModuleProd=numpy.sqrt(rss_x**2+rss_y**2+rss_z**2)*numpy.sqrt(rsa_x**2+rsa_y**2+rsa_z**2)
    ScatAngle=numpy.arccos(ScalarProd/ModuleProd)*180 / numpy.pi

    return ScatAngle-numpy.arccos(interaction.costheta)

def RR(interaction, scale):
    match interaction.DetectorNumber:
        case 1:
            res = 1
        case 2:
            res = 1
        case 3:
            res = 1
        case 4:
            res = 1
    
    Es = numpy.random.default_rng().normal(interaction.E_s, .01, scale)
    Xs = numpy.random.default_rng().normal(interaction.X_s, 1.7/2.355, scale)
    Ys = numpy.random.default_rng().normal(interaction.Y_s, 1.7/2.355, scale)
    Zs = numpy.random.default_rng().normal(interaction.Z_s, 2.5/2.355, scale)
    Ea = numpy.random.default_rng().normal(interaction.E_a, res, scale)
    Xa = numpy.random.default_rng().normal(interaction.X_a, 3.9/2.355, scale)
    Ya = numpy.random.default_rng().normal(interaction.Y_a, 3.9/2.355, scale)
    Za = numpy.random.default_rng().normal(interaction.Z_a, 3.5, scale) #0->Offset
    

    frame = {
        'DetectorNumber': interaction.DetectorNumber,
        'E_s': Es,
        'X_s': Xs,
        'Y_s': Ys,
        'Z_s': Zs,
        'E_a': Ea,
        'X_a': Xa,
        'Y_a': Ya,
        'Z_a': Za,
        'dt': interaction["dt"],
        'ts': interaction["ts"],
        'RR': True,
        'id': interaction["id"]
    }

    return pandas.DataFrame(frame)

class ImagingData:
    def __init__(self, intake_, rr_=1, resolutions_=False):
        self.__intake = intake_
        self.__rr = rr_
        self.__resolutions = resolutions_
        self.__distance = self.metadata()['position']['z']
        self.__dataframe = self._df()
        
    def __repr__(self):
        return f"File:\t{self.__intake.urlpath}\nDistance:\t{self.__distance}\nResolution Recovery:\t{self.__rr}"

    def __str__(self):
        return f"{self.__intake.urlpath}_{self.__distance}"

    def _df(self):
        df_ = self.__intake.read().drop("0",axis=1)
        df_['RR'] = False

        df_.reset_index(inplace=True,drop=True)
        df_['id'] = df_.index

        if self.__rr > 0:
            lrr_ = df_.apply(lambda x: RR(x, self.__rr), axis=1)
            dfrr_ = pandas.concat([lrr_[j] for j in lrr_.index])
            df_ = pandas.concat([df_,dfrr_])

        df_['E_t'] = df_['E_a'] + df_['E_s']
        df_['norm'] = numpy.sqrt(
            (df_.X_s-df_.X_a)**2+
            (df_.Y_s-df_.Y_a)**2+
            (df_.Z_s-df_.Z_a)**2
        )
        df_['X_n'] = (df_.X_s-df_.X_a)/df_['norm']
        df_['Y_n'] = (df_.Y_s-df_.Y_a)/df_['norm']
        df_['Z_n'] = (df_.Z_s-df_.Z_a)/df_['norm']
        df_['costheta'] = df_.apply(fom_costheta, axis=1)
        df_['lambd'] = df_.apply(fom_lambda, axis=1)
        df_['arm'] = df_.apply(lambda x: fom_arm(x, self.__distance), axis=1)

        df_.reset_index(inplace=True,drop=True)
        
        return df_

    def df(self):
        return self.__dataframe
        
    def metadata(self):
        return self.__intake.metadata

    def distance(self):
        return self.__distance

    def dimension(self):
        return len(df.index)

    def save(self):
        file = uproot.recreate(f"{self.__intake.urlpath}_CALIBRATED.root")
        file["COINCIDENCES"] = self.df()
        file["METADATA"] = ROOT.TObjString(str(self.metadata()))

    def dbs(self, query="ilevel_0 in ilevel_0"):
        df = self.df().query(query)
        
        canvas = ROOT.TCanvas()
        canvas.cd()
        
        hist = ROOT.Get_BackProjection_Image(
             100,
            -250.0,
             250.0,
             100,
            -250.0,
             250.0,
            "Backprojection",
            len(df.index),
            df.X_s.to_numpy(),
            df.Y_s.to_numpy(),
            df.Z_s.to_numpy(),
            df.E_s.to_numpy(),
            df.X_a.to_numpy(),
            df.Y_a.to_numpy(),
            df.Z_a.to_numpy(),
            df.E_a.to_numpy(),
            self.distance()
        )

        hist.SetStats(0)

        hist.Draw("colz")

        return hist, canvas

    def soe(self, query="ilevel_0 in ilevel_0"):
        df = self.df().query(query)
        
        canvas = ROOT.TCanvas()
        canvas.cd()
        
        hist = ROOT.Get_SOE_Image(
            500,
            1000,
            len(df.index),
            len(df.index),
            0.0,
            self.distance(),
             60,
            -150.0,
             150.0,
             60,
            -150.0,
             150.0,
            "SOE",
            df.X_s.to_numpy(),
            df.Y_s.to_numpy(),
            df.Z_s.to_numpy(),
            df.E_s.to_numpy(),
            df.X_a.to_numpy(),
            df.Y_a.to_numpy(),
            df.Z_a.to_numpy(),
            df.E_a.to_numpy()
        )

        hist.SetStats(0)

        hist.Draw("colz")

        return hist, canvas
        
    def analytical(self, query="ilevel_0 in ilevel_0"):
        df = self.df().query(query)
        
        canvas = ROOT.TCanvas()
        canvas.cd()
        
        hist = ROOT.Get_Analytical_Image(
             100,
            -250.0,
             250.0,
             100,
            -250.0,
             250.0,
            "Analytical",
            len(df.index),
            df.X_s.to_numpy(),
            df.Y_s.to_numpy(),
            df.Z_s.to_numpy(),
            df.E_s.to_numpy(),
            df.X_a.to_numpy(),
            df.Y_a.to_numpy(),
            df.Z_a.to_numpy(),
            df.E_a.to_numpy(),
            self.distance(),
            30, #degrees,
            0.001,
            0.5
        )

        hist.SetStats(0)

        hist.Draw("colz")

        return hist, canvas