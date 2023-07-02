__version__ = "2023.06.24-1"

import numpy, pandas, pymc, scipy, sympy, ROOT, uproot

class Angle:

    def __init__(self, angle_):
        self.__angle = angle_ /180 * numpy.pi
    
    def __pos__(self):
        return self

    def __mod__(self):
        return self.__angle % (2*numpy.pi)

    def cos(self):
        return numpy.cos(self.__angle)

    def sin(self):
        return numpy.sin(self.__angle)

    def tan(self):
        return numpy.tan(self.__angle)

    def __repr__(self):
        return "A({:f}º)".format(self.__angle /numpy.pi *180)

class Point3:

    def __init__(self, x_, y_, z_):
        self.__x = x_
        self.__y = y_
        self.__z = z_

    def get_x(self):
        return self.__x
    
    def get_y(self):
        return self.__y
    
    def get_z(self):
        return self.__z
    
    def __sub__(self, other_):
        x = self.__x - other_.__x
        y = self.__y - other_.__y
        z = self.__z - other_.__z
        return Vector3(x,y,z)

    def __add__(self, vector_):
        x = self.__x + vector_.get_x()
        y = self.__y + vector_.get_y()
        z = self.__z + vector_.get_z()
        return Point3(x,y,z)
    
    def __pos__(self):
        return self

    def __repr__(self):
        return "P({:f}, {:f}, {:f})".format(self.__x, self.__y, self.__z)

class Vector3:

    def __init__(self, x_, y_, z_):
        self.__x = x_
        self.__y = y_
        self.__z = z_

    def get_x(self):
        return self.__x
    
    def get_y(self):
        return self.__y
    
    def get_z(self):
        return self.__z

    def __add__(self, other_):
        x = self.__x + other_.__x
        y = self.__y + other_.__y
        z = self.__z + other_.__z
        return Vector3(x,y,z)
    
    def __iadd__(self, other_):
        self.__x += other_.__x
        self.__y += other_.__y
        self.__z += other_.__z
        return self
    
    def __sub__(self, other_):
        x = self.__x - other_.__x
        y = self.__y - other_.__y
        z = self.__z - other_.__z
        return Vector3(x,y,z)

    def __isub__(self, other_):
        self.__x -= other_.__x
        self.__y -= other_.__y
        self.__z -= other_.__z
        return self
    
    def __mul__(self, factor_):
        x = self.__x * factor_
        y = self.__y * factor_
        z = self.__z * factor_
        return Vector3(x,y,z)

    def __rmul__(self, factor_):
        x = self.__x * factor_
        y = self.__y * factor_
        z = self.__z * factor_
        return Vector3(x,y,z)
    
    def __truediv__(self, factor_):
        x = self.__x / factor_
        y = self.__y / factor_
        z = self.__z / factor_
        return Vector3(x,y,z)
    
    def __floordiv__(self, factor_):
        x = self.__x // factor_
        y = self.__y // factor_
        z = self.__z // factor_
        return Vector3(x,y,z)
    
    def __mod__(self, factor_):
        x = self.__x % factor_
        y = self.__y % factor_
        z = self.__z % factor_
        return Vector3(x,y,z)
    
    def __pos__(self):
        return self
    
    def __neg__(self):
        x = self.__x * -1.0
        y = self.__y * -1.0
        z = self.__z * -1.0
        return Vector3(x,y,z)

    def __abs__(self):
        return (self.__x**2 + self.__y**2 + self.__z**2)**0.5

    def dot(self, other_):
        return self.__x*other_.__x + self.__y*other_.__y + self.__z*other_.__z

    def __repr__(self):
        return "V({:f}, {:f}, {:f})".format(self.__x, self.__y, self.__z)

    def perp1(self):
        if   self.__x == 0:
            return Vector3(1,0,0)
        elif self.__y == 0:
            return Vector3(0,1,0)
        elif self.__z == 0:
            return Vector3(0,0,1)
        else:
            return Vector3(1,1,-1*(self.__x+self.__y)/self.__z)

    def perp2(self, vector_):
        vect_p = Vector3(
            vector_.get_y()*self.__z-vector_.get_z()*self.__y,
            vector_.get_z()*self.__x-vector_.get_x()*self.__z,
            vector_.get_x()*self.__y-vector_.get_y()*self.__x
        )

        return vect_p/abs(vect_p)
        
class Cone:
    def __init__(self, vertice_, vector_, angle_):
        self.__vertice = vertice_
        self.__vector = vector_
        self.__angle = angle_
        self.__vector_p1 = self.__vector.perp1()    
        self.__vector_p2 = self.__vector.perp2(self.__vector_p1)

    def surface(self, point_):
        dataset = pandas.DataFrame({'Point': point_})

        dataset["vect_s"] = dataset.Point.map(
                                lambda Point: Point - self.__vertice
                            )
        dataset["vect_c"] = dataset.vect_s.map(
                                lambda vect_s: self.__vector * (abs(vect_s)*self.__angle.cos()/abs(self.__vector))
                            )
        dataset["center"] = dataset.vect_c.map(
                                lambda vect_c: self.__vertice + vect_c
                            )
        dataset["dist_c"] = dataset.vect_s.map(
                                lambda vect_s: abs(vect_s)*self.__angle.sin()
                            )
        dataset["dist_p"] = dataset.apply(
                                lambda entry: abs(entry.Point - entry.center),
                                axis=1
                            )
        
        return numpy.allclose(dataset.dist_c,dataset.dist_p)

    def __repr__(self):
        return "Cone(\n\t{},\n\t{},\n\t{}\n)".format(self.__vertice, self.__vector, self.__angle)

    def surface_point(self, prng1_, prng2_, n_=1):
        r = prng1_.random(n_)**0.5
        a = prng2_.random(n_)*2*numpy.pi

        return r, a

    def __call__(self, rand0_, rand1_):

        r = rand0_**0.5
        a = rand1_*2*numpy.pi

        vect_p  = self.__vector_p1*numpy.cos(a)+\
                  self.__vector_p2*numpy.sin(a)
        vect_p /= abs(vect_p)

        factor  = 350
        radius  = r * factor * numpy.tan(self.__angle)

        vect_h  = radius * self.__vector

        vect_a  = vect_p * radius
        
        return self.__vertice + vect_h + vect_a

class Sphere:
    def __init__(self, point_, r_):
        self.__x = point_.get_x()
        self.__y = point_.get_y()
        self.__z = point_.get_z()
        self.__r = r_

    def inside(self, point):
        return (
            (self.__x - point.get_x()) ** 2 +
            (self.__y - point.get_y()) ** 2 +
            (self.__z - point.get_z()) ** 2 <= self.__r
        )

    def volume(self):
        return 4/3 * numpy.pi * self.__r**3

class CubeFOV:
    def __init__(self, xmin_, xmax_, ymin_, ymax_, zmin_, zmax_):
        self.__x  = [xmin_, xmax_]
        self.__y  = [ymin_, ymax_]
        self.__z  = [zmin_, zmax_]
        self.__df = pandas.DataFrame({
            ("i0","x"):[],
            ("i0","y"):[],
            ("i0","z"):[]
        })

    def inside(self, point):
        return (
            (self.__x[0] <= point.get_x() <= self.__x[1]) and
            (self.__y[0] <= point.get_y() <= self.__y[1]) and
            (self.__z[0] <= point.get_z() <= self.__z[1])
        )

    def density(self, point, radius):
        Sphere.inside()/CubeFOV.inside()/Sphere.volume()

        self.__df[iteration].query("@inside_radius()")
        