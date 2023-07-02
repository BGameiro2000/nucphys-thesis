#include "TH2.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TH2D.h"
#include "TCanvas.h"
#include <fstream>
#include <iostream>
#include "TRandom3.h"
#include "TF1.h"
#include "TF2.h"
#include "TMath.h"
#include "TFile.h"

#include <future>
#include <vector>
#include <cstdlib>
using namespace std;

#define N_MAX_DEGREES 200

/*
    HELPERS
*/

Double_t Get_Theta_Compton(const Double_t Es,const Double_t Eg){
  constexpr auto mec2 {0.511e0};//rest mass of electron
  const Double_t cos_theta {1.0e0-mec2*Es/((Eg-Es)*Eg)};
  return acos(cos_theta);
}

Double_t Get_CTheta_Compton(const Double_t Es,const Double_t Eg){
  constexpr auto mec2 {0.511e0};//rest mass of electron
  const Double_t cos_theta {1.0e0-mec2*Es/((Eg-Es)*Eg)};
  return cos_theta;
}

Bool_t Check_Position(const Double_t xlow,const Double_t xhigh,const Double_t ylow,const Double_t yhigh,const Double_t x_pos,const Double_t y_pos){
  if((x_pos>xlow)&&(x_pos<xhigh)&&(y_pos>ylow)&&(y_pos<yhigh)){return true;}
  return false;
}

void Get_New_Position(TRandom3 *r,const Double_t xlow,const Double_t xhigh,const Double_t ylow,const Double_t yhigh,const Double_t xs,const Double_t ys,
		      const Double_t *n,const Double_t A,const Double_t ctheta2,const Double_t zprime,Double_t &x_pos,Double_t &y_pos){
  const Int_t nMaxTries {50};//This parameter can be changed. Smaller values goes in the direction of "faster", but it is not sure the convergence
  x_pos=y_pos=1.0e16;
  auto nTries {0};
  while(!Check_Position(xlow,xhigh,ylow,yhigh,x_pos,y_pos)&&nTries<nMaxTries){
    y_pos=r->Uniform(ylow,yhigh);
    const Double_t u=y_pos-ys;
    const Double_t B=2.0e0*n[0]*(n[1]*u+n[2]*zprime);
    const Double_t C=(n[1]*u+n[2]*zprime)*(n[1]*u+n[2]*zprime)-ctheta2*(u*u+zprime*zprime);
    const Double_t D=B*B-4.0e0*A*C;
    if(D>0.0e0){
      if(r->Rndm()>0.5e0){
	x_pos=(-B+sqrt(D))/(2.0e0*A)+xs;
      }
      else{
	x_pos=(-B-sqrt(D))/(2.0e0*A)+xs;
      }
    }
    else{x_pos=y_pos=1.0e16;}
    nTries++;
  }
}

Double_t Get_Local_Density(TH2D *h,const Double_t x,const Double_t y,const int nRadius){
  const Int_t nx=h->GetXaxis()->FindBin(x);
  const Int_t ny=h->GetYaxis()->FindBin(y);
  Double_t value=0.0e0;
  for(auto i=nx-nRadius;i<=nx+nRadius;i++){
    if(i>0&&i<=h->GetXaxis()->GetNbins()){
      for(auto j=ny-nRadius;j<=ny+nRadius;j++){
	if(j>0&&j<=h->GetYaxis()->GetNbins()){
	  value+=h->GetBinContent(i,j);
	}
      }
    }
  }
  if(value<1.0e0){value=1.0e0;}
  return value;
}

Double_t Get_Compton_Probability(const Double_t Egamma,const Double_t cTheta){
  const Double_t x=cTheta;
  const Double_t gamma=Egamma/0.511e0;
  Double_t value=((1.0e0+x*x)/(2.0e0*pow(1.0e0+gamma*(1.0e0-x),2)))*(1.0e0+gamma*gamma*(1.0e0-x)*(1.0e0-x))/((1.0e0+x*x)*(1.0e0+gamma*(1.0e0-x)));;
  return value;
}

Double_t Get_Legendre_Polynomial_Recurrence(const int nDegree,const Double_t Pn,const Double_t Pn_1,const Double_t x){
  Double_t value=((2.0e0*nDegree+1.0e0)*x*Pn-static_cast<Double_t>(nDegree)*Pn_1)/static_cast<Double_t>(nDegree+1);
  return value;
}

Double_t Get_Polynomial_Factor_Value(const int nDegree,const Double_t Egamma,const Double_t Omega_1,const Double_t Omega_2){
    char name[100];
    sprintf(name,"f_%d",rand() % 10000);
  TF1 *fIntegrando=new TF1(name,"((1+x*x)/(2.0*pow(1.0+[0]*(1.0-x),2)))*(1.0+[0]*[0]*(1.0-x)*(1.0-x)/((1.0+x*x)*(1.0+[0]*(1.0-x))))*pow(ROOT::Math::legendre([1],x),2)",-1.0,1.0);
  fIntegrando->FixParameter(0,Egamma/0.511e0);
  fIntegrando->FixParameter(1,nDegree);
  const Double_t value=fIntegrando->Integral(cos(Omega_2),cos(Omega_1));
  if(fIntegrando!=nullptr){delete fIntegrando;}
  return value;
}

/*
    IMAGING
*/

TH2D *Get_BackProjection_Image(const Int_t nBinsX,const Double_t xlow,const Double_t xhigh,
		const Int_t nBinsY,const Double_t ylow,const Double_t yhigh,
		const char *name,const int nInteractions,
		const Double_t *xs,const Double_t *ys,const Double_t *zs,const Double_t *Es,
		const Double_t *xa,const Double_t *ya,const Double_t *za,const Double_t *Ea,
		const Double_t zp){
  constexpr auto mec2 {0.511e0};//rest mass of electron
  TH2D *hValue=new TH2D(name,name,nBinsX,xlow,xhigh,nBinsY,ylow,yhigh);
  hValue->GetYaxis()->SetTitleSize(0.05);
  hValue->GetXaxis()->SetTitleSize(0.05);
  hValue->GetYaxis()->SetTitleOffset(0.85);
  hValue->GetXaxis()->SetTitleOffset(0.85);
  hValue->GetXaxis()->SetTitle("x [cm]");
  hValue->GetYaxis()->SetTitle("y [cm]");
  for(auto i=0;i<nInteractions;i++){
    //if(i%1000==0){cout<<"Interaction "<<i<<"/"<<nInteractions<<endl;}
    const Double_t Eg=Es[i]+Ea[i];
    const Double_t cos_theta {1.0e0-mec2*Es[i]/((Eg-Es[i])*Eg)};
    const Double_t cos_theta2 {cos_theta*cos_theta};
    const Double_t norm {sqrt((xs[i]-xa[i])*(xs[i]-xa[i])+(ys[i]-ya[i])*(ys[i]-ya[i])+(zs[i]-za[i])*(zs[i]-za[i]))};
    const Double_t n[3] {(xs[i]-xa[i])/norm,(ys[i]-ya[i])/norm,(zs[i]-za[i])/norm};
    const Double_t zprime=zp-zs[i];
    for(auto j=1;j<=hValue->GetYaxis()->GetNbins();j++){
        const Double_t current_y=hValue->GetYaxis()->GetBinCenter(j);
        const Double_t u=current_y-ys[i];
        const Double_t A=n[0]*n[0]-cos_theta2;
        const Double_t B=2.0e0*n[0]*(n[1]*u+n[2]*zprime);
        const Double_t C=(n[1]*u+n[2]*zprime)*(n[1]*u+n[2]*zprime)-cos_theta2*(u*u+zprime*zprime);
        const Double_t Discriminant=B*B-4.0e0*A*C;
        if(Discriminant>0.0e0){
            const Double_t x1=(-B+sqrt(Discriminant))/(2.0e0*A)+xs[i];
            hValue->Fill(x1,current_y);
            const Double_t x2=(-B-sqrt(Discriminant))/(2.0e0*A)+xs[i];
            hValue->Fill(x2,current_y);
        }
    }
    for(auto j=1;j<=hValue->GetXaxis()->GetNbins();j++){
        const Double_t current_x=hValue->GetXaxis()->GetBinCenter(j);
        const Double_t u=current_x-xs[i];
        const Double_t A=n[1]*n[1]-cos_theta2;
        const Double_t B=2.0e0*n[1]*(n[0]*u+n[2]*zprime);
        const Double_t C=(n[0]*u+n[2]*zprime)*(n[0]*u+n[2]*zprime)-cos_theta2*(u*u+zprime*zprime);
        const Double_t Discriminant=B*B-4.0e0*A*C;
        if(Discriminant>0.0e0){
            const Double_t y1=(-B+sqrt(Discriminant))/(2.0e0*A)+ys[i];
            hValue->Fill(current_x,y1);
            const Double_t y2=(-B-sqrt(Discriminant))/(2.0e0*A)+ys[i];
            hValue->Fill(current_x,y2);
        }
    }
  }
  return hValue;
}

////////////////////////////////////////////////////////////////

TH2D *Get_SOE_Image(const int SEED,const Int_t nIterations,const Int_t nStepPerIteration,const Int_t nPhotons,const Double_t radius_local_density,const Double_t z,
		    const Int_t nBinsX,const Double_t xlow,const Double_t xhigh,
		    const Int_t nBinsY,const Double_t ylow,const Double_t yhigh,const char *name,
		    const Double_t *xs,const Double_t *ys,const Double_t *zs,const Double_t *Es,
		    const Double_t *xa,const Double_t *ya,const Double_t *za,const Double_t *Ea){
  TRandom3 *r=new TRandom3(SEED);
  TH2D *hImage=new TH2D(name,name,nBinsX,xlow,xhigh,nBinsY,ylow,yhigh);
  TH2D *hDensity=new TH2D("hDensity","hDensity",nBinsX,xlow,xhigh,nBinsY,ylow,yhigh);
  hDensity->GetYaxis()->SetTitleSize(0.05);
  hDensity->GetXaxis()->SetTitleSize(0.05);
  hDensity->GetYaxis()->SetTitleOffset(0.85);
  hDensity->GetXaxis()->SetTitleOffset(0.85);
  hDensity->GetXaxis()->SetTitle("x [cm]");
  hDensity->GetYaxis()->SetTitle("y [cm]");
  Double_t *x=new Double_t[nPhotons];
  Double_t *y=new Double_t[nPhotons];
  Double_t **n=new Double_t *[nPhotons];
  Double_t *ctheta2=new Double_t[nPhotons];
  Double_t *A=new Double_t[nPhotons];
  Double_t *zprime=new Double_t[nPhotons];
  for(auto i=0;i<nPhotons;i++){
    n[i]=new Double_t[3];
  }
  Int_t nBackscattering=0;
  for(auto j=0;j<nPhotons;j++){
    auto ctheta=Get_CTheta_Compton(Es[j],Es[j]+Ea[j]);
    //if(ctheta<0.0e0){cout<<"Warning: This is a backscattering("<<nBackscattering<<")!: Es= "<<Es[j]<<" Ea= "<<Ea[j]<<endl;nBackscattering++;}
    ctheta2[j]=ctheta*ctheta;
    const Double_t norm {sqrt((xs[j]-xa[j])*(xs[j]-xa[j])+(ys[j]-ya[j])*(ys[j]-ya[j])+(zs[j]-za[j])*(zs[j]-za[j]))};
    n[j][0]=(xs[j]-xa[j])/norm;
    n[j][1]=(ys[j]-ya[j])/norm;
    n[j][2]=(zs[j]-za[j])/norm;
    A[j]=(n[j][0]*n[j][0]-ctheta2[j]);
    zprime[j]=z-zs[j];
  }
  Int_t nRadius=static_cast<Int_t>(radius_local_density/((xhigh-xlow)/static_cast<Double_t>(nBinsX)));
  if(nRadius<1){nRadius=1;}
  //cout<<"Number of bins for radius: "<<nRadius<<endl;
  //cout<<"Number of fotons involved  the calculation: "<<nPhotons<<endl;
  //cout<<"Number of steps per iteration: "<<nStepPerIteration<<endl;
  //cout<<"Number of iterations: "<<nIterations<<endl;

  for(auto j=0;j<nPhotons;j++){
    Get_New_Position(r,xlow,xhigh,ylow,yhigh,xs[j],ys[j],n[j],A[j],ctheta2[j],zprime[j],x[j],y[j]);
    hDensity->Fill(x[j],y[j]);
  }

  //cout<<"Density (OK).....";
  
  for(auto i=0;i<nIterations;i++){
    //if(i%100==0){cout<<"Starting iteration.......("<<i<<"/"<<nIterations<<")...."<<endl;}
    for(auto j=0;j<nStepPerIteration;j++){
      Double_t x_proposed=1.0e16;
      Double_t y_proposed=1.0e16;
      const Int_t current_k=r->Integer(nPhotons);
      const Double_t current_Density=Get_Local_Density(hDensity,x[current_k],y[current_k],radius_local_density);
      Get_New_Position(r,xlow,xhigh,ylow,yhigh,xs[current_k],ys[current_k],n[current_k],A[current_k],ctheta2[current_k],zprime[current_k],x_proposed,y_proposed);
      if(Check_Position(xlow,xhigh,ylow,yhigh,x_proposed,y_proposed)){
	const Double_t proposed_Density=Get_Local_Density(hDensity,x_proposed,y_proposed,nRadius);
	const Double_t Acceptance=min(1.0e0,(proposed_Density+1)/current_Density);
	if(r->Rndm()<Acceptance){
	  hDensity->Fill(x[current_k],y[current_k],-1.0e0);
	  x[current_k]=x_proposed;
	  y[current_k]=y_proposed;
	  hDensity->Fill(x_proposed,y_proposed,1.0e0);
	}
      }
    }
    //if(i%20==0){cout<<"!"<<endl;}
  }
  for(auto j=0;j<nPhotons;j++){
    if(Check_Position(xlow,xhigh,ylow,yhigh,x[j],y[j])){
      hImage->Fill(x[j],y[j]);
    }
  }
  //cout<<"Image filled!"<<endl;
  if(r!=nullptr){delete r;}
  if(hDensity!=nullptr){delete hDensity;}
  if(x!=nullptr){delete [] x;}
  if(y!=nullptr){delete [] y;}
  if(A!=nullptr){delete [] A;}
  if(zprime!=nullptr){delete [] zprime;}
  if(ctheta2!=nullptr){delete [] ctheta2;}
  for(auto i=0;i<nPhotons;i++){if(n[i]!=nullptr){delete n[i];}}
  if(n!=nullptr){delete [] n;}
  return hImage;
}

////////////////////////////////////////////////////////////////

TH2D *Get_Analytical_Image(const Int_t nBinsX, const Double_t xlow, const Double_t xhigh,
        const Int_t nBinsY, const Double_t ylow, const Double_t yhigh,
        const char *name, const int nInteractions,
        const Double_t *xs, const Double_t *ys, const Double_t *zs, const Double_t *Es,
        const Double_t *xa, const Double_t *ya, const Double_t *za, const Double_t *Ea,
        const Double_t z, const Int_t nMaxDegrees, const Double_t Omega_1, const Double_t Omega_2) {
    auto *hValue = new TH2D(name, name, nBinsX, xlow, xhigh, nBinsY, ylow, yhigh);
    hValue->GetYaxis()->SetTitleSize(0.05);
    hValue->GetXaxis()->SetTitleSize(0.05);
    hValue->GetYaxis()->SetTitleOffset(0.85);
    hValue->GetXaxis()->SetTitleOffset(0.85);
    hValue->GetXaxis()->SetTitle("x [cm]");
    hValue->GetYaxis()->SetTitle("y [cm]");
    Double_t *Hn = new Double_t[nMaxDegrees];
    Double_t *Pn = new Double_t[nMaxDegrees];
    Double_t *Pn_ji = new Double_t[nMaxDegrees];
    for (auto k = 0; k < nInteractions; k++) {
        const Double_t theta_c_interaction = Get_Theta_Compton(Es[k], Es[k] + Ea[k]);
        if ((theta_c_interaction > Omega_1)&&(theta_c_interaction < Omega_2)) {
            cout << k << endl;
            const Double_t ctheta_c_interaction = Get_CTheta_Compton(Es[k], Es[k] + Ea[k]);
            for (auto i = 0; i < nMaxDegrees; i++) {
                Hn[i] = Get_Polynomial_Factor_Value(i, Es[k] + Ea[k], Omega_1, Omega_2);
            }//Calculating the factors
            Pn[0] = 1.0e0;
            Pn[1] = ctheta_c_interaction;
            for (auto i = 2; i < nMaxDegrees; i++) {
                Pn[i] = Get_Legendre_Polynomial_Recurrence(i - 1, Pn[i - 1], Pn[i - 2], ctheta_c_interaction);
            }
            const Double_t value_K = Get_Compton_Probability(Es[k] + Ea[k], ctheta_c_interaction);
            const Double_t norm = sqrt((xa[k] - xs[k])*(xa[k] - xs[k])+(ya[k] - ys[k])*(ya[k] - ys[k])+(za[k] - zs[k])*(za[k] - zs[k]));
            const Double_t n[3] = {(xa[k] - xs[k]) / norm, (ya[k] - ys[k]) / norm, (za[k] - zs[k]) / norm};
            const Double_t theta_k = atan(sqrt(n[0] * n[0] + n[1] * n[1]) / n[2]);
            Double_t phi_k = TMath::Pi() / 2.0e0;
            if (abs(n[0]) > 0.0e0) {
                phi_k = atan(abs(n[1] / n[0]));
                if (n[0] < 0.0e0 && n[1] > 0.0e0) {
                    phi_k = TMath::Pi() - phi_k;
                } else if (n[0] < 0.0e0 && n[1] < 0.0e0) {
                    phi_k = TMath::Pi() + phi_k;
                } else if (n[0] > 0.0e0 && n[1] < 0.0e0) {
                    phi_k = 2.0e0 * TMath::Pi() - phi_k;
                }
            } else {
                if (n[1] < 0.0) {
                    phi_k = 3.0e0 * TMath::Pi() / 2.0e0;
                }
            }
            for (auto i = 1; i <= hValue->GetNbinsX(); i++) {
                const Double_t current_x = hValue->GetXaxis()->GetBinCenter(i);
                for (auto j = 1; j <= hValue->GetNbinsY(); j++) {
                    const Double_t current_y = hValue->GetYaxis()->GetBinCenter(j);
                    const Double_t norm_s = sqrt((current_x - xs[k])*(current_x - xs[k])+(current_y - ys[k])*(current_y - ys[k])+(z - zs[k])*(z - zs[k]));
                    const Double_t s[3] = {(current_x - xs[k]) / norm_s, (current_y - ys[k]) / norm_s, (z - zs[k]) / norm_s};
                    const Double_t theta_s = atan(sqrt(s[0] * s[0] + s[1] * s[1]) / s[2]);
                    Double_t phi_s = TMath::Pi() / 2.0e0;
                    if (abs(s[0]) > 0.0e0) {
                        phi_s = atan(abs(s[1] / s[0]));
                        if (s[0] < 0.0e0 && s[1] > 0.0e0) {
                            phi_s = TMath::Pi() - phi_s;
                        } else if (s[0] < 0.0e0 && s[1] < 0.0e0) {
                            phi_s = TMath::Pi() + phi_s;
                        } else if (s[0] > 0.0e0 && s[1] < 0.0e0) {
                            phi_s = 2.0e0 * TMath::Pi() - phi_s;
                        }
                    } else {
                        if (s[1] < 0.0) {
                            phi_s = 3.0e0 * TMath::Pi() / 2.0e0;
                        }
                    }
                    Double_t value = 0.0e0;
                    Double_t cos_ji = cos(theta_s) * cos(theta_k) + sin(theta_s) * sin(theta_k) * cos(phi_s - phi_k);
                    Pn_ji[0] = 1.0e0;
                    Pn_ji[1] = cos_ji;
                    for (auto ii = 2; ii < nMaxDegrees; ii++) {
                        Pn_ji[ii] = Get_Legendre_Polynomial_Recurrence(ii - 1, Pn_ji[ii - 1], Pn_ji[ii - 2], cos_ji);
                    }
                    for (auto ii = 0; ii < nMaxDegrees; ii++) {
                        value += ((2.0e0 * ii + 1) / (TMath::Pi()*4.0e0 * Hn[ii])) * value_K * Pn[ii] * Pn_ji[ii];
                    }
                    hValue->Fill(current_x, current_y, value);
                }
            }
        } else {
            cout << "Interaction: " << k << "/" << nInteractions << " Out of bounds for computation -> " << theta_c_interaction << "[" << Omega_1 << "," << Omega_2 << "]" << endl;
        }
    }
    if (Hn != nullptr) {
        delete [] Hn;
    }
    if (Pn != nullptr) {
        delete [] Pn;
    }
    if (Pn_ji != nullptr) {
        delete [] Pn_ji;
    }
    return hValue;
}