/*
 * This file was developed as part of a Thesis at IFIC-UV, Valencia, Spain
 * Optimization of imaging techniques for background suppression of stellar Nucleo-Synthesis reactions with i-TED
 *
 * Code based on previous iterations
 * 
 * Author:
 *     - BGameiro (Bernardo Gameiro, inbox@bgameiro.me)
 */

/****************
Modules
****************/


#include "Riostream.h"
#include <time.h>
#include <string.h>
#include <cstdlib>
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TF1.h"
#include "TF2.h"
#include "TGraph.h"
#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TFile.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TStopwatch.h"
#include "TApplication.h"
#include "TGaxis.h"
#include "TLatex.h"
#include <string.h>
#include <vector> 
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <TNamed.h>
#include <TPad.h>
#include <limits.h>
#include <random>
#include <TRandom3.h>
#include <TPaveLabel.h>
#include <TPaveText.h>
#include <TCutG.h>
#include <future>   
#include "Math/IFunction.h"

/****************
Constants
****************/

const int ndetectors = 5; // 1 scatter and 4 absorvers

/********
Funstions
********/

int Acquisition_Time(TTree * detector_tree)
{
    const long time_init = detector_tree->GetMinimum("TimeStamp");
    const long time_fin  = detector_tree->GetMaximum("TimeStamp");
    int  time_acq  = round((time_fin-time_init)/pow(10,12));
    
    return time_acq;
}

float Alpha_Activity(TTree * detector_tree) // Seems to have a problem
{
    //float ADCmin_alpha[ndetectors] = {750,750,630,800,600}; //i-TED A
    //float ADCmin_alpha[ndetectors] = {700,600,650,720,650}; //i-TED B
    //float ADCmin_alpha[ndetectors] = {550,600,600,600,650}; //i-TED C
    //float ADCmin_alpha[ndetectors] = {550,600,600,600,650}; //i-TED 
    
    TH1D* h;
    double Alphas = 0.;

    int  time_acq  = Acquisition_Time(detector_tree);
    int En;
    detector_tree->SetBranchAddress("Total_Deposited_Energy", &En);
    
    for(int i=0; i<detector_tree->GetEntries(); i++){
        
        detector_tree->GetEntry(i);
        
        if ((750 < En)  && (En < 1400)) {
          Alphas+=1.;
        } ;
        
    };
    
	cout<<"Integral Alphas: "<<Alphas<<endl;
	Alphas/=time_acq;
	cout<<"Alpha rate (Hz): "<<Alphas<<endl;
    
    return Alphas;
}

