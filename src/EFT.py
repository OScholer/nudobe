#to generate warnings
import warnings

#numerics
import numpy as np
from scipy import optimize
from scipy import integrate
import pandas as pd

#plotting
import matplotlib as mplt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

#0nubb stuff
import PSFs
from NMEs import Load_NMEs

#some functions used
import functions as f

#plotting functions
import plots

#import constants
from constants import *

#import operator running
from RGE import *

#to get files in path
import os

#current working directory as absolute path
import sys
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())
    
#get absolute path of this file
cwd = os.path.abspath(os.path.dirname(__file__))

#generate isotope classes to calculate PSF observables which need the electron wave functions
#isotopes without NMEs are put as comments to speed up the code
#if you add NMEs with isotopes currently commented make sure to uncomment

PSF_scheme_global = "A"

def set_PSF_scheme(scheme = "A", setglobal = True):
    #Z is the proton number in the daughter nucleus i.e. Z = Z_mother + 2
    #Here, masses and energies are in MeV
    U238  = PSFs.isotope(Z = 92, A = 238, Delta_M = 1.144154 + 2*m_e_MeV, scheme = scheme)
    Th232 = PSFs.isotope(Z = 90, A = 232, Delta_M = 0.837879 + 2*m_e_MeV, scheme = scheme)
    #Hg204 = PSFs.isotope(Z = 80, A = 204, Delta_M = 0.419154 + 2*m_e_MeV, scheme = scheme)
    Pt198 = PSFs.isotope(Z = 78, A = 198, Delta_M = 1.049142 + 2*m_e_MeV, scheme = scheme)
    #Os192 = PSFs.isotope(Z = 76, A = 192, Delta_M = 0.408274 + 2*m_e_MeV, scheme = scheme)
    #W186  = PSFs.isotope(Z = 74, A = 186, Delta_M = 0.491643 + 2*m_e_MeV, scheme = scheme)
    #Yb176 = PSFs.isotope(Z = 70, A = 176, Delta_M = 1.088730 + 2*m_e_MeV, scheme = scheme)
    #Er170 = PSFs.isotope(Z = 68, A = 170, Delta_M = 0.655586 + 2*m_e_MeV, scheme = scheme)
    Gd160 = PSFs.isotope(Z = 64, A = 160, Delta_M = 1.730530 + 2*m_e_MeV, scheme = scheme)
    Sm154 = PSFs.isotope(Z = 62, A = 154, Delta_M = 1.250810 + 2*m_e_MeV, scheme = scheme)
    Nd150 = PSFs.isotope(Z = 60, A = 150, Delta_M = 3.371357 + 2*m_e_MeV, scheme = scheme)
    Nd148 = PSFs.isotope(Z = 60, A = 148, Delta_M = 1.928286 + 2*m_e_MeV, scheme = scheme)
    #Nd146 = PSFs.isotope(Z = 60, A = 146, Delta_M = 0.070421 + 2*m_e_MeV, scheme = scheme)
    #Ce142 = PSFs.isotope(Z = 58, A = 142, Delta_M = 1.417175 + 2*m_e_MeV, scheme = scheme)
    Xe136 = PSFs.isotope(Z = 54, A = 136, Delta_M = 2.457984 + 2*m_e_MeV, scheme = scheme)
    Xe134 = PSFs.isotope(Z = 54, A = 134, Delta_M = 0.825751 + 2*m_e_MeV, scheme = scheme)
    Te130 = PSFs.isotope(Z = 52, A = 130, Delta_M = 2.527515 + 2*m_e_MeV, scheme = scheme)
    Te128 = PSFs.isotope(Z = 52, A = 128, Delta_M = 0.866550 + 2*m_e_MeV, scheme = scheme)
    Sn124 = PSFs.isotope(Z = 50, A = 124, Delta_M = 2.291010 + 2*m_e_MeV, scheme = scheme)
    #Sn122 = PSFs.isotope(Z = 50, A = 122, Delta_M = 0.372877 + 2*m_e_MeV, scheme = scheme)
    Cd116 = PSFs.isotope(Z = 48, A = 116, Delta_M = 2.813438 + 2*m_e_MeV, scheme = scheme)
    #Cd114 = PSFs.isotope(Z = 48, A = 114, Delta_M = 0.542493 + 2*m_e_MeV, scheme = scheme)
    Pd110 = PSFs.isotope(Z = 46, A = 110, Delta_M = 2.017234 + 2*m_e_MeV, scheme = scheme)
    #Ru104 = PSFs.isotope(Z = 44, A = 104, Delta_M = 1.301297 + 2*m_e_MeV, scheme = scheme)
    Mo100 = PSFs.isotope(Z = 42, A = 100, Delta_M = 3.034342 + 2*m_e_MeV, scheme = scheme)
    #Mo98  =  PSFs.isotope(Z = 42, A = 98, Delta_M = 0.109935 + 2*m_e_MeV, scheme = scheme)
    Zr96  =  PSFs.isotope(Z = 40, A = 96, Delta_M = 3.348982 + 2*m_e_MeV, scheme = scheme)
    #Zr94  =  PSFs.isotope(Z = 40, A = 94, Delta_M = 1.141919 + 2*m_e_MeV, scheme = scheme)
    #Kr86  =  PSFs.isotope(Z = 36, A = 86, Delta_M = 1.257542 + 2*m_e_MeV, scheme = scheme)
    Se82  =  PSFs.isotope(Z = 34, A = 82, Delta_M = 2.996402 + 2*m_e_MeV, scheme = scheme)
    #Se80  =  PSFs.isotope(Z = 34, A = 80, Delta_M = 0.133874 + 2*m_e_MeV, scheme = scheme)
    Ge76  =  PSFs.isotope(Z = 32, A = 76, Delta_M = 2.039061 + 2*m_e_MeV, scheme = scheme)
    #Zn70  =  PSFs.isotope(Z = 30, A = 70, Delta_M = 0.997118 + 2*m_e_MeV, scheme = scheme)
    Ca48  =  PSFs.isotope(Z = 20, A = 48, Delta_M = 4.266970 + 2*m_e_MeV, scheme = scheme)
    #Ca46  =  PSFs.isotope(Z = 20, A = 46, Delta_M = 0.988576 + 2*m_e_MeV, scheme = scheme)

    #list of isotope classes
    isotopes_dict = {"238U"  : U238,
                     "232Th" : Th232,
                     #"204Hg" : Hg204,
                     "198Pt" : Pt198,
                     #"192Os" : Os192,
                     #"186W"  : W186,
                     #"176Yb" : Yb176,
                     #"170Er" : Er170,
                     "160Gd" : Gd160,
                     "154Sm" : Sm154,
                     "150Nd" : Nd150,
                     "148Nd" : Nd148,
                     #"146Nd" : Nd146,
                     #"142Ce" : Ce142,
                     "136Xe" : Xe136,
                     "134Xe" : Xe134,
                     "130Te" : Te130,
                     "128Te" : Te128,
                     "124Sn" : Sn124,
                     #"122Sn" : Sn122,
                     "116Cd" : Cd116,
                     #"114Cd" : Cd114,
                     "110Pd" : Pd110,
                     #"104Ru" : Ru104,
                     "100Mo" : Mo100,
                     #"98Mo"  : Mo98,
                     "96Zr"  : Zr96,
                     #"94Zr"  : Zr94,
                     #"86Kr"  : Kr86,
                     "82Se"  : Se82,
                     #"80Se"  : Se80,
                     "76Ge"  : Ge76,
                     #"70Zn"  : Zn70,
                     "48Ca"  : Ca48,
                     #"46Ca"  : Ca46
                     }
    #set the PSF_scheme globally
    if setglobal:
        global PSF_scheme_global
        PSF_scheme_global = scheme
        global isotopes
        isotopes = isotopes_dict
    return(isotopes_dict)

#generate dict with isotope classes - needed for angular corr and spectra
isotopes = set_PSF_scheme(PSF_scheme_global)

#list of corresponding names
#I don't know why I didn't make a dict here...
isotope_names = list(isotopes.keys())

#GENERATE THE RUNNING MATRIX FOR LEFT
#create running matrix that connects m_W and xPT scales in LEFT
matrix = np.zeros((len(LEFT_WCs), len(LEFT_WCs)))
idx = 0
for operator in LEFT_WCs:
    LEFT_WCs[operator] = 1
    a = np.array(list(run_LEFT(LEFT_WCs).values()))
    matrix[idx] = a
    idx += 1
    for operators in LEFT_WCs:
        LEFT_WCs[operators] = 0
matrix = matrix.T

#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                         Low-Energy EFT                                            #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


class LEFT(object):
    ################################################################
    # this class generates LEFT models with given Wilson coefficients
    # the WCs are entered at the scale of M_W=80GeV
    # it can calculate the low energy observables of 0nuBB decay
    ################################################################
    def __init__(self, WC, name = None, unknown_LECs = False, method = "IBM2", PSF_scheme = "A", basis = "C"):
        #NME method
        self.method     = method
        
        #operator basis
        self.basis      = basis
        
        #PSF scheme (currently useless)
        self.PSF_scheme = PSF_scheme
        
        # physical constants (see constants.py)
        self.m_N     = m_N
        self.m_e     = m_e
        self.m_e_MEV = m_e_MeV
        self.vev     = vev
        
        #store wilson coefficients
        #there are two possible choices of a LEFT WC basis
        #the C-basis is the preferred one and all calculations are done within this basis
        #however, also the epsilon-basis can be chosen and is translated to the C-basis internally
        #we leave the dict as a comment here so you know how it looks like
        #self.CWC = {#dim3
        #      "m_bb":0, 
        #      #dim6
        #      "SL(6)": 0, "SR(6)": 0, 
        #      "T(6)":0, 
        #      "VL(6)":0, "VR(6)":0, 
        #      #dim7
        #      "VL(7)":0, "VR(7)":0, 
        #      #dim9
        #      "1L(9)":0, "1R(9)":0, 
        #      "1L(9)prime":0, "1R(9)prime":0, 
        #      "2L(9)":0, "2R(9)":0, 
        #      "2L(9)prime":0, "2R(9)prime":0, 
        #      "3L(9)":0, "3R(9)":0, 
        #      "3L(9)prime":0, "3R(9)prime":0, 
        #      "4L(9)":0, "4R(9)":0, 
        #      "5L(9)":0, "5R(9)":0, 
        #      "6(9)":0,
        #      "6(9)prime":0,
        #      "7(9)":0,
        #      "7(9)prime":0,
        #      "8(9)":0,
        #      "8(9)prime":0,
        #      "9(9)":0,
        #      "9(9)prime":0}
        
        self.CWC = LEFT_WCs.copy()
        
        self.EpsilonWC = LEFT_WCs_epsilon.copy()
        
        #self.EpsilonWC = {#dim3
        #                  "m_bb":0, 
        #                  #dim6
        #                  "V+AV+A": 0, "V+AV-A": 0, 
        #                  "TRTR":0, 
        #                  "S+PS+P":0, "S+PS-P":0,
        #                  #dim7
        #                  "VL(7)":0, "VR(7)":0, #copied from C basis
        #                  #dim9
        #                  "1LLL":0, "1LLR":0,
        #                  "1RRL":0, "1RRR":0,
        #                  "1RLL":0, "1RLR":0,
        #                  "2LLL":0, "2LLR":0,
        #                  "2RRL":0, "2RRR":0,
        #                  "3LLL":0, "3LLR":0,
        #                  "3RRL":0, "3RRR":0,
        #                  "3RLL":0, "3RLR":0,
        #                  "4LLR":0, "4LRR":0,
        #                  "4RRR":0, "4RLR":0,
        #                  "5LLR":0, "5LRR":0,
        #                  "5RRR":0, "5RLR":0,
        #                  #redundant operators
        #                  "1LRL":0, "1LRR":0, 
        #                  "3LRL":0, "3LRR":0,
        #                  "4LLL":0, "4LRL":0,
        #                  "4RRL":0, "4RLL":0,
        #                  "TLTL":0,
        #                  "5LLL":0, "5LRL":0,
        #                  "5RRL":0, "5RLL":0,
        #                  #vanishing operators
        #                  "2LRL":0, "2LRR":0, 
        #                  "2RLL":0, "2RLR":0, 
        #                  "TRTL":0, "TLTR":0, 
        #                  #operators not contributing directly
        #                  "V-AV+A": 0, "V-AV-A": 0, 
        #                  "S-PS+P":0, "S-PS-P":0,
        #                 }
        
        
        #get the WCs right
        if basis == "C" or basis == "c":
            for operator in WC:
                self.CWC[operator] = WC[operator]
            self.EpsilonWC = self.change_basis(basis=self.basis, inplace = False)
        elif basis == "E" or basis == "e" or basis == "epsilon" or basis == "Epsilon":
            for operator in WC:
                self.EpsilonWC[operator] = WC[operator]
            self.CWC = self.change_basis(basis=self.basis, inplace = False)
        else:
            warnings.warn("Basis",basis,'is not defined. Choose either "C" for the Grasser basis used in the master formula, or "epsilon" for the old standard basis by Päs et al. Setting the basis to C...')
        
        
        #WC Dict that will store the WCs @ chiPT scale used for calculations
        self.WC = self.CWC.copy()
        
        #store inpit WCs
        self.WC_input = self.WC.copy()
        
        #running WCs from mW to chiPT scale
        self.WC = self._run(WC = self.WC, updown="down")
        
        #store model name
        if name == None:
            self.name = "Model"
        else:
            self.name = name
        
        #Import PSFs
        self.PSFpanda = pd.read_csv(cwd+"/../PSFs/PSFs_"+self.PSF_scheme+".csv")
        self.PSFpanda.set_index("PSFs", inplace = True)
        
        #Import NMEs
        self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        
        #Store the Low Energy Constants (LECs) required
        self.unknown_LECs = unknown_LECs
        if unknown_LECs == True:
            #self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
            #           "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
            #           # all the below are expected to be order 1 in absolute magnitude
            #           "Tprime":1, "Tpipi":1, "1piN":1, "6piN":1, "7piN":1, "8piN":1, "9piN":1, "VLpiN":1, "TpiN":1, 
            #           "1NN":1, "6NN":1, "7NN":1, "VLNN":1, "TNN": 1, "VLE":1, "VLme":1, "VRE":1, "VRme":1, 
            #           # all the below are expected to be order (4pi)**2 in absolute magnitude
            #           "2NN":(4*np.pi)**2, "3NN":(4*np.pi)**2, "4NN":(4*np.pi)**2, "5NN":(4*np.pi)**2, 
            #           # expected to be 1/F_pipi**2 pion decay constant
            #           "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
            #          }
            
            self.LEC = LECs.copy()
            
            self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
            self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
        
        else:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":0, "Tpipi":0, "1piN":0, "6piN":0, "7piN":0, "8piN":0, "9piN":0, "VLpiN":0, "TpiN":0, 
                       "1NN":0, "6NN":1, "7NN":1, "VLNN":0, "TNN": 0, "VLE":0, "VLme":0, "VRE":0, "VRme":0, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":0, "3NN":0, "4NN":0, "5NN":0, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            
            #VpiN and tildeVpiN would vanish in this setting. 
            #We set them to 1, otherwise the dim 9 vector operators would not contribute
            
            self.LEC["VpiN"] = 1#LEC["6piN"] + LEC["8piN"]
            self.LEC["tildeVpiN"] = 1#LEC["7piN"] + LEC["9piN"]
        

        #list of isotope classes
        
        #if the chosen wave function scheme is equal to the global one just set the isotopes to the global parameter
        if self.PSF_scheme == PSF_scheme_global:
            self.isotopes = isotopes
        #if a different wave function scheme is chosen recalculate the isotopes
        else:
            self.set_wave_scheme(self.PSF_scheme)

        #list of corresponding names
        #Again, I don't know why I didn't make a dict here...
        self.isotope_names = np.flip(list(self.NMEs.keys()))
        
    #recalculate PSFs and wave-functions
    def set_wave_scheme(self, scheme = "A"):
        self.PSF_scheme = scheme
        self.isotopes   = set_PSF_scheme(scheme, setglobal = False)
        self.PSFpanda   = pd.read_csv(cwd+"/../PSFs/PSFs_"+scheme+".csv")
        self.PSFpanda.set_index("PSFs", inplace = True)
    
    #This function sets the LECs
    def set_LECs(self, unknown_LECs):
        self.unknown_LECs = unknown_LECs
        if unknown_LECs == True:
            #self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
            #           "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
            #           # all the below are expected to be order 1 in absolute magnitude
            #           "Tprime":1, "Tpipi":1, "1piN":1, "6piN":1, "7piN":1, "8piN":1, "9piN":1, "VLpiN":1, "TpiN":1, 
            #           "1NN":1, "6NN":1, "7NN":1, "VLNN":1, "TNN": 1, "VLE":1, "VLme":1, "VRE":1, "VRme":1, 
            #           # all the below are expected to be order (4pi)**2 in absolute magnitude
            #           "2NN":(4*np.pi)**2, "3NN":(4*np.pi)**2, "4NN":(4*np.pi)**2, "5NN":(4*np.pi)**2, 
            #           # expected to be 1/F_pipi**2 pion decay constant
            #           "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
            #          }

            self.LEC = LECs.copy()
            
            self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
            self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
            
        else:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":0, "Tpipi":0, "1piN":0, "6piN":0, "7piN":0, "8piN":0, "9piN":0, "VLpiN":0, "TpiN":0, 
                       "1NN":0, "6NN":1, "7NN":1, "VLNN":0, "TNN": 0, "VLE":0, "VLme":0, "VRE":0, "VRme":0, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":0, "3NN":0, "4NN":0, "5NN":0, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            self.LEC["VpiN"] = 1         #LEC["6piN"] + LEC["8piN"]
            self.LEC["tildeVpiN"] = 1    #LEC["7piN"] + LEC["9piN"]
    
    #change NME method and return NMEs
    def change_method(self, method = None, inplace = True):
        method = self._set_method_locally(method)
        if method != self.method:
            newNMEs, newNMEpanda, newNMEnames = Load_NMEs(method)
            print("Changing method to "+method)
            if inplace:
                self.NME = newNMEs.copy()
                self.NMEpanda = newNMEpanda.copy()
                self.NMEname = newNMEnames.copy()
            return(newNMEs, newNMEpanda, newNMEnames)
        else:
            return(self.NMEs, self.NMEpanda, self.NMEnames)
    
    #switch NME method locally within a function
    def _set_method_locally(self, method):
        #set the method and import NMEs if necessary
        if method == None:
            method = self.method
            pass
        elif method != self.method and method+".csv" in os.listdir(cwd+"/../NMEs/"):
            pass
        elif method+".csv" not in os.listdir(cwd+"/../NMEs/"):
            warnings.warn("Method",method,"is unavailable. No such file 'NMEs/"+method+".csv'. Keeping current method",self.method)
            method = self.method
        else:
            method = self.method
            pass
        return(method)
    
    #change operator basis
    def change_basis(self, WC = None, basis = None, inplace = True):
    #this functions lets you switch between the C and the epsilon basis
    #if you only want to see the translation and dont want the change to be saved you can set inplace=False
        if basis == None:
            basis = self.basis
            
        if basis in ["C" ,"c"]:
            if WC == None:
                WC = self.CWC
            else:
                for operator in self.CWC:
                    if operator not in WC:
                        WC[operator] = 0
            New_WCs = self.EpsilonWC.copy()
            for operator in New_WCs:
                New_WCs[operator] = 0
                
            #standard mechanism
            New_WCs["m_bb"]   = WC["m_bb"]
                
            #long-range mechanisms
            New_WCs["V+AV+A"] = 1/2 * WC["VR(6)"]
            New_WCs["V+AV-A"] = 1/2 * WC["VL(6)"]
            New_WCs["S+PS+P"] = 1/2 * WC["SR(6)"]
            New_WCs["S+PS-P"] = 1/2 * WC["SL(6)"]
            New_WCs["TRTR"]   = 1/2 * WC["T(6)"]
            
            #short-range scalar
            New_WCs["1LLL"]   =  self.m_N/self.vev * (1/2*WC["2L(9)"] - 1/4*WC["3L(9)"])
            New_WCs["1LLR"]   =  self.m_N/self.vev * (1/2*WC["2R(9)"] - 1/4*WC["3R(9)"])
            New_WCs["1RRR"]   =  self.m_N/self.vev * (1/2*WC["2R(9)prime"] - 1/4*WC["3R(9)prime"])
            New_WCs["1RRL"]   =  self.m_N/self.vev * (1/2*WC["2L(9)prime"] - 1/4*WC["3L(9)prime"])
            New_WCs["1RLL"]   = -self.m_N/self.vev * WC["5L(9)"]
            New_WCs["1RLR"]   = -self.m_N/self.vev * WC["5R(9)"]
            
            New_WCs["2LLL"]   = -self.m_N/(16*self.vev) * WC["3L(9)"]
            New_WCs["2RRL"]   = -self.m_N/(16*self.vev) * WC["3L(9)prime"]
            New_WCs["2LLR"]   = -self.m_N/(16*self.vev) * WC["3R(9)"]
            New_WCs["2RRR"]   = -self.m_N/(16*self.vev) * WC["3R(9)prime"]
            
            New_WCs["3LLL"]   =  self.m_N/(2*self.vev) * WC["1L(9)"]
            New_WCs["3LLR"]   =  self.m_N/(2*self.vev) * WC["1R(9)"]
            New_WCs["3RRL"]   =  self.m_N/(2*self.vev) * WC["1L(9)prime"]
            New_WCs["3RRR"]   =  self.m_N/(2*self.vev) * WC["1R(9)prime"]
            
            New_WCs["3RLL"]   =  self.m_N/(2*self.vev) * WC["4L(9)"]
            New_WCs["3RLR"]   =  self.m_N/(2*self.vev) * WC["4R(9)"]
            
            #short-range vector
            New_WCs["4LLR"]   = 1j*self.m_N/self.vev * WC["9(9)"]
            New_WCs["4RRR"]   = 1j*self.m_N/self.vev * WC["9(9)prime"]
            New_WCs["4LRR"]   = -1j*self.m_N/self.vev * WC["7(9)"]
            New_WCs["4RLR"]   = -1j*self.m_N/self.vev * WC["7(9)prime"]
            
            New_WCs["5LRR"]   =  self.m_N/self.vev * (WC["6(9)"] - 5/3*WC["7(9)"])
            New_WCs["5RLR"]   =  self.m_N/self.vev * (WC["6(9)prime"] - 5/3*WC["7(9)prime"])
            New_WCs["5LLR"]   =  self.m_N/self.vev * (WC["8(9)"] - 5/3*WC["9(9)"])
            New_WCs["5RRR"]   =  self.m_N/self.vev * (WC["8(9)prime"] - 5/3*WC["9(9)prime"])
            
            #dim 7 long-range
            New_WCs["VL(7)"]  = WC["VL(7)"]
            New_WCs["VR(7)"]  = WC["VR(7)"]
        
        elif basis in ["Epsilon", "epsilon", "E", "e"]:
            if WC == None:
                WC = self.EpsilonWC
            else:
                for operator in self.EpsilonWC:
                    if operator not in WC:
                        WC[operator] = 0
            New_WCs = self.CWC.copy()
            for operator in New_WCs:
                New_WCs[operator] = 0
                
            #long-range matching
            New_WCs["m_bb"]  = WC["m_bb"]
            New_WCs["SL(6)"] = 2*WC["S+PS-P"]
            New_WCs["SR(6)"] = 2*WC["S+PS+P"]
            New_WCs["VL(6)"] = 2*WC["V+AV-A"]
            New_WCs["VR(6)"] = 2*WC["V+AV+A"]
            New_WCs["T(6)"]  = 2*WC["TRTR"]            
            New_WCs["VL(7)"]  = WC["VL(7)"]
            New_WCs["VR(7)"]  = WC["VR(7)"]
            
            #short-range matching
            New_WCs["1L(9)"]      = 2*self.vev/self.m_N * WC["3LLL"]
            New_WCs["1R(9)"]      = 2*self.vev/self.m_N * WC["3LLR"]
            New_WCs["1L(9)prime"] = 2*self.vev/self.m_N * WC["3RRL"]
            New_WCs["1R(9)prime"] = 2*self.vev/self.m_N * WC["3RRR"]
            
            New_WCs["2L(9)"]      = 2*self.vev/self.m_N * (WC["1LLL"] - 4*WC["2LLL"])
            New_WCs["2R(9)"]      = 2*self.vev/self.m_N * (WC["1LLR"] - 4*WC["2LLR"])
            New_WCs["2L(9)prime"] = 2*self.vev/self.m_N * (WC["1RRL"] - 4*WC["2RRL"])
            New_WCs["2R(9)prime"] = 2*self.vev/self.m_N * (WC["1RRR"] - 4*WC["2RRR"])
            
            New_WCs["3L(9)"]      = -16*self.vev/self.m_N * WC["2LLL"]
            New_WCs["3R(9)"]      = -16*self.vev/self.m_N * WC["2LLR"]
            New_WCs["3L(9)prime"] = -16*self.vev/self.m_N * WC["2RRL"]
            New_WCs["3R(9)prime"] = -16*self.vev/self.m_N * WC["2RRR"]
            
            New_WCs["4L(9)"]      = 2*self.vev/self.m_N * (WC["3RLL"] + WC["3LRL"]) #including redundancies
            New_WCs["4R(9)"]      = 2*self.vev/self.m_N * (WC["3RLR"] + WC["3LRR"]) #including redundancies
            
            New_WCs["5L(9)"]      = -self.vev/self.m_N * (WC["1RLL"] + WC["1LRL"]) #including redundancies
            New_WCs["5R(9)"]      = -self.vev/self.m_N * (WC["1RLR"] + WC["1LRR"]) #including redundancies
            
            New_WCs["6(9)"]       = self.vev/self.m_N *(   WC["5LRR"] + 1j*5/3 * WC["4LRR"] 
                                              -( WC["5LRL"] + 1j*5/3 * WC["4LRL"])) #including redundancies
            New_WCs["6(9)prime"]  = self.vev/self.m_N *(   WC["5RLR"] + 1j*5/3 * WC["4RLR"]
                                              -( WC["5RLL"] + 1j*5/3 * WC["4RLL"])) #including redundancies
            
            New_WCs["7(9)"]       = 1j*self.vev/self.m_N * (WC["4LRR"] - WC["4LRL"]) #including redundancies
            New_WCs["7(9)prime"]  = 1j*self.vev/self.m_N * (WC["4RLR"] - WC["4RLL"]) #including redundancies
            
            New_WCs["8(9)"]       = self.vev/self.m_N *(   WC["5LLR"] - 1j*5/3 * WC["4LLR"]
                                              -( WC["5LLL"] - 1j*5/3 * WC["4LLL"])) #including redundancies
            New_WCs["8(9)prime"]  = self.vev/self.m_N *(   WC["5RRR"] - 1j*5/3 * WC["4RRR"]
                                              -( WC["5RRL"] - 1j*5/3 * WC["4RRL"])) #including redundancies
            
            New_WCs["9(9)"]       = -1j*self.vev/self.m_N * (WC["4LLR"] - WC["4LLL"]) #including redundancies
            New_WCs["9(9)prime"]  = -1j*self.vev/self.m_N * (WC["4RRR"] - WC["4RRL"]) #including redundancies
            
            
        else:
            print("Unknown basis",basis)
            return
        
        if inplace:
            if basis in ["C" ,"c"]:
                self.basis     = "e"
                self.EpsilonWC = New_WCs
                self.CWC       = WC
                self.WC = self.CWC.copy()
            else:
                self.basis     = "C"
                self.CWC       = New_WCs
                self.EpsilonWC = WC
                self.WC = self.CWC.copy()
                
        return(New_WCs)
                

    #the primed operators as well as the LR run the same
    #use the previously calculated running matrix to run between m_W and Lambda_chi
    def _run(self, WC = None, updown = "down", inplace = False):
        if WC == None:
            WC = self.WC.copy()
        else:
            WCcopy = WC.copy()
            WC = self.WC.copy()
            
            #set 
            for operator in WC:
                WC[operator] = 0
                
            #overwrite with new WCs
            for operator in WCcopy:
                WC[operator] = WCcopy[operator]
        if updown == "down":
            new_WC_values = matrix@np.array(list(WC.values()))
        else:
            new_WC_values = np.linalg.inv(matrix)@np.array(list(WC.values()))
        new_WC = {}
        for idx in range(len(WC)):
            operator = list(WC.keys())[idx]
            new_WC[operator] = new_WC_values[idx]
            
        if inplace:
            self.WC = new_WC
            
        return(new_WC)
        
    #run from and to an arbitrary scale
    def run(self, WC = None, initial_scale = m_W, final_scale = lambda_chi, inplace = False):
    ######################################################################################
    #
    #Define RGEs as differential equations and solve them numerically for the given scales
    #
    ######################################################################################
        if WC == None:
            WC = self.WC.copy()
        else:
            #make a dict with all WCs
            WCs = LEFT_WCs.copy()
            
            #write the non-zero WCs
            for C in WC:
                WCs[C] = WC[C]
            WC = WCs
            pass
        
        new_WCs = run_LEFT(WC, initial_scale=initial_scale, final_scale = final_scale)
        
        #store WCs locally in LEFT class
        if inplace:
            self.WC = new_WCs.copy()
            
        return new_WCs
    
    
    '''
        ####################################################################################################
        
        Define necessary functions to calculate observables
        1. half_lives
        2. hl_ratios
        3. PSF observables
        
        ####################################################################################################
    '''
    
    
    ####################################################################################################
    #                                                                                                  #
    #                                   Half-live calculations                                         #
    #                                                                                                  #
    ####################################################################################################
        
    def t_half(self, isotope, WC = None, method = None):
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
            
        if WC == None:
            WC = self.WC.copy()
            
        #Calculates the half-live for a given isotope and WCs
        amp, M = self.amplitudes(isotope, WC, method)
        element = self.isotopes[isotope]
        
        g_A=self.LEC["A"]
        
        G = self.to_G(isotope)

        #Some PSFs need a rescaling due to different definitions in DOIs paper and 1806...
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2
        G["06"] *= g_06_rescaling
        G["04"] *= g_04_rescaling
        G["09"] *= g_09_rescaling

        #Calculate half-life following eq 38. in 1806.02780
        inverse_result = g_A**4*(G["01"] * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                          - 2 * (G["01"] - G["04"])*(np.conj(amp["nu"])*amp["R"]).real
                          + 4 *  G["02"]* np.absolute(amp["E"])**2
                          + 2 *  G["04"]*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                          - 2 *  G["03"]*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
                          + G["09"] * np.absolute(amp["M"])**2
                          + G["06"] * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)
        return(1/inverse_result)
    
    def amplitudes(self, isotope, WC = None, method=None):
    #calculate transition amplitudes as given in 1806.02780
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
        NMEs, _, __ = self.change_method(method, inplace = False)
        NME = NMEs[isotope].copy()
        
        #set WCs
        if WC == None:
            C = self.WC.copy()
        else:
            C = self.WC.copy()
            
            for x in C:
                C[x] = 0
            for x in WC:
                C[x] = WC[x]
        LEC = self.LEC.copy()

        #generate params (WCs, NMEs) that enter the ME calculation

        #right below eq 21
        C["V(9)"]      = C["6(9)"] + C["8(9)"] + C["6(9)prime"] + C["8(9)prime"]
        C["tildeV(9)"] = C["7(9)"] + C["9(9)"] + C["7(9)prime"] + C["9(9)prime"]

        #eq.25
        C["pipiL(9)"]  = (  LEC["2pipi"]*(C["2L(9)"] + C["2L(9)prime"]) 
                          + LEC["3pipi"]*(C["3L(9)"] + C["3L(9)prime"])
                          - LEC["4pipi"]*C["4L(9)"]
                          - LEC["5pipi"]*C["5L(9)"]
                          - 5/3*(m_pi**2) * LEC["1pipi"]*(  C["1L(9)"] 
                                                       + C["1L(9)prime"]))

        C["pipiR(9)"]  = (  LEC["2pipi"]*(C["2R(9)"] + C["2R(9)prime"]) 
                          + LEC["3pipi"]*(C["3R(9)"] + C["3R(9)prime"])
                          - LEC["4pipi"]*C["4R(9)"]
                          - LEC["5pipi"]*C["5R(9)"]
                          - 5/3*(m_pi**2) * LEC["1pipi"]*(C["1R(9)"] + C["1R(9)prime"]))

        C["piNL(9)"]   = (LEC["1piN"] - 5/6*LEC["1pipi"])*(C["1L(9)"] + C["1L(9)prime"])
        C["piNR(9)"]   = (LEC["1piN"] - 5/6*LEC["1pipi"])*(C["1R(9)"] + C["1R(9)prime"])

        C["NNL(9)"]    = (  LEC["1NN"]*(C["1L(9)"] + C["1L(9)prime"]) 
                          + LEC["2NN"]*(C["2L(9)"] + C["2L(9)prime"])
                          + LEC["3NN"]*(C["3L(9)"] + C["3L(9)prime"])
                          + LEC["4NN"]*C["4L(9)"] 
                          + LEC["5NN"]*C["5L(9)"])
        C["NNR(9)"]    = (  LEC["1NN"]*(C["1R(9)"] + C["1R(9)prime"]) 
                          + LEC["2NN"]*(C["2R(9)"] + C["2R(9)prime"])
                          + LEC["3NN"]*(C["3R(9)"] + C["3R(9)prime"])
                          + LEC["4NN"]*C["4R(9)"] 
                          + LEC["5NN"]*C["5R(9)"])


        #eq. 33
        NME["T"] = NME["TAP"] + NME["TPP"] + NME["TMM"]
        NME["GT"] = NME["GTAA"] + NME["GTAP"] + NME["GTPP"] + NME["GTMM"]
        NME["PS"] = 0.5*NME["GTAP"] + NME["GTPP"] + 0.5*NME["TAP"] + NME["TPP"]
        NME["T6"] = (2 *(LEC["Tprime"] - LEC["TNN"])/LEC["A"]**2 * m_pi**2/self.m_N**2*NME["F,sd"] 
                     - 8*LEC["T"]/LEC["M"] * (NME["GTMM"] + NME["TMM"])
                     + LEC["TpiN"] * m_pi**2/(4*self.m_N**2)*(NME["GTAP,sd"] + NME["TAP,sd"])
                     + LEC["Tpipi"] * m_pi**2/(4*self.m_N**2)*(NME["GTPP,sd"] + NME["TPP,sd"]))


        #store MEs in dictionary to return
        M= {}

        #Matrix Elements

        #eq. 30
        M["nu(3)"] = -V_ud**2*(-                       1/g_A**2 * NME["F"] 
                               +                                  NME["GT"] 
                               +                                  NME["T"] 
                               + 2*m_pi**2 * LEC["nuNN"]/g_A**2 * NME["F,sd"])


        #eq. 31
        M["nu(6)"] = (  V_ud * (         LEC["B"]/self.m_N * (C["SL(6)"] - C["SR(6)"]) 
                                + m_pi**2/(self.m_N * self.vev) * (C["VL(7)"] - C["VR(7)"]))*NME["PS"] 
                      + V_ud * C["T(6)"] * NME["T6"])

        #eq. 32
        M["nu(9)"] = (      (-1/(2*self.m_N**2) * C["pipiL(9)"]) * (  1/2 * NME["GTAP,sd"] 
                                                               +       NME["GTPP,sd"] 
                                                               + 1/2 * NME["TAP,sd"] 
                                                               +       NME["TPP,sd"]) 
                      + m_pi**2/(2*self.m_N**2) * C["piNL(9)"]   * (        NME["GTAP,sd"] 
                                                               +       NME["TAP,sd"])
                      - 2/g_A**2 * m_pi**2/self.m_N**2 * C["NNL(9)"]      * NME["F,sd"])

        #equal to eq. 32 but L --> R see eq.34
        M["R(9)"] = (       (-1/(2*self.m_N**2) * C["pipiR(9)"]) * (  1/2 * NME["GTAP,sd"] 
                                                    +                  NME["GTPP,sd"] 
                                                    +            1/2 * NME["TAP,sd"] 
                                                    +                  NME["TPP,sd"]) 
                      + m_pi**2/(2*self.m_N**2) * C["piNR(9)"]   * (        NME["GTAP,sd"] 
                                                               +       NME["TAP,sd"])
                      - 2/g_A**2 * m_pi**2/self.m_N**2 * C["NNR(9)"]      * NME["F,sd"])

        #eq. 35
        M["EL(6)"] = -V_ud * C["VL(6)"]/3 * (  g_V**2/g_A**2 *       NME["F"] 
                                             +           1/3 * ( 2 * NME["GTAA"] 
                                                                +    NME["TAA"]) 
                                             + 6*LEC["VLE"]/g_A**2 * NME["F,sd"])

        M["ER(6)"] = -V_ud * C["VR(6)"]/3 * (  g_V**2/g_A**2 *       NME["F"] 
                                             - 1/3           * ( 2 * NME["GTAA"] 
                                                                +    NME["TAA"]) 
                                             + 6*LEC["VRE"]/g_A**2 * NME["F,sd"])

        M["meL(6)"] = V_ud*C["VL(6)"]/6 * (  g_V**2/g_A**2   *       NME["F"] 
                                           -           1/3   * (     NME["GTAA"] 
                                                                - 4* NME["TAA"]) 
                                           -             3   * (     NME["GTAP"] 
                                                                +    NME["GTPP"] 
                                                                +    NME["TAP"] 
                                                                +    NME["TPP"])
                                           - 12*LEC["VLme"]/g_A**2 * NME["F,sd"])

        M["meR(6)"] = V_ud*C["VR(6)"]/6 * (  g_V**2/g_A**2 *        NME["F"] 
                                           +           1/3 * (      NME["GTAA"] 
                                                              - 4 * NME["TAA"]) 
                                           +             3 * (      NME["GTAP"] 
                                                              +     NME["GTPP"] 
                                                              +     NME["TAP"] 
                                                              +     NME["TPP"])
                                           - 12*LEC["VRme"]/g_A**2 *NME["F,sd"])

        #eq. 36
        M["M(6)"] = V_ud*C["VL(6)"] * (  2*g_A/LEC["M"] * (                           NME["GTMM"] 
                                                           +                          NME["TMM"]) 
                                       + m_pi**2/self.m_N**2 * (- 2/g_A**2*LEC["VLNN"] *   NME["F,sd"]
                                                           + 1/2*LEC["VLpiN"]     *(  NME["GTAP,sd"] 
                                                                                    + NME["TAP,sd"])))

        M["M(9)"] = m_pi**2/self.m_N**2 * (-2/g_A**2*(LEC["6NN"]*C["V(9)"] + LEC["7NN"]*C["tildeV(9)"])    *    NME["F,sd"] 
                                      + 1/2*(LEC["VpiN"]*C["V(9)"] + LEC["tildeVpiN"]*C["tildeV(9)"]) * (  NME["GTAP,sd"] 
                                                                                                         + NME["TAP,sd"]))





        #generate subamplitudes
        #eq. 29
        A={}
        A["nu"] = (C["m_bb"]/self.m_e * M["nu(3)"] 
                   + self.m_N/self.m_e * M["nu(6)"] 
                   + self.m_N**2/(self.m_e*self.vev) * M["nu(9)"])
        
        A["R"] = self.m_N**2/(self.m_e*self.vev) * M["R(9)"]
        
        A["E"] = M["EL(6)"] + M["ER(6)"]
        
        A["me"] = M["meL(6)"] + M["meR(6)"]
        
        A["M"] = self.m_N/self.m_e * M["M(6)"] + self.m_N**2/(self.m_e*self.vev) * M["M(9)"]


        #return subamplitudes and MEs
        return (A, M)

    def to_G(self, isotope):
        #transform imported PSFs from dataframe into dict
        #format, which is used in the amplitudes function.
        G = {}
        for key in self.PSFpanda[isotope].keys():
            G[key[1:]] = self.PSFpanda[isotope][key]
        return(G)
    
    ####################################################################################################
    #                                                                                                  #
    #                                       PSF observables                                            #
    #                                                                                                  #
    ####################################################################################################
    
    def spectrum(self, 
                 Ebar,             #normalized energy E/Q
                 isotope = "76Ge", #isotope of interest
                 WC      = None,   #you can reset the WCs here @ Lambda_chi
                 method  = None    #you can choose a different NME method if you want
                ):
        #calculates the single electron spectrum (not normalized)        
        if WC == None:
            WC = self.WC.copy()
            
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
        
        #get isotope class
        element = self.isotopes[isotope]
        
        #calculate amplitudes
        amp = self.amplitudes(isotope, WC = WC, method = method)[0]
        
        #Mass difference between mother and daughter nuclei in MeV
        Delta_M = element.Delta_M
        
        #Energy from normalized Energy scale
        E = Ebar * (Delta_M - 2*self.m_e_MEV) + self.m_e_MEV
        
        #rescale PSFs due to different definitions in 1806.02780
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2
        
        #define prefactor
        prefactor = 2*g_A**4*(G_F*V_ud)**4*m_e**2 / (64*np.pi**5*(element.R*1000)**2) #*1000 to adjust 1/MeV to 1/GeV

        def p(E, m = self.m_e_MEV):
            return(np.sqrt(E**2 - m**2))

        result = (element.g_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
                  - 2 * (element.g_01(E, Delta_M - E) - g_04_rescaling * element.g_04(E, Delta_M - E))*(np.conj(amp["nu"])*amp["R"]).real
                  + 4 *  element.g_02(E, Delta_M - E)* np.absolute(amp["E"])**2
                  + 2 *  g_04_rescaling * element.g_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
                  - 2 *  element.g_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
                  + g_09_rescaling * element.g_09(E, Delta_M - E) * np.absolute(amp["M"])**2
                  + g_06_rescaling * element.g_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)* p(E)*p(Delta_M-E)* E * (Delta_M - E)
        return(prefactor*result*1e-12*MeV_to_inverseyear)  
        #1e-12 to compensates for p, E in MeV and output in 1/[yr MeV]
        #g_0k is dimensionless
    

    def angular_corr(self, 
                     Ebar,              #normalized energy E/Q
                     isotope = "76Ge",  #isotope of interest
                     WC      = None,    #you can reset the WCs here @ Lambda_chi
                     method  = None     #you can choose a different NME method if you want
                    ):
        #calculates the angular correlation coefficient for a given normalized electron energy Ebar
        if WC == None:
            WC = self.WC.copy()
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
        
        #get isotope class
        element = self.isotopes[isotope]
        
        #calculate amplitudes
        amp = self.amplitudes(isotope, WC = WC, method = method)[0]
        
        #Mass difference between mother and daughter nuclei in MeV
        Delta_M = element.Delta_M
        
        #Energy from normalized Energy scale
        E = Ebar * (Delta_M - 2*self.m_e_MEV) + self.m_e_MEV
        
        #rescale PSFs due to different definitions in 1806.02780
        g_06_rescaling = self.m_e_MEV*element.R/2
        g_09_rescaling = g_06_rescaling**2
        g_04_rescaling = 9/2

        hs = (element.h_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
              - 2 * (element.h_01(E, Delta_M - E) - g_04_rescaling * element.h_04(E, Delta_M - E))*(np.conj(amp["nu"])*amp["R"]).real
              + 4 *  element.h_02(E, Delta_M - E)* np.absolute(amp["E"])**2
              + 2 *  g_04_rescaling * element.h_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
              - 2 *  element.h_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
              + g_09_rescaling * element.h_09(E, Delta_M - E) * np.absolute(amp["M"])**2
              + g_06_rescaling * element.h_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)

        
        
        gs = (element.g_01(E, Delta_M - E) * (np.absolute(amp["nu"])**2 + np.absolute(amp["R"])**2)
              - 2 * (element.g_01(E, Delta_M - E) - g_04_rescaling * element.g_04(E, Delta_M - E)) * (np.conj(amp["nu"])*amp["R"]).real
              + 4 *  element.g_02(E, Delta_M - E)* np.absolute(amp["E"])**2
              + 2 *  g_04_rescaling * element.g_04(E, Delta_M - E)*(np.absolute(amp["me"])**2 + (np.conj(amp["me"])*(amp["nu"]+amp["R"])).real)
              - 2 *  element.g_03(E, Delta_M - E)*((amp["nu"]+amp["R"])*np.conj(amp["E"]) + 2*amp["me"]*np.conj(amp["E"])).real
              + g_09_rescaling * element.g_09(E, Delta_M - E) * np.absolute(amp["M"])**2
              + g_06_rescaling * element.g_06(E, Delta_M - E) * ((amp["nu"]-amp["R"])*np.conj(amp["M"])).real)
        return (hs/gs)
    
    
    '''
        ####################################################################################################
        
        Define outputting functions for
        1. half_lives
        2. hl_ratios
        3. PSF observables
        
        ####################################################################################################
        
    '''
    
    def _vary_LECs(self, inplace = False): 
    #this function varies the unknown LECs within the appropriate range
        LECs = {}
        for LEC in LECs_unknown:
            if LEC == "nuNN":
                random_LEC = (np.random.rand()+0.5)*LECs_unknown[LEC]
                LECs[LEC] = random_LEC
            else:
                random_LEC = (np.random.choice([1,-1])
                              *((np.sqrt(10)-1/np.sqrt(10))
                                *np.random.rand()+1/np.sqrt(10)
                               )
                              *LECs_unknown[LEC]
                             )
            LECs[LEC] = random_LEC
            
        #set LECs that depend on others
        LECs["VpiN"] = LECs["6piN"] + LECs["8piN"]
        LECs["tildeVpiN"] = LECs["7piN"] + LECs["9piN"]
        if inplace:
            for LEC in LECs:
                self.LEC[LEC] = LECs[LEC]
        else:
            return(LECs)
    
    def half_lives(self, 
                   WC        = None,      #you can reset the WCs here @ Lambda_chi
                   method    = None,      #you can choose a different NME method if you want
                   vary_LECs = False,     #If True the unknown LECs will ne varied n_points times
                   n_points  = 1000       #Number of variations of the unknown LECs
                  ):
        #returns a pandas.DataFrame with all half-lives of the available isotopes for the considered NME method
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
        NMEs, _, __ = self.change_method(method, inplace = False)
        
        #generate DataFrame to store results
        hl = pd.DataFrame([])#, [r"$y$"])
        
        
        #either vary over the unknown LECs or return half_lives for current LECs
        if vary_LECs:
            #generate a backup of the LECs
            LEC_backup = self.LEC.copy()
            hlarrays = {}
            for isotope in list(NMEs.keys()):
                hlarrays[isotope] = np.zeros(n_points)
                
            
            for idx in range(n_points):
                #for each step vary LECs
                self._vary_LECs(inplace = True)
                
                #calculate half_lives in each isotope for given LECs
                for isotope in list(NMEs.keys()):
                    hlarrays[isotope][idx] = self.t_half(isotope, WC, method)
                    
            #fill pandas DataFrame
            for isotope in list(NMEs.keys()):
                hl[isotope] = hlarrays[isotope]
            self.LEC = LEC_backup.copy()
            
        #fill pandas DataFrame for fixed LECs
        else:
            for isotope in list(NMEs.keys()):
                hl[isotope] = [self.t_half(isotope, WC, method)]
        
        return(hl)
    
    
    def ratios(self, 
               reference_isotope = "76Ge",  #half-lives are normalized with regard to this isotopes
               normalized        = True,    #if True normalize ratios to the standard mass mechnism
               WC                = None,    #you can reset the WCs here @ Lambda_chi
               method            = None,    #you can choose a different NME method if you want
               vary_LECs         = False,   #If True the unknown LECs will ne varied n_points times
               n_points          = 100      #Number of variations of the unknown LECs
              ):
        #returns the half-live ratios compared to the standard mass mechanism based on the chosen reference isotope
        if WC == None:
            WC = self.WC.copy()
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
        NMEs, _, __ = self.change_method(method, inplace = False)
        #NME = NMEs[isotope][method].copy()
            
        

        #generate WC dict for mass mechanism
        WC_mbb = self.WC.copy()
        for operator in WC_mbb:
            WC_mbb[operator] = 0
            
        WC_mbb["m_bb"] = 1e-9 #1eV
        
        #normalized to mass mechanism?
        if normalized:
            #get a random seed
            seed = np.random.randint(0, 2**31)
            
            #seed random numbers to have the same for model and mass mechanism
            np.random.seed(seed)
            half_lives      = self.half_lives(vary_LECs = vary_LECs, 
                                              WC = WC,
                                              n_points = n_points)
            np.random.seed(seed)
            half_lives_mass = self.half_lives(vary_LECs = vary_LECs,
                                              n_points = n_points,
                                              WC = {"m_bb" : 1})
            
            ratios_model = half_lives.divide(half_lives[reference_isotope], axis = 0)
            ratios_mass  = half_lives_mass.divide(half_lives_mass[reference_isotope], axis = 0)
            
            ratios = ratios_model.divide(ratios_mass, axis = 0)
        else:
            half_lives = self.half_lives(vary_LECs = vary_LECs, 
                                         WC = WC,
                                         n_points = n_points)
            ratios = half_lives.divide(half_lives[reference_isotope], axis = 0)
        return(ratios)
        
    def plot_ratios(self, 
                    reference_isotope = "76Ge",        #half-lives are normalized with regard to this isotopes
                    normalized        = True,          #if True normalize ratios to the standard mass mechnism
                    show_central      = True,          #Show the median values of the variation
                    WC                = None,          #You can reset the WCs here @ Lambda_chi
                    method            = None,          #Choose a different NME method if desired
                    vary_LECs         = False,         #If True vary the unknown LECs
                    n_points          = 100,           #number of points to plot in the variation
                    alpha             = 0.25,          #alpha value of the datapoints
                    color             = "b",           #Color of the plotted datapoints
                    addgrid           = True,          #If True plot a grid
                    savefig           = False,         #If True save figure as file
                    file              = "ratios.png",  #Filename and path to store figure to
                    dpi               = 300            #set the resolution of the saved figure in dots per inch
                   ):
        #This function generates a scatter plot of the half-life ratios in all isotopes
        
        #set the color
        c = color

        #set the marker
        m = "x"
        
        #get ratios
        if vary_LECs:
            ratios_varied = self.ratios(reference_isotope = reference_isotope, 
                                        normalized = normalized, 
                                        WC = WC, method = method, 
                                        vary_LECs = vary_LECs, 
                                        n_points = n_points)
            if show_central:
                ratios = pd.DataFrame(ratios_varied.median()).T
            else:
                ratios = self.ratios(reference_isotope = reference_isotope, 
                                     normalized = normalized, 
                                     WC = WC, method = method, 
                                     vary_LECs = False)
        else:
            ratios = self.ratios(reference_isotope = reference_isotope, 
                                 normalized = normalized, 
                                 WC = WC, 
                                 method = method)
        
        #generate plot
        fig = plt.figure(figsize=(6.4*1.85, 4.8*2))
        plt.rc("ytick", labelsize = 20)
        plt.rc("xtick", labelsize = 20)
        plt.xlabel(r"$\frac{R^{\mathcal{O}_i}-R^{m_\nu}}{R^{m_\nu}}$", fontsize=30)
        for isotope in ratios:
            if isotope[0] != "1" and isotope[0] != "2":
                isotope_plot = isotope[0:2]
            else:
                isotope_plot = isotope[0:3]
            if isotope == ratios.keys()[0]:
                label = self.name
            else:
                label = None
            plt.scatter(np.log10(ratios[isotope]), isotope_plot, 
                marker = m, color = c, s=150, label = label)
            if vary_LECs:
                plt.scatter(np.log10(ratios_varied[isotope]), 
                            np.repeat(isotope_plot, len(ratios_varied[isotope])), 
                            marker = ".", color = c, s=20, alpha = alpha)
        
        
        if normalized:
            plt.axvline(0, color="k", linewidth=1, label=r"$m_{\beta\beta}$")
            plt.legend(loc="upper right", ncol=1, fontsize = 20)
            plt.xlabel(r"$\log_{10}\frac{R^{\mathcal{O}_i}}{R^{m_{\beta\beta}}}$", fontsize=30)
        else:
            plt.legend([self.name], loc="upper right", ncol=1, fontsize = 20)
            plt.xlabel(r"$\log_{10}R^{\mathcal{O}_i}$", fontsize=30)
        plt.ylabel("A", fontsize = 30)
        plt.tight_layout()
        if addgrid:
            plt.grid(linestyle = "--")
        if savefig:
            plt.savefig(file, dpi=dpi)
        return(fig)
    
    
    def PSF_plot(self, **args):
        #generates both plots of the PSF observables
        #i.e. angular corr. and single e spectrum
        
        #plot spectrum
        fig1 = self.plot_spec(**args)
        
        #plot angular corr
        fig2 = self.plot_corr(**args)
        return(fig1, fig2)
            
    def plot_spec(self, 
                  isotope     = "76Ge",      #isotope to study
                  WC          = None,        #Set WCs, if None self.WC will be used
                  method      = None,        #NME method, if None self.method will be used
                  print_title = False,       #add a title to the plot
                  addgrid     = True,        #If True, plot a grid
                  show_mbb    = True,        #If True, plot mass mechanism for comparison
                  n_points    = 1000,        #number of points to calculate
                  linewidth   = None,        #Width of lines in plot
                  normalize_x = True,        #If True normalize x-axis range [0, 1]
                  savefig     = False,       #If True save figure as file
                  file        = "spec.png",  #Filename and path to store figure to
                  dpi         = 300          #set the resolution of the saved figure in dots per inch
                 ):
        if WC == None:
            WC = self.WC.copy()
        #generates a plot of the single electron spectrum
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
            
        
        #generate WC dict for mass mechanism
        if show_mbb:
            WC_mbb = self.WC.copy()
            for operator in WC_mbb:
                WC_mbb[operator] = 0
            WC_mbb["m_bb"] = 1
        
        #necessary to avoid pole
        epsilon = 1e-6
        
        #get isotope class
        element = self.isotopes[isotope]
        
        #energy range for spectrum
        E = np.linspace(self.m_e_MEV+epsilon, element.Delta_M-self.m_e_MEV-epsilon, n_points)
        
        #normalized energy
        Ebar = (E-self.m_e_MEV)/(element.Delta_M-2*self.m_e_MEV)
        
        #normalization factors for single electron spectra
        integral = integrate.quad(lambda E: self.spectrum(E, isotope = isotope, WC = WC, method = method), 0, 1)
        if show_mbb:
            integral_mbb = integrate.quad(lambda E: self.spectrum(E, isotope = isotope, WC = WC_mbb, method = method), 0, 1)
        
        #set x axis values
        if normalize_x:
            x = Ebar
        else:
            x = E-m_e_MeV
            
        #generate figures
        fig = plt.figure(figsize=(6.4*1.85, 4.8*2))
        
        #set ticksize
        plt.rc("ytick", labelsize = 20)
        plt.rc("xtick", labelsize = 20)
        
        #generate a title for the plot
        if print_title:
            plt.title("Single Electron Spectrum")
            
        #normalized spectrum
        spec = self.spectrum(Ebar, WC = WC, isotope = isotope, method = method)/integral[0]
        
        #mass mechanism comparison
        if show_mbb:
            spec_mbb = self.spectrum(Ebar, WC = WC_mbb, isotope = isotope, method = method)/integral_mbb[0]
        if show_mbb:
            plt.plot(x, spec_mbb, "r", label = r"$m_{\beta\beta}$", linewidth = linewidth)
            
        #plot spectrum
        plt.plot(x, spec, "b", label = self.name, linewidth = linewidth)
        
        #generate legend
        plt.legend(fontsize = 20, loc="upper right")
        
        #set axis limits
        if normalize_x:
            plt.xlim(0,1)
        else:
            plt.xlim(0, element.Delta_M-2*self.m_e_MEV)
        if show_mbb:
            plt.ylim(0, 1.05*max(max(spec),max(spec_mbb)))
        else:
            plt.ylim(0, 1.05*max(spec))
            
        #set axis labels
        if normalize_x:
            plt.xlabel(r"$\overline{\epsilon}$", fontsize = 30)
        else:
            plt.xlabel(r"$T_e$ [MeV]", fontsize = 30)
            
        plt.ylabel(r"$1/\Gamma\; \mathrm{d}\Gamma/\mathrm{d}\epsilon$ [MeV$^{-1}]$", fontsize = 30)
        
        #add grid if desired
        if addgrid:
            plt.grid(linestyle = "--")
            
        #save figure as file
        if savefig:
            plt.savefig(file, dpi = dpi)
        return(fig)
            
    def plot_corr(self, 
                  isotope     = "76Ge",             #isotope to study
                  WC          = None,               #Set WCs, if None self.WC will be used
                  method      = None,               #NME method, if None self.method will be used
                  print_title = False,              #add a title to the plot
                  addgrid     = True,               #If True, plot a grid
                  show_mbb    = True,               #If True, plot mass mechanism for comparison
                  n_points    = 1000,               #number of points to calculate
                  linewidth   = None,               #Width of lines in plot
                  normalize_x = True,               #If True normalize x-axis range [0, 1]
                  savefig     = False,              #If True save figure as file
                  file        = "angular_corr.png", #Filename and path to store figure to
                  dpi         = 300                 #set the resolution of the saved figure in dots per inch
                 ):
        #generates a plot of the angular correlation coefficient
        
        if WC == None:
            WC = self.WC.copy()
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)        
        
        #generate WC dict for mass mechanism
        if show_mbb:
            WC_mbb = self.WC.copy()
            for operator in WC_mbb:
                WC_mbb[operator] = 0
            WC_mbb["m_bb"] = 1
        
        #necessary to avoid pole
        epsilon = 1e-6
        
        #get isotope class
        element = self.isotopes[isotope]
        
        #energy range for spectrum
        E = np.linspace(self.m_e_MEV+epsilon, element.Delta_M-self.m_e_MEV-epsilon, n_points)
        
        #normalized energy
        Ebar = (E-self.m_e_MEV)/(element.Delta_M-2*self.m_e_MEV)
        
        #set x axis values
        if normalize_x:
            x = Ebar
        else:
            x = E-m_e_MeV
        
        #generate figures            
        fig = plt.figure(figsize=(6.4*1.85, 4.8*2))
        
        #set ticksize
        plt.rc("ytick", labelsize = 20)
        plt.rc("xtick", labelsize = 20)
        
        if print_title:
            plt.title("Angular Correlation")
            
        #mass mechanism comparison
        if show_mbb:
            a_corr_mbb = self.angular_corr(Ebar, WC = WC_mbb, isotope = isotope, method = method)
        a_corr = self.angular_corr(Ebar, WC = WC, isotope = isotope, method = method)
        if show_mbb:
            plt.plot(x, a_corr_mbb, "r", label = r"$m_{\beta\beta}$", linewidth = linewidth)
            
        #plot angular correlation
        plt.plot(x, a_corr, "b", label = self.name, linewidth = linewidth)
        
        #generate legend
        plt.legend(fontsize=20, loc="upper right")
        
        #axis range
        if normalize_x:
            plt.xlim(0,1)
        else:
            plt.xlim(0, element.Delta_M-2*self.m_e_MEV)
        plt.ylim(-1,1)
        
        #axis labels
        if normalize_x:
            plt.xlabel(r"$\overline{\epsilon}$", fontsize = 30)
        else:
            plt.xlabel(r"$T_e$ [MeV]", fontsize = 30)
            
        plt.ylabel(r"$a_1/a_0$", fontsize = 30)
        
        #add grid if desired
        if addgrid:
            plt.grid(linestyle = "--")
            
        #save figure to file
        if savefig:
            plt.savefig(file, dpi = dpi)
        return(fig)
    
    def get_limits(self, 
                   half_life,             #experimental limit on the half-life
                   isotope      = "76Ge", #isotope the half-life limit was obtained for
                   method       = None,   #NME method can be changed here
                   groups       = False,  #show only groups of operators with the same limit (not for SMEFT)
                   basis        = None,   #operator basis (epsilon or C)
                   scale        = "up",   #scale the limits should be obtained at ("up" = m_W, "down" = lambd_chi)
                  ):
        #this function can calculate the limits on the different LEFT coefficients for a given experimental half_life and isotope
        #the limits are calculate at the scale "scale" and for the chosen basis
        #we kept this function here (additional to the one in the functions.py file) to make it easier to obtain
        #limits after changing PSFs, LECs or NMEs
        
        #set the method and import NMEs if necessary
        method = self._set_method_locally(method)
            
        if basis == None:
            basis == self.basis
        
        results_2GeV = {}
        results = {}
        scales = {}

        #make a backup so you can overwrite the running afterwards
        WC_backup = self.WC.copy()

        #calculate the limits on the WCs at the scale of 2GeV

        #put operators into groups with the same limit
        if groups:
            if basis in [None, "C", "c"]:
                WCgroups = ["m_bb" , "VL(6)", 
                            "VR(6)", "T(6)", 
                            "SL(6)", "VL(7)", 
                            "1L(9)", 
                            "2L(9)", "3L(9)", 
                            "4L(9)", "5L(9)", 
                            "6(9)", "7(9)"]
                #define labels for plots
                WC_names = {"m_bb"  : "m_bb", 
                            "VL(6)" : "VL(6)", 
                            "VR(6)" : "VR(6)", 
                            "T(6)"  : "T(6)" , 
                            "SL(6)" : "S(6)", 
                            "VL(7)" : "V(7)"     , 
                            "1L(9)" : "S1(9)",
                            "2L(9)" : "S2(9)", 
                            "3L(9)" : "S3(9)", 
                            "4L(9)" : "S4(9)", 
                            "5L(9)" : "S5(9)",
                            "6(9)"  : "V(9)", 
                            "7(9)"  : "Vtilde(9)"
                           }
            else:
                WCgroups = ["m_bb",
                            "V+AV-A",
                            "V+AV+A",
                            "S+PS+P",
                            "TRTR", 
                            "VL(7)", 
                            "1LLL", 
                            "1LRL", 
                            "2LLL",
                            "3LLL", 
                            "3LRL", 
                            "4LLL", 
                            "5LLL"
                           ]
                #define labels for plots
                WC_names = {"m_bb"   : "m_bb", 
                            "V+AV-A" : "V+AV-A", 
                            "V+AV+A" : "V+AV+A", 
                            "S+PS+P" : "S+P_X" , 
                            "TRTR"   : "TRTR", 
                            "VL(7)"  : "V(7)", 
                            "1LLL"   : "1XX",
                            "1LRL"   : "1XY",
                            "2LLL"   : "2XX", 
                            "3LLL"   : "3XX", 
                            "3LRL"   : "3XY", 
                            "4LLL"   : "4XX,XY",
                            "5LLL"   : "5XX,XY"
                           }

            #iterate over relevant WCs (one per group)
            for WC_name in WCgroups:
                #run WC down from high scale if desired
                if scale in ["up", "mW", "m_W", "MW", "M_W"]:
                    #run the WCs down to chipt to get limits on the WCs @ m_W
                    if basis not in [None, "C", "c"]:
                        WC = self.change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                    else:
                        WC = {WC_name : 1}
                    WC = self._run(WC, updown = "down")
                else:
                    if basis not in [None, "C", "c"]:
                        WC = change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                    else:
                        WC = {WC_name : 1}

                #calculate half-life for WC = 1
                hl = self.t_half(WC = WC, method = method, isotope = isotope)

                #get limit on WC
                results[WC_name] = np.sqrt(hl/half_life)

        #get limits for each individual operator
        else:
            if basis in [None, "C", "c"]:
                WCs = self.WC.copy()
            else:
                WCs = self.EpsilonWC.copy()
            for WC_name in WCs:
                #if limits @ high scale are desired run operator down
                if scale in ["up", "mW", "m_W", "MW", "M_W"]:
                    #running only works for C-basis, translate epsilon to C first
                    if basis not in [None, "C", "c"]:
                        WC = self.change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                    else:
                        WC = {WC_name : 1}

                    #run the WCs down to chipt to get limits on the WCs @ m_W
                    WC = self._run(WC, updown = "down")
                else:
                    #translate epsilon to C basis if necessary
                    if basis not in [None, "C", "c"]:
                        WC = change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                    else:
                        WC = {WC_name : 1}

                #calculate half-life for WC = 1
                hl = self.t_half(WC = WC, method = method, isotope = isotope)

                #get limit on WC
                results[WC_name] = np.sqrt(hl/half_life)

        #Get WC Group Label
        if groups:
            res = {}
            for WC in WCgroups:
                WC_name = WC_names[WC]
                res[WC_name] = results[WC]
            results = res.copy()


        #take abs value to get positive results in case the numerical estimate returned a negative
        for result in results:
            results[result] = np.absolute(results[result])

        self.WC = WC_backup.copy()

        #calculate the corresponding scales of new physics assuming naturalness
        if groups:
            for WC in WCgroups:
                WC_group_name = WC_names[WC]

                #standard mechanism
                if WC_group_name == "m_bb":
                    scales[WC_group_name] = vev**2/(np.absolute(results[WC_group_name]))

                else:
                    try:
                        d = int(WC_group_name[-2])
                    except:
                        try:
                            shortcheck = int(WC_group_name[0])
                            d = 9
                        except:
                            shortcheck = False
                            d = 6
                    if basis not in [None, "C", "c"]:
                        if (WC[0] not in ["1", "3"]) and (WC[-2] != "7"):
                            prefactor = 2
                        elif WC[-2] != "7": #dim9 prefactor due to definitions of eps basis
                            prefactor = vev/(4*m_N)
                        else:
                            prefactor = 1
                    else:
                        prefactor = 1
                    scales[WC_group_name] = vev/((prefactor * np.absolute(results[WC_group_name]))**(1/(d-4)))
        else:
            for WC_name in WCs:
                #smeft dim 5 induced
                if WC_name == "m_bb":
                    scales[WC_name] = vev**2/(np.absolute(results[WC_name]))

                else:
                    try:
                        d = int(WC_name[-2])
                    except:
                        try:
                            shortcheck = int(WC_name[0])
                            d = 9
                        except:
                            shortcheck = False
                            d = 6

                    #epsilon basis conventions
                    if basis not in [None, "C", "c"]:
                        if (WC_name[0] not in ["1", "3"]) and (WC_name[-2] != "7"):
                            prefactor = 2
                        elif WC_name[-2] != "7": #dim9 prefactor due to definitions of eps basis
                            prefactor = vev/(4*m_N)
                        else:
                            prefactor = 1
                    else:
                        prefactor = 1

                    scales[WC_name] = vev/((prefactor * np.absolute(results[WC_name]))**(1/(d-4)))

        return(pd.DataFrame({"Limits" : results, "Scales [GeV]" : scales}))
    
    #fancy plots:
    def _m_bb(self, alpha, m_min=1, ordering="NO", dcp=1.36):
        #this function returns the effective electron-neutrino Majorana mass m_bb from m_min for a fixed mass ordering
        
        #majorana phases
        alpha1=alpha[0]
        alpha2=alpha[1]

        #squared mass differences
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3

        #get mass eigenvalues from minimal neutrino mass
        m = m_min
        m1 = m
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)

        m3IO = m
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)

        #create diagonal mass matrices
        M_nu_NO = np.diag([m1,m2,m3])
        M_nu_IO = np.diag([m1IO,m2IO,m3IO])

        #mixing angles
        #s12 = np.sqrt(0.307)
        #s23 = np.sqrt(0.546)
        #s13 = np.sqrt(2.2e-2)

        #c12 = np.cos(np.arcsin(s12))
        #c23 = np.cos(np.arcsin(s23))
        #c13 = np.cos(np.arcsin(s13))

        #mixing marix
        U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                       [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                       [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

        majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

        UPMNS = U @ majorana

        #create non-diagonal mass matrix
        m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
        m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

        if ordering == "NO":
            return(m_BB_NO)
        elif ordering =="IO":
            return(m_BB_IO)
        else:
            return(m_BB_NO,m_BB_IO)
        
        
    def _m_bb_minus(self, alpha, m_min=1, ordering="both", dcp=1.36):
        ##this function returns -m_bb from m_min
        res = self._m_bb(alpha = alpha,
                         m_min = m_min,
                         ordering = ordering, 
                         dcp = dcp)
        
        if ordering == "NO":
            return(-res)
        elif ordering =="IO":
            return(-res)
        else:
            return(-res[0],-res[1])
        
    
    def _m_eff(self, 
               alpha,                       #complex phase of variational WC or if vary_WC == "m_min" this is an array of both majorana phases
               m_min             = 1,       #absolute value of variational WC
               ordering          = "both",  #mass ordering either NO, IO or both
               dcp               = 1.36,    #dirac cp phase
               isotope           = "76Ge",  #name of the isotope
               normalize         = False,   #return value normalized to the standard mass mechanism
               vary_WC           = "m_min", #variational WC
              ):
        #this function returns the effective Majorana mass parameter m_eff not no be confused with m_bb
        #m_bb: Majorana mass in the Lagrangian (determined from m_min and mass ordering)
        #m_eff: Majorana mass which would give the same half-life as given model parameters (determined from WCs and half-life)
        
        
        #make backups
        m_bb_backup = self.WC["m_bb"]
        WC_backup = self.WC.copy()
        
        #majorana phases
        if vary_WC  == "m_min":
            #for m_min you have two majorana phases in the mixing matrix
            #for all other WCs you just need to vary the corresponding phase of the WC
            alpha1=alpha[0]
            alpha2=alpha[1]

            #get mass eigenvalues from minimal neutrino mass in [eV]
            m = m_min
            m1 = m
            m2 = np.sqrt(m1**2+m21)
            m3 = np.sqrt(m2**2+m32)

            m3IO = m
            m2IO = np.sqrt(m3IO**2-m32IO)
            m1IO = np.sqrt(m2IO**2-m21)

            #create diagonal mass matrices
            M_nu_NO = np.diag([m1,m2,m3])
            M_nu_IO = np.diag([m1IO,m2IO,m3IO])

            #mixing marix
            U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                           [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                           [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])
            
            #majorana phases
            majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])
            
            #total mixing matrix
            UPMNS = U@majorana

            #create non-diagonal mass matrix
            m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
            m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

            self.WC["m_bb"] = m_BB_NO*1e-9 #[GeV]
        else:
            if vary_WC == "m_bb":
                factor = 1e-9
            else:
                factor = 1
            self.WC[vary_WC] = np.exp(1j*alpha)*m_min*factor


        G01    = self.to_G(isotope)["01"]
        M3     = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])
        NO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(isotope)**(-1/2)
        
        

        
        if vary_WC == "m_min":
            if normalize:
                NO_eff /= self.WC["m_bb"]
            self.WC["m_bb"] = m_BB_IO*1e-9
            IO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(isotope)**(-1/2)

            if normalize:
                IO_eff /= self.WC["m_bb"]

            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            if len(alpha) > 2:
                self.LEC = LEC_backup.copy()
            if ordering == "NO":
                return(NO_eff)
            elif ordering =="IO":
                return(IO_eff)
            else:
                return(NO_eff,IO_eff)
        else:
            self.WC = WC_backup.copy()
            return(NO_eff)
    
    
    
    def _m_eff_minus(self, 
                     alpha,                       #complex phase of variational WC or if vary_WC == "m_min" this is an array of both majorana phases
                     m_min             = 1,       #absolute value of variational WC
                     ordering          = "both",  #mass ordering either NO, IO or both
                     dcp               = 1.36,    #dirac cp phase
                     isotope           = "76Ge",  #name of the isotope
                     normalize         = False,   #return value normalized to the standard mass mechanism
                     vary_WC           = "m_min", #variational WC
                    ):
        #this function returns the negative of _m_eff and is used for maximization of it
        
        res = self._m_eff(alpha     = alpha,
                          m_min     = m_min, 
                          ordering  = ordering,
                          dcp       = dcp,
                          isotope   = isotope,   
                          normalize = normalize, 
                          vary_WC   = vary_WC, 
                         )
        if ordering == "NO":
            return(-res)
        elif ordering =="IO":
            return(-res)
        else:
            return(-res[0],-res[1])
        
       
    def _m_eff_minmax(self, 
                      m_min, 
                      isotope           = "76Ge",   #isotope of interest
                      ordering          = "both",   #neutrino mass ordering ["NO", "IO", "both"]
                      dcp               = 1.36,     #dirac cp-phase
                      numerical_method  = "Powell", #numerical method for minimization process
                      normalize         = False,    #normalize to standard mass mechanism if desired
                      vary_WC           = "m_min"   #WC to be varied
                     ):
        
        #this function returns the minimum and maximum of the effective majorana mass m_bb_eff
        #m_bb_eff reflects the majorana mass m_bb necessary to generate the same half-live as the input model does
        

        #the neutrino mass from the model needs to be overwritten to be able to produce the plot
        #this is because for the plot we want to be able to vary m_min!
        
        #first we do a backup of the WCs to restore them at the end
        m_bb_backup = self.WC["m_bb"]
        WC_backup = self.WC.copy()
        if vary_WC  == "m_min":
            self.WC["m_bb"] = 0
            
        #alternatively of the variational WC is not m_min you need to set the corresponding WC to 0
        else:
            self.WC[vary_WC ] = 0

        G01 = self.to_G(isotope)["01"]
        M3 = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])

        if ordering == "NO":
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            #get minimal and maximal m_bb by varying phases
            NO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            NO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            return ([NO_min_eff*1e+9, NO_max_eff*1e+9])

        elif ordering == "IO":
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            #get minimal and maximal m_bb by varying phases
            IO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            IO_max_eff = (-scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            return ([IO_min_eff*1e+9, IO_max_eff*1e+9])

        else:
            #get minimal and maximal m_bb by varying phases
            if vary_WC == "m_min":
                pre_alpha = [1,0]
            else:
                pre_alpha = 1
            NO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            
            NO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            
            if vary_WC  == "m_min":
                IO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
                
                IO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            
            
            self.WC["m_bb"] = m_bb_backup
            self.WC = WC_backup.copy()
            
            if normalize:
                if vary_WC  == "m_min":
                    return ([NO_min_eff, NO_max_eff], [IO_min_eff, IO_max_eff]) 
                else:
                    return ([NO_min_eff, NO_max_eff]) 
                    
            else:
                #return in eV
                if vary_WC  == "m_min":
                    return ([NO_min_eff*1e+9, NO_max_eff*1e+9], [IO_min_eff*1e+9, IO_max_eff*1e+9]) 
                else:
                    return ([NO_min_eff*1e+9, NO_max_eff*1e+9])
    
    #get half-life from phases and m_min
    def _t_half(self, 
                alpha,                     #complex phase
                m_min,                     #Absolute value of variational parameter. Given in eV if massive, else dimensionless
                ordering  = "NO",          #NO or IO
                dcp       = 1.36,          #cp-phase dirac neutrinos
                isotope   = "76Ge",        #isotope to calculate
                normalize = False,         #return value normalized to the standard mass mechanism
                vary_WC   = "m_min"        #varuational WC
               ):
        
        #make a backup of WCs
        WC_backup = self.WC.copy()
        
        #separate backup of m_bb
        m_bb_backup = self.WC["m_bb"]
        
        #recalculate m_bb from m_min
        if vary_WC == "m_min":
            self.WC["m_bb"] = self._m_bb(alpha=alpha, m_min=m_min, ordering=ordering, dcp=dcp)*1e-9
            
        #adjust for eV input
        else:
            if vary_WC == "m_bb":
                factor = 1e-9
            else:
                factor = 1
            #recalculate WC to be varied
            self.WC[vary_WC] = np.exp(1j*alpha)*m_min*factor
            
        #get half-life
        t_half = self.t_half(isotope = isotope)
        
        #normalize if desired
        if normalize:
            WCbackup = self.WC.copy()
            for operator in self.WC:
                if operator != "m_bb":
                    self.WC[operator] = 0
            t_half_mbb = self.t_half(isotope = isotope)
            t_half/=t_half_mbb
            self.WC = WCbackup.copy()
            
        #reset WCs from backup
        self.WC = WC_backup.copy()
        
        #reset m_bb from backup
        self.WC["m_bb"]=m_bb_backup
        return(t_half)
    
    #negative half-life (use to maximize half-life)
    def _t_half_minus(self, 
                      alpha,                      #complex phase
                      m_min,                      #value of variational parameter
                      ordering          = "NO",   #NO or IO
                      dcp               = 1.36,   #cp-phase dirac neutrinos
                      isotope           = "76Ge", #isotope to calculate
                      normalize         = False,  #return value normalized to the standard mass mechanism
                      vary_WC           = "m_min"
                     ):
        
        res = self._t_half(alpha, 
                           m_min,   
                           ordering  = ordering,
                           dcp       = dcp,    
                           isotope   = isotope,
                           normalize = normalize,
                           vary_WC   = vary_WC
                          )
        return(-res)
    
    #find the minimum and maximum values of the half-life for a given m_min with respect to the complex phase alpha
    def _t_half_minmax(self, 
                       m_min, 
                       ordering          = "both", 
                       dcp               = 1.36, 
                       isotope           = "76Ge", 
                       numerical_method  = "powell",
                       tol               = None, 
                       normalize         = False, 
                       vary_WC           = "m_min"
                      ):
        #this function gets the min and max t_half
        
        #the phase is a free parameter. Set the initial value
        
        #two phases if m_min is to be varied
        if vary_WC == "m_min":
            pre_alpha = [1,0]
            
        #one phase only if WCs are directly varied
        else:
            pre_alpha = 1
            
        #normal ordering only
        if ordering == "NO":
            #find minimum of NO
            t_half_min_NO = (scipy.optimize.minimize(self._t_half, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #find maximum of NO
            t_half_max_NO = (scipy.optimize.minimize(self._t_half_minus, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #return results
            return([t_half_min_NO, t_half_max_NO])
        
        #inverted ordering only
        elif ordering == "IO":
            #find minimum of IO
            t_half_min_IO = (scipy.optimize.minimize(self._t_half, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #find maximum of IO
            t_half_max_IO = (scipy.optimize.minimize(self._t_half_minus, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #return results
            return([t_half_min_IO, t_half_max_IO])
        
        #both mass orderings
        else:
            #find minimum of NO
            t_half_min_NO = (scipy.optimize.minimize(self._t_half, x0=pre_alpha, args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #find maximum of NO
            t_half_max_NO = (scipy.optimize.minimize(self._t_half_minus, x0=pre_alpha, args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
            
            #check if inverted ordering makes sense
            if vary_WC == "m_min":
                #find minimum of IO
                t_half_min_IO = (scipy.optimize.minimize(self._t_half, x0=pre_alpha, args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
                
                #find maximum of IO
                t_half_max_IO = (scipy.optimize.minimize(self._t_half_minus, x0=pre_alpha, args=(m_min, "IO", dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])
                
                #return NO and IO
                return([t_half_min_NO, t_half_max_NO], [t_half_min_IO, t_half_max_IO])
            else:
                #return only NO if IO didn't make sense
                return([t_half_min_NO, t_half_max_NO])
                
    def WC_variation(self, 
                     isotope           = "76Ge",
                     xaxis             = "m_min", 
                     yaxis             = "t",
                     x_min             = 1e-4,
                     x_max             = 1e+0,
                     n_points          = 100,
                     WC                = None,
                     ordering          = "both",
                     normalize         = False,
                     dcp               = 1.36,
                     numerical_method  = "Powell",        #numerical method for optimization
                    ):
        #This function varies one WC and calculates the min and max half-life or m_eff for different phases
        
        
        #set WCs
        WCbackup = self.WC.copy()
        if WC == None:
            pass
        
        else:
            C = self.WC.copy()
            for x in C:
                C[x] = 0
            for x in WC:
                C[x] = WC[x]
            self.WC = C.copy()
                
        
        if yaxis not in ["t", "m_eff", "1/t"]:
            warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
            yaxis = "t"
        
        #set up x-axis and y-axis arrays
        if xaxis == "m_sum":
            x_min = np.max([x_min, f.m_min_to_m_sum(0)["NO"]])
            x_max = np.max([x_max, f.m_min_to_m_sum(0)["NO"]])
            Msum = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
            M = Msum.copy()
            for idx in range(n_points):
                M[idx] = f.m_sum_to_m_min(Msum[idx])["NO"]
        else:
            M = np.logspace((np.log10(x_min)),(np.log10(x_max)), n_points)
            
        if xaxis == "m_sum":
            MNOsum = M.copy()
            MIOsum = M.copy()
            for idx in range(len(M)):
                m_sum = f.m_min_to_m_sum(M)
                MNOsum = m_sum["NO"]
                MIOsum = m_sum["IO"]
                
        NO_min = np.zeros(n_points)
        NO_max = np.zeros(n_points)
        IO_min = np.zeros(n_points)
        IO_max = np.zeros(n_points)
            
        #choose optimization function
        if yaxis == "m_eff":
            optimize = self._m_eff_minmax
        else:
            optimize = self._t_half_minmax
            
        #get y-axis values
        #or generate minimal and maximal possible values
        for idx in range(n_points):
            m_min = M[idx]
            #Generate Plot with varying mass
            if xaxis  in ["m_min", "m_sum"]:
                [NO_min[idx], NO_max[idx]], [IO_min[idx], IO_max[idx]] = optimize(m_min     = m_min, 
                                                                                  isotope   = isotope, 
                                                                                  ordering  ="both", 
                                                                                  dcp       = dcp,
                                                                                  normalize = normalize, 
                                                                                  vary_WC   = "m_min", 
                                                                                  numerical_method = numerical_method
                                                                                 )

            #Generate Plot with dimensionless WC varried
            else:

                [NO_min[idx], NO_max[idx]] = optimize(m_min     = m_min,
                                                      isotope   = isotope, 
                                                      ordering  = "NO", 
                                                      dcp       = dcp,
                                                      normalize = normalize,
                                                      vary_WC   = xaxis, 
                                                      numerical_method = numerical_method
                                                     )


        #store y-axis points
        if yaxis == "1/t":
            NO_min = 1/np.absolute(NO_min)
            NO_max = 1/np.absolute(NO_max)
            IO_min = 1/np.absolute(IO_min)
            IO_max = 1/np.absolute(IO_max)
        else:
            NO_min = np.absolute(NO_min)
            NO_max = np.absolute(NO_max)
            IO_min = np.absolute(IO_min)
            IO_max = np.absolute(IO_max)
            
        #Set x_axis values
        if xaxis != "m_sum":
            xNO = M
            xIO = M
        else:
            xNO = MNOsum
            xIO = MIOsum
                
        self.WC = WCbackup.copy()
        
        return(pd.DataFrame({"xNO"        : xNO, 
                             yaxis+"_min (NO)" : NO_min, 
                             yaxis+"_max (NO)" : NO_max, 
                             "xIO"        : xIO,
                             yaxis+"_min (IO)" : IO_min, 
                             yaxis+"_max (IO)" : IO_max
                            })
              )
        
    def WC_variation_scatter(self, 
                             isotope           = "76Ge",
                             xaxis             = "m_min", 
                             yaxis             = "t",
                             x_min             = 1e-4,
                             x_max             = 1e+0,
                             n_points          = 10000,
                             WC                = None,
                             ordering          = "both",
                             normalize         = False,
                             dcp               = 1.36,
                             vary_LECs         = False,
                             vary_phases       = True,
                             alpha             = [0,0],
                            ):
        #this function varies some WC and calculates the half-life or m_eff for random phases and/or LECs
        
        #set x-axis range and array
        if xaxis == "m_sum":
            
            #xmin
            x_min = np.max([x_min, f.m_min_to_m_sum(0)["NO"]])
            
            #xmax
            x_max = np.max([x_max, f.m_min_to_m_sum(0)["NO"]])
            
            #x-axis values for sum of neutrino masses
            Msum = np.logspace(np.log10(x_min), np.log10(x_max), 1*n_points)
            
            #make a copy
            M = Msum.copy()
            
            #translate m_sum to m_min
            for idx in range(n_points):
                M[idx] = f.m_sum_to_m_min(Msum[idx])["NO"]
        else:
            M = np.logspace((np.log10(x_min)),(np.log10(x_max)), 1*n_points)
            
        #generate datapoints
        
        #normal ordering
        points = np.zeros((n_points,2))
        
        #inverted ordering
        pointsIO = np.zeros((n_points,2))
        
        #xaxis array
        mspace = M
        
        #make backup of values that are varried
        
        #backup of xaxis WC
        if xaxis not in ["m_min", "m_sum"]:
            WC_backup = self.WC[xaxis]
            
        #backup of m_bb
        m_backup = self.WC["m_bb"]
        
        #backup of LECs
        LEC_backup = self.LEC.copy()
        
        #extract complex phase if not Majorana phases
        if xaxis not in ["m_min", "m_sum"]:
            try:
                alpha = alpha[0]
            except:
                alpha = alpha
        
        #iterate over number of points
        for x in range(n_points):
            #start with either m_min or m_sum on the x-axis to calculate m_bb
            if xaxis in ["m_min", "m_sum"]:
                #take random x-axis point
                m_min = np.random.choice(mspace)
                
                #randomize Majorana phases if they are to be varied
                if vary_phases:
                    alpha = np.pi*np.random.random(2)
                    
                #else set phases to 0
                else:
                    alpha = alpha
                    
                #calculate m_bb
                #normal ordering
                m = self._m_bb(alpha, m_min, "NO")*1e-9
                #inverted ordering
                mIO = self._m_bb(alpha, m_min, "IO")*1e-9
                
                #set WC
                self.WC["m_bb"] = m
                
                if xaxis == "m_sum":
                    
                    #translate m_min to m_sum
                    m_sum = f.m_min_to_m_sum(m_min)
                    
                    #normal ordering
                    msum = m_sum["NO"]
                    
                    #inverted ordering
                    msumIO = m_sum["IO"]
                
            #Else have some Wilson coefficient on the x-axis
            else:
                #choose some x-axis value
                #rescale mass from eV and put as m_bb
                if xaxis == "m_bb":
                    self.WC[xaxis] = np.random.choice(mspace)*1e-9
                    
                #else take x-axis as WC
                else:
                    self.WC[xaxis] = np.random.choice(mspace)
                    
                #vary complex phase
                if vary_phases:
                    alpha = np.pi*np.random.rand()
                    self.WC[xaxis] *= np.exp(1j*alpha)
                else:
                    self.WC[xaxis] *= np.exp(1j*alpha)
                    
            #vary_unknown_LECs
            if vary_LECs == True:
                self._vary_LECs(inplace = True)
            
            #calculate half-life
            t = self.t_half(isotope)
            
            #store y-axis points
            if yaxis == "t":
                points[x][1] = t
            elif yaxis == "1/t":
                points[x][1] = 1/t
            elif yaxis == "m_eff":
                #PSF G01
                G01    = self.to_G(isotope)["01"]
                
                #Mass Mechanism NME
                M3     = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])
                
                #Effective Neutrino Mass
                points[x][1] = self.m_e / (g_A**2*M3*G01**(1/2)) * t**(-1/2) * 1e+9
                
            
            #normalize to mass mechanism if desired
            if normalize and xaxis in ["m_min", "m_bb", "m_sum"]:
                
                #make backup of WCs to later restore
                WC_backup = self.WC.copy()
                
                #loop over WCs and set all but m_bb to 0
                for operator in self.WC:
                    if operator != "m_bb":
                        self.WC[operator]=0
                        
                #calculate half-life for standard mechanism m_bb
                t_half_mbb = self.t_half(isotope)
                
                #reset WCs from backup
                self.WC = WC_backup.copy()
                
                #normalize results
                if yaxis == "t":
                    points[x][1] /= t_half_mbb
                elif yaxis == "1/t":
                    points[x][1] *= t_half_mbb
                elif yaxis == "m_eff":
                    points[x][1] /= (m * 1e+9)
                
            #repeat for inverted ordering
            if xaxis in ["m_min", "m_sum"]:
                self.WC["m_bb"] = mIO
                
                tIO = self.t_half(isotope)
                if xaxis == "m_sum":
                    pointsIO[x][0] = msumIO
                    points[x][0] = msum
                else:
                    pointsIO[x][0] = m_min
                    points[x][0] = m_min
                    
                #store y-axis point
                if yaxis == "t":
                    pointsIO[x][1] = tIO
                elif yaxis == "1/t":
                    pointsIO[x][1] = 1/tIO
                elif yaxis == "m_eff":
                    G01    = self.to_G(isotope)["01"]
                    M3     = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])
                    pointsIO[x][1] = self.m_e / (g_A**2*M3*G01**(1/2)) * tIO**(-1/2) * 1e+9
                    
                #normalize to mass mechanism if desired
                if normalize and xaxis in ["m_min", "m_bb", "m_sum"]:
                    
                    #make backup of WCs to later restore
                    WC_backup = self.WC.copy()
                    
                    #loop over WCs and set all but m_bb to 0
                    for operator in self.WC:
                        if operator != "m_bb":
                            self.WC[operator]=0
                            
                    #calculate half-life for standard mechanism m_bb
                    t_half_mbb = self.t_half(isotope)
                    
                    #reset WCs from backup
                    self.WC = WC_backup.copy()
                    
                    #normalize results
                    if yaxis == "t":
                        pointsIO[x][1] /= t_half_mbb
                    elif yaxis == "1/t":
                        pointsIO[x][1] *= t_half_mbb
                    elif yaxis == "m_eff":
                        pointsIO[x][1] /= (mIO * 1e+9)
            else:
                if xaxis == "m_bb":
                    points[x][0] = np.absolute(self.WC[xaxis])*1e+9
                else:
                    points[x][0] = np.absolute(self.WC[xaxis])
                    
        #restore backup values
        self.WC["m_bb"] = m_backup
        self.LEC = LEC_backup.copy()
        if xaxis not in ["m_min", "m_sum"]:
            self.WC[xaxis] = WC_backup
                    
        #return(points, pointsIO)
        return(pd.DataFrame({"xNO" : points[:,0], 
                             "yNO" : points[:,1],
                             "xIO" : pointsIO[:,0], 
                             "yIO" : pointsIO[:,1]}
                           )
              )
                
    def plot_WC_variation(self, 
                          xaxis             = "m_min",         #WC to vary on the x-axis ["m_min, "m_sum"], or any of the LEFT WCs
                          yaxis             = "t",             #choose from ["t", "m_eff", "1/t"]
                          isotope           = "76Ge",          #Use NMEs from this isotope for the calculation
                          x_min             = 1e-4,            #minimal x-axis value
                          x_max             = 1e+0,            #maximal x-axis value
                          y_min             = None,            #minimal y-axis value
                          y_max             = None,            #maximal y-axis value
                          xscale            = "log",           #x-axis scaling
                          yscale            = "log",           #y-axis scaling
                          n_points          = 100,             #number of points to plot.
                          cosmo             = False,           #plot cosmology limit on m_sum
                          m_cosmo           = 0.15,            #limit on m_sum
                          limits            = None,            #y-axis limits
                          experiments       = None,            #half-life limits from experiments - get converted to y-axis limits
                          ordering          = "both",          #neutrino mass ordering ["both", "NO", "IO"]
                          dcp               = 1.36,            #dirac CP phase
                          numerical_method  = "Powell",        #numerical method for optimization
                          show_mbb          = False,           #if True plot the standard mass mechanism as a reference
                          normalize         = False,           #normalize x-axis to standard mass mechanism
                          colorNO           = "b",             #color of normal ordering
                          colorIO           = "r",             #color of inverted ordering
                          legend            = True,            #plot legend
                          labelNO           = None,            #label for normal ordering in legend
                          labelIO           = None,            #label for inverted ordering in legend
                          autolabel         = True,            #set labels automatically
                          alpha_plot        = 0.5,             #alpha value of main plots
                          alpha_mass        = 0.05,            #alpha value of mass plot of show_mbb = True
                          alpha_cosmo       = 0.1,             #alpha of cosmo limit
                          #return_points     = False,           #if True return plotted points
                          vary_phases       = True,            #vary complex WC phases
                          alpha             = [0,0],           #Majorana phases, or WCs complex phase
                          savefig           = False,           #if True save figure as file
                          file              = "variation.png", #Filename and path to store figure to
                          dpi               = 300              #set the resolution of the saved figure in dots per inch
                         ):
        
        #xaxis defines varied WC
        vary_WC = xaxis
        
        #check if yaxis is allowed
        if yaxis not in ["t", "m_eff", "1/t"]:
            warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
            yaxis = "t"
            
        #color list to use in plot
        colors = [colorNO, colorIO]
        
        if x_min == None:
            x_min = 1e-4
            x_max = 1e+0
        
        
        #check if comparison to mass mechanism is possible
        if vary_WC not in ["m_bb", "m_min", "m_sum"] and show_mbb:
            warnings.warn("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass m_min, the sum of neutrino masses m_sum or m_bb on the x axis. Setting show_mbb = False")
            show_mbb = False
            
        print("Generating Datapoints...")
        
        #get datapoints
        if vary_phases:
            datapoints = self.WC_variation(isotope           = isotope, 
                                           xaxis             = xaxis, 
                                           yaxis             = yaxis, 
                                           x_min             = x_min, 
                                           x_max             = x_max, 
                                           n_points          = n_points, 
                                           WC                = None, 
                                           ordering          = ordering,
                                           normalize         = normalize, 
                                           dcp               = dcp, 
                                           numerical_method  = numerical_method
                                          )
            print("Datapoints Generated")

            #store datapoint arrays
            xNO    = datapoints["xNO"]
            NO_min = datapoints[yaxis+"_min (NO)"]
            NO_max = datapoints[yaxis+"_max (NO)"]
            xIO    = datapoints["xIO"]
            IO_min = datapoints[yaxis+"_min (IO)"]
            IO_max = datapoints[yaxis+"_max (IO)"]
            
        #get datapoints for a fixed phase
        else:
            if xscale == "lin":
                m_min_array = np.linspace(x_min, x_max, n_points)
            else:
                m_min_array = np.logspace(np.log10(x_min), np.log10(x_max), n_points)
                
            xNO    = np.zeros(n_points)
            NO_min = xNO.copy()
            NO_max = xNO.copy()
            xIO    = xNO.copy()
            IO_min = xNO.copy()
            IO_max = xNO.copy()
            if xaxis not in ["m_min", "m_sum"]:
                try:
                    alpha = alpha[0]
                except:
                    pass
                
            for idx in range(n_points):
                if yaxis == "t":
                    m_min       = m_min_array[idx]
                    xNO[idx]    = m_min
                    NO_min[idx] = self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                    NO_max[idx] = NO_min[idx]
                    xIO[idx]    = m_min
                    IO_min[idx] = self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                    IO_max[idx] = IO_min[idx]
                    
                elif yaxis == "1/t":
                    m_min       = m_min_array[idx]
                    xNO[idx]    = m_min
                    NO_min[idx] = 1/self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                    NO_max[idx] = NO_min[idx]
                    xIO[idx]    = m_min
                    IO_min[idx] = 1/self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                    IO_max[idx] = IO_min[idx]
                    
                else:
                    m_min       = m_min_array[idx]
                    xNO[idx]    = m_min
                    NO_min[idx] = self._m_eff(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                    NO_max[idx] = NO_min[idx]
                    xIO[idx]    = m_min
                    IO_min[idx] = self._m_eff(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                    IO_max[idx] = IO_min[idx]
        
        #generate arrays for mass mechanism datapoints
        if show_mbb:
            NO_min_mbb = np.zeros(n_points)
            NO_max_mbb = np.zeros(n_points)
            IO_min_mbb = np.zeros(n_points)
            IO_max_mbb = np.zeros(n_points)
        else:
            NO_min_mbb = None
            NO_max_mbb = None
            IO_min_mbb = None
            IO_max_mbb = None
        
        #get datapoints for mass comparison
        if show_mbb:
            print("Generating Datapoints for mass mechanism...")
            if xaxis == "m_sum":
                xaxismbb = "m_sum"
            else:
                xaxismbb = "m_min"
            datapoints = self.WC_variation(isotope           = isotope, 
                                           xaxis             = xaxismbb, 
                                           yaxis             = yaxis, 
                                           x_min             = x_min, 
                                           x_max             = x_max, 
                                           n_points          = n_points, 
                                           WC                = {}, 
                                           ordering          = ordering,
                                           normalize         = normalize, 
                                           dcp               = dcp, 
                                           numerical_method  = numerical_method)
            xNO_mbb    = datapoints["xNO"]
            NO_min_mbb = datapoints[yaxis+"_min (NO)"]
            NO_max_mbb = datapoints[yaxis+"_max (NO)"]
            xIO_mbb    = datapoints["xIO"]
            IO_min_mbb = datapoints[yaxis+"_min (IO)"]
            IO_max_mbb = datapoints[yaxis+"_max (IO)"]
            print("Datapoints for mass mechanism generated")
            
            
        print("Preparing plot...")
        #set legend labels
        if autolabel:
            if vary_WC  in ["m_min", "m_sum"]:
                NO_label = "NO"
                IO_label =  "IO"
            else:
                NO_label = None
                IO_label =  None
            
        #initiate limits dict
        if limits == None:
            limits = {}
            
        #Adjust experimental limits
        if experiments != None:
            for experiment in experiments:
                #extract yaxis limit
                y_data = experiments[experiment]["limit"]
                    
                if yaxis == "m_eff":
                    #convert half-life into m_bb limit
                    y_data = self.get_limits(y_data, isotope = isotope)["Limits"]["m_bb"]*1e+9
                
                #overwrite yaxis limit
                experiments[experiment]["limit"] = y_data
                
                #store experiments in limits dict
                limits[experiment] = experiments[experiment]
                    
        #define axis labels
        
        #y-axis
        if normalize:
            if yaxis == "m_eff":
                ylabel = r"$\left|\frac{m_{\beta\beta}^{eff}}{m_{\beta\beta}}\right|$"
            elif yaxis == "1/t":
                ylabel = r"$\frac{t_{1/2, m_{\beta\beta}}}{t_{1/2}}$"
            else:
                ylabel = r"$\frac{t_{1/2}}{t_{1/2, m_{\beta\beta}}}$"
                
        else:
            if yaxis == "m_eff":
                ylabel = r"$|m_{\beta\beta}^{eff}|$ [eV]"
            elif yaxis == "1/t":
                ylabel = r"$t_{1/2}^{-1}$ [yr$^{-1}$]"
            else:
                ylabel = r"$t_{1/2}$ [yr]"
            
        #x-axis
        if vary_WC == "m_min":
            xlabel = r"$m_{min}$ [eV]"
        elif vary_WC == "m_sum":
            xlabel = r"$\sum_i m_{i}$ [eV]"
        elif vary_WC == "m_bb":
            xlabel = r"$|m_{\beta\beta}|$ [eV]"
        else:
            xlabel = r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$"
            
            
        #set y-axis limits if not set manually
        if yaxis in ["m_eff", "1/t"]:
            if y_max == None:
                if vary_WC == "m_min":
                    y_max = np.max([np.max(IO_max), np.max(NO_max)])
                else:
                    y_max = np.max(NO_max)

                if normalize:
                    y_min = 1e-4
                    y_max = 1e+4

                y_max = 10**np.ceil(np.log10(y_max))
            if y_min == None:

                if yaxis == "m_eff":
                    y_min = 1e-4*y_max
                elif yaxis == "1/t":
                    scaling = 1e-8
                    if vary_WC in ["m_min", "m_sum"]:
                        y_min = np.max([10**np.floor(np.log10(np.min([np.min(NO_max), np.min(IO_max)]))),scaling*y_max])
                    else:
                        y_min = np.max([10**np.floor(np.log10(np.min(NO_max))),scaling*y_max])

        else:
            if y_min == None:
                if vary_WC in ["m_min", "m_sum"]:
                    y_min = np.min([np.min(IO_max), np.min(NO_max)])
                else:
                    y_min = np.min(NO_max)
                y_min = 10**np.floor(np.log10(y_min))
            if y_max == None:
                y_max = np.min([10**np.ceil(np.log10(np.max([NO_max, IO_max]))),1e+8*y_min])
            
        if xaxis == "m_sum":
            
            #xmin
            x_min = np.max([x_min, f.m_min_to_m_sum(0)["NO"]])
            
            #xmax
            x_max = np.max([x_max, f.m_min_to_m_sum(0)["NO"]])
            
        print("Generating plot")
            
            
        #Generate Figure
        fig = plots.lobster(xNO = xNO, NO_min = NO_min, NO_max = NO_max, vary_WC = vary_WC, 
                            NO_min_mbb = NO_min_mbb, NO_max_mbb = NO_max_mbb, 
                            IO_min_mbb = IO_min_mbb, IO_max_mbb = IO_max_mbb, 
                            NO_label = labelNO, IO_label = labelIO, autolabel = autolabel,
                            x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                            xscale = xscale, yscale = yscale, legend = legend, 
                            xIO = xIO, IO_min = IO_min, IO_max = IO_max, ordering = ordering, 
                            limits = limits, ylabel = ylabel, xlabel = xlabel, 
                            cosmo = cosmo, m_cosmo = m_cosmo, colors = colors, 
                            alpha_plot = alpha_plot, alpha_mass = alpha_mass, alpha_cosmo = alpha_cosmo, 
                            show_mbb = show_mbb, normalize = normalize, 
                            savefig = savefig, file = file, dpi = dpi)
        return(fig)
    
    def plot_WC_variation_scatter(self,  
                                  xaxis             = "m_min",        #WC to vary on the x-axis ["m_min, "m_sum"], or any of the LEFT WCs
                                  yaxis             = "t",            #choose from ["t", "m_eff", "1/t"]
                                  isotope           = "76Ge",         #Use NMEs from this isotope for the calculation
                                  vary_phases       = True,           #vary complex phases
                                  vary_LECs         = False,          #vary unknown LECs
                                  alpha             = [0,0],          #Majorana phases, or WCs complex phase
                                  n_points          = 10000,          #number of points to plot.
                                  markersize        = 0.5, 
                                  x_min             = 1e-4,           #minimal x-axis value
                                  x_max             = 1,              #maximal x-axis value
                                  y_min             = None,           #minimal y-axis value 
                                  y_max             = None,           #maximal y-axis value
                                  xscale            = "log",          #x-axis scaling
                                  yscale            = "log",          #y-axis scaling
                                  limits            = None,           #y-axis limits
                                  experiments       = None,           #half-life limits from experiments - get converted to y-axis limits
                                  ordering          = "both",         #neutrino mass ordering ["both", "NO", "IO"]
                                  dcp               = 1.36,           #dirac CP phase
                                  show_mbb          = False,          #if True plot the standard mass mechanism as a reference
                                  normalize         = False,          #normalize y-axis to standard mass mechanism
                                  cosmo             = False,          #plot cosmology limit on m_sum
                                  m_cosmo           = 0.15,           #limit on m_sum
                                  colorNO           = "b",            #color of normal ordering
                                  colorIO           = "r",            #color of inverted ordering 
                                  legend            = True,           #plot legend
                                  labelNO           = None,           #label for normalordering in legend
                                  labelIO           = None,           #label for inverted ordering in legend
                                  autolabel         = True,           #set labels automatically
                                  alpha_plot        = 1,              #alpha value of main plots
                                  alpha_mass        = 0.05,           #alpha value of mass plot of show_mbb = True
                                  alpha_cosmo       = 0.1,            #alpha of cosmo limit
                                  savefig           = False,          #if True save figure as file
                                  file              = "var_scat.png", #Filename and path to store figure to
                                  dpi               = 300             #set the resolution of the saved figure in dots per inch
                                 ):
        
        vary_WC = xaxis
        if yaxis not in ["t", "m_eff", "1/t"]:
            warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
            yaxis = "t"
            
        colors = [colorNO, colorIO]
        
        
        
        if vary_WC not in ["m_bb", "m_min", "m_sum"] and show_mbb:
            warnings.warn("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass m_min, the sum of neutrino masses m_sum or m_bb on the x axis. Setting show_mbb = False")
            show_mbb = False
            
        print("Generating Datapoints...")
        #get datapoints
        datapoints = self.WC_variation_scatter(isotope           = isotope, 
                                               xaxis             = xaxis, 
                                               yaxis             = yaxis, 
                                               x_min             = x_min, 
                                               x_max             = x_max, 
                                               n_points          = n_points, 
                                               WC                = None, 
                                               ordering          = ordering,
                                               normalize         = normalize, 
                                               dcp               = dcp, 
                                               vary_phases       = vary_phases, 
                                               vary_LECs         = vary_LECs, 
                                               alpha             = alpha
                                              )
        print("Datapoints Generated")
        
        points = np.array([datapoints["xNO"], datapoints["yNO"]]).T
        pointsIO = np.array([datapoints["xIO"], datapoints["yIO"]]).T
            
        #datapoints for mass comparison
        if show_mbb:
            xNO_mbb    = np.zeros(100)
            NO_min_mbb = np.zeros(100)
            NO_max_mbb = np.zeros(100)
            xIO_mbb    = np.zeros(100)
            IO_min_mbb = np.zeros(100)
            IO_max_mbb = np.zeros(100)
        else:
            xNO_mbb    = None
            NO_min_mbb = None
            NO_max_mbb = None
            xIO_mbb    = None
            IO_min_mbb = None
            IO_max_mbb = None
        
        if show_mbb:
            print("Generating Datapoints for mass mechanism...")
            if xaxis == "m_sum":
                xaxismbb = "m_sum"
            else:
                xaxismbb = "m_min"
            datapoints = self.WC_variation(isotope           = isotope, 
                                           xaxis             = xaxismbb, 
                                           yaxis             = yaxis, 
                                           x_min             = x_min, 
                                           x_max             = x_max, 
                                           n_points          = 100, 
                                           WC                = {}, 
                                           ordering          = ordering,
                                           normalize         = normalize, 
                                           dcp               = dcp)
            
            xNO_mbb    = datapoints["xNO"]
            NO_min_mbb = datapoints[yaxis+"_min (NO)"]
            NO_max_mbb = datapoints[yaxis+"_max (NO)"]
            xIO_mbb    = datapoints["xIO"]
            IO_min_mbb = datapoints[yaxis+"_min (IO)"]
            IO_max_mbb = datapoints[yaxis+"_max (IO)"]
            
            print("Datapoints for mass mechanism generated")
            
        print("Preparing plot...")
        #set legend labels
        if autolabel:
            if vary_WC  in ["m_min", "m_sum"]:
                NO_label = "NO"
                IO_label =  "IO"
            else:
                NO_label = None
                IO_label =  None
            
        #initiate limits dict
        if limits == None:
            limits = {}
            
        #Adjust experimental limits
        if experiments != None:
            for experiment in experiments:
                y_data = experiments[experiment]["limit"]
                    
                if yaxis == "m_eff":
                    #convert half-life into m_bb limit
                    y_data = self.get_limits(y_data, isotope = isotope)["Limits"]["m_bb"]*1e+9
                
                experiments[experiment]["limit"] = y_data
                    
                limits[experiment] = experiments[experiment]
                    
        #define axis labels
        
        #y-axis
        if normalize:
            if yaxis == "m_eff":
                ylabel = r"$\left|\frac{m_{\beta\beta}^{eff}}{m_{\beta\beta}}\right|$"
            elif yaxis == "1/t":
                ylabel = r"$\frac{t_{1/2, m_{\beta\beta}}}{t_{1/2}}$"
            else:
                ylabel = r"$\frac{t_{1/2}}{t_{1/2, m_{\beta\beta}}}$"
                
        else:
            if yaxis == "m_eff":
                ylabel = r"$|m_{\beta\beta}^{eff}|$ [eV]"
            elif yaxis == "1/t":
                ylabel = r"$t_{1/2}^{-1}$ [yr$^{-1}$]"
            else:
                ylabel = r"$t_{1/2}$ [yr]"
            
        #x-axis
        if vary_WC == "m_min":
            xlabel = r"$m_{min}$ [eV]"
        elif vary_WC == "m_sum":
            xlabel = r"$\sum_i m_{i}$ [eV]"
        elif vary_WC == "m_bb":
            xlabel = r"$|m_{\beta\beta}|$ [eV]"
        else:
            xlabel = r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$"
            
            
        #set y-axis limits if not set manually
        
        #y-axis limits for scatter plot
        #define y-axis limits if not given
        if y_min == None:
            if vary_WC in ["m_min", "m_sum"]:
                if ordering == "both":
                    y_min = np.min([np.min(points[:,1]), np.min(pointsIO[:,1])])
                elif ordering == "IO":
                    y_min = np.min(pointsIO[:,1])
                else:
                    y_min = np.min(points[:,1])
            else:
                y_min = np.min(points[:,1])
        if y_max == None:
            if vary_WC in ["m_min", "m_sum"]:
                if ordering == "both":
                    y_max = np.max([np.max(points[:,1]), np.max(pointsIO[:,1])])
                elif ordering == "IO":
                    y_max = np.max(pointsIO[:,1])
                else:
                    y_max = np.max(points[:,1])
            else:
                y_max = np.max(points[:,1])
            
        if xaxis == "m_sum":
            
            #xmin
            x_min = np.max([x_min, f.m_min_to_m_sum(0)["NO"]])
            
            #xmax
            x_max = np.max([x_max, f.m_min_to_m_sum(0)["NO"]])
            
        print("Generating plot")
        #Generate Figure
        fig = plots.lobster_scatter(points = points, pointsIO = pointsIO, vary_WC = vary_WC,
                                    x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
                                    xlabel = xlabel, ylabel = ylabel, markersize = markersize, 
                                    xscale = xscale, yscale = yscale, legend = legend,
                                    alpha_plot = alpha_plot, alpha_mass = alpha_mass, alpha_cosmo = alpha_cosmo,
                                    show_mbb = show_mbb, xNO_mbb = xNO_mbb, xIO_mbb = xIO_mbb, ordering = ordering,
                                    cosmo = cosmo, m_cosmo = m_cosmo, NO_min_mbb = NO_min_mbb, 
                                    NO_max_mbb = NO_max_mbb, IO_min_mbb = IO_min_mbb, IO_max_mbb = IO_max_mbb,
                                    NO_label = labelNO, IO_label = labelIO, autolabel = autolabel,
                                    limits = limits, savefig = savefig, file = file,  dpi = dpi)
        
        return(fig)
        
    #same as plot_WC_variation but with yaxis fixed to "m_eff"
    def plot_m_eff(self, **args):
        return(self.plot_WC_variation(yaxis = "m_eff", **args))
        
    #same as plot_WC_variation but with yaxis fixed to "1/t"
    def plot_t_half_inv(self, **args):
        return(self.plot_WC_variation(yaxis = "1/t", **args))
        
    #same as plot_WC_variation but with yaxis fixed to "t"
    def plot_t_half(self, **args):
        return(self.plot_WC_variation(yaxis = "t", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "m_eff"
    def plot_m_eff_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "m_eff", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "t" 
    def plot_t_half_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "t", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "1/t" 
    def plot_t_half_inv_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "1/t", **args))
    
    def plot_limits(self, 
                    experiments, 
                    method      = "IBM2",
                    groups      = True,
                    savefig     = False, 
                    plottype    = "scales"):
        limits = {}
        scales = {}
        for isotope in experiments:
            lims = self.get_limits(experiments[isotope], isotope = isotope)
            limits[isotope] = lims[lims.keys()[0]]
            scales[isotope] = lims[lims.keys()[1]]
                    

        scales = pd.DataFrame(scales)
        limits = pd.DataFrame(limits)
        if groups:
            WC_operator_groups = ["m_bb" , "VL(6)", 
                                  "VR(6)", "T(6)", 
                                  "SL(6)", "VL(7)", "1L(9)", 
                                  "2L(9)", "3L(9)", 
                                  "4L(9)", "5L(9)", 
                                  "6(9)", "7(9)"]

            #define labels for plots
            WC_group_names = [r"$m_{\beta\beta}$"      , r"$C_{VL}^{(6)}$", 
                              r"$C_{VR}^{(6)}$", r"$C_{T}^{(6)}$" , 
                              r"$C_{S6}}$", r"$C_{V7}$"     , r"$C_{S1}^{(9)}$", 
                              r"$C_{S2}^{(9)}$", r"$C_{S3}^{(9)}$", 
                              r"$C_{S4}^{(9)}$", r"$C_{S5}^{(9)}$",
                              r"$C_{V}^{(9)}$", r"$C_{\tilde{V}}^{(9)}$"]

            idx = 0
            for operator in scales.T:
                if operator not in WC_operator_groups or operator == "m_bb":
                    scales.drop(operator, axis=0, inplace=True)
                    limits.drop(operator, axis=0, inplace=True)
                else:
                    idx+=1
                    scales.rename(index={operator:WC_group_names[idx]}, inplace=True)
                    limits.rename(index={operator:WC_group_names[idx]}, inplace=True)
        
        
        if plottype == "scales":
            fig = scales.plot.bar(figsize=(16,6))
            fig.set_ylabel(r"$\Lambda$ [TeV]", fontsize =14)
        else:
            fig = limits.plot.bar(figsize=(16,6))
            fig.set_ylabel(r"$C_X$", fontsize =14)
        fig.set_yscale("log")
        fig.set_xticklabels(WC_group_names[1:], fontsize=14)
        fig.grid(linestyle="--")
        if len(experiments)>10:
            ncol = int(len(experiments)/2)
        else:
            ncol = len(experiments)
        fig.legend(fontsize=12, loc = (0,1.02), ncol=ncol)
        
        if savefig:
            if plottype == "scales":
                fig.get_figure().savefig("scale_limits.png")
            elif plottype == "limits":
                fig.get_figure().savefig("WC_limits.png")
                
        return(fig.get_figure())
    
    #generate readable expression of half-life formula
    def generate_formula(self, isotope, WC = None, method = None, decimal = 2, output = "latex", 
                         unknown_LECs = None, PSF_scheme = None):
        #set NME method
        method = self._set_method_locally(method)
        
        if unknown_LECs == None:
            unknown_LECs = self.unknown_LECs
        if PSF_scheme == None:
            PSF_scheme = self.PSF_scheme
            
        if WC == None:
            WC = []
            for WCs in self.WC:
                if self.WC[WCs] != 0:
                    WC.append(WCs)
        return(f.generate_formula(WC = WC, isotope = isotope, output = output, 
                                  method = method, decimal = decimal, unknown_LECs = unknown_LECs, 
                                  PSF_scheme = PSF_scheme))
    
    #generate half-life matrix
    def generate_matrix(self, isotope, WC = None, method = None, unknown_LECs = None, PSF_scheme = None):
        #set NME method
        method = self._set_method_locally(method)
        
        if unknown_LECs == None:
            unknown_LECs = self.unknown_LECs
        if PSF_scheme == None:
            PSF_scheme = self.PSF_scheme
        
        if WC == None:
            WC = []
            for WCs in self.WC:
                if self.WC[WCs] != 0:
                    WC.append(WCs)
        return(f.generate_matrix(WC = WC, isotope = isotope, method = method, 
                                 unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme))

'''
#####################################################################################################
#                                                                                                   #
#                                                                                                   #
#                                            SMEFT MODEL                                            #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################
'''

class SMEFT(object):
    def __init__(self, 
                 WC,                    #dict of Wilson Coefficients
                 scale        = m_W,    #scale the WCs are defined at
                 name         = None,   #name the model
                 method       = "IBM2", #NME method
                 unknown_LECs = False,  #use unknown LECs? estimates?
                 PSF_scheme   = "A",    #A: uniform charge distribution, approx. wave-func., B: point-like charge, excact wave-func.
                ):
        self.m_N     = m_N
        self.m_e     = m_e
        self.m_e_MEV = m_e_MeV
        self.vev     = vev
        
        
        if scale != m_W:
            if scale < m_W:
                raise ValueError("scale must be greater than or equal to m_W = 80GeV")
        else:
            scale = m_W
            
        self.method = method                       #NME method
        self.name   = name                         #Model name
        self.m_Z    = m_Z                          #Z-Boson Mass in GeV
        self.scale  = scale                        #Matching scale BSM -> SMEFT
        self.unknown_LECs = unknown_LECs           #Use unknown LECs or don't
        self.PSF_scheme = PSF_scheme               #Scheme for Wave functions
        
        self.SMEFT_WCs = SMEFT_WCs.copy()          #see constants.SMEFT_WCs
        
        ########################################################
        #
        #
        #                     Define LECs
        #
        #
        ########################################################
        if unknown_LECs == True:
            #self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
            #           "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
            #           # all the below are expected to be order 1 in absolute magnitude
            #           "Tprime":1, "Tpipi":1, "1piN":1, "6piN":1, "7piN":1, "8piN":1, "9piN":1, "VLpiN":1, "TpiN":1, 
            #           "1NN":1, "6NN":1, "7NN":1, "VLNN":1, "TNN": 1, "VLE":1, "VLme":1, "VRE":1, "VRme":1, 
            #           # all the below are expected to be order (4pi)**2 in absolute magnitude
            #           "2NN":(4*np.pi)**2, "3NN":(4*np.pi)**2, "4NN":(4*np.pi)**2, "5NN":(4*np.pi)**2, 
            #           # expected to be 1/F_pipi**2 pion decay constant
            #           "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
            #          }
            
            self.LEC = LECs.copy()
            
            self.LEC["VpiN"] = self.LEC["6piN"] + self.LEC["8piN"]
            self.LEC["tildeVpiN"] = self.LEC["7piN"] + self.LEC["9piN"]
        
        else:
            self.LEC = {"A":1.271, "S":0.97, "M":4.7, "T":0.99, "B":2.7, "1pipi":0.36, 
                       "2pipi":2.0, "3pipi":-0.62, "4pipi":-1.9, "5pipi":-8, 
                       # all the below are expected to be order 1 in absolute magnitude
                       "Tprime":0, "Tpipi":0, "1piN":0, "6piN":0, "7piN":0, "8piN":0, "9piN":0, "VLpiN":0, "TpiN":0, 
                       "1NN":0, "6NN":1, "7NN":1, "VLNN":0, "TNN": 0, "VLE":0, "VLme":0, "VRE":0, "VRme":0, 
                       # all the below are expected to be order (4pi)**2 in absolute magnitude
                       "2NN":0, "3NN":0, "4NN":0, "5NN":0, 
                       # expected to be 1/F_pipi**2 pion decay constant
                       "nuNN": -1/(4*np.pi) * (self.m_N*1.27**2/(4*0.0922**2))**2*0.6
                      }
            self.LEC["VpiN"] = 1#LEC["6piN"] + LEC["8piN"]
            self.LEC["tildeVpiN"] = 1#LEC["7piN"] + LEC["9piN"]
        
        #SET WCs
        for operator in WC:
            #store SMEFT operators
            #need to be conjugated to have d -> u transitions
            #if operator not in ["LH(5)", "LH(7)"]:
            self.SMEFT_WCs[operator] = WC[operator]#.conjugate()# / self.scale**(int(operator[-2])-4)

        self.WC_input = self.SMEFT_WCs.copy()
        self.WC = self.WC_input.copy()
        
        #if the matching scale of the BSM model is not at m_W, run the operators down to m_W
        if self.scale != m_W:
            print("Running operators down to m_W")
            self.run(self.WC, inplace = True)
        
        
        ########################################################
        #
        #
        #                     Define NMEs
        #
        #
        ########################################################
        self.NMEs, self.NMEpanda, self.NMEnames = Load_NMEs(method)
        
    
        
    #recalculate PSFs and wave-functions
    def set_wave_scheme(self, scheme = "A"):
        if scheme != self.PSF_scheme:
            self.PSF_scheme = scheme
            self.isotopes   = set_PSF_scheme(scheme, setglobal = False)
            self.PSFpanda   = pd.read_csv(cwd+"/../PSFs/PSFs_"+scheme+".csv")
            
            
    def _vary_LECs(self, inplace = False): 
    #this function varies the unknown LECs within the appropriate range
        LECs = {}
        for LEC in LECs_unknown:
            if LEC == "nuNN":
                random_LEC = (np.random.rand()+0.5)*LECs_unknown[LEC]
                LECs[LEC] = random_LEC
            else:
                #random_LEC = variation_range*2*(np.random.rand()-0.5)*LECs[LEC]
                random_LEC = (np.random.choice([1,-1])
                              *((np.sqrt(10)-1/np.sqrt(10))
                                *np.random.rand()+1/np.sqrt(10))*LECs_unknown[LEC])
            LECs[LEC] = random_LEC
            
        #set LECs that depend on others
        LECs["VpiN"] = LECs["6piN"] + LECs["8piN"]
        LECs["tildeVpiN"] = LECs["7piN"] + LECs["9piN"]
        if inplace:
            for LEC in LECs:
                self.LEC[LEC] = LECs[LEC]
        else:
            return(LECs)
    
    
    #switch NME method locally within a function
    def _set_method_locally(self, method):
        #set the method and import NMEs if necessary
        if method == None:
            method = self.method
            pass
        elif method != self.method and method+".csv" in os.listdir(cwd+"/../NMEs/"):
            pass
        elif method+".csv" not in os.listdir(cwd+"/../NMEs/"):
            warnings.warn("Method",method,"is unavailable. No such file 'NMEs/"+method+".csv'. Keeping current method",self.method)
            method = self.method
        else:
            method = self.method
            pass
        return(method)
        
            
    '''
        ################################################################################################
        
        Define RGEs computed in 1901.10302
        
        Note that "scale" in the RGEs refers to log(mu)
        i.e. "scale" = log(mu), while scale in the final 
        functions refers to the actual energy scale i.e. 
        "scale" = mu
        
        ################################################################################################
    '''
    
    ####################################################################################################
    
    def run(self, WC = None, initial_scale = None, final_scale = m_W, inplace = False):
        if initial_scale == None:
            initial_scale = self.scale
        if WC == None:
            WC = self.WC_input.copy()
        else:
            #make a dict with all WCs
            WCs = SMEFT_WCs.copy()
            
            #write the non-zero WCs
            for C in WC:
                WCs[C] = WC[C]
            WC = WCs
            
        #check if dim9 operators are defined
        dim9 = False
        for operator in WC:
            if operator[-2] == "9":
                if WC[operator] != 0:
                    dim9 = True
        if dim9:
            warnings.warn("You defined a non-zero dimension 9 operator. The RGEs of dim9 operators are currently not included in NuBB. Proceeding to run dim7 operators...")
            
        final_WC = run_SMEFT(WC = WC, initial_scale = initial_scale, final_scale = final_scale)
        
        if inplace:
            self.WC = final_WC.copy()
        
        return(final_WC)
        
        
    
    ####################################################################################################
        
        
        
    '''
        Match SMEFT Operators to LEFT at 80GeV = M_W scale
    '''
    def LEFT_matching(self, WC = None):#, scale = 80):
        #match SMEFT WCs onto LEFT WCs at EW scale
        #this script takes some time because it needs to solve the different RGEs
        Lambda = self.scale
        if WC == None:
            WC = self.WC.copy()
            for operator in WC:
                WC[operator]
            
        else:
            C = self.SMEFT_WCs.copy()
            for operator in C:
                C[operator]=0
            for operator in WC:
                C[operator] = WC[operator]
            WC = C.copy()
        
        LEFT_WCs = {"m_bb":0, "SL(6)": 0, "SR(6)": 0, 
                    "T(6)":0, "VL(6)":0, "VR(6)":0, 
                    "VL(7)":0, "VR(7)":0, 
                    "1L(9)":0, "1R(9)":0, 
                    "1L(9)prime":0, "1R(9)prime":0,
                    "2L(9)":0, "2R(9)":0, 
                    "2L(9)prime":0, "2R(9)prime":0, 
                    "3L(9)":0, "3R(9)":0, 
                    "3L(9)prime":0, "3R(9)prime":0,
                    "4L(9)":0, "4R(9)":0, 
                    "5L(9)":0, "5R(9)":0, 
                    "6(9)":0,"6(9)prime":0,
                    "7(9)":0,"7(9)prime":0,
                    "8(9)":0,"8(9)prime":0,
                    "9(9)":0,"9(9)prime":0}
        

        ######################################################################################
        #                                                                                    #
        #                                  dim 3 matching                                    #
        #                                                                                    #
        ######################################################################################
        
        LEFT_WCs["m_bb"] = -vev**2 * WC["LH(5)"] - vev**4/2 * WC["LH(7)"]
        
        ######################################################################################
        #                                                                                    #
        #                                  dim 6 matching                                    #
        #                                                                                    #
        ######################################################################################
        
        LEFT_WCs["VL(6)"] = (vev**3 * V_ud*(- 1j/np.sqrt(2) * WC["LHDe(7)"].conjugate()
                                            + 4*m_e/vev     * WC["LHW(7)"].conjugate())
                             +vev**4 * (2*m_e*V_ud * WC["LLH4W1(9)"].conjugate()
                                        -m_d/4     * WC["deQLH2D(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["VR(6)"] = (vev**3/np.sqrt(2) * WC["LeudH(7)"].conjugate() 
                             - vev**4*m_u/4 * WC["deQLH2D(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["SR(6)"] = (vev**3 *( 1/(2*np.sqrt(2)) * WC["LLQdH1(7)"].conjugate()
                                      -V_ud/2*m_d/vev   * WC["LHD2(7)"].conjugate())
                             +vev**4 * (- m_d*V_ud/2 * WC["LLH4D23(9)"].conjugate()
                                        + m_d*V_ud/4 * WC["LLH4D24(9)"].conjugate()
                                        + m_d/4      * WC["QQLLH2D2(9)"].conjugate()
                                        + m_u/4      * WC["duLLH2D(9)"].conjugate()
                                        + m_e/8      * WC["deQLH2D(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["SL(6)"] = (vev**3 * (  1/(np.sqrt(2))   * WC["LLQuH(7)"].conjugate()
                                       + V_ud/2 * m_u/vev * WC["LHD2(7)"].conjugate())
                             +vev**4 * (+m_u*V_ud/2 * WC["LLH4D23(9)"].conjugate()
                                        -m_u*V_ud/4 * WC["LLH4D24(9)"].conjugate()
                                        -m_u/4      * WC["QQLLH2D2(9)"].conjugate()
                                        -1/4*m_d    * WC["duLLH2D(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["T(6)"]  = (vev**3/(8*np.sqrt(2)) * (2 * WC["LLQdH2(7)"].conjugate()
                                                      +   WC["LLQdH1(7)"].conjugate())
                             +vev**4*(m_e/16 * WC["deQLH2D(9)"].conjugate()))
       
        ######################################################################################
        #                                                                                    #
        #                                  dim 7 matching                                    #
        #                                                                                    #
        ######################################################################################
        
        LEFT_WCs["VL(7)"] = (vev**3*V_ud/2 * (+ 2 * WC["LHD1(7)"].conjugate()
                                              -     WC["LHD2(7)"].conjugate()
                                              + 8 * WC["LHW(7)"].conjugate())
                             +vev**5*(+2* V_ud * WC["LLH4W1(9)"].conjugate()
                                      +V_ud/2  * WC["LLH4D23(9)"].conjugate()
                                      -V_ud/4  * WC["LLH4D24(9)"].conjugate()
                                      -1/4     * WC["QQLLH2D2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["VR(7)"] = ( vev**3 * (-1j * WC["LLduD1(7)"].conjugate())
                             +vev**5 * (1/4 * WC["duLLH2D(9)"].conjugate()))
        
        ######################################################################################
        #                                                                                    #
        #                                  dim 9 matching                                    #
        #                                                                                    #
        ######################################################################################
        
        LEFT_WCs["1L(9)"] = (vev**3 * V_ud**2*(+ 2*WC["LHD1(7)"].conjugate()
                                               + 8*WC["LHW(7)"].conjugate())
                             +vev**5 * (+ 4*V_ud**2 * WC["LLH4W1(9)"].conjugate()
                                        - V_ud**2   * WC["LLH4D24(9)"].conjugate()
                                        - V_ud**2   * WC["LLH4D23(9)"].conjugate()
                                        - V_ud      * WC["QQLLH2D2(9)"].conjugate()
                                        - V_ud/2    * WC["QLQLH2D5(9)"].conjugate()
                                        - V_ud/2    * WC["QLQLH2D2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["1R(9)"] = (vev**5 * (-V_ud**2 * WC["eeH4D2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["1R(9)prime"] = vev**5 * (1/4*WC["ddueue(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["2L(9)"] = vev**5 * (-WC["QuQuLL1(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["2L(9)prime"] = vev**5 * (-WC["dQdQLL1(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["3L(9)"] = vev**5 * (-WC["QuQuLL2(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["3L(9)prime"] = vev**5 * (-WC["dQdQLL2(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["4L(9)"] = (-vev**3 * 2j*V_ud   * WC["LLduD1(7)"].conjugate()
                             +vev**5 * (+ V_ud   * WC["duLLH2D(9)"].conjugate()
                                        - V_ud/2 * WC["dLuLH2D2(9)"].conjugate()
                                        - 1/2    * WC["dQQuLL2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["4R(9)"] = (vev**5 * (-V_ud/2 * WC["deueH2D(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["5L(9)"] = vev**5 * (-1/2 * WC["dQQuLL1(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["6(9)"]  = (vev**5 * (-2/3 * V_ud * WC["dLQeH2D1(9)"].conjugate()
                                       + V_ud/2    * WC["dQLeH2D2(9)"].conjugate()
                                       - 5/12*V_ud    * WC["deQLH2D(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["6(9)prime"]  = (vev**5 * (  1/6 * WC["QudueL2(9)"].conjugate()
                                            + 1/2 * WC["QudueL1(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["7(9)"] = vev**5 * (- V_ud * WC["deQLH2D(9)"].conjugate()
                                     - V_ud * WC["dLQeH2D1(9)"].conjugate())
        
        ######################################################################################
        
        LEFT_WCs["7(9)prime"]  = (vev**5 * (WC["QudueL2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["8(9)"] = (vev**5 * (- V_ud/2 * WC["QueLH2D2(9)"].conjugate()
                                      + V_ud/6 * WC["QeuLH2D2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["8(9)prime"] = (vev**5 * (+ 1/6    * WC["dQdueL2(9)"].conjugate()
                                           + 1/2    * WC["dQdueL1(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["9(9)"] = (vev**5 * (V_ud*WC["QeuLH2D2(9)"].conjugate()))
        
        ######################################################################################
        
        LEFT_WCs["9(9)prime"] = vev**5 * WC["dQdueL2(9)"].conjugate()
        
        ######################################################################################
        
        
        return(LEFT_WCs)
    
    def generate_LEFT_model(self, WC = None, method = None, LEC = None, name = None, PSF_scheme = None):
        if WC == None:
            WC = self.WC.copy()
        if method == None:
            method = self.method
            NMEs = self.NMEs
        elif method != self.method and method+".csv" in os.listdir(cwd+"/../NMEs/"):
            newNMEs, newNMEpanda, newNMEnames = Load_NMEs(method)
            NMEs = newNMEs
        elif method+".csv" not in os.listdir(cwd+"/../NMEs/"):
            warnings.warn("Method",method,"is unavailable. No such file 'NMEs/"+method+".csv'. Keeping current method",self.method)
            method = self.method
            NMEs = self.NMEs
        else:
            method = self.method
            NMEs = self.NMEs
            
        if PSF_scheme == None:
            PSF_scheme = self.PSF_scheme
            
        if name == None:
            name = self.name
        if LEC == None:
            LEC = self.LEC.copy()
            
        LEFT_WCs = self.LEFT_matching(WC)
        model = LEFT(LEFT_WCs, method = method, name = name, PSF_scheme = PSF_scheme)
        model.LEC = LEC
        model.NMEs = NMEs.copy()
        return(model)

    def set_LECs(self, unknown_LECs):
        self.unknown_LECs = unknown_LECs
        
    def t_half(self, isotope, WC = None, method = None):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.t_half(isotope))
        
    def half_lives(self, WC = None, method = None, vary_LECs = False, n_points = 1000):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.half_lives(vary_LECs = vary_LECs, n_points = n_points))
    
    def spectrum(self, Ebar, WC = None, method = None):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.spectrum(Ebar))
    
    def angular_corr(self, Ebar, WC = None, method = None):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.angular_corr(Ebar))
    
    def plot_spec(self, 
                  isotope     = "76Ge",      #isotope to study
                  WC          = None,        #Set WCs, if None self.WC will be used
                  method      = None,        #NME method, if None self.method will be used
                  print_title = False,       #add a title to the plot
                  addgrid     = True,        #If True, plot a grid
                  show_mbb    = True,        #If True, plot mass mechanism for comparison
                  n_points    = 1000,        #number of points to calculate
                  linewidth   = None,        #Width of lines in plot
                  normalize_x = True,        #If True normalize x-axis range [0, 1]
                  savefig     = False,       #If True save figure as file
                  file        = "spec.png",  #Filename and path to store figure to
                  dpi         = 300          #set the resolution of the saved figure in dots per inch
                 ):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.plot_spec(isotope     = isotope, 
                               print_title = print_title, 
                               addgrid     = addgrid, 
                               show_mbb    = show_mbb, 
                               savefig     = savefig, 
                               dpi         = dpi,
                               normalize_x = normalize_x,
                               n_points    = n_points,
                               linewidth   = linewidth,
                               file        = file)
              )
    
    def plot_corr(self, 
                  isotope     = "76Ge",             #isotope to study
                  WC          = None,               #Set WCs, if None self.WC will be used
                  method      = None,               #NME method, if None self.method will be used
                  print_title = False,              #add a title to the plot
                  addgrid     = True,               #If True, plot a grid
                  show_mbb    = True,               #If True, plot mass mechanism for comparison
                  n_points    = 1000,               #number of points to calculate
                  linewidth   = None,               #Width of lines in plot
                  normalize_x = True,               #If True normalize x-axis range [0, 1]
                  savefig     = False,              #If True save figure as file
                  file        = "angular_corr.png", #Filename and path to store figure to
                  dpi         = 300                 #set the resolution of the saved figure in dots per inch
                 ):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.plot_corr(isotope     = isotope, 
                               print_title = print_title, 
                               addgrid     = addgrid, 
                               show_mbb    = show_mbb, 
                               linewidth   = linewidth,
                               dpi         = dpi,
                               normalize_x = normalize_x,
                               n_points    = n_points,
                               savefig     = savefig, 
                               file        = file)
              )
    
    def get_limits(self, 
                   half_life,            #half-life limit
                   isotope,              #isotope the limit is taken from
                   method       = None,  #NME method
                   groups       = False, #group operators into groups with the same limit?
                  ):
        #this function can calculate the limits on the different LEFT coefficients for a given experimental half_life and isotope
        #the limits are calculate at the scale "scale" and for the chosen basis

        results_2GeV = {}
        results = {}
        scales = {}

        #make a backup so you can overwrite the running afterwards
        WC_backup = self.WC.copy()

        #calculate the limits on the WCs at the scale of 2GeV
        if groups:
            WCgroups = list(SMEFT_WCs.keys())
            #define labels for plots
            WC_names = list(SMEFT_WCs.keys())
        else:
            WCgroups = list(SMEFT_WCs.keys())
            #define labels for plots
            WC_names = list(SMEFT_WCs.keys())

        for WC_name in WCgroups:
            WC = {WC_name : 1}
            hl = self.t_half(WC = WC, method = method, isotope = isotope)
            results[WC_name] = np.sqrt(hl/half_life)

            dimension = int(WC_name[-2])
            scale = 1/np.absolute(results[WC_name])**(1/(dimension -4))
            scales[WC_name] = scale

        #only show grouped WCs if desired
        if groups:
            res = {}
            scl = {}
            idx = 0
            print(WC_names)
            for WC in WCgroups:
                WC_name = WC_names[idx]
                res[WC_name] = results[WC]
                scl[WC_name] = scales[WC]
                idx+=1
            results = res.copy()
            scales = scl.copy()


        #take abs value to get positive results in case the numerical estimate returned a negative
        for result in results:
            results[result] = np.absolute(results[result])

        self.WC = WC_backup.copy()

        return(pd.DataFrame({r"Limits [GeV$^{4-d}$]" : results, "Scales [GeV]" : scales}))
    
    def ratios(self, 
               reference_isotope = "76Ge", #half-lives are normalized with regard to this isotopes
               normalized        = True,   #if True normalize ratios to the standard mass mechnism
               WC                = None,   #You can reset the WCs here @ m_W
               method            = None,   #Choose a different NME method if desired
               vary_LECs         = False,  #If True vary the unknown LECs
               n_points          = 100     #number of points to plot in the variation
              ):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.ratios(reference_isotope = reference_isotope, 
                            normalized = normalized, vary_LECs = vary_LECs, 
                            n_points = n_points))
    
    def plot_ratios(self, 
                    reference_isotope = "76Ge",        #half-lives are normalized with regard to this isotopes
                    normalized        = True,          #if True normalize ratios to the standard mass mechnism
                    show_central      = True,          #Show the median values of the variation
                    WC                = None,          #You can reset the WCs here @ m_W
                    method            = None,          #Choose a different NME method if desired
                    vary_LECs         = False,         #If True vary the unknown LECs
                    n_points          = 100,           #number of points to plot in the variation
                    alpha             = 0.25,          #alpha value of the datapoints
                    color             = "b",           #Color of the plotted datapoints
                    addgrid           = True,          #If True plot a grid
                    savefig           = False,         #If True save figure as file
                    file              = "ratios.png",  #Filename and path to store figure to
                    dpi               = 300            #set the resolution of the saved figure in dots per inch
                   ):
        
        model = self.generate_LEFT_model(WC, method, LEC = None)
        
        return(model.plot_ratios(reference_isotope = reference_isotope, 
                                 show_central      = show_central,
                                 normalized        = normalized, 
                                 vary_LECs         = vary_LECs, 
                                 n_points          = n_points, 
                                 color             = color,
                                 addgrid           = addgrid, 
                                 savefig           = savefig, 
                                 alpha             = alpha, 
                                 dpi               = dpi,
                                 file              = file
                                )
              )
    
    def plot_WC_variation(self, 
                          xaxis             = "m_min",   #WC to vary on the x-axis ["m_min, "m_sum"], or any of the LEFT WCs
                          yaxis             = "t",       #choose from ["t", "m_eff", "1/t"]
                          isotope           = "76Ge",    #Use NMEs from this isotope for the calculation
                          x_min             = 1e-4,      #minimal x-axis value
                          x_max             = 1e+0,      #maximal x-axis value
                          y_min             = None,      #minimal y-axis value
                          y_max             = None,      #maximal y-axis value
                          xscale            = "log",     #x-axis scaling
                          yscale            = "log",     #y-axis scaling
                          n_points          = 100,       #number of points to plot.
                          cosmo             = False,     #plot cosmology limit on m_sum
                          m_cosmo           = 0.15,      #limit on m_sum
                          limits            = None,      #y-axis limits
                          experiments       = None,      #half-life limits from experiments - get converted to y-axis limits
                          ordering          = "both",    #neutrino mass ordering ["both", "NO", "IO"]
                          dcp               = 1.36,      #dirac CP phase
                          numerical_method  = "Powell",  #numerical method for optimization
                          show_mbb          = False,     #if True plot the standard mass mechanism as a reference
                          normalize         = False,     #normalize y-axis to standard mass mechanism
                          colorNO           = "b",       #color of normal ordering
                          colorIO           = "r",       #color of inverted ordering
                          legend            = True,      #plot legend
                          labelNO           = None,      #label for normal ordering in legend
                          labelIO           = None,      #label for inverted ordering in legend
                          autolabel         = True,      #set labels automatically
                          alpha_plot        = 0.5,       #alpha value of main plots
                          alpha_mass        = 0.05,      #alpha value of mass plot of show_mbb = True
                          alpha_cosmo       = 0.1,       #alpha of cosmo limit
                          vary_phases       = True,      #vary complex WC phases
                          alpha             = [0,0],     #Majorana phases, or WCs complex phase
                          savefig           = False,     #if True save figure as file
                          file              = "var.png", #Filename and path to store figure to
                          dpi               = 300        #set the resolution of the saved figure in dots per inch
                         ):
        
        if xaxis in ["m_min", "m_sum"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)

            return(model.plot_WC_variation(xaxis            = xaxis, 
                                           yaxis            = yaxis, 
                                           isotope          = isotope, 
                                           x_min            = x_min, 
                                           x_max            = x_max, 
                                           y_min            = y_min, 
                                           y_max            = y_max, 
                                           xscale           = xscale, 
                                           yscale           = yscale,
                                           n_points         = n_points, 
                                           cosmo            = cosmo, 
                                           m_cosmo          = m_cosmo, 
                                           limits           = limits, 
                                           experiments      = experiments, 
                                           ordering         = ordering, 
                                           dcp              = dcp, 
                                           numerical_method = numerical_method,
                                           show_mbb         = show_mbb, 
                                           normalize        = normalize, 
                                           colorNO          = colorNO, 
                                           colorIO          = colorIO, 
                                           legend           = legend, 
                                           labelNO          = labelNO,
                                           labelIO          = labelIO,
                                           autolabel        = autolabel, 
                                           alpha_plot       = alpha_plot,
                                           alpha_mass       = alpha_mass, 
                                           alpha_cosmo      = alpha_cosmo,
                                           vary_phases      = vary_phases,
                                           alpha            = alpha,
                                           savefig          = savefig, 
                                           file             = file, 
                                           dpi              = dpi
                                          )
                  )
        
        else:
            #xaxis defines varied WC
            vary_WC = xaxis

            #check if yaxis is allowed
            if yaxis not in ["t", "m_eff", "1/t"]:
                warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
                yaxis = "t"

            #color list to use in plot
            colors = [colorNO, colorIO]


            #check if comparison to mass mechanism is possible
            if vary_WC not in ["m_bb", "m_min", "m_sum"] and show_mbb:
                warnings.warn("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass m_min, the sum of neutrino masses m_sum or m_bb on the x axis. Setting show_mbb = False")
                show_mbb = False

            #get datapoints
            print("Generating Datapoints...")
            if vary_phases:
                datapoints = self.WC_variation(isotope           = isotope, 
                                               xaxis             = xaxis, 
                                               yaxis             = yaxis, 
                                               x_min             = x_min, 
                                               x_max             = x_max, 
                                               n_points          = n_points, 
                                               WC                = None, 
                                               ordering          = ordering,
                                               normalize         = normalize, 
                                               dcp               = dcp, 
                                               numerical_method  = numerical_method)
                print("Datapoints generated")
                #store datapoint arrays
                xNO    = datapoints["xNO"]
                NO_min = datapoints[yaxis+"_min (NO)"]
                NO_max = datapoints[yaxis+"_max (NO)"]
                xIO    = datapoints["xIO"]
                IO_min = datapoints[yaxis+"_min (IO)"]
                IO_max = datapoints[yaxis+"_max (IO)"]
            
            else:
                if xscale == "lin":
                    m_min_array = np.linspace(x_min, x_max, n_points)
                else:
                    m_min_array = np.logspace(np.log10(x_min), np.log10(x_max), n_points)

                xNO    = np.zeros(n_points)
                NO_min = xNO.copy()
                NO_max = xNO.copy()
                xIO    = xNO.copy()
                IO_min = xNO.copy()
                IO_max = xNO.copy()

                if xaxis not in ["m_min", "m_sum"]:
                    try:
                        alpha = alpha[0]
                    except:
                        pass

                for idx in range(n_points):
                    if yaxis == "t":
                        m_min       = m_min_array[idx]
                        xNO[idx]    = m_min
                        NO_min[idx] = self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                        NO_max[idx] = NO_min[idx]
                        xIO[idx]    = m_min
                        IO_min[idx] = self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                        IO_max[idx] = IO_min[idx]

                    elif yaxis == "1/t":
                        m_min       = m_min_array[idx]
                        xNO[idx]    = m_min
                        NO_min[idx] = 1/self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                        NO_max[idx] = NO_min[idx]
                        xIO[idx]    = m_min
                        IO_min[idx] = 1/self._t_half(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                        IO_max[idx] = IO_min[idx]

                    else:
                        m_min       = m_min_array[idx]
                        xNO[idx]    = m_min
                        NO_min[idx] = self._m_eff(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "NO")
                        NO_max[idx] = NO_min[idx]
                        xIO[idx]    = m_min
                        IO_min[idx] = self._m_eff(alpha = alpha, m_min = m_min, vary_WC = xaxis, ordering = "IO")
                        IO_max[idx] = IO_min[idx]

            #generate arrays for mass mechanism datapoints
            if show_mbb:
                NO_min_mbb = np.zeros(n_points)
                NO_max_mbb = np.zeros(n_points)
                IO_min_mbb = np.zeros(n_points)
                IO_max_mbb = np.zeros(n_points)
            else:
                NO_min_mbb = None
                NO_max_mbb = None
                IO_min_mbb = None
                IO_max_mbb = None

            #get datapoints for mass comparison
            if show_mbb:
                print("Generating Datapoints for mass mechanism...")
                if xaxis == "m_sum":
                    xaxismbb = "m_sum"
                else:
                    xaxismbb = "m_min"
                datapoints = self.WC_variation(isotope           = isotope, 
                                               xaxis             = xaxismbb, 
                                               yaxis             = yaxis, 
                                               x_min             = x_min, 
                                               x_max             = x_max, 
                                               n_points          = n_points, 
                                               WC                = {}, 
                                               ordering          = ordering,
                                               normalize         = normalize, 
                                               dcp               = dcp, 
                                               numerical_method  = numerical_method)
                print("Datapoints for mass mechanism generated")
                xNO_mbb    = datapoints["xNO"]
                NO_min_mbb = datapoints[yaxis+"_min (NO)"]
                NO_max_mbb = datapoints[yaxis+"_max (NO)"]
                xIO_mbb    = datapoints["xIO"]
                IO_min_mbb = datapoints[yaxis+"_min (IO)"]
                IO_max_mbb = datapoints[yaxis+"_max (IO)"]

            print("Preparing plot")
            #set legend labels
            if autolabel:
                if vary_WC  in ["m_min", "m_sum"]:
                    NO_label = "NO"
                    IO_label =  "IO"
                else:
                    NO_label = None
                    IO_label =  None

            #initiate limits dict
            if limits == None:
                limits = {}

            #Adjust experimental limits
            if experiments != None:
                for experiment in experiments:
                    #extract yaxis limit
                    y_data = experiments[experiment]["limit"]

                    if yaxis == "m_eff":
                        #convert half-life into m_bb limit
                        y_data = self.get_limits(y_data, isotope = isotope)["Limits"]["m_bb"]*1e+9

                    #overwrite yaxis limit
                    experiments[experiment]["limit"] = y_data

                    #store experiments in limits dict
                    limits[experiment] = experiments[experiment]

            #define axis labels

            #y-axis
            if normalize:
                if yaxis == "m_eff":
                    ylabel = r"$\left|\frac{m_{\beta\beta}^{eff}}{m_{\beta\beta}}\right|$"
                elif yaxis == "1/t":
                    ylabel = r"$\frac{t_{1/2, m_{\beta\beta}}}{t_{1/2}}$"
                else:
                    ylabel = r"$\frac{t_{1/2}}{t_{1/2, m_{\beta\beta}}}$"

            else:
                if yaxis == "m_eff":
                    ylabel = r"$|m_{\beta\beta}^{eff}|$ [eV]"
                elif yaxis == "1/t":
                    ylabel = r"$t_{1/2}^{-1}$ [yr$^{-1}$]"
                else:
                    ylabel = r"$t_{1/2}$ [yr]"

            #x-axis
            if vary_WC == "m_min":
                xlabel = r"$m_{min}$ [eV]"
            elif vary_WC == "m_sum":
                xlabel = r"$\sum_i m_{i}$ [eV]"
            elif vary_WC == "m_bb":
                xlabel = r"$|m_{\beta\beta}|$ [eV]"
            else:
                xlabel = r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$ $[\mathrm{GeV}^{("+str(4-int(vary_WC[-2]))+")}]$"


            #set y-axis limits if not set manually
            if yaxis in ["m_eff", "1/t"]:
                if y_max == None:
                    if vary_WC == "m_min":
                        y_max = np.max([np.max(IO_max), np.max(NO_max)])
                    else:
                        y_max = np.max(NO_max)

                    if normalize:
                        y_min = 1e-4
                        y_max = 1e+4

                    y_max = 10**np.ceil(np.log10(y_max))
                if y_min == None:

                    if yaxis == "m_eff":
                        y_min = 1e-4*y_max
                    elif yaxis == "1/t":
                        scaling = 1e-8
                        if vary_WC in ["m_min", "m_sum"]:
                            y_min = np.max([10**np.floor(np.log10(np.min([np.min(NO_max), np.min(IO_max)]))),scaling*y_max])
                        else:
                            y_min = np.max([10**np.floor(np.log10(np.min(NO_max))),scaling*y_max])

            else:
                if y_min == None:
                    if vary_WC in ["m_min", "m_sum"]:
                        y_min = np.min([np.min(IO_max), np.min(NO_max)])
                    else:
                        y_min = np.min(NO_max)
                    y_min = 10**np.floor(np.log10(y_min))
                if y_max == None:
                    y_max = np.min([10**np.ceil(np.log10(np.max([NO_max, IO_max]))),1e+8*y_min])

            print("Generating Plot")
            #Generate Figure
            fig = plots.lobster(xNO = xNO, NO_min = NO_min, NO_max = NO_max, vary_WC = vary_WC, 
                                NO_min_mbb = NO_min_mbb, NO_max_mbb = NO_max_mbb, 
                                IO_min_mbb = IO_min_mbb, IO_max_mbb = IO_max_mbb, 
                                NO_label = labelNO, IO_label = labelIO, autolabel = autolabel,
                                x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, 
                                xscale = xscale, yscale = yscale, legend = legend, 
                                xIO = xIO, IO_min = IO_min, IO_max = IO_max, ordering = ordering, 
                                limits = limits, ylabel = ylabel, xlabel = xlabel, 
                                cosmo = cosmo, m_cosmo = m_cosmo, colors = colors, 
                                alpha_plot = alpha_plot, alpha_mass = alpha_mass, alpha_cosmo = alpha_cosmo, 
                                show_mbb = show_mbb, normalize = normalize, 
                                savefig = savefig, file = file, dpi = dpi)
            return(fig)
    
    def plot_WC_variation_scatter(self, 
                                  xaxis             = "m_min",        #WC to vary on the x-axis ["m_min, "m_sum"], or any of the LEFT WCs
                                  yaxis             = "t",            #choose from ["t", "m_eff", "1/t"]
                                  isotope           = "76Ge",         #Use NMEs from this isotope for the calculation
                                  vary_phases       = True,
                                  vary_LECs         = False, 
                                  alpha             = [0,0],          #Majorana phases, or WCs complex phase
                                  n_points          = 10000,          #number of points to plot.
                                  markersize        = 0.5, 
                                  x_min             = 1e-4,           #minimal x-axis value
                                  x_max             = 1,              #maximal x-axis value
                                  y_min             = None,           #minimal y-axis value 
                                  y_max             = None,           #maximal y-axis value
                                  xscale            = "log",          #x-axis scaling
                                  yscale            = "log",          #y-axis scaling
                                  limits            = None,           #y-axis limits
                                  experiments       = None,           #half-life limits from experiments - get converted to y-axis limits
                                  ordering          = "both",         #neutrino mass ordering ["both", "NO", "IO"]
                                  dcp               = 1.36,           #dirac CP phase
                                  show_mbb          = False,          #if True plot the standard mass mechanism as a reference
                                  normalize         = False,          #normalize y-axis to standard mass mechanism
                                  cosmo             = False,          #plot cosmology limit on m_sum
                                  m_cosmo           = 0.15,           #limit on m_sum
                                  colorNO           = "b",            #color of normal ordering
                                  colorIO           = "r",            #color of inverted ordering 
                                  legend            = True,           #plot legend
                                  labelNO           = None,           #label for normalordering in legend
                                  labelIO           = None,           #label for inverted ordering in legend
                                  autolabel         = True,           #set labels automatically
                                  alpha_plot        = 1,              #alpha value of main plots
                                  alpha_mass        = 0.05,           #alpha value of mass plot of show_mbb = True
                                  alpha_cosmo       = 0.1,            #alpha of cosmo limit
                                  savefig           = False,          #if True save figure as file
                                  file              = "var_scat.png", #Filename and path to store figure to
                                  dpi               = 300             #set the resolution of the saved figure in dots per inch
                                 ):
                
        if xaxis in ["m_min", "m_sum"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)

            return(model.plot_WC_variation_scatter(xaxis       = xaxis, 
                                                   yaxis       = yaxis, 
                                                   isotope     = isotope, 
                                                   vary_phases = vary_phases, 
                                                   vary_LECs   = vary_LECs, 
                                                   n_points    = n_points, 
                                                   markersize  = markersize,
                                                   x_min       = x_min, 
                                                   x_max       = x_max, 
                                                   y_min       = y_min, 
                                                   y_max       = y_max, 
                                                   xscale      = xscale,
                                                   yscale      = yscale,
                                                   limits      = limits, 
                                                   experiments = experiments, 
                                                   ordering    = ordering,
                                                   dcp         = dcp, 
                                                   show_mbb    = show_mbb, 
                                                   normalize   = normalize,
                                                   cosmo       = cosmo,
                                                   m_cosmo     = m_cosmo,
                                                   colorNO     = colorNO,
                                                   colorIO     = colorIO, 
                                                   legend      = legend,
                                                   labelNO     = labelNO, 
                                                   labelIO     = labelIO, 
                                                   autolabel   = autolabel, 
                                                   alpha_plot  = alpha_plot,
                                                   alpha_mass  = alpha_mass, 
                                                   alpha_cosmo = alpha_cosmo, 
                                                   savefig     = savefig, 
                                                   file        = file, 
                                                   dpi         = dpi, 
                                                   alpha       = alpha
                                                  )
                  )
        else:
            vary_WC = xaxis
            if yaxis not in ["t", "m_eff", "1/t"]:
                warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
                yaxis = "t"

            colors = [colorNO, colorIO]



            if vary_WC not in ["m_bb", "m_min", "m_sum"] and show_mbb:
                warnings.warn("comparing to mass mechanism only makes sense if you put either the minimal neutrino mass m_min, the sum of neutrino masses m_sum or m_bb on the x axis. Setting show_mbb = False")
                show_mbb = False

            #get datapoints
            print("Generating Datapoints...")
            datapoints = self.WC_variation_scatter(isotope           = isotope, 
                                                   xaxis             = xaxis, 
                                                   yaxis             = yaxis, 
                                                   x_min             = x_min, 
                                                   x_max             = x_max, 
                                                   n_points          = n_points, 
                                                   WC                = None, 
                                                   ordering          = ordering,
                                                   normalize         = normalize, 
                                                   dcp               = dcp, 
                                                   vary_phases       = vary_phases, 
                                                   vary_LECs         = vary_LECs, 
                                                   alpha             = alpha
                                                  )
            print("Datapoints generated")
            points = np.array([datapoints["xNO"], datapoints["yNO"]]).T
            pointsIO = np.array([datapoints["xIO"], datapoints["yIO"]]).T

            #datapoints for mass comparison
            if show_mbb:
                xNO_mbb    = np.zeros(100)
                NO_min_mbb = np.zeros(100)
                NO_max_mbb = np.zeros(100)
                xIO_mbb    = np.zeros(100)
                IO_min_mbb = np.zeros(100)
                IO_max_mbb = np.zeros(100)
            else:
                xNO_mbb    = None
                NO_min_mbb = None
                NO_max_mbb = None
                xIO_mbb    = None
                IO_min_mbb = None
                IO_max_mbb = None

            if show_mbb:
                print("Generating Datapoints for mass mechanism...")
                if xaxis == "m_sum":
                    xaxismbb = "m_sum"
                else:
                    xaxismbb = "m_min"
                datapoints = self.WC_variation(isotope           = isotope, 
                                               xaxis             = xaxismbb, 
                                               yaxis             = yaxis, 
                                               x_min             = x_min, 
                                               x_max             = x_max, 
                                               n_points          = 100, 
                                               WC                = {}, 
                                               ordering          = ordering,
                                               normalize         = normalize, 
                                               dcp               = dcp, 
                                               numerical_method  = "Powell")
                print("Datapoints for mass mechanism generated")

                xNO_mbb    = datapoints["xNO"]
                NO_min_mbb = datapoints[yaxis+"_min (NO)"]
                NO_max_mbb = datapoints[yaxis+"_max (NO)"]
                xIO_mbb    = datapoints["xIO"]
                IO_min_mbb = datapoints[yaxis+"_min (IO)"]
                IO_max_mbb = datapoints[yaxis+"_max (IO)"]

            print("Preparing plot...")
            #set legend labels
            if autolabel:
                if vary_WC  in ["m_min", "m_sum"]:
                    NO_label = "NO"
                    IO_label =  "IO"
                else:
                    NO_label = None
                    IO_label =  None

            #initiate limits dict
            if limits == None:
                limits = {}

            #Adjust experimental limits
            if experiments != None:
                for experiment in experiments:
                    y_data = experiments[experiment]["limit"]

                    if yaxis == "m_eff":
                        #convert half-life into m_bb limit
                        y_data = self.get_limits(y_data, isotope = isotope)["Limits"]["m_bb"]*1e+9

                    experiments[experiment]["limit"] = y_data

                    limits[experiment] = experiments[experiment]

            #define axis labels

            #y-axis
            if normalize:
                if yaxis == "m_eff":
                    ylabel = r"$\left|\frac{m_{\beta\beta}^{eff}}{m_{\beta\beta}}\right|$"
                elif yaxis == "1/t":
                    ylabel = r"$\frac{t_{1/2, m_{\beta\beta}}}{t_{1/2}}$"
                else:
                    ylabel = r"$\frac{t_{1/2}}{t_{1/2, m_{\beta\beta}}}$"

            else:
                if yaxis == "m_eff":
                    ylabel = r"$|m_{\beta\beta}^{eff}|$ [eV]"
                elif yaxis == "1/t":
                    ylabel = r"$t_{1/2}^{-1}$ [yr$^{-1}$]"
                else:
                    ylabel = r"$t_{1/2}$ [yr]"

            #x-axis
            if vary_WC == "m_min":
                xlabel = r"$m_{min}$ [eV]"
            elif vary_WC == "m_sum":
                xlabel = r"$\sum_i m_{i}$ [eV]"
            elif vary_WC == "m_bb":
                xlabel = r"$|m_{\beta\beta}|$ [eV]"
            else:
                xlabel = r"$|C_{"+vary_WC[:-3]+"}^{"+vary_WC[-3:]+"}|$ $[\mathrm{GeV}^{("+str(4-int(vary_WC[-2]))+")}]$"


            #set y-axis limits if not set manually

            #y-axis limits for scatter plot
            #define y-axis limits if not given
            if y_min == None:
                if vary_WC in ["m_min", "m_sum"]:
                    if ordering == "both":
                        y_min = np.min([np.min(points[:,1]), np.min(pointsIO[:,1])])
                    elif ordering == "IO":
                        y_min = np.min(pointsIO[:,1])
                    else:
                        y_min = np.min(points[:,1])
                else:
                    y_min = np.min(points[:,1])
            if y_max == None:
                if vary_WC in ["m_min", "m_sum"]:
                    if ordering == "both":
                        y_max = np.max([np.max(points[:,1]), np.max(pointsIO[:,1])])
                    elif ordering == "IO":
                        y_max = np.max(pointsIO[:,1])
                    else:
                        y_max = np.max(points[:,1])
                else:
                    y_max = np.max(points[:,1])

            print("Generating Plot")
            #Generate Figure
            fig = plots.lobster_scatter(points = points, pointsIO = pointsIO, vary_WC = vary_WC,
                                        x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max,
                                        xlabel = xlabel, ylabel = ylabel, markersize = markersize, 
                                        xscale = xscale, yscale = yscale, legend = legend,
                                        alpha_plot = alpha_plot, alpha_mass = alpha_mass, alpha_cosmo = alpha_cosmo,
                                        show_mbb = show_mbb, xNO_mbb = xNO_mbb, xIO_mbb = xIO_mbb, ordering = ordering,
                                        cosmo = cosmo, m_cosmo = m_cosmo, NO_min_mbb = NO_min_mbb, 
                                        NO_max_mbb = NO_max_mbb, IO_min_mbb = IO_min_mbb, IO_max_mbb = IO_max_mbb,
                                        NO_label = labelNO, IO_label = labelIO, autolabel = autolabel,
                                        limits = limits, savefig = savefig, file = file,  dpi = dpi)
            
            return(fig)
        
    #same as plot_WC_variation but with yaxis fixed to "m_eff"
    def plot_m_eff(self, **args):
        return(self.plot_WC_variation(yaxis = "m_eff", **args))
        
    #same as plot_WC_variation but with yaxis fixed to "1/t"
    def plot_t_half_inv(self, **args):
        return(self.plot_WC_variation(yaxis = "1/t", **args))
        
    #same as plot_WC_variation but with yaxis fixed to "t"
    def plot_t_half(self, **args):
        return(self.plot_WC_variation(yaxis = "t", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "m_eff"
    def plot_m_eff_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "m_eff", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "t" 
    def plot_t_half_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "t", **args))
    
    #same as plot_WC_variation_scatter but with yaxis fixed to "1/t" 
    def plot_t_half_inv_scatter(self, **args):
        return(self.plot_WC_variation_scatter(yaxis = "1/t", **args))
    
    #generate readable expression of half-life formula
    def generate_formula(self, isotope, WC = None, method = None, decimal = 2, output = "latex", 
                         unknown_LECs = None, PSF_scheme = None):
        #set NME method
        method = self._set_method_locally(method)
        
        if unknown_LECs == None:
            unknown_LECs = self.unknown_LECs
        if PSF_scheme == None:
            PSF_scheme = self.PSF_scheme
            
        if WC == None:
            WC = []
            for WCs in self.WC:
                if self.WC[WCs] != 0:
                    WC.append(WCs)
        return(f.generate_formula(WC = WC, isotope = isotope, output = output, 
                                  method = method, decimal = decimal, unknown_LECs = unknown_LECs, 
                                  PSF_scheme = PSF_scheme))
    
    #generate half-life matrix
    def generate_matrix(self, isotope, WC = None, method = None, unknown_LECs = None, PSF_scheme = None):
        #set NME method
        method = self._set_method_locally(method)
        
        if unknown_LECs == None:
            unknown_LECs = self.unknown_LECs
        if PSF_scheme == None:
            PSF_scheme = self.PSF_scheme
        
        if WC == None:
            WC = []
            for WCs in self.WC:
                if self.WC[WCs] != 0:
                    WC.append(WCs)
        return(f.generate_matrix(WC = WC, isotope = isotope, method = method, 
                                 unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme))
    
    
    
        #fancy plots:
    def _m_bb(self, alpha, m_min=1, ordering="NO", dcp=1.36):
        #this function returns the effective electron-neutrino Majorana mass m_bb from m_min for a fixed mass ordering
        
        #majorana phases
        alpha1=alpha[0]
        alpha2=alpha[1]

        #squared mass differences
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3

        #get mass eigenvalues from minimal neutrino mass
        m = m_min
        m1 = m
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)

        m3IO = m
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)

        #create diagonal mass matrices
        M_nu_NO = np.diag([m1,m2,m3])
        M_nu_IO = np.diag([m1IO,m2IO,m3IO])

        #mixing angles
        #s12 = np.sqrt(0.307)
        #s23 = np.sqrt(0.546)
        #s13 = np.sqrt(2.2e-2)

        #c12 = np.cos(np.arcsin(s12))
        #c23 = np.cos(np.arcsin(s23))
        #c13 = np.cos(np.arcsin(s13))

        #mixing marix
        U = np.array([[c12*c13, s12*c13, s13*np.exp(-1j*dcp)], 
                       [-s12*c23-c12*s23*s13*np.exp(1j*dcp), c12*c23-s12*s23*s13*np.exp(1j*dcp), s23*c13], 
                       [s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp), c23*c13]])

        majorana = np.diag([1, np.exp(1j*alpha1), np.exp(1j*alpha2)])

        UPMNS = U @ majorana

        #create non-diagonal mass matrix
        m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
        m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

        if ordering == "NO":
            return(m_BB_NO)
        elif ordering =="IO":
            return(m_BB_IO)
        else:
            return(m_BB_NO,m_BB_IO)
        
        
    def _m_bb_minus(self, alpha, m_min=1, ordering="both", dcp=1.36):
        ##this function returns -m_bb from m_min
        res = self._m_bb(alpha = alpha,
                         m_min = m_min,
                         ordering = ordering, 
                         dcp = dcp)
        
        if ordering == "NO":
            return(-res)
        elif ordering =="IO":
            return(-res)
        else:
            return(-res[0],-res[1])
        
    #this function returns the effective Majorana mass parameter m_eff not no be confused with m_bb
    #m_bb: Majorana mass in the Lagrangian (determined from m_min and mass ordering)
    #m_eff: Majorana mass which would give the same half-life as given model parameters (determined from WCs and half-life)
    def _m_eff(self, 
               alpha,                       #complex phase of variational WC or if vary_WC == "m_min" this is an array of both majorana phases
               m_min             = 1,       #absolute value of variational WC
               ordering          = "both",  #mass ordering either NO, IO or both
               dcp               = 1.36,    #dirac cp phase
               isotope           = "76Ge",  #name of the isotope
               normalize         = False,   #return value normalized to the standard mass mechanism
               vary_WC           = "m_min", #variational WC
              ):
        
        if vary_WC in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model._m_eff(alpha, m_min, ordering, dcp, isotope, normalize, vary_WC))
        
        
        else:
            #make backups
            WC_backup = self.WC.copy()
            #majorana phases
            try:
                alpha = alpha[0]
            except:
                pass
            
            self.WC[vary_WC] = np.exp(1j*alpha)*m_min

            LEFT_model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)

            G01    = LEFT_model.to_G(isotope)["01"]
            M3     = np.abs(LEFT_model.amplitudes(isotope)[1]["nu(3)"])
            NO_eff = self.m_e / (g_A**2*M3*G01**(1/2)) * self.t_half(isotope)**(-1/2)

            self.WC = WC_backup.copy()
            return(NO_eff)
    
    
    
    def _m_eff_minus(self, 
                     alpha,                       #complex phase of variational WC or if vary_WC == "m_min" this is an array of both majorana phases
                     m_min             = 1,       #absolute value of variational WC
                     ordering          = "both",  #mass ordering either NO, IO or both
                     dcp               = 1.36,    #dirac cp phase
                     isotope           = "76Ge",  #name of the isotope
                     normalize         = False,   #return value normalized to the standard mass mechanism
                     vary_WC           = "m_min", #variational WC
                    ):
        
        if vary_WC in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model._m_eff_minus(alpha, m_min, ordering, dcp, isotope, normalize, vary_WC))
        
        else:
            res = self._m_eff(alpha     = alpha,
                              m_min     = m_min, 
                              ordering  = ordering,
                              dcp       = dcp,
                              isotope   = isotope,   
                              normalize = normalize, 
                              vary_WC   = vary_WC, 
                  )
            if ordering == "NO":
                return(-res)
            elif ordering =="IO":
                return(-res)
            else:
                return(-res[0],-res[1])
        
       
    def _m_eff_minmax(self, 
                      m_min, 
                      isotope           = "76Ge",   #isotope of interest
                      ordering          = "both",   #neutrino mass ordering ["NO", "IO", "both"]
                      dcp               = 1.36,     #dirac cp-phase
                      numerical_method  = "Powell", #numerical method for minimization process
                      normalize         = False,    #normalize to standard mass mechanism if desired
                      vary_WC           = "m_min"   #WC to be varied
                     ):
        if vary_WC in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model._m_eff_minmax(m_min, isotope, ordering, dcp, numerical_method, normalize, vary_WC))
        
        #this function returns the effective majorana mass m_bb_eff
        #m_bb_eff reflects the majorana mass m_bb necessary to generate the same half-live as the input model does
        

        #the neutrino mass from the model needs to be overwritten to be able to produce the plot
        #this is because for the plot we want to be able to vary m_min!
        
        else:
            #first we do a backup of the WCs to restore them at the end
            WC_backup = self.WC.copy()

            #alternatively if the variational WC is not m_min you need to set the corresponding WC to 0
            self.WC[vary_WC] = 0

            pre_alpha = 1
            #get minimal and maximal m_bb by varying phases
            NO_min_eff = (scipy.optimize.minimize(self._m_eff, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])
            NO_max_eff = (-scipy.optimize.minimize(self._m_eff_minus, x0=pre_alpha,args=(m_min, "NO", dcp, isotope, normalize, vary_WC), method=numerical_method)["fun"])


            self.WC = WC_backup.copy()
            return ([NO_min_eff*1e+9, NO_max_eff*1e+9])
    
    #get half-life from phases and m_min
    def _t_half(self, 
                alpha, 
                m_min,                     #Absolute value of variational parameter. Given in eV if massive, else dimensionless
                ordering  = "NO",          #NO or IO
                dcp       = 1.36,          #cp-phase dirac neutrinos
                isotope   = "76Ge",        #isotope to calculate
                normalize = False,         #return value normalized to the standard mass mechanism
                vary_WC   = "m_min"        #varuational WC
               ):
        if vary_WC in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model._t_half(alpha, m_min, ordering, dcp, isotope, normalize, vary_WC))
        
        else:
            #make a backup of WCs
            WC_backup = self.WC.copy()


            try:
                alpha = alpha[0]
            except:
                pass

            self.WC[vary_WC] = np.exp(1j*alpha)*m_min

            #get half-life
            t_half = self.t_half(isotope = isotope)

            #normalize if desired
            if normalize:
                WCbackup = self.WC.copy()
                for operator in self.WC:
                    if operator != "LH(5)":
                        self.WC[operator] = 0
                t_half_mbb = self.t_half(isotope = isotope)
                t_half/=t_half_mbb
                self.WC = WCbackup.copy()

            #reset WCs from backup
            self.WC = WC_backup.copy()

            return(t_half)
    
    #negative half-life (use to maximize half-life)
    def _t_half_minus(self, 
                      alpha, 
                      m_min,                      #value of variational parameter
                      ordering          = "NO",   #NO or IO
                      dcp               = 1.36,   #cp-phase dirac neutrinos
                      isotope           = "76Ge", #isotope to calculate
                      normalize         = False,  #return value normalized to the standard mass mechanism
                      vary_WC           = "m_min"
                     ):
        
        res = self._t_half(alpha, 
                           m_min,   
                           ordering=ordering,
                           dcp=dcp,    
                           isotope = isotope,
                           normalize = normalize,
                           vary_WC  = vary_WC
                          )
        return(-res)
    
    #find the minimum and maximum values of the half-life for a given m_min
    def _t_half_minmax(self, 
                       m_min, 
                       ordering          = "both", 
                       dcp               = 1.36, 
                       isotope           = "76Ge", 
                       numerical_method  = "powell",
                       tol               = None, 
                       normalize         = False, 
                       vary_WC           = "m_min"
                      ):
        if vary_WC in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model._t_half_minmax(m_min, ordering, dcp, isotope, numerical_method, tol, normalize, vary_WC))
        
        #the phase is a free parameter. Set the initial value
        pre_alpha = 1
            
        #normal ordering only
        #find minimum of NO
        t_half_min_NO = (scipy.optimize.minimize(self._t_half, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])

        #find maximum of NO
        t_half_max_NO = (scipy.optimize.minimize(self._t_half_minus, x0=pre_alpha, args=(m_min, ordering, dcp, isotope, normalize, vary_WC), method=numerical_method, tol=tol)["fun"])

        #return results
        return([t_half_min_NO, t_half_max_NO])
                
    def WC_variation(self, 
                     isotope           = "76Ge",
                     xaxis             = "m_min", 
                     yaxis             = "t",
                     x_min             = 1e-4,
                     x_max             = 1e+0,
                     n_points          = 100,
                     WC                = None,
                     ordering          = "both",
                     normalize         = False,
                     dcp               = 1.36,
                     numerical_method  = "Powell",        #numerical method for optimization
                    ):
        if xaxis in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model.WC_variation(isotope, xaxis, yaxis, x_min, x_max, n_points, WC, 
                                      ordering, normalize, dcp, numerical_method))
        
        #set WCs
        WCbackup = self.WC.copy()
        if WC == None:
            pass
        
        else:
            C = self.WC.copy()
            for x in C:
                C[x] = 0
            for x in WC:
                C[x] = WC[x]
            self.WC = C.copy()
                
        
        if yaxis not in ["t", "m_eff", "1/t"]:
            warnings.warn("yaxis",yaxis,'is no valid parameter. Choose from ["m_eff", "t", "1/t"]. Setting yaxis = "t"')
            yaxis = "t"
            
        M = np.logspace((np.log10(x_min)),(np.log10(x_max)), n_points)
                
        NO_min = np.zeros(n_points)
        NO_max = np.zeros(n_points)
        IO_min = np.zeros(n_points)
        IO_max = np.zeros(n_points)
            
        #choose optimization function
        if yaxis == "m_eff":
            optimize = self._m_eff_minmax
        else:
            optimize = self._t_half_minmax
            
        #get y-axis values
        #or generate minimal and maximal possible values
        for idx in range(n_points):
            m_min = M[idx]

            #Generate Plot with dimensionless WC varried
            [NO_min[idx], NO_max[idx]] = optimize(m_min     = m_min,
                                                  isotope   = isotope, 
                                                  ordering  = "NO", 
                                                  dcp       = dcp,
                                                  normalize = normalize,
                                                  vary_WC   = xaxis, 
                                                  numerical_method = numerical_method
                                                 )


        #store y-axis points
        if yaxis == "1/t":
            NO_min = 1/np.absolute(NO_min)
            NO_max = 1/np.absolute(NO_max)
            IO_min = 1/np.absolute(IO_min)
            IO_max = 1/np.absolute(IO_max)
        else:
            NO_min = np.absolute(NO_min)
            NO_max = np.absolute(NO_max)
            IO_min = np.absolute(IO_min)
            IO_max = np.absolute(IO_max)
            
        #Set x_axis values
        if xaxis != "m_sum":
            xNO = M
            xIO = M
        else:
            xNO = MNOsum
            xIO = MIOsum
                
        self.WC = WCbackup.copy()
        
        return(pd.DataFrame({"xNO"        : xNO, 
                             yaxis+"_min (NO)" : NO_min, 
                             yaxis+"_max (NO)" : NO_max, 
                             "xIO"        : xIO,
                             yaxis+"_min (IO)" : IO_min, 
                             yaxis+"_max (IO)" : IO_max
                            })
              )
        
    def WC_variation_scatter(self, 
                             isotope           = "76Ge",
                             xaxis             = "m_min", 
                             yaxis             = "t",
                             x_min             = 1e-4,
                             x_max             = 1e+0,
                             n_points          = 10000,
                             WC                = None,
                             ordering          = "both",
                             normalize         = False,
                             dcp               = 1.36,
                             vary_LECs         = False,
                             vary_phases       = True,
                             alpha             = [0,0],
                            ):
        if xaxis in ["m_min", "m_sum", "m_bb"]:
            model = self.generate_LEFT_model(WC = self.WC, method = self.method, LEC = None)
            return(model.WC_variation_scatter(isotope, xaxis, yaxis, x_min, x_max, n_points, WC, 
                                              ordering, normalize, dcp, vary_LECs, vary_phases))
        #set x-axis range and array
        if xaxis == "m_sum":
            
            #xmin
            x_min = np.max([x_min, f.m_min_to_m_sum(0)["NO"]])
            
            #xmax
            x_max = np.max([x_max, f.m_min_to_m_sum(0)["NO"]])
            
            #x-axis values for sum of neutrino masses
            Msum = np.logspace(np.log10(x_min), np.log10(x_max), 1*n_points)
            
            #make a copy
            M = Msum.copy()
            
            #translate m_sum to m_min
            for idx in range(n_points):
                M[idx] = f.m_sum_to_m_min(Msum[idx])["NO"]
        else:
            M = np.logspace((np.log10(x_min)),(np.log10(x_max)), 1*n_points)
            
        #generate datapoints
        
        #normal ordering
        points = np.zeros((n_points,2))
        
        #inverted ordering
        pointsIO = np.zeros((n_points,2))
        
        #xaxis array
        mspace = M
        
        #make backup of values that are varried
        
        #backup of xaxis WC
        if xaxis not in ["m_min", "m_sum"]:
            WC_backup = self.WC[xaxis]
            
        else:
            #backup of m_bb
            m_backup = self.WC["LH(5)"]
        
        #backup of LECs
        LEC_backup = self.LEC.copy()
        
        #extract complex phase if not Majorana phases
        if xaxis not in ["m_min", "m_sum"]:
            try:
                alpha = alpha[0]
            except:
                alpha = alpha
        
        #iterate over number of points
        for x in range(n_points):
            #start with either m_min or m_sum on the x-axis to calculate m_bb
            if xaxis in ["m_min", "m_sum"]:
                #take random x-axis point
                m_min = np.random.choice(mspace)
                
                #randomize Majorana phases if they are to be varied
                if vary_phases:
                    alpha = np.pi*np.random.random(2)
                    
                #else set phases to 0
                else:
                    alpha = alpha
                    
                #calculate m_bb
                #normal ordering
                m = self._m_bb(alpha, m_min, "NO")*1e-9
                #inverted ordering
                mIO = self._m_bb(alpha, m_min, "IO")*1e-9
                
                #set WC
                self.WC["m_bb"] = m
                
                if xaxis == "m_sum":
                    
                    #translate m_min to m_sum
                    m_sum = f.m_min_to_m_sum(m_min)
                    
                    #normal ordering
                    msum = m_sum["NO"]
                    
                    #inverted ordering
                    msumIO = m_sum["IO"]
                
            #Else have some Wilson coefficient on the x-axis
            else:
                #choose some x-axis value
                #rescale mass from eV and put as m_bb
                if xaxis == "m_bb":
                    self.WC[xaxis] = np.random.choice(mspace)*1e-9
                    
                #else take x-axis as WC
                else:
                    self.WC[xaxis] = np.random.choice(mspace)
                    
                #vary complex phase
                if vary_phases:
                    alpha = np.pi*np.random.rand()
                    self.WC[xaxis] *= np.exp(1j*alpha)
                else:
                    self.WC[xaxis] *= np.exp(1j*alpha)
                    
            #vary_unknown_LECs
            if vary_LECs == True:
                self._vary_LECs(inplace = True)
            
            #calculate half-life
            t = self.t_half(isotope)
            
            #store y-axis points
            if yaxis == "t":
                points[x][1] = t
            elif yaxis == "1/t":
                points[x][1] = 1/t
            elif yaxis == "m_eff":
                #PSF G01
                G01    = self.to_G(isotope)["01"]
                
                #Mass Mechanism NME
                M3     = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])
                
                #Effective Neutrino Mass
                points[x][1] = self.m_e / (g_A**2*M3*G01**(1/2)) * t**(-1/2) * 1e+9
                
            
            #normalize to mass mechanism if desired
            if normalize and xaxis in ["m_min", "m_bb", "m_sum"]:
                
                #make backup of WCs to later restore
                WC_backup = self.WC.copy()
                
                #loop over WCs and set all but m_bb to 0
                for operator in self.WC:
                    if operator != "m_bb":
                        self.WC[operator]=0
                        
                #calculate half-life for standard mechanism m_bb
                t_half_mbb = self.t_half(isotope)
                
                #reset WCs from backup
                self.WC = WC_backup.copy()
                
                #normalize results
                if yaxis == "t":
                    points[x][1] /= t_half_mbb
                elif yaxis == "1/t":
                    points[x][1] *= t_half_mbb
                elif yaxis == "m_eff":
                    points[x][1] /= (m * 1e+9)
                
            #repeat for inverted ordering
            if xaxis in ["m_min", "m_sum"]:
                self.WC["m_bb"] = mIO
                
                tIO = self.t_half(isotope)
                if xaxis == "m_sum":
                    pointsIO[x][0] = msumIO
                    points[x][0] = msum
                else:
                    pointsIO[x][0] = m_min
                    points[x][0] = m_min
                    
                #store y-axis point
                if yaxis == "t":
                    pointsIO[x][1] = tIO
                elif yaxis == "1/t":
                    pointsIO[x][1] = 1/tIO
                elif yaxis == "m_eff":
                    G01    = self.to_G(isotope)["01"]
                    M3     = np.abs(self.amplitudes(isotope, self.WC)[1]["nu(3)"])
                    pointsIO[x][1] = self.m_e / (g_A**2*M3*G01**(1/2)) * tIO**(-1/2) * 1e+9
                    
                #normalize to mass mechanism if desired
                if normalize and xaxis in ["m_min", "m_bb", "m_sum"]:
                    
                    #make backup of WCs to later restore
                    WC_backup = self.WC.copy()
                    
                    #loop over WCs and set all but m_bb to 0
                    for operator in self.WC:
                        if operator != "m_bb":
                            self.WC[operator]=0
                            
                    #calculate half-life for standard mechanism m_bb
                    t_half_mbb = self.t_half(isotope)
                    
                    #reset WCs from backup
                    self.WC = WC_backup.copy()
                    
                    #normalize results
                    if yaxis == "t":
                        pointsIO[x][1] /= t_half_mbb
                    elif yaxis == "1/t":
                        pointsIO[x][1] *= t_half_mbb
                    elif yaxis == "m_eff":
                        pointsIO[x][1] /= (mIO * 1e+9)
            else:
                if xaxis == "m_bb":
                    points[x][0] = np.absolute(self.WC[xaxis])*1e+9
                else:
                    points[x][0] = np.absolute(self.WC[xaxis])
                    
        #restore backup values
        self.LEC = LEC_backup.copy()
        if xaxis not in ["m_min", "m_sum"]:
            self.WC[xaxis] = WC_backup
        else:
            self.WC["LH(5)"] = m_backup
                    
        #return(points, pointsIO)
        return(pd.DataFrame({"xNO" : points[:,0], 
                             "yNO" : points[:,1],
                             "xIO" : pointsIO[:,0], 
                             "yIO" : pointsIO[:,1]}
                           )
              )
    