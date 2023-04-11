import numpy as np
import scipy.constants

#helpful stuff from scipy
pc = scipy.constants.physical_constants

#conversion factors for different units
u_to_GeV           = pc["atomic mass constant energy equivalent in MeV"][0] * 1e-3
MeV_to_inverseyear = 1/(2.087e-29)
fm_to_inverse_MeV  = 1/197.3

#energy units
TeV = 1e+3
GeV = 1
MeV = 1e-3
KeV = 1e-6
eV  = 1e-9
meV = 1e-12

###################################################################################
#                                                                                 #
# masses and scales (all numbers are in GeV if not explicitly stated differently) #
#                                                                                 #
###################################################################################

G_F        = pc["Fermi coupling constant"][0] #GeV^-2
vev        = 246*GeV  #Higgs vev
lambda_chi = 2*GeV    #chiPT scale
m_W        = 80*GeV   #W-boson mass (matching scale of SMEFT->LEFT)
m_Z        = 91*GeV   #Z-boson mass (required for RGE running)
m_H        = 125.25*GeV #Higgs-boson mass )(required for SMEFT running) taken from PDG2022

#heavy quarks #taken from PDG 2022 https://pdg.lbl.gov/2022/tables/contents_tables.html
m_t = 172.69*GeV      #top-quark
m_b = 4.18*GeV        #bottom-quark
m_c = 1.27*GeV        #charm-quark

#light quarks are set to 0 except for the matching procedures. If you want to set these to 0 too you need to do so in the EFT.py
m_s = 93.4*MeV        #strange-quark
m_u = 2.16*MeV        #up-quark
m_d = 4.67*MeV        #down-quark

m_pi = 139.57*MeV      #pion mass
m_N  = 0.93*GeV        #nucleon mass scale in GeV
m_p  = 0.938272088*GeV #proton mass

#electron mass
m_e     = pc["electron mass energy equivalent in MeV"][0] * 1e-3 #electron mass in GeV
m_e_MeV = pc["electron mass energy equivalent in MeV"][0]        #electron mass in MeV (used for electron wave functions)


#######################################################################
#                                                                     #
#                           Coupling Constants                        #
#                                                                     #
#######################################################################

#fine structure constant @ mu=0
alpha   = pc["fine-structure constant"][0]

#coupling constants at mu=m_Z
alphas0 = 0.1173
alpha10 = 0.0169225
alpha20 = 0.033735
alphat0 = 0.07514
alphaL0 = 1/(4*np.pi) * m_H**2/(2*vev**2)

#######################################################################
#                                                                     #
#                 Quark and Neutrino Mixing Parameters                #
#                                                                     #
#######################################################################

#quark mixing angles
V_ud = 0.97417

#neutrino mixing angles (s = sin, c=cos)
s12 = np.sqrt(0.307)
s23 = np.sqrt(0.546)
s13 = np.sqrt(2.2e-2)

c12 = np.cos(np.arcsin(s12))
c23 = np.cos(np.arcsin(s23))
c13 = np.cos(np.arcsin(s13))

#squared mass differences [eV^2] (the neutrino mixing is always calculated in eV)
m21   =  7.53e-5
m32   =  2.453e-3
m32IO = -2.546e-3


#######################################################################
#                                                                     #
#                        Low-Energy Constants                         #
#                                                                     #
#######################################################################

g_A = 1.271
g_V = 1

F_pi = 0.0922 * GeV

#all LECs at known values or NDA estimates
LECs = {"A":g_A, 
        "S":0.97, 
        "M":4.7, 
        "T":0.99, 
        "B":2.7, 
        "1pipi":0.36, 
        "2pipi":2.0, 
        "3pipi":-0.62, 
        "4pipi":-1.9, 
        "5pipi":-8, 
        # all the below are expected to be order 1 in absolute magnitude
        "Tprime":1, 
        "Tpipi":1, 
        "1piN":1, 
        "6piN":1, 
        "7piN":1, 
        "8piN":1, 
        "9piN":1, 
        "VLpiN":1,
        "TpiN":1, 
        "1NN":1, 
        "6NN":1, 
        "7NN":1, 
        "VLNN":1, 
        "TNN": 1, 
        "VLE":1, 
        "VLme":1,
        "VRE":1, 
        "VRme":1, 
        # all the below are expected to be order (4pi)**2 in absolute magnitude
        "2NN":(4*np.pi)**2, 
        "3NN":(4*np.pi)**2, 
        "4NN":(4*np.pi)**2,
        "5NN":(4*np.pi)**2, 
        # expected to be 1/F_pipi**2 pion decay constant
        "nuNN": -1/(4*np.pi) * (m_N*g_A**2/(4*F_pi**2))**2*0.6
       }

#known LECs
LECs_known = {"A":g_A, 
              "S":0.97, 
              "M":4.7, 
              "T":0.99, 
              "B":2.7, 
              "1pipi":0.36, 
              "2pipi":2.0, 
              "3pipi":-0.62, 
              "4pipi":-1.9, 
              "5pipi":-8
             }

#Unknown Low-Energy Constants order of magnitude estimate
LECs_unknown = {'Tprime': 1,
                'Tpipi': 1,
                '1piN': 1,
                '6piN': 1,
                '7piN': 1,
                '8piN': 1,
                '9piN': 1,
                'VLpiN': 1,
                'TpiN': 1,
                '1NN': 1,
                '6NN': 1,
                '7NN': 1,
                'VLNN': 1,
                'TNN': 1,
                'VLE': 1,
                'VLme': 1,
                'VRE': 1,
                'VRme': 1,
                '2NN': 157.91367041742973,
                '3NN': 157.91367041742973,
                '4NN': 157.91367041742973,
                '5NN': 157.91367041742973,
                'nuNN': -1/(4*np.pi) * (m_N*g_A**2/(4*F_pi**2))**2*0.6
               }



#######################################################################
#                                                                     #
#                         Wilson Coefficients                         #
#                                                                     #
#######################################################################



SMEFT_WCs = {#dim5                    #Wilson Coefficients of SMEFT
             "LH(5)"      : 0,        #up to dimension 7. We only 
             #dim7                    #list the operators violating
             "LH(7)"     : 0,         #lepton number by 2 units.
             "LHD1(7)"   : 0,
             "LHD2(7)"   : 0,
             "LHDe(7)"   : 0,
             #"LHB(7)"    : 0,
             "LHW(7)"    : 0,
             "LLduD1(7)" : 0,
             #"LLeH(7)"   : 0,
             "LLQdH1(7)" : 0,
             "LLQdH2(7)" : 0,
             "LLQuH(7)" : 0,
             "LeudH(7)"  : 0, 
             #dim9
             #  -6-fermi
             "ddueue(9)"    : 0,
             "dQdueL1(9)"   : 0,
             "dQdueL2(9)"   : 0,
             "QudueL1(9)"   : 0,
             "QudueL2(9)"   : 0,
             "dQQuLL1(9)"   : 0,
             "dQQuLL2(9)"   : 0,
             "QuQuLL1(9)"   : 0,
             "QuQuLL2(9)"   : 0,
             "dQdQLL1(9)"   : 0,
             "dQdQLL2(9)"   : 0,
             #  -other
             "LLH4W1(9)"    : 0,
             "deueH2D(9)"   : 0,
             "dLuLH2D2(9)"  : 0,
             "duLLH2D(9)"   : 0,
             "dQLeH2D2(9)"  : 0,
             "dLQeH2D1(9)"  : 0,
             "deQLH2D(9)"   : 0,
             "QueLH2D2(9)"  : 0,
             "QeuLH2D2(9)"  : 0,
             "QLQLH2D2(9)"  : 0,
             "QLQLH2D5(9)"  : 0,
             "QQLLH2D2(9)"  : 0,
             "eeH4D2(9)"    : 0,
             "LLH4D23(9)"   : 0,
             "LLH4D24(9)"   : 0
             }

SMEFT_WCs_latex = [#dim5           #Wilson Coefficients of SMEFT
                   r"$C_{LH}^{(5)}$",        #up to dimension 7. We only 
                   #dim7                    #list the operators violating
                   r"$C_{LH}^{(7)}$",         #lepton number by 2 units.
                   r"$C_{LHD1}^{(7)}$",
                   r"$C_{LHD2}^{(7)}$",
                   r"$C_{LHDe}^{(7)}$",
                   #"LHB(7)",
                   r"$C_{LHW}^{(7)}$",
                   r"$C_{LLduD1}^{(7)}$",
                   #"LLeH(7)",
                   r"$C_{LLQdH1}^{(7)}$",
                   r"$C_{LLQdH2}^{(7)}$",
                   r"$C_{LLQuH}^{(7)}$",
                   r"$C_{LeudH}^{(7)}$", 
                   #dim9
                   #  -6-fermi
                   r"$C_{ddueue}^{(9)}$",
                   r"$C_{dQdueL1}^{(9)}$",
                   r"$C_{dQdueL2}^{(9)}$",
                   r"$C_{QudueL1}^{(9)}$",
                   r"$C_{QudueL2}^{(9)}$",
                   r"$C_{dQQuLL1}^{(9)}$",
                   r"$C_{dQQuLL2}^{(9)}$",
                   r"$C_{QuQuLL1}^{(9)}$",
                   r"$C_{QuQuLL2}^{(9)}$",
                   r"$C_{dQdQLL1}^{(9)}$",
                   r"$C_{dQdQLL2}^{(9)}$",
                   #  -other
                   r"$C_{LLH^4W1}^{(9)}$",
                   r"$C_{deueH^2D}^{(9)}$",
                   r"$C_{dLuLH^2D2}^{(9)}$",
                   r"$C_{duLLH^2D}^{(9)}$",
                   r"$C_{dQLeH^2D2}^{(9)}$",
                   r"$C_{dLQeH^2D1}^{(9)}$",
                   r"$C_{deQLH^2D}^{(9)}$",
                   r"$C_{QueLH^2D2}^{(9)}$",
                   r"$C_{QeuLH^2D2}^{(9)}$",
                   r"$C_{QLQLH^2D2}^{(9)}$",
                   r"$C_{QLQLH^2D5}^{(9)}$",
                   r"$C_{QQLLH^2D2}^{(9)}$",
                   r"$C_{eeH^4D^2}^{(9)}$",
                   r"$C_{LLH^4D^23}^{(9)}$",
                   r"$C_{LLH^4D^24}^{(9)}$"]

SMEFT_WCs_latex_dict = {}
for idx in range(len(SMEFT_WCs)):
    operator = list(SMEFT_WCs.keys())[idx]
    SMEFT_WCs_latex_dict[operator] = SMEFT_WCs_latex[idx]


LEFT_WCs = {#dim3
            "m_bb"       : 0, 
            #dim6
            "SL(6)"      : 0, 
            "SR(6)"      : 0, 
            "T(6)"       : 0, 
            "VL(6)"      : 0, 
            "VR(6)"      : 0, 
            #dim7
            "VL(7)"      : 0,
            "VR(7)"      : 0, 
            #dim9
            "1L(9)"      : 0, 
            "1R(9)"      : 0, 
            "1L(9)prime" : 0, 
            "1R(9)prime" : 0, 
            "2L(9)"      : 0, 
            "2R(9)"      : 0, 
            "2L(9)prime" : 0, 
            "2R(9)prime" : 0, 
            "3L(9)"      : 0, 
            "3R(9)"      : 0, 
            "3L(9)prime" : 0, 
            "3R(9)prime" : 0, 
            "4L(9)"      : 0, 
            "4R(9)"      : 0, 
            "5L(9)"      : 0, 
            "5R(9)"      : 0, 
            "6(9)"       : 0,
            "6(9)prime"  : 0,
            "7(9)"       : 0,
            "7(9)prime"  : 0,
            "8(9)"       : 0,
            "8(9)prime"  : 0,
            "9(9)"       : 0,
            "9(9)prime"  : 0
            }

#define labels for plots
LEFT_WCs_latex = [r"$m_{\beta\beta}$", 
                  r"$C_{VL}^{(6)}$", 
                  r"$C_{VR}^{(6)}$", 
                  r"$C_{T}^{(6)}$" , 
                  r"$C_{SL}^{(6)}$", 
                  r"$C_{SR}^{(6)}$", 
                  r"$C_{VL}^{(7)}$", 
                  r"$C_{VR}^{(7)}$", 
                  r"$C_{1L}^{(9)}$", 
                  r"$C_{1R}^{(9)}$", 
                  r"${C_{1L}^{(9)}}'$", 
                  r"${C_{1R}^{(9)}}'$", 
                  r"$C_{2L}^{(9)}$", 
                  r"$C_{2R}^{(9)}$", 
                  r"${C_{2L}^{(9)}}'$", 
                  r"${C_{2R}^{(9)}}'$", 
                  r"$C_{3L}^{(9)}$", 
                  r"$C_{3R}^{(9)}$", 
                  r"${C_{3L}^{(9)}}'$", 
                  r"${C_{3R}^{(9)}}'$", 
                  r"${C_{4L}^{(9)}}$", 
                  r"${C_{4R}^{(9)}}$", 
                  r"${C_{5L}^{(9)}}$", 
                  r"${C_{5R}^{(9)}}$", 
                  r"${C_{6}^{(9)}}$",
                  r"${C_{6}^{(9)}}'$",  
                  r"${C_{7}^{(9)}}$",
                  r"${C_{7}^{(9)}}'$", 
                  r"${C_{8}^{(9)}}$",
                  r"${C_{8}^{(9)}}'$",  
                  r"${C_{9}^{(9)}}$",
                  r"${C_{9}^{(9)}}'$"]

LEFT_WCs_latex_dict = {}
for idx in range(len(LEFT_WCs)):
    operator = list(LEFT_WCs.keys())[idx]
    LEFT_WCs_latex_dict[operator] = LEFT_WCs_latex[idx]


LEFT_WCs_epsilon = {#dim3
                    "m_bb":0, 
                    #dim6
                    "V+AV+A": 0, "V+AV-A": 0, 
                    "TRTR":0, 
                    "S+PS+P":0, "S+PS-P":0,
                    #dim7
                    "VL(7)":0, "VR(7)":0, #copied from C basis
                    #dim9
                    "1LLL":0, "1LLR":0,
                    "1RRL":0, "1RRR":0,
                    "1RLL":0, "1RLR":0,
                    "2LLL":0, "2LLR":0,
                    "2RRL":0, "2RRR":0,
                    "3LLL":0, "3LLR":0,
                    "3RRL":0, "3RRR":0,
                    "3RLL":0, "3RLR":0,
                    "4LLR":0, "4LRR":0,
                    "4RRR":0, "4RLR":0,
                    "5LLR":0, "5LRR":0,
                    "5RRR":0, "5RLR":0,
                    #redundant operators
                    "1LRL":0, "1LRR":0, 
                    "3LRL":0, "3LRR":0,
                    "4LLL":0, "4LRL":0,
                    "4RRL":0, "4RLL":0,
                    "TLTL":0,
                    "5LLL":0, "5LRL":0,
                    "5RRL":0, "5RLL":0,
                    #vanishing operators
                    "2LRL":0, "2LRR":0, 
                    "2RLL":0, "2RLR":0, 
                    "TRTL":0, "TLTR":0, 
                    #operators not contributing directly
                    "V-AV+A": 0, "V-AV-A": 0, 
                    "S-PS+P":0, "S-PS-P":0
                    }