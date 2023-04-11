#import necessary stuff
import EFT

import warnings

import matplotlib as mplt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

#import scipy.optimize
from scipy import optimize
import numpy as np
import pandas as pd
from scipy import integrate

#import constants
from constants import *

#to get files in path
import os
#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                           Get WC Limits in single operator scenario                                           #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################
def get_limits_LEFT(half_life,            #half-life limit
                    isotope,              #isotope the limit is taken from
                    method       = None,  #NME method
                    groups       = False, #group operators into groups with the same limit?
                    basis        = None,  #operator basis
                    scale        = "up",  #scale to extract limits at
                    unknown_LECs = False, #use unknown LECs?
                    PSF_scheme   = "A"    #which PSFs to use
                   ):
    #this function can calculate the limits on the different LEFT coefficients for a given experimental half_life and isotope
    #the limits are calculate at the scale "scale" and for the chosen basis
    
    #generate a LEFT model
    model = EFT.LEFT({}, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
    
    #set the method and import NMEs if necessary
    method = model._set_method_locally(method)
    
    model = EFT.LEFT({}, method = method, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)


    results_2GeV = {}
    results = {}
    scales = {}

    #make a backup so you can overwrite the running afterwards
    WC_backup = model.WC.copy()

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
                    WC = model.change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                else:
                    WC = {WC_name : 1}
                WC = model._run(WC, updown = "down")
            else:
                if basis not in [None, "C", "c"]:
                    WC = change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                else:
                    WC = {WC_name : 1}
                    
            #calculate half-life for WC = 1
            hl = model.t_half(WC = WC, method = method, isotope = isotope)
            
            #get limit on WC
            results[WC_name] = np.sqrt(hl/half_life)
            
    #get limits for each individual operator
    else:
        if basis in [None, "C", "c"]:
            WCs = model.WC.copy()
        else:
            WCs = model.EpsilonWC.copy()
        for WC_name in WCs:
            #if limits @ high scale are desired run operator down
            if scale in ["up", "mW", "m_W", "MW", "M_W"]:
                #running only works for C-basis, translate epsilon to C first
                if basis not in [None, "C", "c"]:
                    WC = model.change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                else:
                    WC = {WC_name : 1}
                    
                #run the WCs down to chipt to get limits on the WCs @ m_W
                WC = model._run(WC, updown = "down")
            else:
                #translate epsilon to C basis if necessary
                if basis not in [None, "C", "c"]:
                    WC = change_basis(WC = {WC_name : 1}, inplace = False, basis = "e")
                else:
                    WC = {WC_name : 1}
                    
            #calculate half-life for WC = 1
            hl = model.t_half(WC = WC, method = method, isotope = isotope)
            
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

    model.WC = WC_backup.copy()

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



def get_limits_SMEFT(half_life,            #half-life limit
                     isotope,              #isotope the limit is taken from
                     method       = None,  #NME method
                     groups       = False, #group operators into groups with the same limit?
                     unknown_LECs = False, #use unknown LECs?
                     PSF_scheme   = "A"    #which PSFs to use
                    ):
    #this function can calculate the limits on the different LEFT coefficients for a given experimental half_life and isotope
    #the limits are calculate at the scale "scale" and for the chosen basis
    
    model = EFT.SMEFT({}, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
    
    #set the method and import NMEs if necessary
    method = model._set_method_locally(method)
    
    model = EFT.SMEFT({}, method = method, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)

    results_2GeV = {}
    results = {}
    scales = {}

    #make a backup so you can overwrite the running afterwards
    WC_backup = model.WC.copy()

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
        hl = model.t_half(WC = WC, method = method, isotope = isotope)
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

    model.WC = WC_backup.copy()
    
    return(pd.DataFrame({r"Limits [GeV$^{4-d}$]" : results, "Scales [GeV]" : scales}))
    

#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                        Contour Plot to put limits on 2 operator scenarios                                     #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################

def get_contours(WCx,                      #x-axis WC (independent)
                 WCy,                      #y-axis WC (dependent)
                 half_life,                #half-life limit
                 isotope,                  #isotope of interest
                 method     = "IBM2",      #NME method
                 phase      = 3/4*np.pi,   #complex phase relative between the two WCs
                 n_points   = 5000,        #number of points to be studied
                 x_min      = None,        #minimum x-axis
                 x_max      = None,        #maximum x-axis
                 unknown_LECs = False,
                 PSF_scheme = "A",
                ):
    
    SMEFT_WCs = EFT.SMEFT_WCs
    exp_idx = 0
    limit = half_life
    
    M = generate_matrix(WC = [WCx, WCy], isotope = isotope, method = method, 
                        PSF_scheme = PSF_scheme, unknown_LECs = unknown_LECs)
    
    if WCx in SMEFT_WCs:
        is_SMEFT = True
    else:
        is_SMEFT = False
    
    if is_SMEFT:
        fac = 3
    else:
        fac = 1.5
        
    #find the range of the curve by finding the maximal value 
    radius = fac * np.sqrt(1/M[0,0]*1/limit)
    
    if x_min == None:
        x_min = -radius
        x_max =  radius

    #lists to store contour points
    WCyarray_max = []
    WCyarray_min = []
    WCxarray = []
    for x in np.linspace(x_min,x_max, n_points):
        #find the scale at which to search for the minima

        #Calculate half-life following eq 38. in 1806.02780
        a = M[1,1]
        b = 2*np.cos(phase)*M[0,1]*x
        c = M[0,0]*x**2 - 1/limit


        y = (-b + np.sqrt(b**2-4*a*c+0*1j))/(2*a)
        y2 = (-b - np.sqrt(b**2-4*a*c+0*1j))/(2*a)


        if y.imag == 0 and y2.imag == 0:
            WCxarray.append(x)
            WCyarray_max.append(y.real)
            WCyarray_min.append(y2.real)
    
    if is_SMEFT:
        return(pd.DataFrame({WCx + r" [GeV$^{-"+str(int(WCx[-2])-4)+"}$]": WCxarray, 
                             WCy + " min" + r" [GeV$^{-"+str(int(WCy[-2])-4)+"}$]": WCyarray_min, 
                             WCy + " max" + r" [GeV$^{-"+str(int(WCy[-2])-4)+"}$]": WCyarray_max}
                           )
              )
    else:
        return(pd.DataFrame({WCx: WCxarray, 
                             WCy + " min": WCyarray_min, 
                             WCy + " max": WCyarray_max}
                           )
              )



#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Generate analytical expression for the decay rate                                     #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################
def generate_formula_coefficients(WC, method = "IBM2", unknown_LECs = False, PSF_scheme = "A"):
    #check if the first operator is in SMEFT or LEFT
    if WC[0] in SMEFT_WCs:
        is_SMEFT = True
    else:
        is_SMEFT = False
        
    #coefficients dict
    C = {}

    for WC1 in WC:
        #generate model class
        
        if is_SMEFT:
            #SMEFT model
            model1 = EFT.SMEFT({WC1:1}, method=method, scale = m_W, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
            
        else:
            #LEFT model
            model1 = EFT.LEFT({WC1:1}, method=method, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
        
        #calculate half-life
        thalf1 = model1.half_lives()
        
        #store coefficient
        C[WC1] = 1/thalf1
        
        #additional WCs and interference terms
        for WC2 in WC:
            if WC2 != WC1:
                if is_SMEFT:
                    #single WC model
                    model2 = EFT.SMEFT({WC2:1}, method=method, scale = m_W, 
                                       unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                    
                    #multi WC model
                    model3 = EFT.SMEFT({WC1:1, WC2:1}, method=method, scale = m_W, 
                                       unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                else:
                    #single WC model
                    model2 = EFT.LEFT({WC2:1}, method=method, unknown_LECs = unknown_LECs, 
                                      PSF_scheme = PSF_scheme)
                    
                    #multi WC model
                    model3 = EFT.LEFT({WC1:1, WC2:1}, method=method, unknown_LECs = unknown_LECs, 
                                      PSF_scheme = PSF_scheme)
                    
                #single WC half-lives
                thalf2 = model2.half_lives()
                
                #multi WC with interference half-life
                thalf3 = model3.half_lives()
                
                #store coefficients
                C[WC2] = 1/thalf2
                C[WC1+WC2] = 1/thalf3-(1/thalf2+1/thalf1)
                
    #return coefficient dict
    return(C)

def generate_terms(WC, isotope = "76Ge", method = "IBM2", decimal = 2, 
                   output = "latex", unknown_LECs = False, PSF_scheme = "A"):
    #check if the first operator is in SMEFT or LEFT
    if WC[0] in SMEFT_WCs:
        is_SMEFT = True
    else:
        is_SMEFT = False
        
    #get half-life coefficients
    C = generate_formula_coefficients(WC = WC, method = method, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
    
    #raise error if output is incorrect
    if output not in ["latex", "html"]:
        raise ValueError("output must be either 'latex' or 'html'")
        
    #generate formula terms in output format
    terms = {}
    
    #iterate over WCs
    for WC1 in WC:
        #get order of magnitude
        exponent = int(np.floor(np.log10(C[WC1][isotope][0])))
        
        #get prefactor (rounded)
        prefactor = np.round(C[WC1][isotope][0]*10**(-exponent), decimal)
        
        #scaling (for SMEFT operators)
        if is_SMEFT:
            scaling = int(WC1[-2])-4
            scaling = "~\\mathrm{GeV}^"+str(scaling)
        else:
            scaling = ""
        
        #get WC name in output format
        
        #standard mechanism
        if WC1 == "m_bb":
            #latex output string
            if output == "latex":
                WC1string = "\\frac{m_{\\beta\\beta}}{1\\mathrm{GeV}}"
                
            #html output string
            elif output == "html":
                WC1string = "m<sub>&beta;&beta;</sub>"
                
        #primed WCs
        elif WC1[-5:] == "prime":
            #latex output string
            if output == "latex":
                WC1string = "C_{"+WC1[:-8]+"}^{"+WC1[-8:-5]+"}`"
                
            #html output string
            elif output == "html":
                WC1string = "C<sub>"+WC1[:-8]+"</sub><sup>"+WC1[-8:-5]+"</sup>'"
                
        #all other WCs
        else:
            if output == "latex":
                WC1string = "C_{"+WC1[:-3]+"}^{"+WC1[-3:]+"}"+scaling
            elif output == "html":
                WC1string = "C<sub>"+WC1[:-3]+"</sub><sup>"+WC1[-3:]+"</sup>"
                
        #put components together (prefactor x 10^exponent x |WCname|^2
        if output == "latex":
            terms[WC1] = "$"+str(prefactor)+"~\\mathrm{y}^{-1}\\times 10^{"+str(exponent)+"}\\left|"+WC1string+"\\right|^2$"
        elif output == "html":
            terms[WC1] = str(prefactor)+"y<sup>-1</sup>&times;10<sup>"+str(exponent)+"</sup>|"+WC1string+"|<sup>2</sup>"
            
        #get interference terms
        for WC2 in WC:
            
            if is_SMEFT:
                scaling2 = int(WC2[-2])-4
                scaling2 = "~\\mathrm{GeV}^"+str(scaling2)
            else:
                scaling2 = ""
            if WC2 not in terms:
                #add second WC
                exponent = int(np.floor(np.log10(C[WC2][isotope][0])))
                prefactor = np.round(C[WC2][isotope][0]*10**(-exponent), decimal)
                if WC2 == "m_bb":
                    if output == "latex":
                        WC2string = "~\\frac{m_{\\beta\\beta}}{1~\\mathrm{GeV}}"+scaling
                    elif output == "html":
                        WC2string = "m<sub>&beta;&beta;</sub>"
                elif WC2[-5:] == "prime":
                    if output == "latex":
                        WC2string = "C_{"+WC2[:-8]+"}^{"+WC2[-8:-5]+"}`"
                    elif output == "html":
                        WC2string = "C<sub>"+WC2[:-8]+"</sub><sup>"+WC2[-8:-5]+"</sup>'"
                else:
                    if output == "latex":
                        WC2string = "C_{"+WC2[:-3]+"}^{"+WC2[-3:]+"}"
                    elif output == "html":
                        WC2string = "C<sub>"+WC2[:-3]+"</sub><sup>"+WC2[-3:]+"</sup>"
                if output == "latex":
                    terms[WC2] = "$"+str(prefactor)+"~\\mathrm{y}^{-1}\\times 10^{"+str(exponent)+"}\\left|"+WC2string+"\\right|^2$"
                elif output == "html":
                    terms[WC2] = str(prefactor)+"y<sup>-1</sup>&times;10<sup>"+str(exponent)+"</sup>|"+WC2string+"|<sup>2</sup>"
                    
            
            if WC2+WC1 not in terms and WC1 != WC2:
                #add interference terms
                if C[WC1+WC2][isotope][0] != 0:
                    exponent = int(np.floor(np.log10(np.abs(C[WC1+WC2][isotope][0]))))
                    prefactor = np.round(C[WC1+WC2][isotope][0]*10**(-exponent), decimal)
                    if output == "latex":
                        terms[WC1+WC2] = ("$"+str(prefactor)+"~\\mathrm{y}^{-1}\\times 10^{"+str(exponent)+"} \\mathrm{Re}\\left["+WC1string+"({"+WC2string+"})^*\\right]$")
                    if output == "html":
                        terms[WC1+WC2] = (str(prefactor)+"y<sup>-1</sup>&times; 10<sup>"+str(exponent)+"</sup> Re&#91;"+WC1string+"("+WC2string+")<sup>&#42;</sup>&#93;")
                    
    return(terms)

def generate_formula(WC, isotope = "76Ge", method = "IBM2", decimal = 2, 
                     output = "latex", unknown_LECs = False, PSF_scheme = "A"):
    #Generate Terms
    terms = generate_terms(WC = WC, isotope = isotope, output = output, 
                           method = method, decimal = decimal, 
                           unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
    
    #Write Equation
    if output == "latex":
        formula = r"$T_{1/2}^{-1} = "
    elif output == "html":
        formula = "T<sub>1/2</sub><sup>-1</sup> = "
        
    #Add Terms
    for WCs in WC:
        if output == "latex":
            formula+="+"+terms[WCs][1:-1]
        elif output == "html":
            formula+=" +"+terms[WCs]
            
    #Add Interference Terms
    for WC1 in WC:
        for WC2 in WC:
            if WC2 != WC1 and WC1+WC2 in terms:
                if output == "latex":
                    if terms[WC1+WC2][1] != "-":
                        formula += "+"
                    formula+=terms[WC1+WC2][1:-1]
                elif output == "html":
                    if terms[WC1+WC2][0] != "-":
                        formula += " +"
                    else:
                        formula += " "
                    formula+=terms[WC1+WC2]
    if output == "latex":
        formula+="$"
        
    #return formula
    return(formula)


#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Generate matrix for decay rate C^dagger M C = T^-1                                    #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################

def generate_matrix_coefficients(WC, isotope = "76Ge", method = "IBM2", unknown_LECs = False, PSF_scheme = "A"):
    #check if WCs are SMEFT or LEFT
    if WC[0] in SMEFT_WCs:
        is_SMEFT = True
    else:
        is_SMEFT = False
    C = {}

    for WC1 in WC:
        if is_SMEFT:
            model1 = EFT.SMEFT({WC1:1}, method=method, scale = m_W, 
                               unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
        else:
            model1 = EFT.LEFT({WC1:1}, method=method, unknown_LECs = unknown_LECs, 
                              PSF_scheme = PSF_scheme)
        thalf1 = model1.half_lives()[isotope][0]
        for WC2 in WC:
            if is_SMEFT:
                model2 = EFT.SMEFT({WC2:1}, method=method, scale = m_W, 
                                   unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                model3 = EFT.SMEFT({WC1:1, WC2:1}, method=method, scale = m_W, 
                                   unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
            else:
                model2 = EFT.LEFT({WC2:1}, method=method, 
                                  unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                model3 = EFT.LEFT({WC1:1, WC2:1}, method=method, 
                                  unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                
            thalf2 = model2.half_lives()[isotope][0]
            thalf3 = model3.half_lives()[isotope][0]
            if WC1 == WC2:
                C[WC1+WC2] = 1/thalf1
            else:
                C[WC1+WC2] = 1/2*(1/thalf3-(1/thalf2+1/thalf1))
    return(C)

def generate_matrix(WC, isotope = "76Ge", method = "IBM2", unknown_LECs = False, PSF_scheme = "A"):
    C = generate_matrix_coefficients(WC, isotope, method, unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
    
    M = np.zeros([len(WC), len(WC)])
    for idx1 in range(len(WC)):
        try:
            WC1 = list(WC.keys())[idx1]
        except:
            WC1 = WC[idx1]
        for idx2 in range(len(WC)):
            try:
                WC2 = list(WC.keys())[idx2]
            except:
                WC2 = WC[idx2]
            value = C[WC1+WC2]
            M[idx1, idx2] = value
    return(M)



#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                         Neutrino Physics Formulae (mixing and masses).                                        #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################


def m_bb(alpha, m_min=1, ordering="NO", dcp=1.36):
    #this function returns m_bb from m_min and the majorana mixing matrix

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

    UPMNS = U@majorana

    #create non-diagonal mass matrix
    m_BB_NO = np.abs(UPMNS[0,0]**2*m1+UPMNS[0,1]**2*m2+UPMNS[0,2]**2*m3)
    m_BB_IO = np.abs(UPMNS[0,0]**2*m1IO+UPMNS[0,1]**2*m2IO+UPMNS[0,2]**2*m3IO)

    if ordering == "NO":
        return(m_BB_NO)
    elif ordering =="IO":
        return(m_BB_IO)
    else:
        return(m_BB_NO,m_BB_IO)
    
    
    
#translate the sum of neutrino masses to the minimal neutrino mass
def m_sum_to_m_min(m_sum):
    def m_sum_NO(m_min, m_sum):
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3
        
        m1 = m_min
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)
        
        msum = m1+m2+m3
        return(msum-m_sum)
    
    def m_sum_IO(m_min, m_sum):
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3

        m3IO = m_min
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)
        
        msum = m1IO+m2IO+m3IO
        return(msum-m_sum)
    
    m_min_NO = optimize.root(m_sum_NO, x0 = 0.05, args = [m_sum]).x[0]
    m_min_IO = optimize.root(m_sum_IO, x0 = 0.05, args = [m_sum]).x[0]
    return({"NO" : m_min_NO, "IO" : m_min_IO})


#translate the minimal neutrino mass to the sum of neutrino masses
def m_min_to_m_sum(m_min):
    def m_sum_NO(m_min):
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3
        
        m1 = m_min
        m2 = np.sqrt(m1**2+m21)
        m3 = np.sqrt(m2**2+m32)
        
        msum = m1+m2+m3
        return(msum)
    
    def m_sum_IO(m_min):
        #m21 = 7.53e-5
        #m32 = 2.453e-3
        #m32IO = -2.546e-3

        m3IO = m_min
        m2IO = np.sqrt(m3IO**2-m32IO)
        m1IO = np.sqrt(m2IO**2-m21)
        
        msum = m1IO+m2IO+m3IO
        return(msum)
    
    return({"NO" : m_sum_NO(m_min), "IO" : m_sum_IO(m_min)})


#Neutrino Mixing Matrix
def U_PMNS(CP_phase = 1.36*np.pi, alpha = [0,0]):
    dcp = CP_phase
    U = np.array([[                            c12*c13,                             s12*c13, s13*np.exp(-1j*dcp)], 
                  [-s12*c23-c12*s13*s23*np.exp(1j*dcp),  c12*c23-s12*s23*s13*np.exp(1j*dcp),             s23*c13], 
                  [ s12*s23-c12*c23*s13*np.exp(1j*dcp), -c12*s23-s12*c23*s13*np.exp(1j*dcp),             c23*c13]])
    
    majorana = np.diag([1, np.exp(1j*alpha[0]), np.exp(1j*alpha[1])])
    
    U = U@majorana
    return(U)
