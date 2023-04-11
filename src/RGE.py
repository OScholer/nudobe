#Here we define the Renormalization Group Equations of the different LEFT and SMEFT operators etc.
import numpy as np

from scipy import integrate

from constants import *

#################################################################################################
#                                                                                               #
#                                                                                               #
#                                      General Couplings                                        #
#                                                                                               #
#                                                                                               #
#################################################################################################

#running of couplings 1901.10302 eq. 23
def RGEalpha(scale, alpha, n_g = None):
    if n_g == None:
        #quark masses
        masses = np.array([m_u, m_d, m_s, m_c, m_b, m_t])

        #find index where scale mu sits (= number of quarks lighter than scale mu)
        n_f = np.searchsorted(2*masses, np.exp(scale))

        n_g = n_f/2
    
    #RGE matrix
    M1 = np.zeros((5,5))
    M1[0,0]      = 1/(2*np.pi)*(1/10 + 4/3 * n_g)
    M1[0,1]      = 0
    M1[0,2]      = 0
    M1[0,3]      = 0
    M1[0,4]      = 0


    M1[1,0]      = 0
    M1[1,1]      = 1/(2*np.pi)*(-43/6 + 4/3*n_g)
    M1[1,2]      = 0
    M1[1,3]      = 0
    M1[1,4]      = 0


    M1[2,0]      = 0
    M1[2,1]      = 0
    M1[2,2]      = 1/(2*np.pi)*(-11 + 4/3*n_g)
    M1[2,3]      = 0
    M1[2,4]      = 0


    M1[3,0]      = 1/(2*np.pi)*(-17/20)
    M1[3,1]      = 1/(2*np.pi)*(-  9/4)
    M1[3,2]      = 1/(2*np.pi)*(-    8)
    M1[3,3]      = 1/(2*np.pi)*(+  9/2)
    M1[3,4]      = 0


    M1[4,0]      = 1/(4*np.pi) * (- 9/5)
    M1[4,1]      = 1/(4*np.pi) * (-   9)
    M1[4,2]      = 0
    M1[4,3]      = 1/(4*np.pi) * (+  12)
    M1[4,4]      = 1/(4*np.pi) * (+  24)


    M2 = np.zeros(5)
    alpha12 = np.array(alpha[0:2])
    M2[-1] = 1/(8*np.pi) * np.dot(alpha12, np.dot(np.array([[27/100, 9/10], [0, 9/4]]), alpha12))

    return(np.dot(M1, alpha)*alpha + M2)

##############################################

def alpha(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = m_Z, n_g = None):
    alpha_new = integrate.solve_ivp(RGEalpha, [np.log(mu0), np.log(scale)], alpha0, method = "RK45", rtol = 1e-4, args = [n_g]).y[:,-1]
    return(alpha_new)

##############################################
#U(1) running

def alpha1(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = m_Z, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[0])

##############################################
#SU(2) running

def alpha2(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = m_Z, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[1])

##############################################
#SU(3) running

def alpha3(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = m_Z, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[2])

##############################################
#SU(3) running

def alpha_s(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = m_Z, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[2])

##############################################
#top yukawa

def alpha_t(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = 91, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[3])

##############################################
#quartic higgs coupling

def alpha_lambda(scale, alpha0 = [alpha10, alpha20, alphas0, alphat0, alphaL0], mu0 = 91, n_g = None):
    return(alpha(scale, alpha0, n_g = n_g)[4])

##############################################

#################################################################################################
#                                                                                               #
#                                                                                               #
#                                    RGEs of LEFT Operators                                     #
#                                                                                               #
#                                                                                               #
#################################################################################################

'''
    #############################################################################################
    Define RGEs from 1806.02780 and define a "running" 
    function to run WCs from m_W down to chiPT scale.
    Note that "scale" in the RGEs refers to log(mu)
    i.e. "scale" = log(mu), while scale in the final 
    running functions refers to the actual energy scale 
    i.e. "scale" = mu
    #############################################################################################
'''

######################################################################################
#
#Define RGEs as differential equations and solve them numerically for the given scales
#
######################################################################################

#running couplings from m_W down to 2GeV
#define ODEs to solve i.e. define
# dC / dln(mu)

##############################################

#SL(6) and SR(6)
def RGEC6_S(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    return (-6*C_F * alpha_s(mu)/(4*np.pi) * C)

##############################################

#T(6)
def RGEC6_T(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    return (2*C_F * alpha_s(mu)/(4*np.pi) * C)

##############################################

#1L(9), 1R(9), 1L(9)prime, 1R(9)prime
def RGEC9_1(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    return 6*(1-1/n_c)*alpha_s(mu)/(4*np.pi) * C

##############################################

#2L(9), 2R(9), 2L(9)prime, 2R(9)prime, 3L(9), 3R(9), 3L(9)prime, 3R(9)prime
def RGEC9_23(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    M = np.array([[8 + 2/n_c - 6*n_c, -4-8/n_c + 4*n_c], [4 - 8/n_c, 4 + 2/n_c + 2*n_c]])
    return alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

##############################################

#4L(9), 4R(9), 5L(9), 5R(9)
def RGEC9_45(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    M = np.array([[6/n_c, 0],[-6, -12*C_F]])
    return alpha_s(mu)/(4*np.pi) * np.dot(M, C)

##############################################

#6(9), 6(9)prime, 7(9), 7(9)prime, 8(9), 8(9)prime, 9(9), 9(9)prime
def RGEC9_67_89(ln_mu, C):
    mu = np.exp(ln_mu)
    n_c = 3
    C_F = (n_c**2 - 1) / (2*n_c)
    M = np.array([[-2*C_F*(3*n_c - 4)/n_c, 2*C_F * (n_c + 2)*(n_c - 1) / n_c**2], 
                  [4*(n_c-2)/n_c, (4 - n_c + 2*n_c**2 + n_c**3)/n_c**2]])
    return alpha_s(mu)/(4*np.pi) *(np.dot(M, C))

##############################################



#################################################################################################
#                                                                                               #
#                                                                                               #
#                                    Solving the LEFT RGEs                                      #
#                                                                                               #
#                                                                                               #
#################################################################################################

def run_LEFT(WC,                          #Wilson Coefficients as dict, see constants.LEFT_WCs
             initial_scale = m_W,         #run WCs from here
             final_scale   = lambda_chi   #to here
            ):
    WCs = LEFT_WCs.copy()
    
    for C in WC:
        WCs[C] = WC[C]
        
    WC = WCs
    
    final_WC = WC.copy()
    
    C6_SL_sol = integrate.solve_ivp(RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [WC["SL(6)"]], 
                                    method = "RK45", rtol = 1e-4)
    C6_SR_sol = integrate.solve_ivp(RGEC6_S, [np.log(initial_scale), np.log(final_scale)], [WC["SR(6)"]], 
                                    method = "RK45", rtol = 1e-4)
    C6_T_sol  = integrate.solve_ivp(RGEC6_T, [np.log(initial_scale), np.log(final_scale)], [WC["T(6)"]], 
                                    method = "RK45", rtol = 1e-4)

    final_WC["SL(6)"] = C6_SL_sol.y[0][-1]
    final_WC["SR(6)"] = C6_SR_sol.y[0][-1]
    final_WC["T(6)"]  = C6_T_sol.y[0][-1]

    C9_1_sol_L       = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1L(9)"]], 
                                           method = "RK45", rtol = 1e-4)
    C9_1_prime_sol_L = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1L(9)prime"]], 
                                           method = "RK45", rtol = 1e-4)

    final_WC["1L(9)"]      = C9_1_sol_L.y[0][-1]
    final_WC["1L(9)prime"] = C9_1_prime_sol_L.y[0][-1]

    C9_1_sol_R       = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1R(9)"]], 
                                           method = "RK45", rtol = 1e-4)
    C9_1_prime_sol_R = integrate.solve_ivp(RGEC9_1, [np.log(initial_scale), np.log(final_scale)], [WC["1R(9)prime"]], 
                                           method = "RK45", rtol = 1e-4)

    final_WC["1R(9)"]      = C9_1_sol_R.y[0][-1]
    final_WC["1R(9)prime"] = C9_1_prime_sol_R.y[0][-1]

    C9_23_sol_L       = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], [WC["2L(9)"], WC["3L(9)"]], 
                                            method = "RK45", rtol = 1e-4)
    C9_23_prime_sol_L = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], 
                                            [WC["2L(9)prime"], WC["3L(9)prime"]], 
                                            method = "RK45", rtol = 1e-4)

    final_WC["2L(9)"]      = C9_23_sol_L.y[0][-1]
    final_WC["2L(9)prime"] = C9_23_prime_sol_L.y[0][-1]
    final_WC["3L(9)"]      = C9_23_sol_L.y[1][-1]
    final_WC["3L(9)prime"] = C9_23_prime_sol_L.y[1][-1]

    C9_23_sol_R       = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], 
                                            [WC["2R(9)"], WC["3R(9)"]], 
                                            method = "RK45", rtol = 1e-4)
    C9_23_prime_sol_R = integrate.solve_ivp(RGEC9_23, [np.log(initial_scale), np.log(final_scale)], 
                                            [WC["2R(9)prime"], WC["3R(9)prime"]], 
                                            method = "RK45", rtol = 1e-4)

    final_WC["2R(9)"]      = C9_23_sol_R.y[0][-1]
    final_WC["2R(9)prime"] = C9_23_prime_sol_R.y[0][-1]
    final_WC["3R(9)"]      = C9_23_sol_R.y[1][-1]
    final_WC["3R(9)prime"] = C9_23_prime_sol_R.y[1][-1]

    C9_45_sol_L = integrate.solve_ivp(RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [WC["4L(9)"], WC["5L(9)"]], 
                                      method = "RK45", rtol = 1e-4)

    final_WC["4L(9)"] = C9_45_sol_L.y[0][-1]
    final_WC["5L(9)"] = C9_45_sol_L.y[1][-1]

    C9_45_sol_R = integrate.solve_ivp(RGEC9_45, [np.log(initial_scale), np.log(final_scale)], [WC["4R(9)"], WC["5R(9)"]], 
                                      method = "RK45", rtol = 1e-4)

    final_WC["4R(9)"] = C9_45_sol_R.y[0][-1]
    final_WC["5R(9)"] = C9_45_sol_R.y[1][-1]

    C9_67_sol       = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["6(9)"], WC["7(9)"]], 
                                          method = "RK45", rtol = 1e-4)
    C9_67_prime_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], 
                                          [WC["6(9)prime"], WC["7(9)prime"]],
                                          method = "RK45", rtol = 1e-4)

    final_WC["6(9)"]      = C9_67_sol.y[0][-1]
    final_WC["6(9)prime"] = C9_67_prime_sol.y[0][-1]
    final_WC["7(9)"]      = C9_67_sol.y[1][-1]
    final_WC["7(9)prime"] = C9_67_prime_sol.y[1][-1]

    C9_89_sol       = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], [WC["8(9)"], WC["9(9)"]], 
                                          method = "RK45", rtol = 1e-4)
    C9_89_prime_sol = integrate.solve_ivp(RGEC9_67_89, [np.log(initial_scale), np.log(final_scale)], 
                                          [WC["8(9)prime"], WC["9(9)prime"]], 
                                          method = "RK45", rtol = 1e-4)

    final_WC["8(9)"]      = C9_89_sol.y[0][-1]
    final_WC["8(9)prime"] = C9_89_prime_sol.y[0][-1]
    final_WC["9(9)"]      = C9_89_sol.y[1][-1]
    final_WC["9(9)prime"] = C9_89_prime_sol.y[1][-1]

    return final_WC


#################################################################################################
#                                                                                               #
#                                                                                               #
#                                    RGEs of SMEFT Operators                                    #
#                                                                                               #
#                                                                                               #
#################################################################################################
'''
    #############################################################################################

    Define RGEs computed in 1901.10302

    Note that "scale" in the RGEs refers to log(mu)
    i.e. "scale" = log(mu), while scale in the final 
    functions refers to the actual energy scale i.e. 
    "scale" = mu

    #############################################################################################
'''
##############################################

def RGELLduD1(scale, C):
    scale = np.exp(scale)
    rge = 1/(4*np.pi) * (  1/10 * alpha1(scale) 
                         -  1/2 * alpha2(scale))*C
    return(rge)

##############################################

def C_LLduD1(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C = integrate.solve_ivp(RGELLduD1, [np.log(initial_scale), np.log(final_scale)], [WC["LLduD1(7)"]], 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C)

####################################################################################################

def RGELHDe(scale, C):
    scale = np.exp(scale)
    rge = 1/(4*np.pi) * (- 9/10 * alpha1(scale) 
                         +    6 * alpha_lambda(scale) 
                         +    9 * alpha_t(scale))*C
    return(rge)

##############################################

def C_LHDe(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C = integrate.solve_ivp(RGELHDe, [np.log(initial_scale), np.log(final_scale)], [WC["LHDe(7)"]], 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C)

####################################################################################################

def RGELeudH(scale, C):
    scale = np.exp(scale)
    rge = 1/(4*np.pi) * (- 69/20 * alpha1(scale) 
                         -   9/4 * alpha2(scale) 
                         +     3 * alpha_t(scale))*C
    return(rge)

##############################################

def C_LeudH(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C = integrate.solve_ivp(RGELeudH, [np.log(initial_scale), np.log(final_scale)], [WC["LeudH(7)"]], 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C)

####################################################################################################

def RGELLQuH(scale, C):
    scale = np.exp(scale)
    rge = 1/(4*np.pi) * (  1/20 * alpha1(scale) 
                         -  3/4 * alpha2(scale) 
                         -    8 * alpha3(scale)
                         +    3 * alpha_t(scale))*C
    return(rge)

##############################################

def C_LLQuH(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C = integrate.solve_ivp(RGELLQuH, [np.log(initial_scale), np.log(final_scale)], [WC["LLQuH(7)"]], 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C)  

####################################################################################################

def RGELLQdH12(scale, C):
    scale = np.exp(scale)
    LLQdH11 = (  13/20 * alpha1(scale) 
               +   9/4 * alpha2(scale) 
               -     8 * alpha3(scale) 
               +     3 * alpha_t(scale))

    LLQdH12 = (      6 * alpha2(scale))

    LLQdH21 = (-    4/3 * alpha1(scale) 
               +   16/3 * alpha3(scale))

    LLQdH22 = (- 121/60 * alpha1(scale) 
               -   15/4 * alpha2(scale) 
               +    8/3 * alpha3(scale) 
               +      3 * alpha_t(scale))

    rge = np.array([[LLQdH11, LLQdH12], 
                    [LLQdH21, LLQdH22]])

    return(1/(4*np.pi)*np.dot(rge, C))

##############################################

def C_LLQdH1(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C0 = [WC["LLQdH1(7)"], WC["LLQdH2(7)"]]
    C = integrate.solve_ivp(RGELLQdH12, [np.log(initial_scale), np.log(final_scale)], y0 = C0, 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C) 


def C_LLQdH2(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    C0 = [WC["LLQdH1(7)"], WC["LLQdH2(7)"]]
    C = integrate.solve_ivp(RGELLQdH12, [np.log(initial_scale), np.log(final_scale)], y0 = C0, 
                            method = "RK45", rtol = 1e-4).y[1][-1]
    return(C) 

####################################################################################################

def RGE_LHD1_LHD2_LHW(scale, C):
    scale = np.exp(scale)
    LHD1_LHD1 = (-  9/10 * alpha1(scale) 
                 +  11/2 * alpha2(scale) 
                 +     6 * alpha_t(scale))

    LHD1_LHD2 = (- 33/20 * alpha1(scale) 
                 -  19/4 * alpha2(scale) 
                 -     2 * alpha_lambda(scale))
    LHD1_LHW  = 0

    LHD2_LHD1 = (-     8 * alpha2(scale))

    LHD2_LHD2 = (   12/5 * alpha1(scale) 
                 +     3 * alpha2(scale) 
                 +     4 * alpha_lambda(scale)
                 +     6 * alpha_t(scale))

    LHD2_LHW  = 0

    LHW_LHD1  = (    5/8 * alpha2(scale))

    LHW_LHD2  = (-  9/80 * alpha1(scale)
                 + 11/16 * alpha2(scale))

    LHW_LHW   = (-   6/5 * alpha1(scale) 
                 +  13/2 * alpha2(scale) 
                 +     4 * alpha_lambda(scale) 
                 +     6 * alpha_t(scale))


    rge = np.array([[LHD1_LHD1, LHD1_LHD2, LHD1_LHW], 
                    [LHD2_LHD1, LHD2_LHD2, LHD2_LHW], 
                    [ LHW_LHD1,  LHW_LHD2,  LHW_LHW]])

    return(1/(4*np.pi)*np.dot(rge, C))

##############################################

def C_LHD1(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    
    C0 = [WC["LHD1(7)"], WC["LHD2(7)"], WC["LHW(7)"]]
    C = integrate.solve_ivp(RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], y0 = C0, 
                            method = "RK45", rtol = 1e-4).y[0][-1]
    return(C)

##############################################

def C_LHD2(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    
    C0 = [WC["LHD1(7)"], WC["LHD2(7)"], WC["LHW(7)"]]
    C = integrate.solve_ivp(RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], y0 = C0, 
                            method = "RK45", rtol = 1e-4).y[1][-1]
    return(C)

##############################################

def C_LHW(WC, initial_scale, final_scale = m_W):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs
    
    C0 = [WC["LHD1(7)"], WC["LHD2(7)"], WC["LHW(7)"]]
    C = integrate.solve_ivp(RGE_LHD1_LHD2_LHW, [np.log(initial_scale), np.log(final_scale)], y0 = C0, 
                            method = "RK45", rtol = 1e-4).y[2][-1]
    return(C)

####################################################################################################

def run_SMEFT(WC, initial_scale, final_scale = m_W, inplace = False):
    #make a dict with all WCs
    WCs = SMEFT_WCs.copy()

    #write the non-zero WCs
    for C in WC:
        WCs[C] = WC[C]
    WC = WCs

    final_WC = WC.copy()
    final_WC["LHDe(7)"]   = C_LHDe(WC   = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LHW(7)"]    = C_LHW(WC    = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LHD1(7)"]   = C_LHD1(WC   = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LHD2(7)"]   = C_LHD2(WC   = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LeudH(7)"]  = C_LeudH(WC  = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LLQdH1(7)"] = C_LLQdH1(WC = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LLQdH2(7)"] = C_LLQdH2(WC = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LLQuH(7)"]  = C_LLQuH(WC  = WC, initial_scale = initial_scale, final_scale = final_scale)
    final_WC["LLduD1(7)"] = C_LLduD1(WC = WC, initial_scale = initial_scale, final_scale = final_scale)

    if inplace:
        WC = final_WC.copy()

    return(final_WC)