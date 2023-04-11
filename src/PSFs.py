#!Note! In this file we use MeV units
#See 1506.07145 for the definitions of the wave functions and PSFs
#Note that we do not include V_ud into the PSFs, instead we include them in the sub-amplitudes (see EFT.py)

import numpy as np
import pandas as pd

#numerical 
from scipy.special import gamma as Gamma
GAMMA = Gamma
from scipy.constants import physical_constants as pc
from scipy import integrate

#physical constants and units
from constants import *

#use mpmath for hyp1f1 (sadly scipy doesn't allow complex input)
import mpmath as mp
mp.dps = 50; mp.pretty = True


#to get files in path
import os

#current working directory as absolute path
import sys
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())
    
#get absolute path of this file
cwd = os.path.abspath(os.path.dirname(__file__))


#isotope classes generate wave functions and PSFs
class isotope(object):
    def __init__(self, 
                 Z,           #charge number of mother isotope
                 A,           #mass number of mother isotope
                 Delta_M,     #mass difference between mother and daughter isotope
                 scheme = "A" #wave-function scheme (A = uniform charge distribution, B = point-like charge distribution)
                ):
        #store parameters
        self.Z       = Z + 2 # We need Z after the decay hence +2
        self.A       = A
        self.Delta_M = Delta_M
        self.E_max   = Delta_M
        self.scheme  = scheme
        self.alpha   = alpha
        self.m_e     = m_e_MeV
        self.R       = 1.2 * A**(1/3) * fm_to_inverse_MeV #radius of the nucleus
        self.Q       = self.Delta_M-2*self.m_e            #Q-value of the decay
        '''_______________________________________'''
        '''                                       '''
        ''' Define Wave Functions to compute PSFs '''
        '''_______________________________________'''  
        
        #uniform charge-distribution
        if self.scheme == "A":
            '''Define Helper Functions'''
            def gamma(k, Z):
                return(np.sqrt(k**2 - (self.alpha*Z)**2))

            def y(E, Z):
                p = np.sqrt(E**2 - self.m_e**2)
                return(self.alpha*Z*(E/p))

            def F(k, E, r, Z): #note that F_k = F(k+1)
                p = np.sqrt(E**2 - self.m_e**2)
                return(((Gamma(2*k+1)/(Gamma(k)*Gamma(1+2*gamma(k, Z))))**2
                       * (2*p*r)**(2*(gamma(k, Z)-1))
                       * np.exp(np.pi*y(E, Z))
                       * np.absolute(Gamma(gamma(k, Z)+1j*y(E, Z)))**2
                      ))

            '''Define Wavefunctions'''
            #S-Wave
            def g_m(E, r, Z):
                return(np.sqrt(F(1, E, r, Z)*((E+self.m_e)/(2*E))))

            def f_p(E, r, Z):            
                return(np.sqrt(F(1, E, r, Z)*((E-self.m_e)/(2*E))))

            #P-Wave
            def g_p(E, r, Z):
                return(np.sqrt(F(1, E, r, Z)*((E-self.m_e)/(2*E)))
                       * (self.alpha*Z/2 + (E+self.m_e)*r/3))

            def f_m(E, r, Z):
                return(-np.sqrt(F(1, E, r, Z)*((E+self.m_e)/(2*E)))
                       * (self.alpha*Z/2 + (E-self.m_e)*r/3))
            
            self.eps = 0
        
        #point-like charge distribution
        elif self.scheme == "B":
            def F(k):
                k += 1
                return(GAMMA(2*k+1)/(GAMMA(k)*GAMMA(1+2*gamma)))
            
            def hyp1f1(a,b,x):
                try:
                    return(complex(mp.hyp1f1(a,b,x)))
                except:
                    result = np.zeros(len(x), dtype = complex)
                    for idx in range(len(x)):
                        result[idx] = complex(mp.hyp1f1(a[idx],b,x[idx]))
                    return(result)

            def g(kappa, E, r, Z):
                k     = np.absolute(kappa)
                p     = np.sqrt(E**2 - m_e_MeV**2)
                
                try:
                    if len(p)>0:
                        p *= mp.mpc(1)
                except:
                    pass
                try:
                    if len(r)>0:
                        r *= mp.mpc(1)
                except:
                    pass
                gamma = np.sqrt(k**2 - (alpha*Z)**2)
                y     = alpha*Z*E/p
                ksi   = -1/(2*1j)*np.log((gamma-1j*y)/(kappa-1j*y*m_e_MeV/E))
                return(kappa/k * 1/(p*r) * np.sqrt((E + m_e_MeV)/(2*E)) 
                       * np.absolute(GAMMA(1+gamma+1j*y))/GAMMA(1+2*gamma) 
                       * (2*p*r)**gamma * np.exp(np.pi*y/2) 
                       * np.imag(np.exp((1j*(p*r+ksi))) * hyp1f1(gamma - 1j*y, 1+2*gamma, -2*1j*p*r)))

            def f(kappa, E, r, Z):
                k     = np.absolute(kappa)
                p     = np.sqrt(E**2 - m_e_MeV**2)
                try:
                    if len(p)>0:
                        p *= mp.mpc(1)
                except:
                    pass
                try:
                    if len(r)>0:
                        r *= mp.mpc(1)
                except:
                    pass
                gamma = np.sqrt(k**2 - (alpha*Z)**2)
                y     = alpha*Z*E/p
                ksi   = -1/(2*1j)*np.log((gamma-1j*y)/(kappa-1j*y*m_e_MeV/E))
                return(kappa/k * 1/(p*r) * np.sqrt((E - m_e_MeV)/(2*E)) 
                       * np.absolute(GAMMA(1+gamma+1j*y))/GAMMA(1+2*gamma) 
                       * (2*p*r)**gamma * np.exp(np.pi*y/2) 
                       * np.real(np.exp((1j*(p*r+ksi))) * hyp1f1(gamma - 1j*y, 1+2*gamma, -2*1j*p*r)))

            '''Define Wavefunctions'''
            #S-Wave
            def g_m(E, r, Z):
                return(g(-1, E, r, Z))

            def f_p(E, r, Z):            
                return(f(1, E, r, Z))

            #P-Wave
            def g_p(E, r, Z):
                return(g(1, E, r, Z))

            def f_m(E, r, Z):
                return(f(-1, E, r, Z))
            
            self.eps = 0
            
        #write wave-functions
        self.g_m = g_m
        self.g_p = g_p
        self.f_m = f_m
        self.f_p = f_p
        
        
        
        
    '''_______________________________________'''
    '''                                       '''
    '''Define Helper Functions to compute PSFs'''
    '''_______________________________________'''
    
    
    '''Wavefunction Combinations'''
    def Css(self, E):
        return (self.g_m(E, self.R, self.Z) * self.f_p(E, self.R, self.Z))
    
    def Css_m(self, E):
        return (self.g_m(E, self.R, self.Z)**2 - self.f_p(E, self.R, self.Z)**2)
    
    def Css_p(self, E):
        return (self.g_m(E, self.R, self.Z)**2 + self.f_p(E, self.R, self.Z)**2)
    
    def Csp_f(self, E):
        return (self.f_m(E, self.R, self.Z) * self.f_p(E, self.R, self.Z))
    
    def Csp_m(self, E):
        return (self.g_m(E, self.R, self.Z)*self.f_m(E, self.R, self.Z) 
                - self.g_p(E, self.R, self.Z)*self.f_p(E, self.R, self.Z))
    
    def Csp_p(self, E):
        return (self.g_m(E, self.R, self.Z)*self.f_m(E, self.R, self.Z) 
                + self.g_p(E, self.R, self.Z)*self.f_p(E, self.R, self.Z))
    
    def Cpp(self, E):
        return (self.g_p(E, self.R, self.Z)*self.f_m(E, self.R, self.Z))
    
    def Csp_g(self, E):
        return (self.g_m(E, self.R, self.Z)*self.g_p(E, self.R, self.Z))
    
    def Cpp_m(self, E):
        return (self.g_p(E, self.R, self.Z)**2 - self.f_m(E, self.R, self.Z)**2)
    
    def Cpp_p(self, E):
        return (self.g_p(E, self.R, self.Z)**2 + self.f_m(E, self.R, self.Z)**2)
    
    
    '''Angular Distribution Functions'''
    
    def h_01(self, E_1, E_2):
        return (-4*self.Css(E_1)*self.Css(E_2))
    
    def h_02(self, E_1, E_2):
        return (2*(E_1-E_2)**2/(self.m_e**2)*self.Css(E_1)*self.Css(E_2))
    
    def h_03(self, E_1, E_2):
        return 0
    
    def h_04(self, E_1, E_2):
        return (-2/(3*self.m_e*self.R)*(self.Csp_f(E_1)*self.Css(E_2) 
                                      + self.Csp_f(E_2)*self.Css(E_1)
                                      + self.Csp_g(E_2)*self.Css(E_1) 
                                      + self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_05(self, E_1, E_2):
        return (4/(self.m_e*self.R) * (self.Csp_f(E_1)*self.Css(E_2) 
                                       + self.Csp_f(E_2)*self.Css(E_1) 
                                       + self.Csp_g(E_2)*self.Css(E_1) 
                                       + self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_06(self, E_1, E_2):
        return 0
    
    def h_07(self, E_1, E_2):
        return (-16/(self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Css(E_2) 
                                       + self.Csp_f(E_2)*self.Css(E_1) 
                                       - self.Csp_g(E_2)*self.Css(E_1) 
                                       - self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_08(self, E_1, E_2):
        return (-8/(self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Csp_g(E_2) 
                                       + self.Csp_f(E_2)*self.Csp_g(E_1) 
                                       + self.Css(E_1)*self.Cpp(E_2) 
                                       + self.Css(E_2)*self.Cpp(E_1)))
    
    def h_09(self, E_1, E_2):
        return (32/(self.m_e*self.R)**2 *self.Css(E_1)*self.Css(E_2))
    
    def h_010(self, E_1, E_2):
        return (-9/2*self.h_010tilde(E_1, E_2) - self.h_02(E_1, E_2))
    
    def h_011(self, E_1, E_2):
        return (9*self.h_011tilde(E_1, E_2) + 1/9*self.h_02(E_1, E_2) + self.h_010tilde(E_1, E_2))
    
    '''with'''
    
    def h_010tilde(self, E_1, E_2):
        return (2*(E_1-E_2)/(3*self.m_e**2*self.R) * (self.Csp_f(E_1)*self.Css(E_2) 
                                                     - self.Csp_f(E_2)*self.Css(E_1) 
                                                     + self.Csp_g(E_2)*self.Css(E_1) 
                                                     - self.Csp_g(E_1)*self.Css(E_2)))
    
    def h_011tilde(self, E_1, E_2):
        return (-2/(3*self.m_e*self.R)**2 * (self.Csp_f(E_1)*self.Csp_f(E_2) 
                                             + self.Csp_g(E_2)*self.Csp_g(E_1)
                                             + self.Css(E_1)*self.Cpp(E_2) 
                                             + self.Css(E_2)*self.Cpp(E_1)))
    
    '''Components of Phase Space Factors'''
    def g_01(self, E_1, E_2):
        return (self.Css_p(E_1)*self.Css_p(E_2))
    
    def g_11(self, E_1, E_2):
        return self.g_01(E_1, E_2)
    
    def g_02(self, E_1, E_2):
        return ((E_1 - E_2)**2/(2*self.m_e**2) * (self.Css_p(E_1)*self.Css_p(E_2) 
                                               - self.Css_m(E_1)*self.Css_m(E_2)))
    
    def g_03(self, E_1, E_2):
        return ((E_1-E_2)/self.m_e * (self.Css_p(E_1)*self.Css_m(E_2) 
                                      - self.Css_p(E_2)*self.Css_m(E_1)))
    
    def g_04(self, E_1, E_2):
        return (1/(3*self.m_e*self.R) * (-self.Css_m(E_1)*self.Csp_m(E_2) 
                                         - self.Css_m(E_2)*self.Csp_m(E_1) 
                                         + self.Css_p(E_1)*self.Csp_p(E_2) 
                                         + self.Css_p(E_2)*self.Csp_p(E_1)) - self.g_03(E_1, E_2)/9)
    
    def g_05(self, E_1, E_2):
        return (-2/(self.m_e*self.R) * (self.Css_m(E_1)*self.Csp_m(E_2) 
                                         + self.Css_m(E_2)*self.Csp_m(E_1) 
                                         + self.Css_p(E_1)*self.Csp_p(E_2) 
                                         + self.Css_p(E_2)*self.Csp_p(E_1)))
    
    def g_06(self, E_1, E_2):
        return (4/(self.m_e*self.R) * (self.Css_p(E_1)*self.Css_m(E_2) 
                                       + self.Css_p(E_2)*self.Css_m(E_1)))
    
    def g_07(self, E_1, E_2):
        return (-8/(self.m_e*self.R)**2 * (self.Css_p(E_1)*self.Csp_m(E_2) 
                                           + self.Css_p(E_2)*self.Csp_m(E_1) 
                                           + self.Css_m(E_1)*self.Csp_p(E_2) 
                                           + self.Css_m(E_2)*self.Csp_p(E_1)))
    
    def g_08(self, E_1, E_2):
        return (2/(self.m_e*self.R)**2 * (-self.Cpp_m(E_1)*self.Css_m(E_2) 
                                          - self.Cpp_m(E_2)*self.Css_m(E_1) 
                                          + self.Cpp_p(E_1)*self.Css_p(E_2)
                                          + self.Cpp_p(E_2)*self.Css_p(E_1) 
                                          + 2*self.Csp_m(E_1)*self.Csp_m(E_2) 
                                          + 2*self.Csp_p(E_1)*self.Csp_p(E_2)))
    
    def g_09(self, E_1, E_2):
        return (8/(self.m_e*self.R)**2 * (self.Css_p(E_1)*self.Css_p(E_2) 
                                          + self.Css_m(E_1)*self.Css_m(E_2)))
    
    def g_010(self, E_1, E_2):
        return (-9/2*self.g_010tilde(E_1, E_2) - self.g_02(E_1, E_2))
    
    def g_011(self, E_1, E_2):
        return (9*self.g_011tilde(E_1, E_2) + 1/9 * self.g_02(E_1, E_2) + self.g_010tilde(E_1, E_2))
    
    '''with'''
    
    def g_010tilde(self, E_1, E_2):
        return ((E_1 - E_2)/(3*self.m_e**2*self.R) * (-self.Css_p(E_1)*self.Csp_m(E_2) 
                                                      + self.Css_p(E_2)*self.Csp_m(E_1) 
                                                      + self.Css_m(E_1)*self.Csp_p(E_2) 
                                                      - self.Css_m(E_2)*self.Csp_p(E_1)))
    
    def g_011tilde(self, E_1, E_2):
        return (1/(18*self.m_e**2*self.R**2) * (self.Cpp_m(E_1)*self.Css_m(E_2) 
                                             + self.Cpp_m(E_2)*self.Css_m(E_1)
                                             + self.Cpp_p(E_1)*self.Css_p(E_2) 
                                             + self.Cpp_p(E_2)*self.Css_p(E_1) 
                                             - 2*self.Csp_m(E_1)*self.Csp_m(E_2) 
                                             + 2*self.Csp_p(E_1)*self.Csp_p(E_2)))
    
    '''________________________________________'''
    '''                                        '''
    '''   Calculation of Phase Space Factors   '''
    '''________________________________________'''
    
    def PSFs(self):
        g = [self.g_01, self.g_02, self.g_03, 
             self.g_04, self.g_05, self.g_06, 
             self.g_07, self.g_08, self.g_09, 
             self.g_010, self.g_011]
        
        h = [self.h_01, self.h_02, self.h_03, 
             self.h_04, self.h_05, self.h_06, 
             self.h_07, self.h_08, self.h_09, 
             self.h_010, self.h_011]
        
        PSFs = []
        
        '''Constants'''
        #fermi constant G_F in 1/MeV^2
        G_beta = G_F*1e-6 #usually G_beta = G_F*V_ud. Instead we put V_ud into the sub-amplitudes (see EFT.py)
        
        #common prefactor for all G_0k
        prefactor = G_beta**4*self.m_e**2 / (64 * np.pi**5 * np.log(2) * self.R**2)
        
        #momentum
        def p(E, m = self.m_e):
            return (np.sqrt(E**2-m**2))
        
        #depending on angle
        G_theta_0k = []
        
        #independent of angle
        G_0k = []
        
        #do integrations to get PSFs
        for k in range(11):
            G_theta = integrate.quad(lambda E_1: (h[k](E_1, self.Delta_M - E_1) 
                                                  * p(E_1)*p(self.Delta_M-E_1)
                                                  * E_1 * (self.Delta_M - E_1)
                                                 ), 
                                     self.m_e + self.Q*self.eps, self.Delta_M-self.m_e-self.Q*self.eps)
            G = integrate.quad(lambda E_1: (g[k](E_1, self.Delta_M - E_1) 
                                            * p(E_1)*p(self.Delta_M-E_1)
                                            * E_1 * (self.Delta_M - E_1)
                                           ), 
                               self.m_e + self.Q*self.eps, self.Delta_M-self.m_e-self.Q*self.eps)
            
            #add to lists
            G_theta_0k.append(G_theta)
            G_0k.append(G)
            
        return (G_0k, G_theta_0k, 2*prefactor, np.log(2)*prefactor) #G_theta = G_theta[0]*prefactor *log(2), G = G * prefactor * 2
    
    #this function returns a pandas DataFrame containing the PSFs G_0k and the error on G_0k as obtained from scipy.integrate_quad
    def G0k(self):
        G_0k, G_theta_0k, prefactor, _ = self.PSFs()
        results = np.array(G_0k)[:,0] * prefactor * MeV_to_inverseyear
        error   = np.array(G_0k)[:,1] * prefactor * MeV_to_inverseyear
        index = []
        for x in range(11):
            index.append("G0"+str(x+1)+"")
        return(pd.DataFrame({r"$G_{0k}$" : results, r"$\Delta G_{0k}$" : error}, 
                            index = index)
              )



#this function generates the csv file for the PSFs given a wave-function scheme 
#(currently "A" = uniform charge density, or "B" = point-like charge distribution)
def make_PSFs(savefile = True, file = cwd+"/../PSFs/PSFs_A.csv", scheme = "A"):
    isotopes = {"238U"  : isotope(Z = 92, A = 238, Delta_M = 1.144154 + 2*m_e_MeV, scheme = scheme),
                "232Th" : isotope(Z = 90, A = 232, Delta_M = 0.837879 + 2*m_e_MeV, scheme = scheme),
                "204Hg" : isotope(Z = 80, A = 204, Delta_M = 0.419154 + 2*m_e_MeV, scheme = scheme),
                "198Pt" : isotope(Z = 78, A = 198, Delta_M = 1.049142 + 2*m_e_MeV, scheme = scheme),
                "192Os" : isotope(Z = 76, A = 192, Delta_M = 0.408274 + 2*m_e_MeV, scheme = scheme),
                "186W"  : isotope(Z = 74, A = 186, Delta_M = 0.491643 + 2*m_e_MeV, scheme = scheme),
                "176Yb" : isotope(Z = 70, A = 176, Delta_M = 1.088730 + 2*m_e_MeV, scheme = scheme),
                "170Er" : isotope(Z = 68, A = 170, Delta_M = 0.655586 + 2*m_e_MeV, scheme = scheme),
                "160Gd" : isotope(Z = 64, A = 160, Delta_M = 1.730530 + 2*m_e_MeV, scheme = scheme),
                "154Sm" : isotope(Z = 62, A = 154, Delta_M = 1.250810 + 2*m_e_MeV, scheme = scheme),
                "150Nd" : isotope(Z = 60, A = 150, Delta_M = 3.371357 + 2*m_e_MeV, scheme = scheme),
                "148Nd" : isotope(Z = 60, A = 148, Delta_M = 1.928286 + 2*m_e_MeV, scheme = scheme),
                "146Nd" : isotope(Z = 60, A = 146, Delta_M = 0.070421 + 2*m_e_MeV, scheme = scheme),
                "142Ce" : isotope(Z = 58, A = 142, Delta_M = 1.417175 + 2*m_e_MeV, scheme = scheme),
                "136Xe" : isotope(Z = 54, A = 136, Delta_M = 2.457984 + 2*m_e_MeV, scheme = scheme),
                "134Xe" : isotope(Z = 54, A = 134, Delta_M = 0.825751 + 2*m_e_MeV, scheme = scheme),
                "130Te" : isotope(Z = 52, A = 130, Delta_M = 2.527515 + 2*m_e_MeV, scheme = scheme),
                "128Te" : isotope(Z = 52, A = 128, Delta_M = 0.866550 + 2*m_e_MeV, scheme = scheme),
                "124Sn" : isotope(Z = 50, A = 124, Delta_M = 2.291010 + 2*m_e_MeV, scheme = scheme),
                "122Sn" : isotope(Z = 50, A = 122, Delta_M = 0.372877 + 2*m_e_MeV, scheme = scheme),
                "116Cd" : isotope(Z = 48, A = 116, Delta_M = 2.813438 + 2*m_e_MeV, scheme = scheme),
                "114Cd" : isotope(Z = 48, A = 114, Delta_M = 0.542493 + 2*m_e_MeV, scheme = scheme),
                "110Pd" : isotope(Z = 46, A = 110, Delta_M = 2.017234 + 2*m_e_MeV, scheme = scheme),
                "104Ru" : isotope(Z = 44, A = 104, Delta_M = 1.301297 + 2*m_e_MeV, scheme = scheme),
                "100Mo" : isotope(Z = 42, A = 100, Delta_M = 3.034342 + 2*m_e_MeV, scheme = scheme),
                "98Mo"  : isotope(Z = 42, A = 98,  Delta_M = 0.109935 + 2*m_e_MeV, scheme = scheme),
                "96Zr"  : isotope(Z = 40, A = 96,  Delta_M = 3.348982 + 2*m_e_MeV, scheme = scheme),
                "94Zr"  : isotope(Z = 40, A = 94,  Delta_M = 1.141919 + 2*m_e_MeV, scheme = scheme),
                "86Kr"  : isotope(Z = 36, A = 86,  Delta_M = 1.257542 + 2*m_e_MeV, scheme = scheme),
                "82Se"  : isotope(Z = 34, A = 82,  Delta_M = 2.996402 + 2*m_e_MeV, scheme = scheme),
                "80Se"  : isotope(Z = 34, A = 80,  Delta_M = 0.133874 + 2*m_e_MeV, scheme = scheme),
                "76Ge"  : isotope(Z = 32, A = 76,  Delta_M = 2.039061 + 2*m_e_MeV, scheme = scheme),
                "70Zn"  : isotope(Z = 30, A = 70,  Delta_M = 0.997118 + 2*m_e_MeV, scheme = scheme),
                "48Ca"  : isotope(Z = 20, A = 48,  Delta_M = 4.266970 + 2*m_e_MeV, scheme = scheme),
                "46Ca"  : isotope(Z = 20, A = 46,  Delta_M = 0.988576 + 2*m_e_MeV, scheme = scheme)
               }
    
    #pandas DataFrame containing the PSFs for each isotope
    PSFpanda = pd.DataFrame({"PSFs" : ["G01", "G02", "G03",
                                       "G04", "G05", "G06",
                                       "G07", "G08", "G09",
                                       "G010", "G011"]})
    
    #pandas DataFrame containing the errors on the PSFs for each isotope
    Errorpanda = pd.DataFrame({"PSFs" : ["G01", "G02", "G03",
                                         "G04", "G05", "G06",
                                         "G07", "G08", "G09",
                                         "G010", "G011"]})
    
    #same as PSFpanda but for the angular dependent part
    PSFThetapanda = pd.DataFrame({"PSFs" : ["G01", "G02", "G03",
                                            "G04", "G05", "G06",
                                            "G07", "G08", "G09",
                                            "G010", "G011"]})
    
    #same as Errorpanda but for the angular dependent part
    ErrorThetapanda = pd.DataFrame({"PSFs" : ["G01", "G02", "G03",
                                              "G04", "G05", "G06",
                                              "G07", "G08", "G09",
                                              "G010", "G011"]})
    
    #iterate over isotopes to write to the pandas DataFrames
    for iso in isotopes:
        name = iso
        PSFs = isotopes[iso].PSFs()
        
        #write DataFrames
        PSFpanda[name] = np.array(PSFs[0])[:,0] * MeV_to_inverseyear * PSFs[2]
        Errorpanda[name] = np.array(PSFs[0])[:,1] * MeV_to_inverseyear * PSFs[2]
        PSFThetapanda[name] = np.array(PSFs[1])[:,0] * MeV_to_inverseyear * PSFs[3]
        ErrorThetapanda[name] = np.array(PSFs[1])[:,1] * MeV_to_inverseyear * PSFs[3]
    
    
    PSFpanda.set_index("PSFs", inplace = True)
    Errorpanda.set_index("PSFs", inplace = True)
    PSFThetapanda.set_index("PSFs", inplace = True)
    ErrorThetapanda.set_index("PSFs", inplace = True)
    
    #save file if desired
    if savefile:
        PSFpanda.to_csv(file)
        
    #return results
    return(PSFpanda, Errorpanda, PSFThetapanda, ErrorThetapanda)