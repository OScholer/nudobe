from constants import *
import pandas as pd


#to get files in path
import os

#current working directory as absolute path
import sys
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())
    
#get absolute path of this file
cwd = os.path.abspath(os.path.dirname(__file__))


def Load_NMEs(method):
    #Read NMEs from csv file
    NMEpanda = pd.read_csv(cwd+"/../NMEs/"+method+".csv", 
                           delimiter = "  ,  |  , |  ,| ,  | , | ,|,  |, |,", 
                           engine = "python"
                          )
    NMEpanda.set_index("NME", inplace = True)
    
    #adjust typing to float and the definition of short-range NMEs
    for j in range(len(NMEpanda.index)):
        if j == 5:
            pass
        else:
            NMEpanda.iloc[j] = NMEpanda.iloc[j].astype(float)
            #fix short-range conventions
            if j >= 9:
                NMEpanda.iloc[j] *= m_e*m_p/(m_pi**2)
                
    #List of NME names
    NMEnames = ["F", 
                "GTAA", "GTAP", "GTPP", "GTMM", 
                "TAA", "TAP", "TPP", "TMM" , 
                "F,sd", 
                "GTAA,sd", "GTAP,sd", "GTPP,sd" , 
                "TAP,sd" , "TPP,sd"]
    
    #NMEs in dict format
    NMEs = {}
    for column in NMEpanda:
        element = column
        NME = {}
        for idx in range(len(NMEnames)):
            try:
                NME[NMEnames[idx]] = float(NMEpanda[column][idx])
            except:
                NME[NMEnames[idx]] = 0
        try:
            NMEs[element] = NME
        except:
            NMEs[element] = NME
            
    return (NMEs, NMEpanda, NMEnames)

