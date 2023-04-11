#to get files in path
import os

#current working directory as absolute path
import sys
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())
    
#get absolute path of this file
cwd = os.path.abspath(os.path.dirname(__file__))

sys.path.append(cwd+"/src/")

import EFT
import functions
import constants
import plots
import PSFs
import RGE