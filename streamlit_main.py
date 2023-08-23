#Streamlit generates the user-interface
import streamlit as st

#Import external tools
from math import floor, log10
import numpy as np
import pandas as pd
from scipy import integrate

#Typing
import base64

#Import nudobe Parts
#current working directory as absolute path
import sys
import os
if not hasattr(sys.modules[__name__], '__file__'):
    __file__ = inspect.getfile(inspect.currentframe())
    
#get absolute path of this file
cwd = os.path.abspath(os.path.dirname(__file__))

sys.path.append(cwd+"/src/")
#from nudobe 
import EFT
#from nudobe 
import functions as f
#from nudobe 
import plots
#from nudobe 
import constants
from constants import *

import matplotlib


####################################################################################################
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
#                                         Welcome Screen                                           #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

#Title
#st.title("The tool which must not be named")
st.title(r"$\nu$DoBe - Online")

#Contact
st.markdown('''If you use results of this tool in your scientific work, please add a citation to:  **<a href="https://arxiv.org/abs/2304.05415">arxiv:2304.05415</a>**. <br>
               For more advanced analyses, you can download the full code from <a href = "https://github.com/OScholer/nudobe">gitHub</a>.<br> 
               You have any suggestions/comments on how to improve the tool?<br> => Contact:<br>
               Oliver Scholer: scholer@mpi-hd.mpg.de<br> 
               Lukas Graf: lukas.graf@berkeley.edu<br> 
               Jordy de Vries: j.devries4@uva.nl
            ''', unsafe_allow_html=True)

#Selectbox to choose what to do
path_option = st.selectbox("Please specify what you would like to do:", options = ["-", "Define a model", "Study operator limits"])



####################################################################################################
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
#                                 DEFINE A MODEL IN SMEFT OR LEFT                                  #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

if path_option == "Define a model":
    #Model Name
    name = st.text_input("If you want you can give your model name. This name will be displayed in all plots", value="Model")
    
    #Sidebar NME Method
    method = st.sidebar.selectbox("Which NME approximation do you want to use?", options = ["IBM2", "QRPA", "SM"], help = "Currently we allow for 3 different sets of nuclear matrix elements (NMEs): IBM2: F. Deppisch et al., 2020, arxiv:2009.10119 | QRPA: J. Hyvärinen and J. Suhonen, 2015, Phys. Rev. C 91, 024613 | Shell Model (SM): J. Menéndez, 2018, arXiv:1804.02105")
    
    #Sidebar EFT Choice
    model_option = st.sidebar.selectbox("Do you wish to define your model in terms of LEFT or SMEFT Wilson coefficients?", options = ["-","LEFT", "SMEFT"])
    ################################################################################################
    #                                                                                              #
    #                                        Low-Energy EFT                                        #
    #                                                                                              #
    ################################################################################################
    if model_option == "LEFT":
        #Allow for complex phases of WCs?
        phases = st.sidebar.checkbox("Allow for complex phases?", help = "If you check this box you can set complex phases for each Wilson coefficient.")
        
        #Generate Dicts
        LEFT_WCs = {}
        cols = {}
        
        
        ############################################################################################
        #                                  WC Parameter Input                                      #
        ############################################################################################
        st.sidebar.subheader("Effective Neutrino Mass")
        
        #Loop over all WCs to generate sidebar input fields
        for WC in EFT.LEFT_WCs:
            #Prefactors and such
            if WC == "m_bb":
                factor = 1e-12
                text = "meV"
                val = 100.
            elif WC[-2] == "6":
                factor = 1e-9
                text = "10^-9"
                val = 0.
            else:
                factor = 1e-6
                text = "10^-6"
                val = 0.
                
            #Absolute Value Input
            LEFT_WCs[WC] = st.sidebar.number_input(WC+" ["+text+"]", value = val)*factor
            
            #Complex Phase Input
            if phases:
                LEFT_WCs[WC] *= np.exp(1j*st.sidebar.number_input(WC+" phase [pi]")*np.pi)
                st.sidebar.write("________________________________")
        
        #Generate Model from Input
        LEFT_model = EFT.LEFT(LEFT_WCs, method=method, name = name)
        
        ############################################################################################
        #                                     Output Screen                                        #
        ############################################################################################
        #Half-lives
        st.subheader("Half-lives")
        #Generate Half-Lives
        hl = LEFT_model.half_lives()
        
        #Rename Index Label
        hl.rename(index = {0:"10^24 years"}, inplace = True)
        
        #Round Half-Life Values to 2 digits
        if np.inf not in hl.values:
            hl = hl.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
            
        #Downloadable CSV
        def get_table_download_link_csv(df):
            #csv = df.to_csv(index=False)
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            #b64 = base64.b64encode(csv.encode()).decode() 
            b64 = base64.b64encode(csv).decode()
            href = f'Download half-lives as <a href="data:file/csv;base64,{b64}" download="LEFT_model_half_lives.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="LEFT_model_half_lives.tex" target="_blank">.tex</a> file.'
            return href
        
        #Download Button
        st.markdown(get_table_download_link_csv(hl.T), unsafe_allow_html=True)
        
        #Show Table
        st.table(hl.T*1e-24)
        
        #Angular Correlation
        st.subheader("Angular correlation")
        st.latex(r"\frac{\mathrm{d}\Gamma}{\mathrm{d}\cos\theta\mathrm{d}\overline{\epsilon}_1} = a_0\left(1+\frac{a_1}{a_0}\cos\theta\right)")
        
        #Initial Isotope is 76Ge
        #Get Index of 76Ge in NME list
        ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        
        #Select Box to choose the isotope of interest
        plot_isotope = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                    key = "angularcorrisotope")
        
        #Checkbox if comparison to mass mechanism is desired
        show_mbb1 = st.checkbox("Compare to mass mechanism?", key="show_mbb1")
        
        #Generate Figure
        fig_angular_corr = LEFT_model.plot_corr(show_mbb=show_mbb1, isotope = plot_isotope)
        
        #Show Figure
        st.pyplot(fig_angular_corr)
        
        #Electron Spectra
        st.subheader("Normalized single electron spectrum")
        st.latex(r'''\frac{\mathrm{d}\Gamma}{\mathrm{d}\epsilon_1} 
                     \left(\left\{C_i\right\}, \overline{\epsilon}\right) \propto \sum_k g_{0k}
                     \left(\epsilon, \Delta M - \epsilon, R\right)
                     \left|A_{k}(\{C_i\})\right|^2p_1 p_2 \epsilon\left(\Delta M-\epsilon\right)''')
        
        #ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        
        #Select Box to choose the isotope of interest
        plot_isotope2 = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "spectraisotope")
        #integral = integrate.quad(lambda E: LEFT_model.spectrum(E), 0, 1)[0]
        #st.line_chart({name: np.real(LEFT_model.spectra(np.linspace(1e-5,1-1e-5, 1000))/integral)})
        
        #Checkbox if comparison to mass mechanism is desired
        show_mbb2 = st.checkbox("Compare to mass mechanism?", key="show_mbb2")
        
        #Generate Figure
        fig_spec = LEFT_model.plot_spec(show_mbb=show_mbb2, isotope = plot_isotope2)
        
        #Show Figure
        st.pyplot(fig_spec)
        
        #Half-Life Ratios
        st.subheader("Half-life ratios")
        
        #Define the reference Isotope for Ratio Calculation
        reference_isotope = st.selectbox("Choose the reference isotope:", options = LEFT_model.isotope_names, index = ge_idx)
        
        #Set Number of Option Cols
        ratio_option_cols = st.columns(2)
        
        #Set Plot Option Checkboxes
        
        #Mass Comparisson
        compare = ratio_option_cols[0].checkbox("Compare to mass mechanism?", help = "If you check this box we will normalize the ratios to the mass mechanisms ratio values")
        
        #LEC Variation
        vary_LECs = ratio_option_cols[1].checkbox("Vary unknown LECs?", help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from log_10(O) to log10(O+1)) . g_nuNN will be varied 50% around it's theoretical estimate.")
        
        #Number of LEC Variations
        if vary_LECs:
            n_points = st.number_input("How many variations do you want to run? Remember: The higher this number the longer the calculation takes..." , value=100)
        else:
            n_points = 1
            
        #Generate Figure
        fig = LEFT_model.plot_ratios(vary_LECs = vary_LECs, n_points = n_points, 
                                normalized = compare, reference_isotope = reference_isotope)
        
        #Show Figure
        st.pyplot(fig)
        
        #Variation of Wilson Coefficients
        st.subheader("Vary single Wilson coefficients")
        
        
        ##Define Plotting Functions so that a plotting Loop can be generated
        def plots(plotidx):
            #Plot type
            plotoptions = st.selectbox("Choose additional figures you want to see. These plots take a few seconds...", 
                                        options = ["-", "m_eff", "half_life", "1/half_life"], key = "chooseplottype"+str(plotidx))
            
            #
            if plotoptions in ["m_eff", "half_life", "1/half_life"]:
                
                #Index of 76Ge in NME list
                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                
                #Generate Option Columns
                plot_cols = st.columns(3)
                
                #Define Isotope of Interest
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, key = "isotope"+str(plotidx))
                
                #Choose between scatter or line plot
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], key = "plottype"+str(plotidx), help = "Scatter plots vary all the relevant parameters and generate a number of scenarios while line plots calculate the minimum and maximum by running an optimization algorithm. If you want to vary also the LECs you will need to choose scatter plots.")
                
                #X-axis WC
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = np.append(["m_min", "m_sum"], np.array(list(LEFT_model.WC.keys()))[np.array(list(LEFT_model.WC.values()))!=0]), key = "vary"+str(plotidx), help = "Choose the Wilson coefficient you want to vary on the x-axis")
                
                #Show Cosmo Limit?
                show_cosmo = False
                m_cosmo = 0.15
                
                #Generate Line Plot
                if scatter_or_line == "Line":
                    
                    #columns for xaxis input (min, max n_points)
                    xlim_cols = st.columns(3)
                    
                    #xaxis = minimal neutrino mass
                    if vary_WC == "m_min":
                        
                        #xmin Input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax Input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min 10^...[meV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #xaxis = effective neutrino mass
                    elif vary_WC == "m_bb":
                        
                        #xmin input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_bb 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_bb 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #xaxis == m_sum
                    elif vary_WC == "m_sum":
                        
                        #xmin input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_sum 10^...[eV]", value = -2., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_sum 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #dimension 6 operator input
                    elif vary_WC[-2] == "6":
                        
                        #xmin
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -11., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                        #xmax
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -5., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #dimension 7 and 9 operator input
                    else:
                        #xmin
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -7., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                        #xmax
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -2., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #yaxis limits checkbox
                    choose_ylim = xlim_cols[2].checkbox("Set y-axis limits", help = "You can either let the code choose the y-axis limits or choose them yourself by checking this box.", key = "ylim checkbox"+str(plotidx))
                    
                    #show yaxis fields if box checked
                    ylim_cols = st.columns(2)
                    
                    #ymin
                    y_min =  None
                    
                    #ymax
                    y_max = None
                    
                    #choose ymin and ymax if box checked
                    if choose_ylim:
                        #yaxis input columns
                        ylim_cols = st.columns(3)
                        
                        #ymin
                        y_min = 10**ylim_cols[0].number_input("Minimum y-axis limit exponent", value = -4., key = "ymin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #ymax
                        y_max = 10**ylim_cols[1].number_input("Maximum m_min exponent [meV]", value = 0., key = "ymax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #compare to mass mechanism - preset value
                    show_mbb = False
                    
                    #normalize to mass mechanism - preset value
                    normalize_to_mass = False
                    
                    #show cosmology limit on y axis - preset value
                    show_cosmo = False
                    
                    #allow for additional input if neutrino mass is on the x-axis
                    if vary_WC in ["m_min", "m_sum"]:
                        option_cols = st.columns(2)
                        
                        #show mass mechanism
                        show_mbb = option_cols[0].checkbox("Compare to mass mechanism?", key =plotoptions+"show_mbb"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        
                        #Normalize y-axis to mass mechanism
                        normalize_to_mass = option_cols[1].checkbox("Normalize to mass mechanism?", key =plotoptions+"normalize"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        
                        #Cosmology Limit
                        cosmo_options = st.columns(2)
                        
                        #Show Cosmo Limit?
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+"show_cosmo"+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        
                        #cosmo limit - preset value
                        m_cosmo = 0.15
                        
                        #allow for input of cosmo limit
                        if show_cosmo:
                            #Cosmo Limit on m_sum
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    
                    #Generate Figure
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                    normalize = normalize_to_mass, 
                                                    xaxis = vary_WC, n_points = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                    normalize = normalize_to_mass, 
                                                    xaxis = vary_WC, n_points = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv(cosmo=show_cosmo, isotope = plot_isotope, 
                                                         show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                         normalize = normalize_to_mass, 
                                                         xaxis = vary_WC, n_points = 200, 
                                                         x_min = x_min, x_max = x_max, 
                                                         y_min = y_min, y_max = y_max)
                else:
                    xlim_cols = st.columns(3)
                    if vary_WC == "m_min":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min exponent [eV]", value = -4., key = "xmin"+str(plotidx))
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min exponent [meV]", value = 0., key = "xmax"+str(plotidx))
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                        
                    #generate option cols for plot options
                    option_cols = st.columns(4)
                    
                    #allow for variation of unknown LECs
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"vary_LECs"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from 1/sqrt(10) to sqrt(10) times the estimate . g_nuNN will be varied 50% around it's theoretical estimate.")
                    
                    #allow for variation of relative complex phase
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"vary_phases"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                    
                    #number of scatterd points
                    n_points = xlim_cols[2].number_input("Number of points", value = 10000, step = 1, min_value = 0, key =plotoptions+"npoints"+str(plotidx))
                        
                    show_mbb = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC == "m_min":
                        show_mbb = option_cols[2].checkbox("Compare to mass mechanism?", key =plotoptions+"show_mbb"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        normalize_to_mass = option_cols[3].checkbox("Normalize to mass mechanism?", key =plotoptions+"normalize"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        cosmo_options = st.columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+"show_cosmo"+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                            show_mbb = show_mbb, n_points = n_points, 
                                                            normalize = normalize_to_mass,
                                                            cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        
                    if plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                             show_mbb = show_mbb, n_points = n_points, 
                                                             normalize = normalize_to_mass,
                                                             cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                             vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        
                    if plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                            show_mbb = show_mbb, n_points = n_points, 
                                                            normalize = normalize_to_mass,
                                                            cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        

                st.pyplot(fig)
            return(plotoptions)
        plotoptions = ""
        plotidx = 0
        while plotoptions != "-":
            plotoptions = plots(plotidx)
            plotidx +=1


    ################################################################################################
    #                                                                                              #
    #                                     Standard Model EFT                                       #
    #                                                                                              #
    ################################################################################################
    elif model_option == "SMEFT":
        #complex phases
        phases = st.sidebar.checkbox("Allow for complex phases?", help = "If you check this box you can set complex phases for each Wilson coefficient.")
        
        #multiple scales?
        scale_options = st.sidebar.selectbox("Does your model generate SMEFT operators at multiple scales? If 'Yes' you will need to define a scale for each operator.", options = ["No", "Yes"])
        
        #multiple scales?
        if scale_options == "Yes":
            multiscales = True
            
        #single scale
        else:
            multiscales = False
            #define scale
            scale = st.sidebar.number_input("Set the scale at which your SMEFT model is generated [TeV].", value=50)*1e+3
        
        #dimensionless WCs
        st.sidebar.write("Set the dimensionless Wilson coefficients:")
        
        #counter (dimensions)
        ctr = 0
        
        #iterate over all SMEFT operators
        for operator in SMEFT_WCs:
            #dimension 7 operators
            if operator[-2] == "7":
                if ctr == 0:
                    #generate sidebar text
                    st.sidebar.subheader("Dimension 7")
                    
                    #add to counter
                    ctr+=1
                
                #set dimension
                dimension = 7
                
            #dimension 9 operators
            if operator[-2] == "9":
                if ctr == 1:
                    #generate sidebar text
                    st.sidebar.subheader("Dimension 9")
                    
                    #add to counter
                    ctr+=1
                
                #set dimension
                dimension = 9
                
            #dimension 5 operators
            if operator == "LH(5)":
                #generate sidebar text
                st.sidebar.subheader("Dimension 5")
                
                #set dimension
                dimension = 5
                
            #if multiple scales are allowed add scale input
            if multiscales:
                scale = st.sidebar.number_input("Scale [TeV].", value=50, key=operator)*1e+3
            
            #need scaling for LH(5) because it is much less scale suppressed than higher dimensional operators
            if scale <=10*1e+4:
                LLHHfactor = 1e-12
            elif 10*1e+4<scale and scale <= 10*1e+7:
                LLHHfactor = 1e-9
            elif 10*1e+7<scale and scale <= 10000*1e+10:
                LLHHfactor = 1e-6
            elif 10*1e+10<scale and scale <= 10000*1e+13:
                LLHHfactor = 1e-3
            else:
                LLHHfactor = 1
                
            if operator == "LH(5)":
                #set WC
                SMEFT_WCs[operator] = st.sidebar.number_input(operator+" ["+str(LLHHfactor)+"]", value = 1.)*LLHHfactor
                
                #complex phase
                if phases:
                    SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                ##############
                
                #add dimensional suppression
                SMEFT_WCs[operator] /= scale**(dimension-4)
                st.sidebar.write("_______________________")
                    
            else:
                #set WC
                SMEFT_WCs[operator] = st.sidebar.number_input(operator)
                
                #complex phase
                if phases:
                    SMEFT_WCs[operator] *= np.exp(1j*st.sidebar.number_input(operator+" phase [pi]")*np.pi)
                    
                #dimensional suppression
                SMEFT_WCs[operator] /= scale**(dimension-4)
                st.sidebar.write("_______________________")
                
        #Generate SMEFT Model
        SMEFT_model = EFT.SMEFT(SMEFT_WCs, scale, method = method, name=name)
        
        #Generate LEFT Model from SMEFT
        LEFT_model = EFT.LEFT(SMEFT_model.LEFT_matching(), name=name, method=method)
        
        #Half-Lives
        st.subheader("Half-lives")
        
        #calculate half-lives
        hl = SMEFT_model.half_lives()
        
        #set index
        hl.rename(index = {0: "10^24 years"}, inplace = True)
        
        #round results
        if np.inf not in hl.values:
            hl = hl.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
            
        #download csv table
        def get_table_download_link_csv(df):
            csv = df.to_csv().encode()
            latex = df.to_latex().encode()
            b64 = base64.b64encode(csv).decode()
            href = f'Download half-lives as <a href="data:file/csv;base64,{b64}" download="SMEFT_model_half_lives.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="SMEFT_model_half_lives.tex" target="_blank">.tex</a> file.'
            return href
        
        #generate download links
        st.markdown(get_table_download_link_csv(hl.T), unsafe_allow_html=True)
        
        #show half-life table
        st.table(hl.T*1e-24)
        
        #angula correlation
        st.subheader("Angular correlation")
        st.latex(r"\frac{\mathrm{d}\Gamma}{\mathrm{d}\cos\theta\mathrm{d}\overline{\epsilon}_1} = a_0\left(1+\frac{a_1}{a_0}\cos\theta\right)")
        
        #Germanium index in list of isotopes
        ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
        
        #choose isotope to study from selectbox
        plot_isotope = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "angularcorrisotope")
        
        #decide if mass mechanism should be shown for comparison
        show_mbb1 = st.checkbox("Compare to mass mechanism?", key="show_mbb1")
        
        #generate figure
        fig_angular_corr = LEFT_model.plot_corr(show_mbb=show_mbb1, isotope = plot_isotope)
        
        #show figure
        st.pyplot(fig_angular_corr)
        
        #normalized electron spectra
        st.subheader("Normalized single electron spectrum")
        st.latex(r'''\frac{\mathrm{d}\Gamma}{\mathrm{d}\epsilon_1} 
                     \left(\left\{C_i\right\}, \overline{\epsilon}\right) \propto \sum_k g_{0k}
                     \left(\epsilon, \Delta M - \epsilon, R\right)
                     \left|A_{k}(\{C_i\})\right|^2p_1 p_2 \epsilon\left(\Delta M-\epsilon\right)''')
        
        #choose isotope to study from selectbox
        plot_isotope2 = st.selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, 
                                     key = "spectraisotope")
        
        #decide if mass mechanism should be shown for comparison
        show_mbb2 = st.checkbox("Compare to mass mechanism?", key="show_mbb2")
        
        #generate figure
        fig_spec = LEFT_model.plot_spec(show_mbb=show_mbb2, isotope = plot_isotope2)
        
        #show figure
        st.pyplot(fig_spec)
        
        #Half-Life Ratios
        st.subheader("Half-life ratios")
        
        #choose reference isotope from selectbox
        reference_isotope = st.selectbox("Choose the reference isotope:", options = LEFT_model.isotope_names, index = ge_idx)
        
        #generate option columns
        ratio_option_cols = st.columns(2)
        
        #show mass mechanism?
        compare = ratio_option_cols[0].checkbox("Compare to mass mechanism?")
        
        #vary unknown LECs?
        vary_LECs = ratio_option_cols[1].checkbox("Vary unknown LECs?")
        
        #number of LEC variations
        if vary_LECs:
            n_points = st.number_input("How many variations do you want to run? Remember: The higher this number the longer the calculation takes..." , value=100)
            
        else:
            n_points = 1
            
        #Generate Figure
        fig = LEFT_model.plot_ratios(vary_LECs = vary_LECs, n_points = n_points, 
                                     normalized = compare, reference_isotope = reference_isotope)
        
        #show figure
        st.pyplot(fig)
        
        #Half-Life Plots with Variation of WC on xaxis
        st.subheader("Vary single Wilson coefficients")
        def plots(plotidx):
            #Plot type
            plotoptions = st.selectbox("Choose additional figures you want to see. These plots take a few seconds...", 
                                        options = ["-", "m_eff", "half_life", "1/half_life"], key = "chooseplottype"+str(plotidx))
            
            #
            if plotoptions in ["m_eff", "half_life", "1/half_life"]:
                
                #Index of 76Ge in NME list
                ge_idx = int(np.where(LEFT_model.isotope_names=="76Ge")[0][0])
                
                #Generate Option Columns
                plot_cols = st.columns(3)
                
                #Define Isotope of Interest
                plot_isotope = plot_cols[0].selectbox("Choose an isotope:", options = LEFT_model.isotope_names, index = ge_idx, key = "isotope"+str(plotidx))
                
                #Choose between scatter or line plot
                scatter_or_line = plot_cols[1].selectbox("Choose the plot-type", options = ["Scatter", "Line"], key = "plottype"+str(plotidx), help = "Scatter plots vary all the relevant parameters and generate a number of scenarios while line plots calculate the minimum and maximum by running an optimization algorithm. If you want to vary also the LECs you will need to choose scatter plots.")
                
                #X-axis WC
                vary_WC = plot_cols[2].selectbox("X-axis WC", options = ["m_min", "m_sum", "m_bb"], key = "vary"+str(plotidx), help = "Choose the Wilson coefficient you want to vary on the x-axis")
                
                #Show Cosmo Limit?
                show_cosmo = False
                m_cosmo = 0.15
                
                #Generate Line Plot
                if scatter_or_line == "Line":
                    
                    #columns for xaxis input (min, max n_points)
                    xlim_cols = st.columns(3)
                    
                    #xaxis = minimal neutrino mass
                    if vary_WC == "m_min":
                        
                        #xmin Input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax Input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #xaxis = effective neutrino mass
                    elif vary_WC == "m_bb":
                        
                        #xmin input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_bb 10^...[eV]", value = -4., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_bb 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #xaxis == m_sum
                    elif vary_WC == "m_sum":
                        
                        #xmin input
                        x_min = 10**xlim_cols[0].number_input("Minimum m_sum 10^...[eV]", value = -2., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #xmax input
                        x_max = 10**xlim_cols[1].number_input("Maximum m_sum 10^...[eV]", value = 0., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #dimension 6 operator input
                    elif vary_WC[-2] == "6":
                        
                        #xmin
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -11., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                        #xmax
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -5., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #dimension 7 and 9 operator input
                    else:
                        #xmin
                        x_min = 10**xlim_cols[0].number_input("Minimum C_"+vary_WC+" 10^...", value = -7., key = "xmin"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                        #xmax
                        x_max = 10**xlim_cols[1].number_input("Maximum C_"+vary_WC+" 10^...", value = -2., key = "xmax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #yaxis limits checkbox
                    choose_ylim = xlim_cols[2].checkbox("Set y-axis limits", help = "You can either let the code choose the y-axis limits or choose them yourself by checking this box.", key = "ylim checkbox"+str(plotidx))
                    
                    #show yaxis fields if box checked
                    ylim_cols = st.columns(2)
                    
                    #ymin
                    y_min =  None
                    
                    #ymax
                    y_max = None
                    
                    #choose ymin and ymax if box checked
                    if choose_ylim:
                        #yaxis input columns
                        ylim_cols = st.columns(3)
                        
                        #ymin
                        y_min = 10**ylim_cols[0].number_input("Minimum y-axis limit exponent", value = -4., key = "ymin"+str(plotidx), help = "This sets the minimum limit on the x axis as 10^a. Preset: a=-4")
                        
                        #ymax
                        y_max = 10**ylim_cols[1].number_input("Maximum y-axis limit exponent", value = 0., key = "ymax"+str(plotidx), help = "This sets the maximum limit on the x axis as 10^a. Preset: a=0")
                        
                    #compare to mass mechanism - preset value
                    show_mbb = False
                    
                    #normalize to mass mechanism - preset value
                    normalize_to_mass = False
                    
                    #show cosmology limit on y axis - preset value
                    show_cosmo = False
                    
                    #allow for additional input if neutrino mass is on the x-axis
                    if vary_WC in ["m_min", "m_sum"]:
                        option_cols = st.columns(2)
                        
                        #show mass mechanism
                        show_mbb = option_cols[0].checkbox("Compare to mass mechanism?", key =plotoptions+"show_mbb"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        
                        #Normalize y-axis to mass mechanism
                        normalize_to_mass = option_cols[1].checkbox("Normalize to mass mechanism?", key =plotoptions+"normalize"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        
                        #Cosmology Limit
                        cosmo_options = st.columns(2)
                        
                        #Show Cosmo Limit?
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+"show_cosmo"+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        
                        #cosmo limit - preset value
                        m_cosmo = 0.15
                        
                        #allow for input of cosmo limit
                        if show_cosmo:
                            #Cosmo Limit on m_sum
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    
                    #Generate Figure
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                    normalize = normalize_to_mass, 
                                                    xaxis = vary_WC, n_points = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half(cosmo=show_cosmo, isotope = plot_isotope, 
                                                    show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                    normalize = normalize_to_mass, 
                                                    xaxis = vary_WC, n_points = 200, 
                                                    x_min = x_min, x_max = x_max, 
                                                    y_min = y_min, y_max = y_max)
                    elif plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv(cosmo=show_cosmo, isotope = plot_isotope, 
                                                         show_mbb = show_mbb, m_cosmo = m_cosmo,
                                                         normalize = normalize_to_mass, 
                                                         xaxis = vary_WC, n_points = 200, 
                                                         x_min = x_min, x_max = x_max, 
                                                         y_min = y_min, y_max = y_max)
                else:
                    xlim_cols = st.columns(3)
                    if vary_WC == "m_min":
                        x_min = 10**xlim_cols[0].number_input("Minimum m_min exponent [eV]", value = -4., key = "xmin"+str(plotidx))
                        x_max = 10**xlim_cols[1].number_input("Maximum m_min exponent [eV]", value = 0., key = "xmax"+str(plotidx))
                    elif vary_WC == "m_bb":
                        x_min = xlim_cols[0].number_input("Minimum m_bb [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_bb [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC == "m_sum":
                        x_min = xlim_cols[0].number_input("Minimum m_sum [meV]", value = 0.1, key = "xmin"+str(plotidx))*1e-3
                        x_max = xlim_cols[1].number_input("Maximum m_sum [meV]", value = 1000., key = "xmax"+str(plotidx))*1e-3
                    elif vary_WC[-2] == "6":
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-9]", value = 0.1, key = "xmin"+str(plotidx))*1e-9
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-9]", value = 1000., key = "xmax"+str(plotidx))*1e-9
                    else:
                        x_min = xlim_cols[0].number_input("Minimum C_"+vary_WC+" [1e-6]", value = 0.1, key = "xmin"+str(plotidx))*1e-6
                        x_max = xlim_cols[1].number_input("Maximum C_"+vary_WC+" [1e-6]", value = 1000., key = "xmax"+str(plotidx))*1e-6
                        
                    #generate option cols for plot options
                    option_cols = st.columns(4)
                    
                    #allow for variation of unknown LECs
                    vary_LECs = option_cols[0].checkbox("Vary unknown LECs?", key =plotoptions+"vary_LECs"+str(plotidx), help = "If you check this box we will vary all unknown LECs around their order of magnitude estimate O (i.e. from 1/sqrt(10) to sqrt(10) times the estimate . g_nuNN will be varied 50% around it's theoretical estimate.")
                    
                    #allow for variation of relative complex phase
                    vary_phases = option_cols[1].checkbox("Vary phase?", key =plotoptions+"vary_phases"+str(plotidx), value=True, help = "If you check this box we will vary the complex phase of the operator chosen for the x-axis.")
                    
                    #number of scatterd points
                    n_points = xlim_cols[2].number_input("Number of points", value = 10000, step = 1, min_value = 0, key =plotoptions+"npoints"+str(plotidx))
                        
                    show_mbb = False
                    normalize_to_mass = False
                    show_cosmo = False
                    if vary_WC in ["m_min", "m_sum"]:
                        show_mbb = option_cols[2].checkbox("Compare to mass mechanism?", key =plotoptions+"show_mbb"+str(plotidx), value=False, help = "If you check this box we will plot the contribution of the standard mass mechanism for comparison.")
                        normalize_to_mass = option_cols[3].checkbox("Normalize to mass mechanism?", key =plotoptions+"normalize"+str(plotidx), value=False, help = "If you check this box we will normalize the y-axis with respect to the contributions of the standard mass mechanism.")
                        cosmo_options = st.columns(2)
                        show_cosmo = cosmo_options[0].checkbox("Show cosmology limit?", key =plotoptions+"show_cosmo"+str(plotidx), help = "This plots a grey area excluded from cosmology limits on the sum of neutrino masses translated to the corresponding minimal neutrino mass in normal ordering.")
                        if show_cosmo:
                            m_cosmo = cosmo_options[1].number_input("Limit on the sum of neutrino masses [meV]", help="Preset limit: S.R. Choudhury and S. Hannestad, 2019, arxiv:1907.12598", value = 150, key = "m_cosmo"+str(plotidx))*1e-3
                    if plotoptions == "m_eff":
                        fig = LEFT_model.plot_m_eff_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                            show_mbb = show_mbb, n_points = n_points, 
                                                            normalize = normalize_to_mass,
                                                            cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        
                    if plotoptions == "half_life":
                        fig = LEFT_model.plot_t_half_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                             show_mbb = show_mbb, n_points = n_points, 
                                                             normalize = normalize_to_mass,
                                                             cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                             vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        
                    if plotoptions == "1/half_life":
                        fig = LEFT_model.plot_t_half_inv_scatter(xaxis = vary_WC, vary_phases = vary_phases, 
                                                            show_mbb = show_mbb, n_points = n_points, 
                                                            normalize = normalize_to_mass,
                                                            cosmo = show_cosmo, m_cosmo = m_cosmo, isotope = plot_isotope, 
                                                            vary_LECs = vary_LECs, x_min = x_min, x_max = x_max)
                        

                st.pyplot(fig)
            return(plotoptions)
        plotoptions = ""
        plotidx = 0
        while plotoptions != "-":
            plotoptions = plots(plotidx)
            plotidx +=1



####################################################################################################
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
#                                    STUDY LIMITS ON OPERATORS                                     #
#                                                                                                  #
#                                                                                                  #
#                                                                                                  #
####################################################################################################

elif path_option == "Study operator limits":
    #experimental hl limits for each isotope in 10^24y
    isotope_limits = {"238U"  : 0., 
                      "232Th" : 0., 
                      "198Pt" : 0., 
                      "160Gd" : 0., 
                      "154Sm" : 0., 
                      "150Nd" : 0.18,  #arXiv:0810.0248
                      "148Nd" : 0., 
                      "136Xe" : 230.,   #arXiv:2203.02139
                      "134Xe" : 0.019,  #arXiv:1704.05042
                      "130Te" : 32.,    #arXiv:1912.10966
                      "128Te" : 0.11,  #arXiv:hep-ex/0211071
                      "124Sn" : 0., 
                      "116Cd" : 0.19,  #arXiv:1601.05578 
                      "110Pd" : 0., 
                      "100Mo" : 1.5,   #arXiv:2011.13243
                      "96Zr"  : 9.2e-3, #arXiv:0906.2694
                      #"82Se"  : 0.023        #arXiv:1806.05553
                      "82Se"  : 2.4,   #arXiv:1802.07791
                      "76Ge"  : 180,     #arXiv:2009.06079
                      "48Ca"  : 5.3e-2  #arXiv:0810.4746
                      }
    
    reference_limits = {"238U"  : None, 
                       "232Th" : None, 
                       "198Pt" : None, 
                       "160Gd" : None, 
                       "154Sm" : None, 
                       "150Nd" : "NEMO collaboration, 2008, arXiv:0810.0248", 
                       "148Nd" : None, 
                       "136Xe" : "KamLAND-Zen Collaboration, 2016, arXiv:2203.02139", 
                       "134Xe" : "EXO-200 Collaboration, 2017, arXiv:1704.05042", 
                       "130Te" : "CUORE Collaboration, 2019, arXiv:1912.10966", 
                       "128Te" : "C. Arnaboldi et al., 2002, arXiv:hep-ex/0211071", 
                       "124Sn" : None, 
                       "116Cd" : "Aurora experiment, F.A. Danevich et al., 2016, arXiv:1601.05578", 
                       "110Pd" : None, 
                       "100Mo" : "CUPID-Mo Experiment, E. Armengaud et al., 2020, arXiv:2011.13243", 
                       "96Zr"  : "NEMO-3, J.Argyriades et al., 2009, arXiv:0906.2694",
                       #"82Se"  : 0.023        #arXiv:1806.05553
                       "82Se"  : "CUPID-0 Collaboration, 2018, arXiv:1802.07791",
                       "76Ge"  : "GERDA Collaboration, 2020, arXiv:2009.06079",
                       "48Ca"  : "S.Umehara et al., 2008, arXiv:0810.4746"
                       }
    
    #Single WC Limits
    st.subheader("Limits on single Wilson coefficients:")
    
    #Explanatory text
    st.write("The below table shows the limits assuming only one Wilson coefficient at a time to be present. The results are rounded to 3 significant digits.")
    
    #Select NME method
    method = st.sidebar.selectbox("Which NME approximation do you want to use?", options = ["IBM2", "QRPA", "SM"], help = "Currently we allow for 3 different sets of nuclear matrix elements (NMEs): IBM2: F. Deppisch et al., 2020, arxiv:2009.10119 | QRPA: J. Hyvärinen and J. Suhonen, 2015, Phys. Rev. C 91, 024613 | Shell Model (SM): J. Menéndez, 2018, arXiv:1804.02105")
    
    #Select EFT
    model_option = st.sidebar.selectbox("Do you want study limits on LEFT or SMEFT operators?", options = ["LEFT", "SMEFT"])
    
    
    #if model_option == "LEFT":
    #Generate LEFT model
    LEFT_model = EFT.LEFT({}, method = method)

    #list of isotopes available
    isotopes = LEFT_model.isotope_names

    #Experimental Limit in LEFT
    st.sidebar.subheader("Experimental Limits")

    #Explanatory text
    st.sidebar.write("Please enter the experimental limits for each isotope. The initial values represent the current experimental limits that we could find. We try to keep these limits as recent as possible. If we missed some limit please contact us. [10^24 years]")

    #Enter Half-Life Limits from Experiments
    experiments = {} #{isotope : half-life}
    for isotope in isotopes:
        experiments[isotope] = st.sidebar.number_input(isotope, 
                                                       key=isotope, 
                                                       value = isotope_limits[isotope], 
                                                       step=None, help=reference_limits[isotope])*1e+24

    #Generate Progress Bar
    my_bar = st.progress(0)
    percent_complete = 0

    #DataFrame that includes WC Limits
    limits = pd.DataFrame()

    #DataFrame that includes scale limits
    scales = pd.DataFrame()

    #Show only operator groups with the same limits
    groups = st.checkbox("Show only groups?", help = "Instead of showing the limits for all single Wilson coefficients you can choose to summarize those that give the same contributions. This is only relevant for LEFT operators.", 
                         value = True)

    #Iterate over experiments
    for isotope in experiments:
        if experiments[isotope]>0:
            #calculate limits and scales
            if model_option == "LEFT":
                limitdf = f.get_limits_LEFT(experiments[isotope], isotope=isotope, groups = groups, method = method)
            else:
                limitdf = f.get_limits_SMEFT(experiments[isotope], isotope=isotope, groups = groups, method = method)

            #extract limits
            limits[isotope] = limitdf[limitdf.keys()[0]].values

            #extract scales
            scales[isotope] = limitdf[limitdf.keys()[1]].values

        #update progress percentage
        percent_complete += 1/(len(experiments))

        #round progress percentage
        progress = np.round(percent_complete,2)

        #generate/update progress bar
        my_bar.progress(progress)

    #round limits
    limits = limits.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))

    #round scales
    scales = limits.applymap(lambda x: round(x, 2 - int(floor(log10(abs(x))))))
    
    #add operator names to limits DataFrame
    limits["Operators"] = list(limitdf.index)
        
    #adjust scaling of operator limits
    
    #for LEFT
    if model_option == "LEFT":
  
        #add meV statement to m_bb WC
        limits["Operators"][0] = "m_bb [meV]"
        
        #add scaling factor to the remaining WCs
        for idx in range(len(limits["Operators"][1:])):
            limits["Operators"][idx+1] += " [10^-9]"
        
        #multiplication factors
        multi = 1e+9*np.ones(len(limits["Operators"]))
        multi[0] *= 1e+3
            
    #for SMEFT
    else:
        multi = np.ones(len(limits["Operators"]))
        for idx in range(len(limits["Operators"])):
            operator = limits["Operators"][idx]
            if operator == "LH(5)":
                dimension = 5
                factor = "1e-15 GeV^{-1}"
                multi[idx] *= 1e+15
            else:
                dimension = int(operator[-2])
                if dimension == 7:
                    factor = "1e-15 GeV^{-3}"
                    multi[idx] *= 1e+15
                else:
                    factor = "1e-18 GeV^{-5}"
                    multi[idx] *= 1e+18
            #add factor to description
            limits["Operators"][idx] += " ["+factor+"]"
            
    #set operator names as index
    limits.set_index("Operators", inplace = True)
    
    #multiply prefactors
    limits = limits.multiply(multi, axis = 0)
    
    #generate download links as csv or tex
    def get_table_download_link_csv(df):
        csv = df.to_csv().encode()
        latex = df.to_latex().encode()
        b64 = base64.b64encode(csv).decode()
        if model_option == "LEFT":
            href = f'Download limits as <a href="data:file/csv;base64,{b64}" download="LEFT_operator_limits.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="LEFT_operator_limits.tex" target="_blank">.tex</a> file.'
        else:
            href = f'Download limits as <a href="data:file/csv;base64,{b64}" download="SMEFT_operator_limits.csv" target="_blank">.csv</a> or as <a href="data:file/latex;base64,{b64}" download="LEFT_operator_limits.tex" target="_blank">.tex</a> file.'
        return href
    
    #create download links
    st.markdown(get_table_download_link_csv(limits), unsafe_allow_html=True)
    
    #show table
    st.table(limits)
    
    #make limit plot
    st.subheader("Plot limits")
    
    #generate plots of scales corresponding to the limits or the limits directly
    plottype = st.selectbox("You can either plot the limits directly or the corresponding high energy scale assuming naturalness.", options = ["scales", "limits"])
    
    #generate dict for experimentallimits
    plotexps = {}
    
    #need a counter to count experiments
    counter = 0
    
    #only use non-zero limits for plots. These show in the checkboxes
    checkbox_experiments = {}
    for experiment in experiments:
        if experiments[experiment] > 0:
            checkbox_experiments[experiment] = {"half-life" : experiments[experiment], 
                                                "isotope"   : experiment , 
                                                "label"     : experiment}
            counter += 1
    
    #isotope checkbox columns
    cols1 = st.columns(8)
    if counter > 8:
        cols2 = st.columns(8)

    if counter > 16:
        cols3 = st.columns(8)
    
    #Filter Experiments that have a positive checkbox
    idx = 0
    for experiment in checkbox_experiments:
        if experiment in ["76Ge", "130Te", "136Xe"]:
            preset = True
        else:
            preset = False
        if idx<8:
            col = cols1[idx]
        elif idx >=8 and idx < 16:
            col = cols2[idx-8]
        else:
            col=cols3[idx-16]
        plotexp = col.checkbox(experiment, value=preset)
        idx += 1
        if plotexp:
            plotexps[experiment] = checkbox_experiments[experiment]
            
    #Generate Figure
    if model_option == "LEFT":
        limit_fig = plots.limits_LEFT(plotexps, plottype=plottype, method = method, groups = groups, savefig = False)
    else:
        limit_fig = plots.limits_SMEFT(plotexps, plottype=plottype, method = method, groups = groups, savefig = False)
    
    #Show Figure
    st.pyplot(limit_fig)
    
    #2 Operator Contour Limits
    st.subheader("Limits with 2 active operators")
    
    #Explanatory text
    st.markdown("Below you can generate limit plots assuming 2 different operators at a time to be present.")
    
    #Function to generate contour plot
    def plot_contours(plotidx):
        counter = 0
        #only use non-zero limits for plots. These show in the checkboxes
        checkbox_experiments = {}
        for experiment in experiments:
            if experiments[experiment] > 0:
                checkbox_experiments[experiment] = {"half-life" : experiments[experiment], 
                                                    "isotope"   : experiment , 
                                                    "label"     : experiment}
                counter += 1
        
        #generate checkbox columns for isotope selection
        cols1 = st.columns(8)
        if counter > 8:
            cols2 = st.columns(8)

        if counter > 16:
            cols3 = st.columns(8)
            
        #fill dict with isotopes to be plotted
        idx = 0
        plotexp = {}
        plotexps_contour = {}
        for experiment in checkbox_experiments:
            if experiment in []:#, "130Te", "136Xe"]:
                preset = True
            else:
                preset = False
            if idx<8:
                col = cols1[idx]
            elif idx >=8 and idx < 16:
                col = cols2[idx-8]
            else:
                col=cols3[idx-16]
            plotexp = col.checkbox(experiment, value=preset, key = "contour_isotope"+str(idx)+str(plotidx))
            idx += 1
            if plotexp:
                plotexps_contour[experiment] = checkbox_experiments[experiment]
        
        #columns with WC Choice
        contour_cols = st.columns(2)
        
        #x-axis WC input
        if model_option == "LEFT":
            WCx = contour_cols[0].selectbox("X-axis WC", options = np.array(list(LEFT_WCs.keys())), key = "WCx"+str(plotidx))
        else:
            WCx = contour_cols[0].selectbox("X-axis WC", options = np.array(list(SMEFT_WCs.keys())), key = "WCx"+str(plotidx))
        
        #y-axis WC inpit
        if model_option == "LEFT":
            WCy = contour_cols[1].selectbox("Y-axis WC", 
                                            options = np.array(list(LEFT_WCs.keys())) [np.array(list(LEFT_WCs.keys()))!=WCx], 
                                            key = "WCy"+str(plotidx))
        else:
            WCy = contour_cols[1].selectbox("Y-axis WC", 
                                            options = np.array(list(SMEFT_WCs.keys())) [np.array(list(SMEFT_WCs.keys()))!=WCx], 
                                            key = "WCy"+str(plotidx))
            
        #Columns with plot options
        options_cols = st.columns(2)
        
        #vary relative complex phase
        vary_phase = options_cols[0].checkbox("Vary phase", help = "If you check this box we will vary the relative phase between the two Wilson coefficients", key = "varyphases"+str(plotidx))
        
        #number of phase variations
        n_vary = 1
        
        if vary_phase:
            phase = 0
            show_variation = options_cols[1].checkbox("Show detailled variation", help = "If you check this box we will display how the variation of the relative phase deforms the contour limit.", key = "showvary"+str(plotidx))
            if show_variation:
                n_vary = 5
            else:
                n_vary = 2
        else:
            phase = options_cols[1].number_input("Phase [Pi]", value = 1/4, key = "contour_phase"+str(plotidx))*np.pi
        plotted = False
        #st.write(plotexps_contour)
        if plotexps_contour != {}:
            #generate figure
            fig = plots.contours(WCx, WCy, limits = plotexps_contour, method = method, 
                                 n_vary=n_vary, varyphases = vary_phase, n_points = 400, phase=phase)
            
            #show figure
            st.pyplot(fig)
            plotted = True
        return(plotted)
    plotted = True
    plotidx = 0
    while plotted:
        plotted = plot_contours(plotidx)
        plotidx +=1
        
