#import necessary stuff
import matplotlib as mplt
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as pltcolors

#Standard Matplotlib Colormap. Use this if no colors are given by the user
colormap = list(pltcolors.TABLEAU_COLORS.values())

import numpy as np

import pandas as pd

from constants import *

import warnings

import functions as f

#to find m_min from m_sum
from scipy import optimize

#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                           Generate lobster plot for m_eff, t, t_inv                                           #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################

################################
#          Line Plot           #
################################
def lobster(xNO,                               #Datapoints x-axis Normal Ordering
            NO_max,                            #Datapoints Normal Ordering max
            NO_min,                            #Datapoints Normal Ordering min
            vary_WC           = "m_min",       #xaxis WC
            NO_min_mbb        = None,          #Datapoints for mass comparison NO min
            NO_max_mbb        = None,          #Datapoints for mass comparison NO max
            xIO               = None,          #Datapoints IO
            IO_max            = None,          #Datapoints IO
            IO_min            = None,          #Datapoints IO
            IO_min_mbb        = None,          #Datapoints for mass comparison IO min
            IO_max_mbb        = None,          #Datapoints for mass comparison IO max
            x_min             = None,          #min x-axis range
            x_max             = None,          #max x-axis range
            y_min             = None,          #min y-axis range
            y_max             = None,          #max y-axis range
            ylabel            = None,          #label on y-axis
            xlabel            = None,          #label on x-axis
            xscale            = "log",         #x-axis scaling (lin, log)
            yscale            = "log",         #x-axis scaling (lin, log)
            legend            = True,          #If True show legend
            NO_label          = None,          #Legend Label for NO points
            IO_label          = None,          #Legend Label for IO points
            autolabel         = True,          #generate legend labels automatically
            ordering          = "both",        #show one or 2 orderings (NO, IO, both)
            limits            = None,          #experimental limits {label : {limit, color, linestyle, linewidth, alpha, fill, label}}
            cosmo             = False,         #show x-axis limit on m_sum from cosmology
            m_cosmo           = 0,             #x-axis limit on m_sum
            alpha_cosmo       = None,          #alpha value for cosmo limit
            colors            = None,          #plot color list [NOcolor, IOcolor]
            alpha_plot        = None,          #alpha for main plot
            show_mbb          = False,         #show mass mechanism for comparison
            alpha_mass        = None,          #alpha for mass plot
            normalize         = False,         #normalized y-axis?
            savefig           = False,         #save figure to file
            file              = "lobster.png", #filename
            dpi               = 300            #resolution in dots per inch
           ):

    #generate figure
    fig = plt.figure(figsize=(9,8))
    
    #define tick fontsize
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
                
    #set legend labels
    if autolabel:
        if vary_WC  in ["m_min", "m_sum"]:
            NO_label = "NO"
            IO_label =  "IO"
        else:
            NO_label = None
            IO_label =  None

    #Plot only normal mass ordering
    #This is also used for dimensionless WCs
    if ordering == "NO":
            
        #plot lines
        plt.plot(xNO,NO_min, colors[0], linewidth = 1)
        plt.plot(xNO,NO_max, colors[0], linewidth = 1)
            
        #plot filled areas
        plt.fill_between(xNO, NO_max, NO_min, linewidth = 1, 
                         facecolor = list(mplt.colors.to_rgb(colors[0]))+[alpha_plot],
                         edgecolor = list(mplt.colors.to_rgb(colors[0]))+[1],
                         label = NO_label)

    #Plot only inverted mass ordering (Only for dimensionfull WC)
    elif ordering == "IO" and (vary_WC == "m_min" or vary_WC == "m_sum"):
            
        #plot lines
        plt.plot(xIO,IO_min, colors[1], linewidth = 1)
        plt.plot(xIO,IO_max, colors[1], linewidth = 1)
            
        #plot filled areas
        plt.fill_between(xIO, IO_max, IO_min, linewidth = 1, 
                         facecolor = list(mplt.colors.to_rgb(colors[1]))+[alpha_plot],
                         edgecolor = list(mplt.colors.to_rgb(colors[1]))+[1],
                         label = IO_label)
    
    #Plot both mass orderings (Only for dimensionfull WC)
    else:
        #Normal Ordering
            
        #plot lines
        plt.plot(xNO,NO_min, colors[0], linewidth = 1)
        plt.plot(xNO,NO_max, colors[0], linewidth = 1)
            
        #plot filled areas
        plt.fill_between(xNO, NO_max, NO_min, linewidth = 1, 
                         facecolor = list(mplt.colors.to_rgb(colors[0]))+[alpha_plot],
                         edgecolor = list(mplt.colors.to_rgb(colors[0]))+[1],
                         label = NO_label)
        
        #Inverted Ordering (if neutrino mass is on the x-axis)
        if vary_WC in ["m_min", "m_sum"]:
            
            #plot lines
            plt.plot(xIO,IO_min, colors[1], linewidth = 1)
            plt.plot(xIO,IO_max, colors[1], linewidth = 1)
            
            #plot filled areas
            plt.fill_between(xIO, IO_max, IO_min, linewidth = 1, 
                             facecolor = list(mplt.colors.to_rgb(colors[1]))+[alpha_plot],
                             edgecolor = list(mplt.colors.to_rgb(colors[1]))+[1],
                             label = IO_label)

    #Plot mass mechanism for comparison
    if show_mbb:
        #show only normal ordering
        if ordering == "NO":
            
            #plot lines
            plt.plot(xNO,NO_min_mbb, "k", linewidth = 1)
            plt.plot(xNO,NO_max_mbb, "k", linewidth = 1)
            
            #plot filled areas
            plt.fill_between(xNO, NO_max_mbb, NO_min_mbb, 
                             linewidth = 1, 
                             facecolor = [0,0,0, alpha_mass],
                             edgecolor = [0,0,0,1],
                             label = r"$m_{\beta\beta}$")
        #inverted ordering
        elif ordering == "IO" and (vary_WC == "m_min" or vary_WC == "m_sum"):
            
            #plot lines
            plt.plot(xIO, IO_min_mbb, "k", linewidth = 1)
            plt.plot(xIO, IO_max_mbb, "k", linewidth = 1)
            
            #plot filled areas
            plt.fill_between(xIO, IO_max_mbb, IO_min_mbb, 
                             linewidth = 1, 
                             facecolor = [0,0,0, alpha_mass],
                             edgecolor = [0,0,0,1],
                             label=r"$m_{\beta\beta}$")
            
        #both orderings
        else:
            
            #plot lines
            plt.plot(xNO,NO_min_mbb, "k", linewidth = 1)
            plt.plot(xNO,NO_max_mbb, "k", linewidth = 1)
            plt.plot(xIO,IO_min_mbb, "k", linewidth = 1)
            plt.plot(xIO,IO_max_mbb, "k", linewidth = 1)
            
            #plot filled areas
            plt.fill_between(xIO, IO_max_mbb, IO_min_mbb, 
                             linewidth = 1, 
                             facecolor = [0,0,0, alpha_mass],
                             edgecolor = [0,0,0,1],
                             label=r"$m_{\beta\beta}$")
            plt.fill_between(xNO, NO_max_mbb, NO_min_mbb, 
                             linewidth = 1, 
                             facecolor = [0,0,0, alpha_mass],
                             edgecolor = [0,0,0,1])

    #set axis scaling (lin, log)
    plt.yscale(yscale)
    plt.xscale(xscale)
    
    #set axis range (min, max)
    plt.ylim(y_min,y_max)
    plt.xlim(x_min,x_max)
    
    #set axis labels
    plt.ylabel(ylabel, fontsize = 20)
    plt.xlabel(xlabel, fontsize = 20)
    
    plt.tight_layout()
    
    #plot cosmo limit
    if cosmo:
        if vary_WC != "m_sum":
            def m_sum(m_min):
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = optimize.root(m_sum, x0 = 0.05).x[0]
        else:
            cosmo_limit = m_cosmo
            
        #fill from cosmology ecluded area
        plt.fill_betweenx([y_min,y_max], [x_max], [cosmo_limit], alpha=alpha_cosmo, color="k")
        
    #plot experimental limits
    limitkeys = ["limit", "label", "color", "linestyle", "linewidth", "alpha", "fill", "addtext"]
    if limits != None:
        for limit in limits:
            #check if y-axis limit is defined
            if "limit" not in limits[limit]:
                warnings.warn("'limit' for "+limit+" must be set. Cannot plot limits.")  
            else:
                y_data = limits[limit]["limit"]
                
            try:
                #try to get color
                color = limits[limit]["color"]
            except:
                #set color to grey if none given
                color = "grey"
                
            try:
                #try to get a linestyle
                linestyle = limits[limit]["linestyle"]
            except:
                #set to normal line if none given
                linestyle = "-"
                
            try:
                #try to get linewidth
                linewidth = limits[limit]["linewidth"]
            except:
                #set linewidth = 1 if none given
                linewidth = 1
                
            try:
                #check if excluded area should be filled
                fill = limits[limit]["fill"]
            except:
                #if not given don't fill excluded area
                fill = False
                
            try:
                #add text to plot?
                addtext = limits[limit]["addtext"]
            except:
                #don't add text if not given
                addtext = False
                
            try:
                #try to get alpha value for filled area and line
                alpha = limits[limit]["alpha"]
            except:
                #if no alpha given use preset values
                if fill == True:
                    alpha = 0.25
                else:
                    alpha = 1
            try:
                #try to get a label for the legend
                label = limits[limit]["label"]
            except:
                #if no legend label is given don't show one
                label = None
                
            if fill:
                if ylabel == r"$t_{1/2}$ [yr]":
                    #fill from exp. limit excluded area (min y-axis value to limit)
                    plt.fill_between(xNO, y_min, y_data, color = color, alpha = alpha, 
                                     linestyle = linestyle, linewidth = linewidth, label = label)
                else:
                    #fill from exp. limit excluded area (max y-axis value to limit)
                    plt.fill_between(xNO, y_max, y_data, color = color, alpha = alpha, 
                                     linestyle = linestyle, linewidth = linewidth, label = label)
            else:
                #show limit as line
                plt.axhline(y_data, color = color, alpha = alpha, 
                            linestyle = linestyle, linewidth = linewidth, label = label)
            if addtext:
                plt.text(x = x_min, y = y_data, s = limit, fontsize = 15)
                
    #set ticksize on axes
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
    
    #generate legend
    if vary_WC in ["m_min", "m_sum"] and legend:
        plt.legend(fontsize=20)
        
    #save figure if desired
    if savefig:
        plt.savefig(file, dpi=dpi)
        
    #return figure
    return(fig)


################################
#         Scatter Plot         #
################################
def lobster_scatter(points,                                    #main datapoints
                    pointsIO        = None,                    #datapoints for inverted mass ordering
                    vary_WC         = "m_min",                 #WC to vary on the x-axis
                    x_min           = None,                    #minimum x-axis range
                    x_max           = None,                    #maxiumum x-axis range
                    y_min           = None,                    #minimum y-axis range
                    y_max           = None,                    #maximum y-axis range
                    xlabel          = None,                    #x-axis label
                    ylabel          = None,                    #y-axis label
                    xscale          = "log",                   #x-axis scaling (lin, log)
                    yscale          = "log",                   #y-axis scaling (lin, log)
                    markersize      = 0.15,                    #markersize for scatter points
                    ordering        = "both",                  #neutrino mass ordering from ["both", "NO", "IO"]
                    colorNO         = "b",                     #color of normal ordering
                    colorIO         = "r",                     #color of inverted ordering
                    legend          = True,                    #If True show legend
                    NO_label        = None,                    #Legend Label for NO points
                    IO_label        = None,                    #Legend Label for IO points
                    autolabel       = True,                    #generate legend labels automatically
                    alpha_plot      = 1,                       #alpha value for main plot
                    alpha_mass      = 0.15,                    #alpha value for mass mechanism
                    alpha_cosmo     = 0.1,                     #alpha value for cosmo limit
                    show_mbb        = False,                   #if True plot mass mechanism for comparison
                    cosmo           = False,                   #if True limit from cosmology is plotted
                    m_cosmo         = 0.15,                    #cosmo limit
                    xNO_mbb         = None,
                    NO_min_mbb      = None,                    #min mass mechanism values for normal ordering
                    NO_max_mbb      = None,                    #max mass mechanism values for normal ordering
                    xIO_mbb         = None,
                    IO_min_mbb      = None,                    #min mass mechanism values for inverted ordering
                    IO_max_mbb      = None,                    #max mass mechanism values for inverted ordering
                    limits          = None,                    #experimental limits to plot
                    savefig         = False,                   #if True, save figure under file
                    file            = "lobster_scattered.png", #filename to save figure to
                    dpi             = 300                      #figure resolution in dots per inch
                   ):
    #generate figure
    fig = plt.figure(figsize=(9,8))
    
    #define tick fontsize
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
    
            
    #set legend labels
    if autolabel:
        if vary_WC  in ["m_min", "m_sum"]:
            NO_label = "NO"
            IO_label =  "IO"
        else:
            NO_label = None
            IO_label =  None
    
    #plot scatter points
    if ordering in ["both", "NO"]:
        plt.plot(points[:,0],points[:,1], ".", color = colorNO, alpha = alpha_plot, markersize = markersize)
    if vary_WC in ["m_min", "m_sum"] and ordering in ["both", "IO"]:
        plt.plot(pointsIO[:,0],pointsIO[:,1], ".", color=colorIO, alpha = alpha_plot, markersize = markersize)
        
    #plot mass mechanism for comparison if desired
    if show_mbb:
        if ordering == "NO":
            plt.plot(xNO_mbb,NO_min_mbb, "k", linewidth = 1)
            plt.plot(xNO_mbb,NO_max_mbb, "k", linewidth = 1)
            plt.fill_between(xNO_mbb, NO_max_mbb, NO_min_mbb, color="k", alpha=alpha_mass, label = r"$m_{\beta\beta}$")

        elif ordering == "IO":
            plt.plot(xIO_mbb,IO_min_mbb, "k", linewidth = 1)
            plt.plot(xIO_mbb,IO_max_mbb, "k", linewidth = 1)
            plt.fill_between(xIO_mbb, IO_max_mbb, IO_min_mbb, color="k", alpha=alpha_mass, label=r"$m_{\beta\beta}$")
        else:
            plt.plot(xNO_mbb,NO_min_mbb, "k", linewidth = 1)
            plt.plot(xNO_mbb,NO_max_mbb, "k", linewidth = 1)
            plt.plot(xIO_mbb,IO_min_mbb, "k", linewidth = 1)
            plt.plot(xIO_mbb,IO_max_mbb, "k", linewidth = 1)

            plt.fill_between(xIO_mbb, IO_max_mbb, IO_min_mbb,
                             facecolor=(0,0,0,alpha_mass), 
                             linewidth = 1, 
                             edgecolor = (0,0,0,1),
                             label=r"$m_{\beta\beta}$"
                            )
            plt.fill_between(xNO_mbb, NO_max_mbb, NO_min_mbb,
                             facecolor=(0,0,0,alpha_mass), 
                             linewidth = 1, 
                             edgecolor = (0,0,0,1)
                            )
        
    #set axis scales
    plt.yscale(yscale)
    plt.xscale(xscale)
    
    #set axis label
    plt.xlabel(xlabel, fontsize = 20)
    plt.ylabel(ylabel, fontsize = 20)
    
    #set axis range
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.tight_layout()
    
    #plot experimental limits
    limitkeys = ["limit", "label", "color", "linestyle", "linewidth", "alpha", "fill", "addtext"]
    if limits != None:
        for limit in limits:
            if "limit" not in limits[limit]:
                warnings.warn("'limit' for "+limit+" must be set. Cannot plot limits.")  
            else:
                y_data = limits[limit]["limit"]
            try:
                color = limits[limit]["color"]
            except:
                color = "grey"
            try:
                linestyle = limits[limit]["linestyle"]
            except:
                linestyle = "-"
            try:
                linewidth = limits[limit]["linewidth"]
            except:
                linewidth = 1
            try:
                fill = limits[limit]["fill"]
            except:
                fill = False
            try:
                addtext = limits[limit]["addtext"]
            except:
                addtext = False
            try:
                alpha = limits[limit]["alpha"]
            except:
                if fill == True:
                    alpha = 0.25
                else:
                    alpha = 1
            try:
                label = limits[limit]["label"]
            except:
                label = None
            if fill:
                if ylabel == r"$t_{1/2}$ [yr]":
                    plt.fill_between(np.logspace(np.log10(x_min), np.log10(x_max), 100), 
                                     y_min, y_data, color = color, alpha = alpha, 
                                     label = label, linewidth = linewidth, linestyle = linestyle)
                else:
                    plt.fill_between(np.logspace(np.log10(x_min), np.log10(x_max), 100), 
                                     y_max, y_data, color = color, alpha = alpha, 
                                     label = label, linewidth = linewidth, linestyle = linestyle)
            else:
                plt.axhline(y_data, linewidth = linewidth, color = color, alpha = alpha, 
                            label = label, linestyle = linestyle)
                
            if addtext:
                plt.text(x = x_min, y = y_data, s = experiment, fontsize = 15)
              
    #plot cosmo limit
    if cosmo:
        if vary_WC != "m_sum":
            def m_sum(m_min):
                msum = m_min + np.sqrt(m_min**2+m21) + np.sqrt(m_min**2 + m21 + m32)
                return(msum-m_cosmo)
            cosmo_limit = optimize.root(m_sum, x0 = 0.05).x[0]
        else:
            cosmo_limit = m_cosmo
        m_cosmo
        plt.fill_betweenx([y_min,y_max], [x_max], [cosmo_limit], alpha = alpha_cosmo, color="k")
        
    
    #generate legend
    
    if vary_WC in ["m_min", "m_sum"] and legend:
        labels = fig.get_axes()[0].get_legend_handles_labels()[0]
        if ordering == "NO":
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=NO_label,
                                  markerfacecolor=colorNO, markersize=10)]
        elif ordering == "IO":
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=IO_label,
                                  markerfacecolor=colorIO, markersize=10)]
        else:
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=NO_label,
                                  markerfacecolor=colorNO, markersize=10),
                           Line2D([0], [0], marker='o', color='w', label=IO_label,
                                  markerfacecolor=colorIO, markersize=10)]
        for element in labels:
            legend_elements.append(element)
            
        plt.legend(handles = legend_elements, fontsize=20)
    plt.tight_layout()

    if savefig:
        plt.savefig(file, dpi = dpi)
    return(fig)

#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                           Generate Operator Limits Plot                                                       #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################
def limits_LEFT(experiments,                 #{label :{half-life, isotope, label}}
                method       = "IBM2",       #NME method
                unknown_LECs = False,        #use unknown LECs?
                PSF_scheme   = "A",          #which PSFs to use
                groups       = True,         #plot only the operator groups that have the same limits
                plottype     = "scales",     #"scales" or "limits"
                savefig      = False,        #save figure as file
                file         = "limits.png", #file to save figure to
                dpi          = 300           #resolution in dots per inch
               ):
    
    #make dicts to store limits in
    limits = {}
    scales = {}
    
    #iterate over experiments
    for exp in experiments:
        #read isotope from exp dict
        isotope = experiments[exp]["isotope"] 
        
        #read half-life limit from exp dict
        limit = experiments[exp]["half-life"]
        
        #try to real legend label
        try:
            label = experiments[exp]["label"]
        except:
            label = exp
            
        #calculate limits
        lims = f.get_limits_LEFT(limit, isotope = isotope, method = method, 
                                 unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
        
        #store limits in dicts
        limits[label] = lims[lims.keys()[0]]
        scales[label] = lims[lims.keys()[1]]
        
    #generate DataFrame
    scales = pd.DataFrame(scales)
    limits = pd.DataFrame(limits)
    
    if not groups:
        WC_operator_groups = LEFT_WCs_latex_dict.copy()

    else:
        WC_operator_groups = {"m_bb"  : r"$m_{\beta\beta}$", 
                              "VL(6)" : r"$C_{VL}^{(6)}$", 
                              "VR(6)" : r"$C_{VR}^{(6)}$", 
                              "T(6)"  : r"$C_{T}^{(6)}$", 
                              "SL(6)" : r"$C_{S}^{(6)}}$", 
                              "VL(7)" : r"$C_{V}^{(7)}$", 
                              "1L(9)" : r"$C_{S1}^{(9)}$", 
                              "2L(9)" : r"$C_{S2}^{(9)}$", 
                              "3L(9)" : r"$C_{S3}^{(9)}$", 
                              "4L(9)" : r"$C_{S4}^{(9)}$", 
                              "5L(9)" : r"$C_{S5}^{(9)}$", 
                              "6(9)"  : r"$C_{V}^{(9)}$", 
                              "7(9)"  : r"$C_{\tilde{V}}^{(9)}$"
                             }

    if groups:
        idx = 0
        for operator in scales.T:
            if operator not in WC_operator_groups or operator == "m_bb":
                scales.drop(operator, axis=0, inplace=True)
                limits.drop(operator, axis=0, inplace=True)
            else:
                idx+=1
                scales.rename(index=WC_operator_groups, inplace=True)
                limits.rename(index=WC_operator_groups, inplace=True)
    
    else:
        idx = 0
        for operator in scales.T:
            if operator == "m_bb":
                scales.drop(operator, axis=0, inplace=True)
                limits.drop(operator, axis=0, inplace=True)
            else:
                idx+=1
                scales.rename(index=WC_operator_groups, inplace=True)
                limits.rename(index=WC_operator_groups, inplace=True)
                
                
    plt.rc("ytick", labelsize = 20)
    plt.rc("xtick", labelsize = 20)
    
    if plottype == "scales":
        fig = (scales/1000).plot.bar(figsize=(16,6))
        fig.set_ylabel(r"$\Lambda$ [TeV]", fontsize = 20)
    else:
        fig = limits.plot.bar(figsize=(16,6))
        fig.set_ylabel(r"$C_X$", fontsize = 20)
    fig.set_yscale("log")
    fig.grid(linestyle="--")
    if len(experiments)>10:
        ncol = int(len(experiments)/2)
    else:
        ncol = len(experiments)
    fig.legend(fontsize=20, loc = (0,1.02), ncol=ncol)
    
    plt.tight_layout()
    if savefig:
        fig.get_figure().savefig(file, dpi = dpi)

    return(fig.get_figure())


def limits_SMEFT(experiments,                 #{label :{half-life, isotope, label}}
                 method       = "IBM2",       #NME method
                 unknown_LECs = False,        #use unknown LECs?
                 PSF_scheme   = "A",          #which PSFs to use
                 groups       = False,        #plot only the operator groups that have the same limits
                 plottype     = "scales",     #"scales" or "limits"
                 savefig      = False,        #save figure as file
                 file         = "limits.png", #file to save figure to
                 dpi          = 300           #resolution in dots per inch
                ):
    limits = {}
    scales = {}
    for exp in experiments:
        isotope = experiments[exp]["isotope"] 
        limit = experiments[exp]["half-life"]
        try:
            label = experiments[exp]["label"]
        except:
            label = exp
            
        lims = f.get_limits_SMEFT(limit, isotope = isotope, method = method, 
                                  unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
        limits[label] = lims[lims.keys()[0]]
        scales[label] = lims[lims.keys()[1]]

    scales = pd.DataFrame(scales)
    limits = pd.DataFrame(limits)
    if not groups:
        WC_operator_groups = SMEFT_WCs_latex_dict.copy()
    else:

        WC_operator_groups = SMEFT_WCs_latex_dict.copy()
        
    if groups:
        idx = 0
        for operator in scales.T:
            if operator not in WC_operator_groups or operator == "LH(5)":
                scales.drop(operator, axis=0, inplace=True)
                limits.drop(operator, axis=0, inplace=True)
            else:
                idx+=1
                scales.rename(index=WC_operator_groups, inplace=True)
                limits.rename(index=WC_operator_groups, inplace=True)
    else:
        idx = 0
        for operator in scales.T:
            if operator == "LH(5)":
                scales.drop(operator, axis=0, inplace=True)
                limits.drop(operator, axis=0, inplace=True)
            else:
                idx+=1
                scales.rename(index=WC_operator_groups, inplace=True)
                limits.rename(index=WC_operator_groups, inplace=True)
    plt.rc("ytick", labelsize = 20)
    plt.rc("xtick", labelsize = 20)
    
    if plottype == "scales":
        fig = (scales/1000).plot.bar(figsize=(16,6))
        fig.set_ylabel(r"$\Lambda$ [TeV]", fontsize =20)
    else:
        fig = limits.plot.bar(figsize=(16,6))
        fig.set_ylabel(r"$C_X$", fontsize =20)
    fig.set_yscale("log")
    fig.grid(linestyle="--")
    if len(experiments)>10:
        ncol = int(len(experiments)/2)
    else:
        ncol = len(experiments)
    fig.legend(fontsize=20, loc = (0,1.02), ncol=ncol)

    plt.tight_layout()
    if savefig:
        fig.get_figure().savefig(file, dpi = dpi)

    return(fig.get_figure())

#################################################################################################################################
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#                                           Generate Contour plot for 2 WCs                                                     #
#                                                                                                                               #
#                                                                                                                               #
#                                                                                                                               #
#################################################################################################################################

def contours(WCx,                                     #x-axis WC
             WCy,                                     #y-axis WC
             limits           = {"KamLAND-Zen": {"half-life" : 2.3e+26, 
                                                 "isotope"   : "136Xe", 
                                                 "color"     : "b", 
                                                 "label"     : "KamLAND-Zen", 
                                                 "linewidth" : 1, 
                                                 "linealpha" : 1, 
                                                 "linestyle" : "-",
                                                 "alpha"     : None
                                                }
                                },                    #experimental limits
             method           = "IBM2",               #NME method
             unknown_LECs     = False,                #use unknown LECs?
             PSF_scheme       = "A",                  #which PSFs to use
             n_points         = 5000,                 #number if points on the x-axis, will be cut if NaNs are within x-axis range
             phase            = 0,                    #relative complex phase between the two operators
             varyphases       = False,                #if True the complex phases will be varied
             n_vary           = 5,                    #number of phase variations
             x_min            = None,                 #minimum x_range
             x_max            = None,                 #maximum x_range
             savefig          = False,                #if True save figure as file
             file             = "contour_limits.png", #name of the saved file
             dpi              = 300                   #resolution of stored file in dots per inch
            ):
    
    if (x_min == None or x_max == None) and x_min != x_max:
        warnings.warn("You need to set both x_min and x_max or let the code choose both! Resetting x_min, x_max")
        x_min = None
        x_max = None
        
    if WCx in SMEFT_WCs:
        is_SMEFT = True
    else:
        is_SMEFT = False
        
    #sort experimental limits by half-life
    lim    = limits.copy()
    labels = []
    colors = []
    limits = {}
    for key in sorted(lim, key=lambda x: (lim[x]['half-life'])):
        limits[key] = lim[key]
        
    
    #generate contour points
    fig = plt.figure(figsize=(9, 8))
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
    
    #iterate over experiments
    exp_idx = 0
    radius = {}
    
    #iterate over experimental limits
    for limit in limits:
        try:
            hl      = limits[limit]["half-life"]
        except:
            raise ValueError("'half-life' is not defined in "+limit+" dict.")
        try:
            isotope = limits[limit]["isotope"]
        except:
            raise ValueError("'isotope' is not defined in "+limit+" dict.")
        try:
            color = limits[limit]["color"]
        except:
            color = colormap[exp_idx]
        try:
            label = limits[limit]["label"]
        except:
            label = None
        try:
            if limits[limit]["alpha"] == None:
                if varyphases:
                    alpha=np.min([1/(2*n_vary), 1/(2*len(limits))])
                else:
                    alpha=np.min([1/(2*1), 1/(2*len(limits))])
            else:
                alpha = limits[limit]["alpha"]
        except:
            if varyphases:
                alpha=np.min([1/(2*n_vary), 1/(2*len(limits))])
            else:
                alpha=np.min([1/(2*1), 1/(2*len(limits))])
        try:
            linewidth = limits[limit]["linewidth"]
        except:
            linewidth = 1
        try:
            linealpha = limits[limit]["linealpha"]
        except:
            linealpha = alpha
        try:
            linestyle = limits[limit]["linestyle"]
        except:
            linestyle = "-"
            
            
        color = list(pltcolors.to_rgb(color))
        edgecolor = color+[linealpha]
        facecolor = color+[alpha]
        
        
        M = f.generate_matrix(WC = [WCx, WCy], isotope = isotope, method = method, 
                              unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
        if is_SMEFT:
            fac = 3
        else:
            fac = 1.5
            
        #find the range of the curve by finding the maximal value 
        radius = fac * np.sqrt(1/M[0,0]*1/hl)
        if x_min == None:
            x_min = -radius
            x_max =  radius
        
        if varyphases:
            #Generate Phase Array
            phases = np.linspace(0, np.pi, n_vary)
            
            #Vary over phases
            for idx in range(n_vary):
                
                #Set phase
                phase = phases[idx]
                
                #Get contour values
                contours = f.get_contours(WCx, WCy, half_life = hl, isotope = isotope, phase = phase, method = method,
                                        n_points = n_points, x_min = x_min , x_max = x_max, 
                                        unknown_LECs = unknown_LECs, PSF_scheme = PSF_scheme)
                
                #x-axis
                xplot    = contours[contours.keys()[0]]
                
                #yaxis min
                contour  = contours[contours.keys()[1]]
                
                #y-axis max
                contour2 = contours[contours.keys()[2]]
                
                #X-Axis == Standard Mechanism
                if WCx == "m_bb":
                    #x-axis label
                    plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    
                    #y-axis label
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                    
                    #Fill Contour Area
                    plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), 
                                     edgecolor = edgecolor, facecolor = facecolor, 
                                     linewidth = linewidth, linestyle = linestyle)
                    
                #Y-Axis == Standard Mechanism
                elif WCy == "m_bb":
                    
                    #y-axis label
                    plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                    
                    #x-axis label
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                    
                    #fill contour area
                    plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, 
                                     edgecolor = edgecolor, facecolor = facecolor, 
                                     linewidth = linewidth, linestyle = linestyle)
                    
                #Non Standard Mechanisms on all axes or SMEFT
                else:
                    #SMEFT
                    if is_SMEFT:
                        #x-axis Operator Dimension
                        xdimension = int(WCx[-2])
                        
                        #y-axis Operator Dimension
                        ydimension = int(WCy[-2])
                        
                        #x-axis label
                        plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$ [TeV$^{-"+str(xdimension-4)+"}$]", fontsize = 20)
                        
                        #y-axis label
                        plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$ [TeV$^{-"+str(ydimension-4)+"}$]", fontsize = 20)
                        
                        #fill contour area
                        plt.fill_between((1e+3)**(xdimension-4)*np.array(xplot), 
                                         (1e+3)**(ydimension-4)*np.array(contour), 
                                         (1e+3)**(ydimension-4)*np.array(contour2), 
                                         edgecolor = edgecolor, facecolor = facecolor, 
                                         linewidth = linewidth, linestyle = linestyle)
                    
                    #LEFT
                    else:
                        #x-axis Operator Dimension
                        plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                        
                        #y-axis Operator Dimension
                        plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                        
                        #fill contour area
                        plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), 
                                         edgecolor = edgecolor, facecolor = facecolor, 
                                         linewidth = linewidth, linestyle = linestyle)
        
        #fixed complex phases
        else:
            contours = f.get_contours(WCx, WCy, half_life = hl, isotope = isotope, phase = phase, method = method,
                                    n_points = n_points, x_min = x_min , x_max = x_max)
            #x-axis
            xplot    = contours[contours.keys()[0]]

            #yaxis min
            contour  = contours[contours.keys()[1]]

            #y-axis max
            contour2 = contours[contours.keys()[2]]
            
            #X-Axis == Standard Mechanism
            if WCx == "m_bb":
                #x-axis label
                plt.xlabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                
                #y-axis label
                plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                        
                #fill contour area
                plt.fill_between(np.array(xplot)*1e+9, np.array(contour), np.array(contour2), 
                                 edgecolor = edgecolor, facecolor = facecolor, 
                                 linewidth = linewidth, linestyle = linestyle)
            
            #Y-Axis == Standard Mechanism
            elif WCy == "m_bb":
                #y-axis label
                plt.ylabel(r"$m_{\beta\beta}$ [eV]", fontsize = 20)
                
                #x-axis label
                plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                        
                #fill contour area
                plt.fill_between(np.array(xplot), np.array(contour)*1e+9, np.array(contour2)*1e+9, 
                                 edgecolor = edgecolor, facecolor = facecolor, 
                                 linewidth = linewidth, linestyle = linestyle)
                    
            #Non Standard Mechanisms on all axes or SMEFT
            else:
                #SMEFT
                if is_SMEFT:
                    #x-axis Operator Dimension
                    xdimension = int(WCx[-2])
                    
                    #y-axis Operator Dimension
                    ydimension = int(WCy[-2])
                    
                    #x-axis label
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$ [TeV$^{-"+str(xdimension-4)+"}$]", fontsize = 20)
                    
                    #y-axis label
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$ [TeV$^{-"+str(ydimension-4)+"}$]", fontsize = 20)
                    
                    #fill contour area
                    plt.fill_between((1e+3)**(xdimension-4)*np.array(xplot), 
                                     (1e+3)**(ydimension-4)*np.array(contour), 
                                     (1e+3)**(ydimension-4)*np.array(contour2), 
                                     edgecolor = edgecolor, facecolor = facecolor, 
                                     linewidth = linewidth, linestyle = linestyle)
                    
                #LEFT
                else:
                    #x-axis label
                    plt.xlabel(r"$C_{"+WCx[:-3]+"}^{"+WCx[-3:]+"}$", fontsize = 20)
                    
                    #y-axis label
                    plt.ylabel(r"$C_{"+WCy[:-3]+"}^{"+WCy[-3:]+"}$", fontsize = 20)
                    
                    #fill contour area
                    plt.fill_between(np.array(xplot), np.array(contour), np.array(contour2), 
                                     edgecolor = edgecolor, facecolor = facecolor, 
                                     linewidth = linewidth, linestyle = linestyle)
                        
        #add 1 to counter idx
        exp_idx += 1
        
        #store label in list
        labels.append(label)
        
        #store color in list
        colors.append(color)
        
    #x-axis range
    if WCx == "m_bb":
        plt.xlim([x_min*1e+9, x_max*1e+9])
    elif is_SMEFT:
        plt.xlim([x_min*1e+3**(xdimension-4), x_max*1e+3**(xdimension-4)])
        
    else:
        plt.xlim([x_min, x_max])
       
    #Set Legend options
    legend_elements = []
    for exp_idx in range(len(limits)):
        legend_elements.append(Line2D([0], [0], color = colors[exp_idx], 
                                      label=labels[exp_idx],
                                      markerfacecolor=colors[exp_idx], markersize=10))
        
    #Generate Legend
    plt.legend(handles = legend_elements, fontsize=20)
    
    #Ticksize
    plt.rc("ytick", labelsize = 15)
    plt.rc("xtick", labelsize = 15)
    
    plt.tight_layout()
    
    #Save Figure to File
    if savefig:
        if file == None:
            file = "contours_"+WCx+"_"+WCy+".png"
        plt.savefig(file)
    return(fig)
