#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 13:52:53 2018

@author: dagoret
"""
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import pandas as pd
import glob
from matplotlib.backends.backend_pdf import PdfPages 
from scipy import interpolate
from astropy.io import fits
import sys,os
PATH_SPECTRACTORSIM='../../SpectractorSim'
sys.path.append(PATH_SPECTRACTORSIM)
from spectractorsim import *

import re,os
import numpy as np
from locale import *
setlocale(LC_NUMERIC, '') 

os.path.dirname(__file__) 

REL_PATH=os.path.dirname(__file__) # relative directory path
ABS_PATH=os.path.abspath(__file__) # absolute file path
PYFILE_NAME=os.path.basename(__file__) # the file name only

print 'REL_PATH=',REL_PATH
print 'ABS_PATH=',ABS_PATH
print 'PYFILE_NAME=',PYFILE_NAME

PATH_SPECTRACTORSIM=os.path.join(REL_PATH,'../../SpectractorSim')
sys.path.append(PATH_SPECTRACTORSIM)
from spectractorsim import *

print 'PATH_SPECTRACTORSIM=',PATH_SPECTRACTORSIM
#---------------------------------------------------------------------------------------
def get_index_from_filename(ffilename,the_searchtag):
    """
    Function  get_index_from_filename(ffilename,the_searchtag)
    
    Extract index number from filename.
    
    input :
            ffilename : input filename with its path
            the_searchtag : the regular expression 
            
    output : index number
            
    example of regular expression:
        SelectTagRe='^reduc_%s_([0-9]+)_spectrum.fits$' % (date)
        with date="20170530"
    
    """
    
    fn=os.path.basename(ffilename)
    sel_index= int(re.findall(the_searchtag,fn)[0])
    return sel_index
#--------------------------------------------------------------------------------------
def Convert_InFloat(arr_str):
    """
    In the logbook the decimal point is converted into a comma.
    Then one need to replace the coma by a decimal point and then convert the string into a number
    """
    arr=[ atof(x.replace(",",".")) for x in arr_str]
    arr=np.array(arr)
    return arr
#-----------------------------------------------------------------------------------------
def roll_zeropad(a, shift, axis=None):
    """
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.

    Examples
    --------
    >>> x = np.arange(10)
    >>> roll_zeropad(x, 2)
    array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    >>> roll_zeropad(x, -2)
    array([2, 3, 4, 5, 6, 7, 8, 9, 0, 0])

    >>> x2 = np.reshape(x, (2,5))
    >>> x2
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])
    >>> roll_zeropad(x2, 1)
    array([[0, 0, 1, 2, 3],
           [4, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2)
    array([[2, 3, 4, 5, 6],
           [7, 8, 9, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=0)
    array([[0, 0, 0, 0, 0],
           [0, 1, 2, 3, 4]])
    >>> roll_zeropad(x2, -1, axis=0)
    array([[5, 6, 7, 8, 9],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 1, axis=1)
    array([[0, 0, 1, 2, 3],
           [0, 5, 6, 7, 8]])
    >>> roll_zeropad(x2, -2, axis=1)
    array([[2, 3, 4, 0, 0],
           [7, 8, 9, 0, 0]])

    >>> roll_zeropad(x2, 50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, -50)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> roll_zeropad(x2, 0)
    array([[0, 1, 2, 3, 4],
           [5, 6, 7, 8, 9]])

    """
    a = np.asanyarray(a)
    if shift == 0: return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res
#----------------------------------------------------------------------------------------
def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: 
        return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    #return y
    if len(x)%2==0: # even case
        return y[(window_len/2):-(window_len/2)] 
    else:           #odd case
        return y[(window_len/2-1):-(window_len/2)] 
    
#--------------------------------------------------------------------------------
def PlotAirmass(obs):
    plt.figure(figsize=(15,5))
    plt.plot(obs["index"],Convert_InFloat(obs["airmass"]),'bo')
    plt.xlabel("image index number")
    plt.ylabel("airmass")
    plt.title('Airmass vs image index number')
   
    plt.grid(b=True,which='major', linestyle='-', linewidth=1, color='black')
# Customize the minor grid
    plt.grid(b=True,which='minor', linestyle=':', linewidth=0.5, color='grey')
    
    #plt.tick_params(which='both', # Options for both major and minor ticks
    #            top='on', # turn off top ticks
    #            left='on', # turn off left ticks
    #            right='on',  # turn off right ticks
    #            bottom='on') # turn off bottom ticks  
    plt.show()
    
#---------------------------------------------------------------------------------
#  GetDisperserTransmission
#-------------------------------------------------------------------------------
def PlotSpectraDataSim(the_filelist,the_obs,the_searchtag,wlshift,the_title,FLAG_WL_CORRECTION,Flag_corr_wl=False):

    jet =plt.get_cmap('jet') 
    VMAX=len(the_filelist)
    cNorm  = colors.Normalize(vmin=0, vmax=VMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    num=0
    for the_file in the_filelist:
        num=num+1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                time_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                time_correction=0
            
            hdu = fits.open(the_file)
            data=hdu[0].data
            wl=data[0]+time_correction
            fl=data[1]
            colorVal = scalarMap.to_rgba(num,alpha=1)
            plt.plot(wl,fl,c=colorVal,label=str(idx))
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")   
    #plt.legend()
#---------------------------------------------------------------------------------------------
def PlotSpectraRatioDataDivSim(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_title,
                     FLAG_WL_CORRECTION,Flag_corr_wl=False,XMIN=400,XMAX=1000.,YMIN=0,YMAX=0):
    
    jet = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=len(the_filelist))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    num=0
    
    all_Y_max=[]
    all_Y_min=[]
    
    for the_file in the_filelist:  # loop on reconstruted spectra
        num+=1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:   
            
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            
            # check if tthe index is in the disperser indexes 
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            data1=hdu1[0].data
            header1=hdu1[0].header
            filter1=header1["FILTER1"]
            
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            efunc = interpolate.interp1d(wl1, err1)
            
            fl0=func(WL)
            er0=efunc(WL)
            
            ratio=fl0/fl2
            eratio=er0/fl2
            
            #search nan
            #bad_bins=np.where(np.logical_or(np.isnan(ratio),np.isnan(eratio)))[0]
            #good_bins=np.where(np.logical_and(~np.isnan(ratio),~np.isnan(eratio)))[0]
            
            bad_bins=np.where(np.logical_or(~np.isfinite(ratio),~np.isfinite(eratio)))[0]
            good_bins=np.where(np.logical_and(np.isfinite(ratio),np.isfinite(eratio)))[0]
            
            #print 'bad_bins=',bad_bins
            #print 'good_bins=',good_bins
            
            the_X=WL[good_bins]
            the_Y=ratio[good_bins]
            the_EY=eratio[good_bins]
        
            #print 'the_X=',the_X
            #print 'the_Y=',the_Y
            #print 'the_EY=',the_EY
            
        
            colorVal = scalarMap.to_rgba(num)
            #plt.plot(WL,ratio,'-',color=colorVal)
            
            #plt.fill_between(WL,y1=ratio-1.96*eratio,y2=ratio+1.96*eratio,facecolor='grey',alpha=0.5)
            #plt.errorbar(WL,ratio,yerr=eratio,fmt = '-',markersize = 1,color=colorVal,zorder = 300,antialiased = True)
            
            plt.fill_between(the_X,y1=the_Y-1.96*the_EY,y2=the_Y+1.96*the_EY,facecolor='grey',alpha=0.5)
            plt.errorbar(the_X,the_Y,yerr=the_EY,fmt = '-',markersize = 1,color=colorVal,zorder = 300,antialiased = True)
            #plt.plot(the_X,the_Y,yerr=the_EY, '.')
            
            #sel_iii=np.where(np.logical_and(WL>=XMIN,WL<=XMAX))[0]
            
            sel_iii=np.where(np.logical_and(the_X>=XMIN,the_X<=XMAX))[0]
            
            #print 'sel_iii=',sel_iii
            
            the_Y_max=the_Y[sel_iii].max()*1.5
            the_Y_min=the_Y[sel_iii].min()/1.5
            
            #print 'the_Y_min =',the_Y_min,' the_Y_max =',the_Y_max
            
            all_Y_max.append(the_Y_max)
            all_Y_min.append(the_Y_min)
     
    all_Y_max=np.array(all_Y_max)
    all_Y_min=np.array(all_Y_min)
    
    the_Y_min=all_Y_min.min()
    the_Y_max=all_Y_max.max()
    
    plt.xlim(XMIN,XMAX)
    
    if YMIN==0 and YMAX==0 :
        plt.ylim(the_Y_min,the_Y_max)
    else:
        plt.ylim(0,YMAX)
        
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")   
    plt.ylabel("spectra ratio")  
#-----------------------------------------------------------------------------------------
def PlotSpectraLogRatioDataDivSim(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_title,FLAG_WL_CORRECTION,Flag_corr_wl=False,XMIN=400,XMAX=1000.,YMIN=0,YMAX=0):
    
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=len(the_filelist))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    all_Y_max=[]
    all_Y_min=[]
    num=0
    for the_file in the_filelist:  # loop on reconstruted spectra
        num+=1
        idx=get_index_from_filename(the_file,the_searchtag)
        
        if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
        else:
                wl_correction=0
        
        
        if idx in the_selected_indexes:                # check if tthe index is in the disperser indexes 
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            data1=hdu1[0].data
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            efunc = interpolate.interp1d(wl1, err1)
            
            fl0=func(WL)
            er0=efunc(WL)
          
            #the_Y=2.5*(np.log10(fl0)-np.log10(fl2))
            the_Y=fl0/fl2
            the_Y_err=er0/fl2
            
            ratio=fl0/fl2
            eratio=er0/fl2
            
            bad_bins=np.where(np.logical_or(~np.isfinite(ratio),~np.isfinite(eratio)))[0]
            good_bins=np.where(np.logical_and(np.isfinite(ratio),np.isfinite(eratio)))[0]
            
            #print 'bad_bins=',bad_bins
            #print 'good_bins=',good_bins
            
            the_X=WL[good_bins]
            the_Y=ratio[good_bins]
            the_EY=eratio[good_bins]
            
            
            colorVal = scalarMap.to_rgba(num)
            #plt.semilogy(WL,the_Y,'o',color=colorVal)
            plt.semilogy(the_X,the_Y,'-',color=colorVal)
            
            sel_iii=np.where(np.logical_and(the_X>=XMIN,the_X<=XMAX))[0]
            
            the_Y_max=np.max(the_Y[sel_iii])*3.0
            the_Y_min=np.min(the_Y[sel_iii])/3.0
          
            all_Y_max.append(the_Y_max)
            all_Y_min.append(the_Y_min)
            
    plt.xlim(XMIN,XMAX)
    
    all_Y_max=np.array(all_Y_max)
    all_Y_min=np.array(all_Y_min)
    
    the_Y_min=all_Y_min.min()
    the_Y_max=all_Y_max.max()
    
    
    if YMIN==0 and YMAX==0 :
        plt.ylim(the_Y_min,the_Y_max)
    else:
        plt.ylim(YMIN,YMAX)
    
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")  
    plt.ylabel("Spectra ratio")  
    #plt.grid(True,which="majorminor",ls="-", color='0.65')
    #plt.grid(True,which="both",ls="-")
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
#------------------------------------------------------------------------------------    
def SaveSpectraRatioDataDivSim(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_ratio_file,FLAG_WL_CORRECTION,Flag_corr_wl=False):
    
    all_ratio_arr=np.zeros((1,len(WL)))
    all_ratio_arr[0,:]=WL
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser    
    for the_file in the_filelist:  # loop on reconstruted spectra
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:                # check if tthe index is in the disperser indexes 
            
            
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            header1=data1=hdu1[0].header
            airmass=header1["airmass"]
            target=header1["target"]
            
            data1=hdu1[0].data
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            fl0=func(WL)
            ratio=fl0/fl2
            
            
            
            new_ratio=np.expand_dims(ratio, axis=0)
            all_ratio_arr=np.append(all_ratio_arr,new_ratio,axis=0)
            
            
    hdu = fits.PrimaryHDU(all_ratio_arr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(the_ratio_file,overwrite=True)
    
    return all_ratio_arr          
#---------------------------------------------------------------------------------------    
#  GetDisperserTransmissionSmooth       
#--------------------------------------------------------------------------------- 
def PlotSpectraDataSimSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,FLAG_WL_CORRECTION,Flag_corr_wl=False,Wwidth=21):

    jet =plt.get_cmap('jet') 
    VMAX=len(the_filelist)
    cNorm  = colors.Normalize(vmin=0, vmax=VMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    num=0
    for the_file in the_filelist:
        num=num+1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            hdu = fits.open(the_file)
            data=hdu[0].data
            wl=data[0]+wl_correction
            fl=data[1]
            err=data[2]
            
            # extend range for (wl1,fl1)
            wl=np.insert(wl,0,WL[0])
            fl=np.insert(fl,0,0.)
            err=np.insert(err,0,0.)
            
            wl=np.append(wl,WL[-1])
            fl=np.append(fl,0.)
            err=np.append(err,0.)
            
            func = interpolate.interp1d(wl, fl)
            efunc = interpolate.interp1d(wl, err) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            fl_smooth=smooth(fl0,window_len=Wwidth)
            
            
            colorVal = scalarMap.to_rgba(num,alpha=1)
            plt.plot(WL,fl_smooth,c=colorVal,label=str(idx))
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")   
    plt.ylabel("smoothed spectra")   
    #plt.legend()
#-----------------------------------------------------------------------------------------
def PlotSpectraRatioDataDivSimSmooth(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_title,FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                     XMIN=400,XMAX=1000.,YMIN=0,YMAX=0,Wwidth=21):
    
    jet =plt.get_cmap('jet') 
    VMAX=len(the_filelist)
    cNorm  = colors.Normalize(vmin=0, vmax=VMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    num=0
    
    all_Y_max=[]
    all_Y_min=[]
    
    for the_file in the_filelist:  # loop on reconstruted spectra
        num+=1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:   
            
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            
            # check if tthe index is in the disperser indexes 
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            data1=hdu1[0].data
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            efunc = interpolate.interp1d(wl1, err1) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            f1_smooth=smooth(fl0,window_len=Wwidth)
            f2_smooth=smooth(fl2,window_len=Wwidth)
            
            ef1_smooth=smooth(er0,window_len=Wwidth)
            
            ratio=f1_smooth/f2_smooth
            eratio=ef1_smooth/f2_smooth
            
            
            bad_bins=np.where(np.logical_or(~np.isfinite(ratio),~np.isfinite(eratio)))[0]
            good_bins=np.where(np.logical_and(np.isfinite(ratio),np.isfinite(eratio)))[0]
            
            
            the_X=WL[good_bins]
            the_Y=ratio[good_bins]
            the_EY=eratio[good_bins]
            
           
            
            sel_iii=np.where(np.logical_and(the_X>=XMIN,the_X<=XMAX))[0]
            
            the_Y_max=the_Y[sel_iii].max()*1.5
            the_Y_min=the_Y[sel_iii].min()/1.5
            
            all_Y_max.append(the_Y_max)
            all_Y_min.append(the_Y_min)
            
            colorVal = scalarMap.to_rgba(num,alpha=1)
            
            plt.plot(the_X,the_Y,color=colorVal)
            plt.errorbar(the_X,the_Y,yerr=the_EY,fmt = 'o',markersize = 1,color=colorVal,zorder = 300,antialiased = True)
            
            
    plt.xlim(XMIN,XMAX)
    
    all_Y_max=np.array(all_Y_max)
    all_Y_min=np.array(all_Y_min)
    
    the_Y_min=all_Y_min.min()
    the_Y_max=all_Y_max.max()
    
    if YMIN==0 and YMAX==0 :
        plt.ylim(0.,the_Y_max)
    else:
        plt.ylim(YMIN,YMAX)
        
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")   
    plt.ylabel("spectra ratio")  
#----------------------------------------------------------------------------------------
def PlotSpectraLogRatioDataDivSimSmooth(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_title,
                        FLAG_WL_CORRECTION,Flag_corr_wl=False,XMIN=400,XMAX=1000.,YMIN=0,YMAX=0,Wwidth=21):
    
    jet =plt.get_cmap('jet') 
    VMAX=len(the_filelist)
    cNorm  = colors.Normalize(vmin=0, vmax=VMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    plt.figure(figsize=(10,8))
    num=0
    all_Y_max=[]
    all_Y_min=[]
    for the_file in the_filelist:  # loop on reconstruted spectra
        num+=1
        idx=get_index_from_filename(the_file,the_searchtag)
        
        if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
        else:
                wl_correction=0
        
        
        if idx in the_selected_indexes:                # check if tthe index is in the disperser indexes 
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            data1=hdu1[0].data
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            efunc = interpolate.interp1d(wl1, err1) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            f1_smooth=smooth(fl0,window_len=Wwidth)
            f2_smooth=smooth(fl2,window_len=Wwidth)
            
            ef1_smooth=smooth(er0,window_len=Wwidth)
            
            ratio=f1_smooth/f2_smooth
            eratio=ratio*(ef1_smooth/f2_smooth)
            
            bad_bins=np.where(np.logical_or(~np.isfinite(ratio),~np.isfinite(eratio)))[0]
            good_bins=np.where(np.logical_and(np.isfinite(ratio),np.isfinite(eratio)))[0]
            
            
            the_X=WL[good_bins]
            the_Y=ratio[good_bins]
            the_EY=eratio[good_bins]
            

            
            sel_iii=np.where(np.logical_and(the_X>=XMIN,the_X<=XMAX))[0]
           
            the_Y_max=(np.max(the_Y[sel_iii]))*3.0
            the_Y_min=(np.min(the_Y[sel_iii]))/3.0
            all_Y_max.append(the_Y_max)
            all_Y_min.append(the_Y_min)
            
            
            colorVal = scalarMap.to_rgba(num,alpha=1)
            
            plt.semilogy(the_X,the_Y,color=colorVal)
            
            
    plt.xlim(XMIN,XMAX)
    
    all_Y_max=np.array(all_Y_max)
    all_Y_min=np.array(all_Y_min)
    
    the_Y_min=all_Y_min.min()
    the_Y_max=all_Y_max.max()
    
    if YMIN==0 and YMAX==0 :
        print 'the_Y_min,the_Y_max =',the_Y_min,the_Y_max
        plt.ylim(the_Y_min,the_Y_max)
    else:
        plt.ylim(YMIN,YMAX)
    
    
    #plt.grid(True,which="majorminor",ls="-", color='0.65')
    #plt.grid(True,which="both",ls="-")
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")  
    plt.ylabel("Spectra ratio (mag)")  
#------------------------------------------------------------------------------------    
def SaveSpectraRatioDataDivSimSmooth(the_filelist,path_tosims,the_obs,the_searchtag,wlshift,the_ratio_file,FLAG_WL_CORRECTION,Flag_corr_wl=False,Wwidth=21):
    
    all_ratio_arr=np.zeros((1,len(WL)))
    all_ratio_arr[0,:]=WL
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser    
    for the_file in the_filelist:  # loop on reconstruted spectra
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:                # check if tthe index is in the disperser indexes 
            
            
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            
            basefn=os.path.basename(the_file)                  # basename of reconstruced spectra
            basefn2=basefn.replace('reduc','specsim')  # reconstruct the simulation filename
            the_filesim=os.path.join(path_tosims,basefn2)  # add the path for the simulated file
            
            hdu1 = fits.open(the_file)
            header1=data1=hdu1[0].header
            airmass=header1["airmass"]
            target=header1["target"]
            
            data1=hdu1[0].data
            wl1=data1[0]+wl_correction
            fl1=data1[1]
            err1=data1[2]
            
            # extend range for (wl1,fl1)
            wl1=np.insert(wl1,0,WL[0])
            fl1=np.insert(fl1,0,0.)
            err1=np.insert(err1,0,0.)
            
            wl1=np.append(wl1,WL[-1])
            fl1=np.append(fl1,0.)
            err1=np.append(err1,0.)
            
            hdu2 = fits.open(the_filesim)
            data2=hdu2[0].data
            wl2=data2[0]
            fl2=data2[1]
            
            func = interpolate.interp1d(wl1, fl1)
            efunc = interpolate.interp1d(wl1, err1) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            ratio=fl0/fl2
            
            f1_smooth=smooth(fl0,window_len=Wwidth)
            f2_smooth=smooth(fl2,window_len=Wwidth)
            ratio=f1_smooth/f2_smooth
            
            new_ratio=np.expand_dims(ratio, axis=0)
            all_ratio_arr=np.append(all_ratio_arr,new_ratio,axis=0)
            
            
    hdu = fits.PrimaryHDU(all_ratio_arr)
    hdul = fits.HDUList([hdu])
    hdul.writeto(the_ratio_file,overwrite=True)
    
    return all_ratio_arr 
#----------------------------------------------------------------------------------
#
#--------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21):

    
    # color according wavelength
    jet =plt.get_cmap('jet') 
    if (ZMIN==0 and ZMAX==0):
        WLMIN=300.
        WLMAX=600.
    elif ZMIN==0:
        WLMIN=WL.min()
        WLMAX=ZMAX
    elif ZMAX==0:
        WLMIN=ZMIN
        WLMAX=600.
    else:
        WLMIN=ZMIN
        WLMAX=ZMAX
        
    cNorm  = colors.Normalize(vmin=WLMIN, vmax=WLMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    attenuation=np.zeros((len(the_filelist)+1,2+len(WL)))
    att_err=np.zeros((len(the_filelist)+1,2+len(WL)))
    
    
   
    num=0
    numsel=0
    for the_file in the_filelist:
        num=num+1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:
            numsel+=1
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            hdu = fits.open(the_file)
            header=hdu[0].header
            airmass=header["AIRMASS"]
            data=hdu[0].data
            wl=data[0]+wl_correction
            fl=data[1]
            err=data[2]
            
            
            
            # extend range for (wl1,fl1)
            wl=np.insert(wl,0,WL[0])
            fl=np.insert(fl,0,0.)
            err=np.insert(err,0,0.)
            
            wl=np.append(wl,WL[-1])
            fl=np.append(fl,0.)
            err=np.append(err,0.)
            
            func = interpolate.interp1d(wl, fl)
            efunc = interpolate.interp1d(wl, err) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            fl_smooth=smooth(fl0,window_len=Wwidth)
            errfl_smooth=smooth(er0,window_len=Wwidth)
            
            attenuation[numsel,0]=idx
            attenuation[numsel,1]=airmass
            attenuation[numsel,2:]=fl_smooth            
            att_err[numsel,2:]=errfl_smooth
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    #print 'all_airmasses',all_airmasses
    #print 'all_indexes',all_imgidx
    
    # selection where airmass are OK
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    #print 'good indexes =',good_indexes
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]
    
    #print 'sel_airmasses',sel_airmasses
    #print 'sel_indexes',sel_imgidx
    
    airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    #print 'airmass-min = ',sel_airmasses[airmassmin_index]
    
    # loop on wavelength bins
    #plt.figure(figsize=(15,4))
    #plt.plot(sel_imgidx,sel_airmasses,'o')
    #plt.show()
    
    plt.figure(figsize=(15,8))
    # loop on wavelength indexes
    for idx_wl in np.arange(2,len(WL)+2): 
        if WL[idx_wl-2]<WLMIN:
            continue
        if WL[idx_wl-2]>WLMAX:
            break
        colorVal = scalarMap.to_rgba(WL[idx_wl-2],alpha=1)
        att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        plt.semilogy(sel_airmasses,sel_attenuation[:,idx_wl],'o-',c=colorVal)
          
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    plt.title(the_title)
    plt.xlabel("airmass")   
    plt.ylabel("intensity in erg/cm2/s:nm")   
    #plt.legend()  
    plt.show() 
#----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationSmoothBin(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Bwidth=20):

    
    # color according wavelength
    jet =plt.get_cmap('jet') 
    if (ZMIN==0 and ZMAX==0):
        WLMIN=300.
        WLMAX=600.
    elif ZMIN==0:
        WLMIN=WL.min()
        WLMAX=ZMAX
    elif ZMAX==0:
        WLMIN=ZMIN
        WLMAX=600.
    else:
        WLMIN=ZMIN
        WLMAX=ZMAX
        
    cNorm  = colors.Normalize(vmin=WLMIN, vmax=WLMAX)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    
    the_selected_indexes=the_obs["index"].values  # get the array of index for that disperser
    
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    attenuation=np.zeros((len(the_filelist)+1,2+len(WL)))
    att_err=np.zeros((len(the_filelist)+1,2+len(WL)))
    
    
   
    num=0
    numsel=0
    for the_file in the_filelist:
        num=num+1
        idx=get_index_from_filename(the_file,the_searchtag)
        if idx in the_selected_indexes:
            numsel+=1
            if FLAG_WL_CORRECTION and Flag_corr_wl:
                wl_correction=wlshift[wlshift["index"]==idx].loc[:,"wlshift"].values[0]
            else:
                wl_correction=0
            
            hdu = fits.open(the_file)
            header=hdu[0].header
            airmass=header["AIRMASS"]
            data=hdu[0].data
            wl=data[0]+wl_correction
            fl=data[1]
            err=data[2]
            
            
            
            # extend range for (wl1,fl1)
            wl=np.insert(wl,0,WL[0])
            fl=np.insert(fl,0,0.)
            err=np.insert(err,0,0.)
            
            wl=np.append(wl,WL[-1])
            fl=np.append(fl,0.)
            err=np.append(err,0.)
            
            func = interpolate.interp1d(wl, fl)
            efunc = interpolate.interp1d(wl, err) 
            
            fl0=func(WL)
            er0=efunc(WL)
            
            fl_smooth=smooth(fl0,window_len=Wwidth)
            errfl_smooth=smooth(er0,window_len=Wwidth)
            
            attenuation[numsel,0]=idx
            attenuation[numsel,1]=airmass
            attenuation[numsel,2:]=fl_smooth            
            att_err[numsel,2:]=errfl_smooth
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    #print 'all_airmasses',all_airmasses
    #print 'all_indexes',all_imgidx
    
    # selection where airmass are OK
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    #print 'good indexes =',good_indexes
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]
    
    #print 'sel_airmasses',sel_airmasses
    #print 'sel_indexes',sel_imgidx
    
    airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    #print 'airmass-min = ',sel_airmasses[airmassmin_index]
    
    # loop on wavelength bins
    #plt.figure(figsize=(15,4))
    #plt.plot(sel_imgidx,sel_airmasses,'o')
    #plt.show()
    
    plt.figure(figsize=(15,8))
    # loop on wavelength indexes
    for idx_wl in np.arange(2,len(WL)+2,Bwidth): 
        if WL[idx_wl-2]<WLMIN:
            continue
        if WL[idx_wl-2]>WLMAX:
            break
        colorVal = scalarMap.to_rgba(WL[idx_wl-2],alpha=1)
        idx_startwl=idx_wl
        idx_stopwl=min(idx_wl+Bwidth-1,sel_attenuation.shape[1])
        
        thelabel="{:d}-{:d} nm".format(WL[idx_startwl-2],WL[idx_stopwl-2] )
        
        FluxBin=sel_attenuation[:,idx_startwl:idx_stopwl]
        FluxAver=np.average(FluxBin,axis=1)
        att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        plt.semilogy(sel_airmasses,FluxAver,'o-',c=colorVal,label=thelabel)
          
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    plt.title(the_title)
    plt.xlabel("airmass")   
    plt.ylabel("intensity in erg/cm2/s:nm")   
    plt.legend(loc='best')  
    if XMIN==0 and XMAX==0:
        plt.xlim(0.7,sel_airmasses.max())
    elif XMIN==0:
        plt.xlim(0.7,XMAX)
    elif XMAX==0:
        plt.xlim(XMIN,sel_airmasses.max())
    else:
        plt.xlim(XMIN,XMAX)
    plt.show()               