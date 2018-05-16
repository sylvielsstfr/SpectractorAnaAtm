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

from scipy.optimize import curve_fit

import pysynphot as S

import sys,os
#PATH_SPECTRACTORSIM='../../SpectractorSim'
#sys.path.append(PATH_SPECTRACTORSIM)
#PATH_SPECTRACTORSIM='../../Spectractor'
#sys.path.append(PATH_SPECTRACTOR)

#from spectractorsim import *
#from spectractor import *
import re
import numpy as np
from locale import *
setlocale(LC_NUMERIC, '') 

os.path.dirname(__file__) 

REL_PATH=os.path.dirname(__file__) # relative directory path
ABS_PATH=os.path.abspath(__file__) # absolute file path
PYFILE_NAME=os.path.basename(__file__) # the file name only

#print 'REL_PATH=',REL_PATH
#print 'ABS_PATH=',ABS_PATH
#print 'PYFILE_NAME=',PYFILE_NAME

PATH_SPECTRACTORSIM=os.path.join(REL_PATH,'../../SpectractorSim')
sys.path.append(PATH_SPECTRACTORSIM)
PATH_SPECTRACTOR=os.path.join(REL_PATH,'../../Spectractor')
sys.path.append(PATH_SPECTRACTOR)
from spectractorsim import *
from spectractor import *

#print 'PATH_SPECTRACTORSIM=',PATH_SPECTRACTORSIM

CALSPEC_FILENAMES={'HD111980':'hd111980_stis_003.fits','HD205905':'hd205905_stis_003.fits',
                   'HD160617':'hd160617_stis_003.fits','HD185975':'hd185975_stis_003.fits'}
#---------------------------------------------------------------------------------------
def GetSED(starname,sedunit='flam'):
    """
    Get SED from starname
    """
    
    filename=CALSPEC_FILENAMES[starname]
    
    fullfilename = os.path.join(os.environ['PYSYN_CDBS'], 'calspec', filename)
    sp = S.FileSpectrum(fullfilename)
    sp.convert(sedunit)
    wl=sp.wave/10.
    flux=sp.flux*10.
    func = interpolate.interp1d(wl, flux)
    return func(WL),sedunit
#---------------------------------------------------------------------------------------
def GetSEDSmooth(starname,Wwidth=21,sedunit='flam'):
    """
    Get smoothed SED from starname
    """
    
    flux,sedunit=GetSED(starname,sedunit=sedunit)
    fluxsmoothed=smooth(flux,window_len=Wwidth)
    return fluxsmoothed,sedunit
#--------------------------------------------------------------------------------------
def PlotSED(starname,Wwidth=21,sedunit='flam',scale='lin'):
    
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    
    flux,unit1=GetSED(starname,sedunit=sedunit)
    fluxsm,unit2=GetSEDSmooth(starname,Wwidth=Wwidth,sedunit=sedunit)
    
    if scale=='lin':
        ax1.plot(WL,flux,'b-')
    else:
        ax1.semilogy(WL,flux,'b-')
    ax1.set_xlabel('$\lambda$ (nm)')
    ax1.set_ylabel(unit1)
    ax1.set_title('sed')
    ax1.grid(True,which='both')

    
    
    if scale=='lin':
        ax2.plot(WL,fluxsm,'b-')
    else:
        ax2.semilogy(WL,fluxsm,'b-')
    
    ax2.set_xlabel('$\lambda$ (nm)')
    ax2.set_ylabel(unit2)
    ax2.set_title('smoothed sed')
    ax2.grid(True,which='both')

    plt.suptitle(starname)
    plt.show()
#---------------------------------------------------------------------------------------
def GetTarget(starname):
    target = Target(starname)
    target.plot_spectra()


#------------------------------------------------------------------------------------
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
            err=data[2]
           
            colorVal = scalarMap.to_rgba(num,alpha=1)
            
            plt.fill_between(wl,y1=fl-1.96*err,y2=fl+1.96*err,facecolor='grey',alpha=0.5)
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
            errfl_smooth=smooth(er0,window_len=Wwidth)
            
            colorVal = scalarMap.to_rgba(num,alpha=1)
            
            plt.fill_between(WL,y1=fl_smooth-1.96*errfl_smooth,y2=fl_smooth+1.96*errfl_smooth,facecolor='grey',alpha=0.5)
            
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
#  GetDisperserAttenuation_smooth.ipynb
#--------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Mag=True):

    
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
        
        
        if not Mag:
            plt.semilogy(sel_airmasses,sel_attenuation[:,idx_wl],'o-',c=colorVal)
        else:
            plt.plot(sel_airmasses,2.5*np.log10(sel_attenuation[:,idx_wl]),'o-',c=colorVal)
          
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    plt.title(the_title)
    plt.xlabel("airmass")  
    
    if not Mag:
        plt.ylabel("intensity in erg/cm2/s/nm")   
    else:
        plt.ylabel("intensity in erg/cm2/s/nm (mag)") 
    #plt.legend()  
    plt.show() 
#----------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationSmoothBin(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Bwidth=20,Mag=True):

    
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
    
    #------------------------------------------------------------------------
    # attenuation container 
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    #-------------------------------------------------------------------------
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
    sel_err=att_err[good_indexes,:]
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
    
    ################### Plot the figure ###############################################
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
        
        # slice of  flux in wavelength bins
        FluxBin=sel_attenuation[:,idx_startwl:idx_stopwl]
        FluxBinErr=sel_err[:,idx_startwl:idx_stopwl]
        # get the average of flux in that big wl bin
        FluxAver=np.average(FluxBin,axis=1)
        FluxAverErr=np.average(FluxBinErr,axis=1)
        
        Y0=FluxAver
        Y1=Y0-FluxAverErr
        Y2=Y0+FluxAverErr
                
        # get the attenuation for the airmass-min
        att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        
        if not Mag:
            plt.fill_between(sel_airmasses,y1=Y1,y2=Y2, where=Y1>0 ,color='grey', alpha=0.3 )        
            plt.yscale( "log" )
        
            # plot the attenuation wrt airmass
            plt.semilogy(sel_airmasses,Y0,'o-',c=colorVal,label=thelabel)
        else:
            newY0=2.5*np.log10(Y0)
            newY1=2.5*np.log10(Y1)
            newY2=2.5*np.log10(Y2)
            
            plt.fill_between(sel_airmasses,y1=newY1,y2=newY2, where=Y1>0 ,color='grey', alpha=0.3 ) 
            plt.plot(sel_airmasses,newY0,'o-',c=colorVal,label=thelabel)
                      
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.3)
    plt.title(the_title)
    plt.xlabel("airmass")  
    
    if not Mag:
        plt.ylabel("intensity in erg/cm2/s/nm")   
    else:
        plt.ylabel("intensity in erg/cm2/s/nm (mag)") 
        
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
    
#------------------------------------------------------------------------------------------    
#  GetDisperserAttenuationRatio_smooth.ipynb    
#------------------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationRatioSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Mag=True):

    
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
    all_sed=np.zeros((len(the_filelist)+1,2+len(WL)))
    
    
   
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
            objectname=header["TARGET"]
            
            # extract smoothed SED
            sedsm,sedunit=GetSEDSmooth(objectname,Wwidth=Wwidth)
            
            
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
            
            # smoothed SED
            all_sed[numsel,0]=idx
            all_sed[numsel,1]=airmass
            all_sed[numsel,2:]=sedsm 
            
            
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    # selection 
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]  
    sel_sed=all_sed[good_indexes,:]
    
    airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    
    
    plt.figure(figsize=(15,8))
    # loop on wavelength indexes
    for idx_wl in np.arange(2,len(WL)+2): 
        if WL[idx_wl-2]<WLMIN:
            continue
        if WL[idx_wl-2]>WLMAX:
            break
        colorVal = scalarMap.to_rgba(WL[idx_wl-2],alpha=1)
        att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        
        if not Mag:
            plt.semilogy(sel_airmasses,sel_attenuation[:,idx_wl]/sel_sed[:,idx_wl],'o-',c=colorVal)
        else:
            plt.plot(sel_airmasses,2.5*np.log10(sel_attenuation[:,idx_wl]/sel_sed[:,idx_wl]),'o-',c=colorVal)
          
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    
    plt.title(the_title)
    plt.xlabel("airmass")   
    if not Mag:
        plt.ylabel("intensity in data/ intensity of SED")  
    else:
        plt.ylabel("intensity in data/ intensity of SED (mag)")  
    #plt.legend()  
    plt.show() 
#----------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------
def PlotSpectraDataSimAttenuationRatioSmoothBin(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Bwidth=20,Mag=True):

    
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
    
    #------------------------------------------------------------------------
    # attenuation container 
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    #-------------------------------------------------------------------------
    attenuation=np.zeros((len(the_filelist)+1,2+len(WL)))
    att_err=np.zeros((len(the_filelist)+1,2+len(WL)))
    all_sed=np.zeros((len(the_filelist)+1,2+len(WL)))
    
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
            objectname=header["TARGET"]
            
            # extract smoothed SED
            sedsm,sedunit=GetSEDSmooth(objectname,Wwidth=Wwidth)
            
            
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
            
            # attenuation in data
            attenuation[numsel,0]=idx
            attenuation[numsel,1]=airmass
            attenuation[numsel,2:]=fl_smooth  
            
            # error on attenuation
            att_err[numsel,2:]=errfl_smooth
            
            # smoothed SED
            all_sed[numsel,0]=idx
            all_sed[numsel,1]=airmass
            all_sed[numsel,2:]=sedsm 
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    
    
    # selection where airmass are OK
    #---------------------------------
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_err=att_err[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]
    
    sel_sed=all_sed[good_indexes,:]
    
    

    
    airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    #print 'airmass-min = ',sel_airmasses[airmassmin_index]
    
    # loop on wavelength bins
    #plt.figure(figsize=(15,4))
    #plt.plot(sel_imgidx,sel_airmasses,'o')
    #plt.show()
    
    ################### Plot the figure ###############################################
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
        
        # slice of  flux in wavelength bins
        FluxBin=sel_attenuation[:,idx_startwl:idx_stopwl]
        FluxBinSED=sel_sed[:,idx_startwl:idx_stopwl]
        
        FluxBinErr=sel_err[:,idx_startwl:idx_stopwl]
       
                 
        # get the average of flux in that big wl bin
        FluxAver=np.average(FluxBin,axis=1)
        FluxAverSED=np.average(FluxBinSED,axis=1)
        FluxAverErr=np.average(FluxBinErr,axis=1)
        
        Y0=FluxAver/FluxAverSED
        Y1=Y0-FluxAverErr/FluxAverSED
        Y2=Y0+FluxAverErr/FluxAverSED
        
        # get the attenuation for the airmass-min
        att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        
        if not Mag:        
            plt.fill_between(sel_airmasses,y1=Y1,y2=Y2, where=Y1>0 ,color='grey', alpha=0.3 )        
            plt.yscale( "log" )
        
            # plot the attenuation wrt airmass
            plt.semilogy(sel_airmasses,Y0,'o-',c=colorVal,label=thelabel)
        else:
            
            newY0=2.5*np.log10(Y0)
            newY1=2.5*np.log10(Y1)
            newY2=2.5*np.log10(Y2)
            
            plt.fill_between(sel_airmasses,y1=newY1,y2=newY2, where=Y1>0 ,color='grey', alpha=0.3 )        
            
            # plot the attenuation wrt airmass
            plt.plot(sel_airmasses,newY0,'o-',c=colorVal,label=thelabel)
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    plt.title(the_title)
    plt.xlabel("airmass")   
    if not Mag:  
        plt.ylabel("intensity in data / intensity in SED") 
    else:
        plt.ylabel("intensity in data / intensity in SED (mag)") 
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
    
#-------------------------------------------------------------------------------------    
#   FIT
#----------------------------------------------------------------------------------    
#--------------------------------------------------------------------------------------------   
def bougline(x, a, b):
    return a*x + b
#-----------------------------------------------------------------------------------
def Varbougline(x,popt,pcov):

    Var=x*pcov[0,0]+pcov[1,1]+x*(pcov[0,1]+pcov[1,0])
    return Var
#---------------------------------------------------------------------------------    
    
#---------------------------------------------------------------------------------------------    
def FitBougherLine(theX,theY,theSigY):
    
    
    bad_bins=np.where(np.logical_or(np.isnan(theY),np.isnan(theSigY)))[0]
    
    if len(bad_bins)>0:
        print 'bad_bins',bad_bins
    
    good_bins=np.where(np.logical_and(~np.isnan(theY),~np.isnan(theSigY)))[0]
    
    theX=theX[good_bins]
    theY=theY[good_bins]
    theSigY=theSigY[good_bins]
    
    
    # range to return  the fit
    xfit=np.linspace(0,theX.max()*1.1,20)    
   
    
    # find a first initialisation approximation with polyfit
    theZ = np.polyfit(theX,theY, 1)
    
    
    # TEST IF  ERROR are null or negative (example simulation)
    if np.any(theSigY<=0):
        pol = np.poly1d(theZ)
        yfit=pol(xfit)
        errfit=np.zeros(len(xfit))
        
    else:
        # do the fit including the errors
        popt, pcov = curve_fit(bougline, theX, theY,p0=theZ,sigma=theSigY,absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
    
        #print "popt = ",popt,' pcov',pcov,' perr',perr
    
    
        #compute the chi-sq
        chi2=np.sum( ((bougline(theX, *popt) - theY) / theSigY)**2)
        #redchi2=(chi2)/(len(theY)-2)
    
        #chi2sum=(Yfit-np.array(theY))**2/np.array(theSigY)**2
        #chi2=np.average(chi2sum)*chi2sum.shape[0]/(chi2sum.shape[0]-3)
        #print 'chi2',chi2
    
    
        #p = np.poly1d(theZ)
        #yfit=p(xfit)
    
        pol = np.poly1d(popt)
        yfit=pol(xfit)
        errfit=np.sqrt(Varbougline(xfit,popt,pcov))
    
    return xfit,yfit,errfit
    
        
#------------------------------------------------------------------------------------------
def FitSpectraDataSimAttenuationRatioSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Mag=True):

    
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
    all_sed=np.zeros((len(the_filelist)+1,2+len(WL)))
    
    
   
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
            objectname=header["TARGET"]
            
            # extract smoothed SED
            sedsm,sedunit=GetSEDSmooth(objectname,Wwidth=Wwidth)
            
            
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
            
            # smoothed SED
            all_sed[numsel,0]=idx
            all_sed[numsel,1]=airmass
            all_sed[numsel,2:]=sedsm 
            
            
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    # selection 
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]  
    sel_sed=all_sed[good_indexes,:]
    
    #airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    
    
    plt.figure(figsize=(15,8))
    # loop on wavelength indexes
    for idx_wl in np.arange(2,len(WL)+2): 
        if WL[idx_wl-2]<WLMIN:
            continue
        if WL[idx_wl-2]>WLMAX:
            break
        colorVal = scalarMap.to_rgba(WL[idx_wl-2],alpha=1)
        #att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        
        if not Mag:
            plt.semilogy(sel_airmasses,sel_attenuation[:,idx_wl]/sel_sed[:,idx_wl],'o-',c=colorVal)
        else:
            plt.plot(sel_airmasses,2.5*np.log10(sel_attenuation[:,idx_wl]/sel_sed[:,idx_wl]),'o-',c=colorVal)
          
            
    
    plt.grid(b=True, which='major', color='k', linestyle='-',lw=1)
    plt.grid(b=True, which='minor', color='grey', linestyle='--',lw=0.5)
    
    plt.title(the_title)
    plt.xlabel("airmass")   
    if not Mag:
        plt.ylabel("intensity in data/ intensity of SED")  
    else:
        plt.ylabel(" 2.5 x log1°(intensity in data/ intensity of SED) (mag)")  
    plt.xlim(0.,sel_airmasses.max()*1.1)  
    
    
    plt.show() 
#----------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------
def FitSpectraDataSimAttenuationRatioSmoothBin(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Bwidth=20,Mag=True):
    """
    
    FitSpectraDataSimAttenuationRatioSmoothBin(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                                        FLAG_WL_CORRECTION,Flag_corr_wl=False,
                                        XMIN=0,XMAX=0,YMIN=0,YMAX=0,ZMIN=0,ZMAX=0,Wwidth=21,Bwidth=20,Mag=True)
    
    Fit the bouguer lines on a set of spectra given in the file_list and which are selected in the obs dataset
    
    task: show the plot of fit
    
    input: 
        the_filelist,
        the_obs,
        the_searchtag,
        wlshift,
        the_title,
        FLAG_WL_CORRECTION,
        Flag_corr_wl=False,
        XMIN=0,XMAX=0,
        YMIN=0,YMAX=0,
        ZMIN=0,ZMAX=0,
        Wwidth=21,
        Bwidth=20,
        Mag=True
    
    return:
    
    
    """

    
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
    
    ##########################################################################################
    # attenuation container 
    # attenuation rows: the file index, the airmass, the attenuation for each WL
    ##########################################################################################
    attenuation=np.zeros((len(the_filelist)+1,2+len(WL)))
    att_err=np.zeros((len(the_filelist)+1,2+len(WL)))
    all_sed=np.zeros((len(the_filelist)+1,2+len(WL)))
    
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
            objectname=header["TARGET"]
            
            # extract smoothed SED
            sedsm,sedunit=GetSEDSmooth(objectname,Wwidth=Wwidth)
            
            
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
            
            # attenuation in data
            attenuation[numsel,0]=idx
            attenuation[numsel,1]=airmass
            attenuation[numsel,2:]=fl_smooth  
            
            # error on attenuation
            att_err[numsel,2:]=errfl_smooth
            
            # smoothed SED
            all_sed[numsel,0]=idx
            all_sed[numsel,1]=airmass
            all_sed[numsel,2:]=sedsm 
      
    AIRMASS_MIN=attenuation[1:,1].min()
    AIRMASS_MAX=attenuation[1:,1].max()
    
    all_airmasses=attenuation[1:,1]
    all_imgidx=attenuation[1:,0]
    
    
    
    # selection where airmass are OK
    #---------------------------------
    good_indexes=np.where(attenuation[:,1]>0)[0]
    
    
    sel_attenuation=attenuation[good_indexes,:]
    sel_err=att_err[good_indexes,:]
    sel_airmasses=sel_attenuation[:,1]
    sel_imgidx=sel_attenuation[:,0]
    
    sel_sed=all_sed[good_indexes,:]
    
    

    
    #airmassmin_index=np.where(sel_airmasses==sel_airmasses.min())[0][0]
    #print 'airmass-min = ',sel_airmasses[airmassmin_index]
    
    # loop on wavelength bins
    #plt.figure(figsize=(15,4))
    #plt.plot(sel_imgidx,sel_airmasses,'o')
    #plt.show()
    
    ###################################################################################
    ################### Plot the figure ###############################################
    ###################################################################################
    
    # collections to return
    
    all_WL=[]
    all_Y=[]
    all_EY=[]
    
    
    
    
    plt.figure(figsize=(18,10))
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
        
        WLBins=WL[idx_startwl-2:idx_stopwl-2]
        
        # slice of  flux in wavelength bins
        FluxBin=sel_attenuation[:,idx_startwl:idx_stopwl]
        FluxBinSED=sel_sed[:,idx_startwl:idx_stopwl]
        
        FluxBinErr=sel_err[:,idx_startwl:idx_stopwl]
       
                 
        # get the average of flux in that big wl bin
        
        if len(FluxBin>0):
        
            FluxAver=np.average(FluxBin,axis=1)
            FluxAverSED=np.average(FluxBinSED,axis=1)
            FluxAverErr=np.average(FluxBinErr,axis=1)
        
            Y0=FluxAver/FluxAverSED
            Y1=Y0-FluxAverErr/FluxAverSED
            Y2=Y0+FluxAverErr/FluxAverSED
        
            # get the attenuation for the airmass-min
            #att_airmassmin=sel_attenuation[airmassmin_index,idx_wl]
        
            if not Mag:        
                plt.fill_between(sel_airmasses,y1=Y1,y2=Y2, where=Y1>0 ,color='grey', alpha=0.3 )        
                plt.yscale( "log" )
        
                # plot the attenuation wrt airmass
                plt.semilogy(sel_airmasses,Y0,'o-',c=colorVal,label=thelabel)
            else:
            
                newY0=2.5*np.log10(Y0)
                newY1=2.5*np.log10(Y1)
                newY2=2.5*np.log10(Y2)
            
                plt.fill_between(sel_airmasses,y1=newY1,y2=newY2, where=Y1>0 ,color='grey', alpha=0.3 )        
            
                # plot the attenuation wrt airmass
                plt.plot(sel_airmasses,newY0,'o-',c=colorVal,label=thelabel)
            
                Xfit,Yfit,YFitErr=FitBougherLine(sel_airmasses,newY0,theSigY=(newY2-newY1)/2.)
                plt.plot(Xfit,Yfit,'-',c=colorVal)
                plt.plot(Xfit,Yfit+YFitErr,':',c=colorVal)
                plt.plot(Xfit,Yfit-YFitErr,':',c=colorVal)
            
                all_WL.append(np.average(WLBins)) # average wavelength in that bin
                all_Y.append(Yfit[0])            # Y for first airmass z=0
                all_EY.append(YFitErr[0])          # EY extracpolated for that airmass z=0
            
    else:
        print 'FitSpectraDataSimAttenuationRatioSmoothBin : skip bins:',WL[idx_startwl-2],'-',WL[idx_stopwl-2]
        print 'FluxBin=',FluxBin
        
    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.grid(b=True, which='minor', color='red', linestyle='--')
    plt.title(the_title)
    plt.xlabel("airmass")   
    if not Mag:  
        plt.ylabel("intensity in data / intensity in SED") 
    else:
        plt.ylabel("intensity in data / intensity in SED (mag)") 
    plt.legend(loc='right', prop={'size':10})  
    
    if XMAX==0:
        plt.xlim(0.,AIRMASS_MAX*1.3)
    else:
        plt.xlim(0.,XMAX)
    plt.show() 
    
    return np.array(all_WL),np.array(all_Y),np.array(all_EY)
    
#-------------------------------------------------------------------------------------        
def PlotOpticalThroughput(wl,thrpt,err,title):
    
    plt.figure(figsize=(10,6))
    plt.title(title)
    plt.errorbar(wl,thrpt,yerr=err,fmt='o',color='blue',ecolor='red')
    
    plt.xlabel('$\lambda$ (nm)' )
    plt.ylabel('total throughput (mag)')
    #plt.grid(b=True, which='major', color='black', linestyle='-')
    #plt.grid(b=True, which='minor', color='red', linestyle='--')
    plt.grid(b=True, which='both')
    plt.show()
    
#-------------------------------------------------------------------------------------
#   EQUIVALENT WIDTH
#----------------------------------------------------------------------------------------------
def ShowEquivalentWidth(wl,spec,wl1,wl2,wl3,wl4,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    ##--------------
    ## Figure 1
    ##------------
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'r-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'r-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,'o')
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=1)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    #--------------
    # Figure 2
    #-----------
    axarr[1].plot(wl_cut,ratio,'b-')
    axarr[1].plot([wl2,wl2],[0,1.2],'r-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'r-.',lw=2)
    axarr[1].grid(True)
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width
#-----------------------------------------------------------------------------------------------------
def ShowEquivalentWidth2(wl,spec,wl1,wl2,wl3,wl4,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    #############
    ## Figure 1
    ############
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'r-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'r-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,'o')
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=1)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    #-------------
    # Figure 2
    #-----------
    axarr[1].plot(wl_cut,ratio,color='blue')
    axarr[1].plot([wl2,wl2],[0,1.2],'r-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'r-.',lw=2)
    axarr[1].plot([wl1,wl4],[1,1],'g--',lw=1)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width    

#--------------------------------------------------------------------------------------------------
def ShowEquivalentWidthwthStatErrTOREMOVE(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3,thelabel='ShowEquivalentWidthwthStatErr'):
    """
    ******************************************************************************
    ShowEquivalentWidthwthStatErr : show how the equivalent width must be computed
    
    - with errors
    - with non linear method
    
    *********************************************************************************
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    #############
    ### Figure 1
    #############
    plt.figure()
    
    plt.plot(wl_cut,spec_cut,'b-')
    plt.errorbar(wl_cut,spec_cut,yerr=spec_cut_err,color='red',fmt='.')
    
    # vertical bars
    plt.plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    plt.plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    
    # continuum fit
    #------------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))

    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    y_cont_err=specerr[continuum_indexes]
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
   
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory    
    z_cont_fit=popt_p
      

    pol_cont_fit=np.poly1d(z_cont_fit)
    
    # fitted curve with its error
    #---------------------------------
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    errorfill(fit_line_x,fit_line_y,fit_line_y_err, color='grey',ax=plt)
    plt.errorbar(x_cont,y_cont,yerr=y_cont_err,fmt='.',color='blue')
    plt.plot(fit_line_x,fit_line_y,'g--',lw=1)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' ADU per second')
    
    plt.grid(True)
    plt.title(thelabel)
    
    # compute the ratio spectrum/continuum
    #-------------------------------------
    full_continum=pol_cont_fit(wl_cut)    
    
    full_continum_err= []
    for wl in wl_cut:
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    
    
    
    ###################
    ### Second figure
    #################
    plt.figure()
    plt.plot(wl_cut,ratio,lw=2,color='blue')
    plt.errorbar(wl_cut,ratio,yerr=ratio_err,fmt='.',lw=2,color='red')
    plt.plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    plt.plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    plt.grid(True)
    plt.ylim(0.8*ratio.min(),1.2*ratio.max())
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' no unit')
    plt.title(thelabel)
    
    # Compute now the equavalent width
    #--------------------------------
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin in wavelength

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width and its error
    #-------------------------------------------------
    absorption_band=wl_bin_size*(1-ratio)
    absorption_band_error=wl_bin_size*ratio_err
    
    equivalent_width= absorption_band.sum()
    
    # quadratic sum of errors for each wl bin
    equivalend_width_error=np.sqrt((absorption_band_error*absorption_band_error).sum() )
    
    return equivalent_width,equivalend_width_error
#----------------------------------------------------------------------------------------------

def ShowEquivalentWidthNonLinearTOREMOVE(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,thelabel='ShowEquivalentWidthNonLinear'):
    """
    ShowEquivalentWidthNonLinear : show how the equivalent width must be computed
    Do not use stats, for simulation by exammple
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
   
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    plt.figure()
    plt.plot(wl_cut,spec_cut,'r-')
    plt.plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    plt.plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg,rcond=2.0e-16*len(x_cont))
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    plt.plot(x_cont,y_cont,marker='.',color='blue',lw=0)
    plt.plot(fit_line_x,fit_line_y,'g--',lw=2)
    
    plt.grid(True)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' ADU per second')
    plt.title(thelabel)
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    plt.figure()
    plt.plot(wl_cut,ratio,'b-')
    plt.plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    plt.plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    plt.grid(True)
    plt.xlabel(' wavelength (nm)')
    plt.ylabel(' No unit')
    plt.ylim(0.8*ratio.min(),1.2*ratio.max())
    plt.title(thelabel)
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    return equivalent_width
#---------------------------------------------------------------------------------------------
def ShowEquivalentWidthNonLinear2(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    ################
    ## Figure 1
    #################
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,marker='.',color='red')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,marker='.',color='blue',lw=0)
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=2)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    axarr[0].set_ylabel('ADU per second')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    external_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    
    
    ############
    # Figure 2
    ###########
    
    axarr[1].plot(wl_cut,ratio,marker='.',color='red')
    axarr[1].plot(wl_cut[external_indexes],ratio[external_indexes],marker='.',color='blue',lw=0)
    
    axarr[1].plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    axarr[1].set_ylabel('No unit')
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width    

#---------------------------------------------------------------------------------------------    

def ShowEquivalentWidthNonLinearwthStatErr(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ***********************************************************************
    ShowEquivalentWidth2NonLinearwthStatErr : 
    
    Fit equivalent width:
    - Non linear
    - with errors
    
    ************************************************************************
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    order=ndeg+1
    
    #######################
    ## Figure 1
    #########################
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    # plot points
    #axarr[0].plot(wl_cut,spec_cut,'b-')
    axarr[0].errorbar(wl_cut,spec_cut,yerr=spec_cut_err,color='red',fmt='.',lw=1)
    axarr[0].plot(wl_cut,spec_cut,'r-',lw=1)
    # plot vertical bars
    axarr[0].plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    # continuum fit
    #----------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    

    # error for continum
    y_cont_err=specerr[continuum_indexes]
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory
    
    z_cont_fit=popt_p
      
   
    pol_cont_fit=np.poly1d(z_cont_fit)
    

    # compute the fit and propagate the error
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    
    errorfill(fit_line_x,fit_line_y,fit_line_y_err,color='grey',ax=axarr[0])
    axarr[0].errorbar(x_cont,y_cont,yerr=y_cont_err,fmt='.',color='blue')
    
       
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    axarr[0].set_ylabel('ADU per second')
    
    # compute the ratio spectrum/continuum and its error
    # -------------------------------------
    full_continum=pol_cont_fit(wl_cut)  
    
    full_continum_err= []
    external_indexes = []
    idx=0
    for wl in wl_cut:
        if wl<wl2 or wl>wl3:
            external_indexes.append(idx)
        idx+=1
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    
    
    
    
    ##################
    # Figure 2
    ###################
    
    axarr[1].plot(wl_cut,ratio,lw=1,color='red')
    axarr[1].errorbar(wl_cut,ratio,yerr=ratio_err,fmt='.',lw=2,color='red')
    axarr[1].errorbar(wl_cut[external_indexes],ratio[external_indexes],yerr=ratio_err[external_indexes],fmt='.',lw=0,color='blue')
    
    
    axarr[1].plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    axarr[1].set_xlabel('$\lambda$ (nm)')
    axarr[1].set_ylabel('ratio : no unit')
    
    
    #compute the equivalent width
    #-----------------------------
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width and its error (units nm because wl in nm)
    # ----------------------------------------------
    absorption_band=wl_bin_size*(1-ratio)
    absorption_band_error=wl_bin_size*ratio_err
    equivalent_width= absorption_band.sum()
    equivalent_width_err=np.sqrt((absorption_band_error**2).sum())
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width,equivalent_width_err
#--------------------------------------------------------------------------------------------
def ShowEquivalentWidthNonLinear(wl,spec,wl1,wl2,wl3,wl4,ndeg=3,label='absortion line',fsize=(12,4)):
    """
    ShowEquivalentWidth : show how the equivalent width must be computed
    """
    
    f, axarr = plt.subplots(1,2,figsize=fsize)
    
    ################
    ## Figure 1
    #################
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
    
    axarr[0].plot(wl_cut,spec_cut,marker='.',color='red')
    axarr[0].plot([wl2,wl2],[ymin,ymax],'k-.',lw=2)
    axarr[0].plot([wl3,wl3],[ymin,ymax],'k-.',lw=2)
    
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    axarr[0].plot(x_cont,y_cont,marker='.',color='blue',lw=0)
    axarr[0].plot(fit_line_x,fit_line_y,'g--',lw=2)
    
    axarr[0].grid(True)
    axarr[0].set_xlabel('$\lambda$ (nm)')
    axarr[0].set_ylabel('ADU per second')
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    
    external_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    
    
    ############
    # Figure 2
    ###########
    
    axarr[1].plot(wl_cut,ratio,marker='.',color='red')
    axarr[1].plot(wl_cut[external_indexes],ratio[external_indexes],marker='.',color='blue',lw=0)
    
    axarr[1].plot([wl2,wl2],[0,1.2],'k-.',lw=2)
    axarr[1].plot([wl3,wl3],[0,1.2],'k-.',lw=2)
    axarr[1].grid(True)
    axarr[1].set_ylim(0.8*ratio.min(),1.2*ratio.max())
    
    axarr[1].set_xlabel('$\lambda$ (nm)')
    axarr[1].set_ylabel('No unit')
    
    NBBins=len(wl_cut)
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin

    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                       
    # calculation of equivalent width
    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()
    
    
    title = 'Equivalent width computation for {}'.format(label)
    f.suptitle(title)
    
    return equivalent_width


#--------------------------------------------------------------------------------------------------
def ComputeEquivalentWidth(wl,spec,wl1,wl2,wl3,wl4):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,1)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    

    # compute bin size in the band
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                                  
    
    # calculation of equivalent width    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()    
    
    return equivalent_width
#--------------------------------------------------------------------------------------------- 
def ComputeEquivalentWidthNonLinear(wl,spec,wl1,wl2,wl3,wl4,ndeg=3):
    """
    ComputeEquivalentWidth : compute the equivalent width must be computed
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
        
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg,rcond=2.0e-16*len(x_cont))
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    
    
    # compute the ratio spectrum/continuum
    full_continum=pol_cont_fit(wl_cut)    
    ratio=spec_cut/full_continum
    

    # compute bin size in the band
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
                                  
    
    # calculation of equivalent width    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum()    
    
    return equivalent_width
#----------------------------------------------------------------------------------------------
def ComputeEquivalentWidthNonLinearwthStatErr(wl,spec,specerr,wl1,wl2,wl3,wl4,ndeg=3):
    """
    ************************************************************************************
    ComputeEquivalentWidthNonLinearwthStatErr : compute the equivalent width must be computed
    
    *************************************************************************************
    """
    selected_indexes=np.where(np.logical_and(wl>=wl1,wl<=wl4))
    
    # extract
    wl_cut=wl[selected_indexes]
    spec_cut=spec[selected_indexes]
    spec_cut_err=specerr[selected_indexes]
    
    
    ymin=spec_cut.min()
    ymax=spec_cut.max()
     
    # continuum fit
    #---------------
    continuum_indexes=np.where(np.logical_or(np.logical_and(wl>=wl1,wl<=wl2),np.logical_and(wl>=wl3,wl<wl4)))
    x_cont=wl[continuum_indexes]
    y_cont=spec[continuum_indexes]
    y_cont_err=specerr[continuum_indexes]
    
    y_w=1./y_cont_err
    y_w[np.where(y_cont==0)]=0. # erase the empty bins
   
    
    popt_p , pcov_p= np.polyfit(x_cont, y_cont,ndeg,w=y_w,full=False,cov=True,rcond=2.0e-16*len(x_cont)) #rcond mandatory    
    z_cont_fit=popt_p
    
    
    
    z_cont_fit=np.polyfit(x_cont, y_cont,ndeg)
        
    pol_cont_fit=np.poly1d(z_cont_fit)
    
    fit_line_x=np.linspace(wl1,wl4,50)
    fit_line_y=pol_cont_fit(fit_line_x)
    fit_line_y_err = []
    for thex in fit_line_x:
        dfdx = [ thex**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        fit_line_y_err.append(propagated_error)
    fit_line_y_err=np.array(fit_line_y_err)
    
    
    # compute the ratio spectrum/continuum and its error
    full_continum=pol_cont_fit(wl_cut)    
    
    full_continum_err= []
    for wl in wl_cut:
        dfdx = [ wl**thepow for thepow in np.arange(ndeg,-1,-1)]
        dfdx=np.array(dfdx)
        propagated_error=np.dot(dfdx.T,np.dot(pcov_p,dfdx))
        full_continum_err.append(propagated_error)
    full_continum_err=np.array(full_continum_err)
    
    
    ratio=spec_cut/full_continum
    # error not correlated    
    ratio_err=ratio*np.sqrt( (spec_cut_err/spec_cut)**2+ (full_continum_err/full_continum)**2)
    
    

    # compute bin size in the band
    wl_shift_right=np.roll(wl_cut,1)
    wl_shift_left=np.roll(wl_cut,-1)
    wl_bin_size=(wl_shift_left-wl_shift_right)/2. # size of each bin    
    outside_band_indexes=np.where(np.logical_or(wl_cut<wl2,wl_cut>wl3))
    wl_bin_size[outside_band_indexes]=0  # erase bin width outside the band
    
    
    
    # calculation of equivalent width    
    absorption_band=wl_bin_size*(1-ratio)
    equivalent_width= absorption_band.sum() 
    
    # return equavalent width error
    absorption_band_error=wl_bin_size*ratio_err
    # quadratic sum of errors for each wl bin
    equivalend_width_error=np.sqrt((absorption_band_error*absorption_band_error).sum() )
    
    return equivalent_width,equivalend_width_error
#----------------------------------------------------------------------------------------------------- 
def ShowAllEquivalentWidth(all_wl,all_spec,all_filt,wl1,wl2,wl3,wl4,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-L)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidth(wl,spectrum,wl1,wl2,wl3,wl4,label=newlabel,fsize=(9,3))
#-----------------------------------------------------------------------------------------------        
def ShowAllEquivalentWidthNonLinear(all_wl,all_spec,all_filt,wl1,wl2,wl3,wl4,ndeg=3,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-N)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidthNonLinear(wl,spectrum,wl1,wl2,wl3,wl4,ndeg=ndeg,label=newlabel,fsize=(9,3))
        
        
#-----------------------------------------------------------------------------------------------        
def ShowAllEquivalentWidthNonLinearwthStatErr(all_wl,all_spec,all_spec_err,all_filt,wl1,wl2,wl3,wl4,ndeg=3,label='absorption line'):
        
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        err=all_spec_err[index]
        
        newlabel = label+ ' spec {} with disp {} (fit BG-N+ stat err)'.format(index,all_filt[index]) 
        
        ShowEquivalentWidthNonLinearwthStatErr(wl,spectrum,err,wl1,wl2,wl3,wl4,ndeg=ndeg,label=newlabel,fsize=(9,3))
#----------------------------------------------------------------------------------------------------        
        
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidth(all_wl,all_spec,wl1,wl2,wl3,wl4):
    
    EQW_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        eqw=ComputeEquivalentWidth(wl,spectrum,wl1,wl2,wl3,wl4)
        EQW_coll.append(eqw)
        
    return np.array(EQW_coll)
#--------------------------------------------------------------------------------------------
        
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidthNonLinear(all_wl,all_spec,wl1,wl2,wl3,wl4,ndeg=3):
    
    EQW_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        eqw=ComputeEquivalentWidthNonLinear(wl,spectrum,wl1,wl2,wl3,wl4,ndeg=ndeg)
        EQW_coll.append(eqw)
        
    return np.array(EQW_coll)
#--------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------
def ComputeAllEquivalentWidthNonLinearwthStatErr(all_wl,all_spec,all_spec_err,wl1,wl2,wl3,wl4,ndeg=3):
    
    EQW_coll = []
    EQWErr_coll = []
    
    NBSPECTRA=len(all_spec)
    
    for index in np.arange(0,NBSPECTRA):        
        spectrum=all_spec[index]
        wl=all_wl[index]
        err=all_spec_err[index]
        
        eqw,eqw_err=ComputeEquivalentWidthNonLinearwthStatErr(wl,spectrum,err,wl1,wl2,wl3,wl4,ndeg=ndeg)
        EQW_coll.append(eqw)
        EQWErr_coll.append(eqw_err)
        
    EQW_coll=np.array(EQW_coll)    
    EQWErr_coll=np.array(EQWErr_coll)    
    return EQW_coll, EQWErr_coll
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
def PlotEquivalentWidthVsAirMass(all_eqw_width,all_eqw_width_sim,all_am,all_filt,tagabsline,dir_top_img,figname,spec_err=None,EQWMIN=2.,EQWMAX=5.):
    """
    """
    am0 = []
    eqw0 = []
    err0 = []
    
    am1 = []
    eqw1 = []
    err1 = []
    
    am2 = []
    eqw2 = []
    err2 = []
    
    am3 = []
    eqw3 = []
    err3 = []
    
    am4 = []
    eqw4 = []
    err4 = []
    
    
    for index,eqw in np.ndenumerate(all_eqw_width):
        idx=index[0]
        grating_name=all_filt[idx]
        am=all_am[idx]
        err=0
        #if spec_err != None:
        if spec_err.any():
            err=spec_err[idx]
        
        
        if re.search(Disp_names[0],grating_name):
            am0.append(am)
            eqw0.append(eqw)
            err0.append(err)
        elif re.search(Disp_names[1],grating_name):
            am1.append(am)
            eqw1.append(eqw)
            err1.append(err)
        elif re.search(Disp_names[2],grating_name):
            am2.append(am)
            eqw2.append(eqw)
            err2.append(err)
        elif re.search(Disp_names[3],grating_name):
            am3.append(am)
            eqw3.append(eqw)
            err3.append(err)
        elif re.search(Disp_names[4],grating_name):
            am4.append(am)
            eqw4.append(eqw)
            err4.append(err)
        else:
            print 'disperser ',grating_name,' not found'
            
    fig=plt.figure(figsize=(20,8))

    ax=fig.add_subplot(1,1,1)
    
    ax.errorbar(am0,eqw0,err0, fmt='o',color='red')
    ax.plot(am0,eqw0,marker='o',color='red',lw=1,label=Disp_names[0])
    ax.errorbar(am1,eqw1,err1, fmt='o',color='blue')
    ax.plot(am1,eqw1,marker='o',color='blue',lw=1,label=Disp_names[1])
    ax.errorbar(am2,eqw2,err2, fmt='o',color='green')
    ax.plot(am2,eqw2,marker='o',color='green',lw=1,label=Disp_names[2])
    ax.errorbar(am3,eqw3,err3, fmt='o',color='cyan')
    ax.plot(am3,eqw3,marker='o',color='cyan',lw=1,label=Disp_names[3])
    ax.errorbar(am4,eqw4,err4, fmt='o',color='magenta')
    ax.plot(am4,eqw4,marker='o',color='magenta',lw=1,label=Disp_names[4])
    
    ax.plot(all_am,all_eqw_width_sim,marker='o',color='black',lw=1,label='Sim')

    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())   
    #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='grey', linewidth=1.0)
    #ax.grid(b=True, which='minor', color='grey', linewidth=0.5)

    ax.set_ylabel('Equivalent width (nm)')
    ax.set_xlabel('AirMass')
    ax.set_ylim(EQWMIN,EQWMAX)


    title='Equivalent Width for {} vs airmass'.format(tagabsline)

    plt.title(title)
    plt.legend(loc='best')

    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)
 #--------------------------------------------------------------------------------------------  

def PlotEquivalentWidthVsTime(all_eqw_width,all_eqw_width_sim,all_am,all_dt,all_filt,tagabsline,dir_top_img,figname,spec_err=None,EQWMIN=2.,EQWMAX=5.):
    """
    """
    am0 = []
    eqw0 = []
    tim0 = []
    err0 = []
    
    am1 = []
    eqw1 = []
    tim1=[]
    err1 = []
    
    am2 = []
    eqw2 = []
    tim2=[]
    err2 = []
    
    am3 = []
    eqw3 = []
    tim3=[]
    err3 = []
    
    am4 = []
    eqw4 = []
    tim4=[]
    err4 = []
    
    NDATA=len(all_eqw_width)
    
    date_range = all_dt[NDATA-1] - all_dt[0]
    
    
    for index,eqw in np.ndenumerate(all_eqw_width):
        idx=index[0]
        grating_name=all_filt[idx]
        am=all_am[idx]
        err=0
        #if spec_err != None:
        if spec_err.any():
            err=spec_err[idx]
        
        
        
        if re.search(Disp_names[0],grating_name):
            am0.append(am)
            eqw0.append(eqw)
            tim0.append(all_dt[idx])
            err0.append(err)
        elif re.search(Disp_names[1],grating_name):
            am1.append(am)
            eqw1.append(eqw)
            tim1.append(all_dt[idx])
            err1.append(err)
        elif re.search(Disp_names[2],grating_name):
            am2.append(am)
            eqw2.append(eqw)
            tim2.append(all_dt[idx])
            err2.append(err)
        elif re.search(Disp_names[3],grating_name):
            am3.append(am)
            eqw3.append(eqw)
            tim3.append(all_dt[idx])
            err3.append(err)
        elif re.search(Disp_names[4],grating_name):
            am4.append(am)
            eqw4.append(eqw)
            tim4.append(all_dt[idx])
            err4.append(err)
        else:
            print 'disperser ',grating_name,' not found'
            
    fig=plt.figure(figsize=(20,8))

    ax=fig.add_subplot(1,1,1)

    ax.errorbar(tim0,eqw0,err0, fmt='o',color='red')
    ax.plot_date(tim0,eqw0,'-',color='red',lw=1,label=Disp_names[0])
    ax.errorbar(tim1,eqw1,err1, fmt='o',color='blue')
    ax.plot_date(tim1,eqw1,'-',color='blue',lw=1,label=Disp_names[1])
    ax.errorbar(tim2,eqw2,err2, fmt='o',color='green')
    ax.plot_date(tim2,eqw2,'-',color='green',lw=1,label=Disp_names[2])
    ax.errorbar(tim3,eqw3,err3, fmt='o',color='cyan')
    ax.plot_date(tim3,eqw3,'-',color='cyan',lw=1,label=Disp_names[3])
    ax.errorbar(tim4,eqw4,err4, fmt='o',color='magenta')
    ax.plot_date(tim4,eqw4,'-',color='magenta',lw=1,label=Disp_names[4])
    
    ax.plot(all_dt,all_eqw_width_sim,'o',color='black')
    ax.plot_date(all_dt,all_eqw_width_sim,'-',color='black',lw=1,label='Sim')
    
    
    date_range = all_dt[NDATA-1] - all_dt[0]

    if date_range > datetime.timedelta(days = 1):
        ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(1,32), interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.get_xaxis().set_minor_locator(mdates.HourLocator(byhour=range(0,24,2)))
        #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,2)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_xaxis().set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,5)))
    
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='k', linewidth=2.0)
    ax.grid(b=True, which='minor', color='grey', linewidth=0.5)
    ax.set_ylabel('Equivalent width (nm)')
    ax.set_xlabel('time')
#    ax.set_ylim(0.,5.)
    ax.set_ylim(EQWMIN,EQWMAX)



    title='Equivalent Width for {} vs time'.format(tagabsline)

    plt.title(title)
    plt.legend(loc='best')
    
    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)
#-----------------------------------------------------------------------------------------------------
    
#--------------------------------------------------------------------------------------------
def PlotEquivalentWidthRatioVsAirMass(all_eqw_widthratio,all_eqw_widthratio_sim,all_am,all_filt,tagabsline,dir_top_img,figname,ratio_err=None,RATIOMIN=2.,RATIOMAX=5.):
    """
    """
    am0 = []
    eqwr0 = []
    err0 = []
    
    am1 = []
    eqwr1 = []
    err1 = []
    
    am2 = []
    eqwr2 = []
    err2 = []
    
    am3 = []
    eqwr3 = []
    err3 = []
    
    am4 = []
    eqwr4 = []
    err4 = []
    
    
    for index,eqwr in np.ndenumerate(all_eqw_widthratio):
        idx=index[0]
        grating_name=all_filt[idx]
        am=all_am[idx]
        err=0
        
        if ratio_err.any():
            err=ratio_err[idx]
        #if ratio_err != None:
        #    err=ratio_err[idx]
        
        
        if re.search(Disp_names[0],grating_name):
            am0.append(am)
            eqwr0.append(eqwr)
            err0.append(err)
        elif re.search(Disp_names[1],grating_name):
            am1.append(am)
            eqwr1.append(eqwr)
            err1.append(err)
        elif re.search(Disp_names[2],grating_name):
            am2.append(am)
            eqwr2.append(eqwr)
            err2.append(err)
        elif re.search(Disp_names[3],grating_name):
            am3.append(am)
            eqwr3.append(eqwr)
            err3.append(err)
        elif re.search(Disp_names[4],grating_name):
            am4.append(am)
            eqwr4.append(eqwr)
            err4.append(err)
        else:
            print 'disperser ',grating_name,' not found'
            
    fig=plt.figure(figsize=(20,8))

    ax=fig.add_subplot(1,1,1)
    
    ax.errorbar(am0,eqwr0,err0, fmt='o',color='red')
    ax.plot(am0,eqwr0,marker='o',color='red',lw=1,label=Disp_names[0])
    ax.errorbar(am1,eqwr1,err1, fmt='o',color='blue')
    ax.plot(am1,eqwr1,marker='o',color='blue',lw=1,label=Disp_names[1])
    ax.errorbar(am2,eqwr2,err2, fmt='o',color='green')
    ax.plot(am2,eqwr2,marker='o',color='green',lw=1,label=Disp_names[2])
    ax.errorbar(am3,eqwr3,err3, fmt='o',color='cyan')
    ax.plot(am3,eqwr3,marker='o',color='cyan',lw=1,label=Disp_names[3])
    ax.errorbar(am4,eqwr4,err4, fmt='o',color='magenta')
    ax.plot(am4,eqwr4,marker='o',color='magenta',lw=1,label=Disp_names[4])
    
    ax.plot(all_am,all_eqw_widthratio_sim,marker='o',color='black',lw=1,label='Sim')

    #ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())   
    #ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())

    ax.grid(b=True, which='major', color='grey', linewidth=1.0)
    #ax.grid(b=True, which='minor', color='grey', linewidth=0.5)

    ax.set_ylabel('Equivalent width ratio')
    ax.set_xlabel('AirMass')

    ax.set_ylim(RATIOMIN,RATIOMAX)

    title='Equivalent Width Ratio for {} vs airmass'.format(tagabsline)

    plt.title(title)
    plt.legend(loc='best')

    figfilename=os.path.join(dir_top_img,figname)
    fig.savefig(figfilename)    
#----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------- 
def PlotEQWDataSimSmooth(the_filelist,the_obs,the_searchtag,wlshift,the_title,
                         FLAG_WL_CORRECTION,Flag_corr_wl=False,Wwidth=21,
                         wl1=880,wl2=900,wl3=1000,wl4=1020,mod='lin',deg=2):

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
            header=hdu[0].header
            filt=header["FILTER2"]
            
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
            
            
            thetitle=the_title+' '+str(idx)
            
            colorVal = scalarMap.to_rgba(num,alpha=1)
            
            if mod=='lin' or mod=='linear':
                ShowEquivalentWidth2(WL,fl_smooth,wl1,wl2,wl3,wl4,label=thetitle)
            else:
                ShowEquivalentWidthNonLinear(WL,fl_smooth,wl1,wl2,wl3,wl4,ndeg=deg,label=thetitle)
            
            #plt.fill_between(WL,y1=fl_smooth-1.96*errfl_smooth,y2=fl_smooth+1.96*errfl_smooth,facecolor='grey',alpha=0.5)
            
            
    plt.grid()    
    plt.title(the_title)
    plt.xlabel("$\lambda$ (nm)")   
    plt.ylabel("smoothed spectra")   
    #plt.legend()