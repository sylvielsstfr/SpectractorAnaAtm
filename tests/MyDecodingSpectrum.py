#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 10:49:32 2018

@author: dagoret
"""
# config package
import matplotlib.pyplot as plt
import sys
import os
from astropy.io import fits

#  config Spectractor
PATH_SPECTRACTOR='../../Spectractor'
sys.path.append(PATH_SPECTRACTOR)
#rom spectractor import *


# input fits file
infile='test/reduc_20170531_074_spectrum.fits'
infile_fullpath=os.path.join(PATH_SPECTRACTOR,infile)

# read file
hdul = fits.open(infile_fullpath)


# info
hdul.info()
print hdul[0].header

# Extract data
image_data = hdul[0].data


# Decode data
wl=image_data[0,:]
spec=image_data[1,:]
err=image_data[2,:]


# Plot Spectrum
plt.figure(figsize=(15,10))
plt.errorbar(wl,spec,yerr=err,color='blue')
plt.grid()
plt.show()

