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

#  config Spectractor
PATH_SPECTRACTOR='../../Spectractor'
sys.path.append(PATH_SPECTRACTOR)
from spectractor import *


# input fits file
infile='test/reduc_20170531_074_spectrum.fits'
infile_fullpath=os.path.join(PATH_SPECTRACTOR,infile)


# Show spectrum
s = Spectrum(infile_fullpath)
s.plot_spectrum() 