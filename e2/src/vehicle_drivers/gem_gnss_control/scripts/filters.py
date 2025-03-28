#!/usr/bin/env python3

#================================================================
# File name: filters.py                                                                  
# Description: class to filter noise out of speed controls                                                                
# Author: Hang Cui, John Pohovey
# Email: hangcui3@illinois.edu, jpohov2@illinois.edu                                                                   
# Date created: 08/02/2021                                                                
# Date last modified: 03/14/2025                                                          
# Version: 1.0                                                                                                                               
# Python version: 3.8                                                             
#================================================================

from __future__ import print_function

# Python Headers
import os 
import csv
import time
import math
import numpy as np
from numpy import linalg as la
import scipy.signal as signal


class OnlineFilter(object):

    def __init__(self, cutoff, fs, order):
        
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Get the filter coefficients 
        self.b, self.a = signal.butter(order, normal_cutoff, btype='low', analog=False)

        # Initialize
        self.z = signal.lfilter_zi(self.b, self.a)
    
    def get_data(self, data):
        filted, self.z = signal.lfilter(self.b, self.a, [data], zi=self.z)
        return filted
