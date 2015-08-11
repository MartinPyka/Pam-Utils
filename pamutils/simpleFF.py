# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:32:35 2014

@author: martin
"""

import nest
import matplotlib.pyplot as mp
import matplotlib

import numpy as np

import zipfile

import pamutils.pam2nest as pam2nest
import pamutils.nest_vis as nest_vis
import pamutils.network


EXPORT_PATH = '/home/rebekka/Projekte/simpleFF_Data/'
DELAY_FACTOR = 4.36

def loadSimpleFF():
    filename = 'simpleFF'    
    weights = [5.]
    w_sd = [1.]
    delay = [4.36]
    d_sd = [1.]
    
    nest.ResetKernel()
    
    net = pamutils.network.Network(EXPORT_PATH + filename + '.zip')
    net.createNetwork(weights, w_sd, delay, d_sd, 
                      output_prefix = EXPORT_PATH + filename) 
    net.stimulusCue(200, 20, 1)
    net.exportSpikes(EXPORT_PATH + filename + '.csv', [0,200])  

    # Exporting the two csv files as one zip 
    zf = zipfile.ZipFile(filename + '_Blender.zip', mode = 'w')
    zf.write(filename + '.csv')
    for i in range(0, len(net.m['connections'])): 
        zf.write(filename + '_d' + str(i) + '.csv')
    
    zf.close()
    
if __name__ == "__main__":
    loadSimpleFF()
    mp.show()
    #analyseNetwork()
    #analyseDelays_FullHippo()    
    #analyseConns_hippocampus()
    #analyseConns_simple()
    
    
