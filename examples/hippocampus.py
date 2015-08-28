# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:32:35 2014

@author: martin
"""


import matplotlib.pyplot as mp
import matplotlib

from nest import *
import nest.voltage_trace
import nest.raster_plot
import numpy as np

import pamutils.pam2nest as pam2nest
import pamutils.nest_vis as nest_vis

import cProfile

EXPORT_PATH = './'
DELAY_FACTOR = 4.36


def analyseNetwork():
    m = pam2nest.import_connections(EXPORT_PATH + 'hippocampus_full.zip')
    print("Loading done")
    nest_vis.printNeuronGroups(m)
    nest_vis.printConnections(m)
    
    w_means = [9., 9., 9., 10., 4.0, 5.0, 4.0, 4.0, 4.0]
    w_sds = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    d_means = [1., 1., 1., 4., 4., 4., 4., 1., 1.]
    d_sds = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    ngs = pam2nest.CreateNetwork(m, 'izhikevich',
                                 w_means, w_sds,
                                 d_means, d_sds)
    
    print(len(ngs))
    
    noise         = Create("poisson_generator", 50)
    
    voltmeter       = Create("voltmeter", 2)
    espikes         = Create("spike_detector")
    
    SetStatus(noise, [{'start': 100., 'stop': 120., 'rate': 100.0}])

    Connect(noise, ngs[3][:50], conn_spec='one_to_one' , syn_spec = {'model': 'static_synapse', 'weight': 2000., 'delay': 1.})

    ConvergentConnect(ngs[0] + ngs[1] + ngs[2] + ngs[3] + ngs[4] + ngs[5],espikes)
    
    Simulate(1000.0)
        
    nest.raster_plot.from_device(espikes, hist=False)
    nest.raster_plot.show()    

   
if __name__ == "__main__":
    cProfile.run('analyseNetwork()')
    
    
