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


EXPORT_PATH = '~/PAM/results/'
DELAY_FACTOR = 4.36


def analyseNetwork():
    m = pam2nest.import_connections(EXPORT_PATH + 'hippocampus_full.zip')
    
    #nest_vis.plotDelayHistograms(m)
    nest_vis.printNeuronGroups(m)
    nest_vis.printConnections(m)
    
    #nest_vis.plotConnectionDelayHistogram(
    #    [m['connections'][0][3],
    #     m['connections'][0][4],
    #     m['connections'][0][5],
    #     m['connections'][0][6]
    #    ],
    #    m, range(0, 1),
    #    20
    #    )

    #matrix = nest_vis.getConnectionMatrix(m, 2)
    #mp.figure()
    #mp.imshow(matrix)
    
        
    #mp.show() 
    weights = [9., 9., 9., 5., 4.0, 5.0, 4.0, 4.0, 4.0]
    ngs = pam2nest.CreateNetwork(m, 'izhikevich', weights, DELAY_FACTOR)
    
    len(ngs)
    # conn = FindConnections([ngs[3][0]])
    
    
    noise         = Create("poisson_generator", 50)
    dc_1            = nest.Create('dc_generator')
    
    voltmeter       = Create("voltmeter", 2)
    espikes         = Create("spike_detector")
    
    SetStatus(noise, [{'start': 0., 'stop': 10., 'rate': 800.0}])
    SetStatus(dc_1, {'start': 10., 'stop': 10.5, 'amplitude': 100.})

    #DivergentConnect(dc_1, ngs[3][:50], weight=[2.], delay=[1.])
    #DivergentConnect(noise, ngs[2], weight=[2.], delay=[1.])
    Connect(noise, ngs[3][:50], params={'weight': 20., 'delay': 1.})
    #    Connect([voltmeter[0]], [g1[0]])
    #    Connect([voltmeter[1]], [g2[int(m['c'][0][0][0])]])
    #  
    ConvergentConnect(ngs[0] + ngs[1] + ngs[2] + ngs[3] + ngs[4] + ngs[5],espikes)
    #
    Simulate(1000.0)
    #    
    ##    nest.voltage_trace.from_device([voltmeter[0]])
    ##    nest.voltage_trace.from_device([voltmeter[1]])
    ##    nest.voltage_trace.show()  
    #    
    nest.raster_plot.from_device(espikes, hist=False)
    nest.raster_plot.show()    

def getDelays(m, index):
    
    d = m['d'][index]
    c = np.array(d)
    c = c.flatten()
    return c


def plotDelays(m, index):
    d = getDelays(m, index )
    mp.hist(d[d>0], 50, range=(0., 10.))
    mp.title(m['neurongroups'][0][m['connections'][0][index][1]][0] + ' - ' +
             m['neurongroups'][0][m['connections'][0][index][2]][0])    

def analyseDelays_FullHippo():
    m_full = pam2nest.import_connections(EXPORT_PATH + 'test_fullhipp.zip')
    nest_vis.printConnections(m_full)
    nest_vis.printNeuronGroups(m_full)
    
    ylim = 3000
    
    mp.figure()
    plotDelays(m_full, 3)
    mp.ylim((0.0, ylim))
    
    mp.figure()
    plotDelays(m_full, 4)
    mp.ylim((0.0, ylim))
        
    mp.figure()
    plotDelays(m_full, 5)  
    mp.ylim((0.0, ylim))
    
    mp.show()
    

    
def connpartHistograms(c, segments, skip, ylim, r_min=-250, r_max=250):
    """ plots the connectivity histograms of parts of the connectivity matrix
    c           : n x m connectivity matrix
    segments    : number of segments
    skip        : number of neurons that should be skipped at the beginning
                  end
    """
    part = c[skip:-skip]
    segment_size = int(np.floor(len(part) / segments))
    mp.figure()
    
    colors = ['#b12913', '#d0854a', '#dadc6c', '#98ee52']
    

    results = []
    for i in range(0, segments):
        data = np.array([])
        
        for counter, row in enumerate(part[i*segment_size:(i+1)*segment_size]):
            corrected = np.where(row > 0)[0] - (float(c.shape[1]) / float(c.shape[0])) * ((i*segment_size) + counter + skip)
            data = np.concatenate((data, corrected))
        results.append(data)
    
    for i in range(0,len(results)):
        mp.subplot(4, 1, i+1)
        mp.hist(results[i], 30, range=(r_min, r_max), color=colors[i])
        mp.ylim(0.0, ylim)
        mp.xlim(r_min, r_max)
        

def analyseConns_hippocampus():
    m_test = pam2nest.import_connections(EXPORT_PATH + 'test.zip')
    nest_vis.printConnections(m_test)
    nest_vis.printNeuronGroups(m_test)
    
    perms, names = pam2nest.import_UVfactors(EXPORT_PATH + "permutations.zip")
    print(names)
    perm_dg = perms[0][0]
    perm_ca3 = perms[1][0]
    perm_ca1 = perms[2][0]
    
    ylim = 3000
    
    #mp.figure()
    #plotDelays(m_test, 0)
    #mp.ylim((0.0, ylim))
    
    range_min = [-65, -62, -110]
    range_max = [60, 62, 65]       
    
    my_cmap = matplotlib.cm.get_cmap('binary')
    
    connections = nest_vis.getConnectionMatrix(m_test, 0)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_dg, perm_ca3)
    connpartHistograms(perm_connections, 4, 25, 800, range_min[0], range_max[0])
    #nest_vis.save(EXPORT_PATH + 'conn_dg_ca3_hist')

    mp.figure()    
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_dg_ca3_matrix')
    
    
    connections = nest_vis.getConnectionMatrix(m_test, 1)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_ca3, perm_ca3)
    connpartHistograms(perm_connections, 4, 25, 140, range_min[1], range_max[1])
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca3_hist')
    mp.figure()    
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca3_matrix')
    
    
    connections = nest_vis.getConnectionMatrix(m_test, 2)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_ca3, perm_ca1)
    connpartHistograms(perm_connections, 4, 25, 300, range_min[2], range_max[2])
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca1_hist')
    mp.figure()
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca1_matrix')
    
    mp.show()
    
   
def analyseConns_simple():
    m_test = pam2nest.import_connections(EXPORT_PATH + 'hippocampus_simple.zip')
    nest_vis.printConnections(m_test)
    nest_vis.printNeuronGroups(m_test)
    
    perms, names = pam2nest.import_UVfactors(EXPORT_PATH + "permutations_simple_v.zip")
    print(names)
    perm_dg = perms[0][0]
    perm_ca3 = perms[1][0]
    perm_ca1 = perms[2][0]
    
    ylim = 3000
    
    #mp.figure()
    #plotDelays(m_test, 0)
    #mp.ylim((0.0, ylim))
    
    range_min = [-65, -62, -110]
    range_max = [60, 62, 65]    
    
    my_cmap = matplotlib.cm.get_cmap('binary')
    
    connections = nest_vis.getConnectionMatrix(m_test, 0)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_dg, perm_ca3)
    connpartHistograms(perm_connections, 4, 25, 500, range_min[0], range_max[0])
    #nest_vis.save(EXPORT_PATH + 'conn_dg_ca3_simple_hist')
    mp.figure()    
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_dg_ca3_simple_matrix')
    
    
    connections = nest_vis.getConnectionMatrix(m_test, 1)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_ca3, perm_ca3)
    connpartHistograms(perm_connections, 4, 25, 200, range_min[1], range_max[1])
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca3_simple_hist')
    mp.figure()    
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca3_simple_matrix')
    
    
    connections = nest_vis.getConnectionMatrix(m_test, 2)    
    perm_connections = nest_vis.permuteConnectionMatrix(connections, perm_ca3, perm_ca1)
    connpartHistograms(perm_connections, 4, 25, 300, range_min[2], range_max[2])
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca1_simple_hist')
    mp.figure()
    mp.imshow(perm_connections, cmap=my_cmap)
    #nest_vis.save(EXPORT_PATH + 'conn_ca3_ca1_simple_matrix')
    
    mp.show()   
    
if __name__ == "__main__":

    analyseNetwork()
    #analyseDelays_FullHippo()    
    #analyseConns_hippocampus()
    #analyseConns_simple()
    
    
