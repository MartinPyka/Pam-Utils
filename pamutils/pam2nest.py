# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 16:27:10 2014

This is a helper module with some functions to fastly use the data generated
by PAM in Nest

@author: martin
"""

import csv
import StringIO
import io
import logging
import os
import zipfile
import nest
import copy
import numpy as np

logger = logging.getLogger(__package__)

def axon_diameter_unm(axon_diameter):
    # the velocity function mimics the plot of figure 2 for unmyelinated fibers in
    # Waxman 1980, Muscle and Nerve. 1 / velocity is the delay for one mm
    #y = 2.2*sqrt(d)
    return np.log(axon_diameter * 2 + 1)*2

def axon_diameter_unm_inv(mm_ms):
    return (np.exp(mm_ms/2.)-1)/2

def axon_diameter_m(axon_diameter):
    # the velocity function mimics the plot of figure 2 for myelinated fibers in
    # Waxman 1980, Muscle and Nerve. 1 / velocity is the delay for one mm    
    return (axon_diameter * (100./18.))

def axon_diameter_m_inv(mm_ms):
    return (mm_ms * (18./100.))

def delayModel_axondiameter_unm(axon_diameter, sd, number=1):
    """ Computes the delay per mm from unmyelinated axons based on 
    log-normal distributions
    axon_diameter        : mean axon_diameter (for a lognormal distribution)
    sd                   : standard deviation
    returns the delay in ms per mm 
    """
    if number == 1:
        ad = np.random.lognormal(np.log(axon_diameter), sd, 1)[0]
    else:
        ad = np.random.lognormal(np.log(axon_diameter), sd, number)

    return (1. / axon_diameter_unm(ad))

def delayModel_axondiameter_m(axon_diameter, sd, number=1):
    """ Computes the delay per mm from myelinated axons based on 
    log-normal distributions
    axon_diameter        : mean axon_diameter (for a lognormal distribution)
    sd                   : standard deviation
    returns the delay in ms per mm 
    """
    # the velocity function mimics the plot of figure 1 for myelinated fibers in
    # Waxman 1980, Muscle and Nerve. 1 / velocity is the delay for one mm
    if number == 1:
        ad = np.random.lognormal(np.log(axon_diameter), sd, 1)[0]
    else:
        ad = np.random.lognormal(np.log(axon_diameter), sd, number)
    
    return (1. / axon_diameter_m(ad))

def delayModel_delayDistribLogNormal(delay, sd, number = 1):
    """ Simple log-normal distribution of delays. Skips the part with the axon
    diameters for simplicity 
    delay                 : mean delay (for a lognormal distribution)
    sd                    : standard deviation
    returns the delay in ms per mm 
    """
    if number == 1:
        return np.random.lognormal(np.log(delay), sd, number)[0]
    else:
        return np.random.lognormal(np.log(delay), sd, number)

def delayModel_delayDistribNormal(delay, sd, number=1):
    """ Simple log-normal distribution of delays. Skips the part with the axon
    diameters for simplicity 
    delay                 : mean delay (for a lognormal distribution)
    sd                    : standard deviation
    returns the delay in ms per mm 
    """
    return np.random.randn(1)[0]*sd + delay
    

def Connect(g1, g2, connections, delays, 
            w_mean, w_sd, 
            d_mean, d_sd, 
            syn_model,
            delayModel = delayModel_delayDistribNormal):
    """ Connects group g1 of neurons with group g2 of neurons using the
    connection- and delay-matrix generated by PAM
    
    g1,
    g2              : Groups of neurons
    
    connnections    : Connection matrix generated in PAM, which is a 
                      n * s matrix (n pre-synaptic neurons times s synapses)
    
    delays          : Delay matrix generated in PAM
    
    w_mean          : The mean weight
    w_sd            : The standard deviation for the weight
    
    d_mean          : Mean-delay
    d_sd            : standard deviation for the delays
    
    syn_model       : synapse-models for the connections
    
    
    Example:
        
        Connect(g1, g2, m['c'][0], m['d'][0], 2.0, DELAY_FACTOR)
    """
    
    delay_matrix = copy.deepcopy(connections)
    
    for i in range(0, len(connections)):
        
        delay_mm = max(delayModel(d_mean, d_sd), 0.1)

        for j in range(0, len(connections[i])):
            # if a synapse has really been created
            if connections[i][j] >= 0:
                weight = np.random.randn(1)[0] * w_sd + w_mean
                if w_mean > 0:
                    weight = max(weight, 0.1)
                if w_mean < 0:
                    weight = min(weight, 0.1)
                    
                # delay = np.random.randn(1)[0] * d_sd + d_mean
                delay = max(delays[i][j] * delay_mm, 0.1)
                nest.Connect([g1[i]], [g2[connections[i][j]]], syn_spec = {'model': syn_model, 'weight': weight, 'delay': delay})
#                nest.DataConnect([g1[i]], 
#                                 params=[{'target': [float(g2[connections[i][j]])],
#                                         'weight': [np.random.randn(1)[0] * w_sd + w_mean], 
#                                         'delay':  [delay]}],
#                                 model=syn_model)                
                delay_matrix[i][j] = delay
    return delay_matrix


def Connect_noDistance(g1, g2, connections, delays, 
            w_mean, w_sd, 
            d_mean, d_sd,
            syn_model,
            delayModel = delayModel_delayDistribNormal):
    """ Connects group g1 of neurons with group g2 of neurons using the
    connection- and delay-matrix generated by PAM
    
    g1,
    g2              : Groups of neurons
    
    connnections    : Connection matrix generated in PAM, which is a 
                      n * s matrix (n pre-synaptic neurons times s synapses)
    
    delays          : Delay matrix generated in PAM
    
    w_mean          : The mean weight
    w_sd            : The standard deviation for the weight
    
    d_mean          : Mean-delay
    d_sd            : standard deviation for the delays
    
    syn_model       : synapse-models for the connections
    
    
    Example:
        
        Connect(g1, g2, m['c'][0], m['d'][0], 2.0, DELAY_FACTOR)
    """
    
    delay_matrix = copy.deepcopy(connections)
    
    for i in range(0, len(connections)):
        
        delay_mm = max(delayModel(d_mean, d_sd), 0.1)

        for j in range(0, len(connections[i])):
            # if a synapse has really been created
            if connections[i][j] >= 0:
                weight = np.random.randn(1)[0] * w_sd + w_mean
                if w_mean > 0:
                    weight = max(weight, 0)
                if w_mean < 0:
                    weight = min(weight, 0)
                    
                # delay = np.random.randn(1)[0] * d_sd + d_mean
                delay = max(delays[i][j], 0.1)
                nest.Connect([g1[i]], [g2[connections[i][j]]], 
                             params={'weight': np.random.randn(1)[0] * w_sd + w_mean, 
                                     'delay':  delay},
                             model=syn_model)
#                nest.DataConnect([g1[i]], 
#                                 params=[{'target': [float(g2[connections[i][j]])],
#                                         'weight': [np.random.randn(1)[0] * w_sd + w_mean], 
#                                         'delay':  [delay]}],
#                                 model=syn_model)                6
                delay_matrix[i][j] = delay
    return delay_matrix

def Connect_noDistance_noConnection(g1, g2, connections, delays, 
            w_mean, w_sd, 
            d_mean, d_sd, 
            syn_model,
            delayModel = delayModel_delayDistribNormal
            ):
    """ Connects group g1 of neurons with group g2 of neurons using the
    connection- and delay-matrix generated by PAM
    
    g1,
    g2              : Groups of neurons
    
    connnections    : Connection matrix generated in PAM, which is a 
                      n * s matrix (n pre-synaptic neurons times s synapses)
    
    delays          : Delay matrix generated in PAM
    
    w_mean          : The mean weight
    w_sd            : The standard deviation for the weight
    
    d_mean          : Mean-delay
    d_sd            : standard deviation for the delays
    
    syn_model       : synapse-models for the connections
    
    
    Example:
        
        Connect(g1, g2, m['c'][0], m['d'][0], 2.0, DELAY_FACTOR)
    """
    
    delay_matrix = copy.deepcopy(connections)
    
    for i in range(0, len(connections)):
        
        delay_mm = max(delayModel(d_mean, d_sd), 0.1)
        l_connections = len(connections[i])

        for j in range(0, len(connections[i])):
            # randomly assign a pre-synaptic neuron to a post-synaptic neuron
            target_id = np.floor(np.random.rand(1) * len(connections[i]))[0].astype(int)
            weight = np.random.randn(1)[0] * w_sd + w_mean
            if w_mean > 0:
                weight = max(weight, 0)
            if w_mean < 0:
                weight = min(weight, 0)
                
            # delay = np.random.randn(1)[0] * d_sd + d_mean
            delay = max(delays[i][j], 0.1)
            nest.Connect([g1[i]], [g2[target_id]], 
                         params={'weight': np.random.randn(1)[0] * w_sd + w_mean, 
                                 'delay':  delay},
                         model=syn_model)
#                nest.DataConnect([g1[i]], 
#                                 params=[{'target': [float(g2[connections[i][j]])],
#                                         'weight': [np.random.randn(1)[0] * w_sd + w_mean], 
#                                         'delay':  [delay]}],6
#                                 model=syn_model)                
            delay_matrix[i][j] = delay
    return delay_matrix


def CreateNetwork(
                  data, 
                  neuron_type, 
                  w_mean=[1.], w_sd=[1.], 
                  d_mean=[1.], d_sd=[1.],
                  syn_model = ['static_synapse'],
                  distrib = [delayModel_delayDistribNormal],
                  connectModel = Connect,
                  delayfile = ''):
    """ Creates the NEST-network for the dataset generated by PAM
    data         : the network model obtained from PAM
    w_mean, w_sd : mean weights and standard deviation for each mapping or
                   one value for all mappings
    d_mean, d_sd : mean dealy and standard deviation for each mapping or
                   one value for all mappings
    syn_model    : synapse model
    delayfile_p  : prefix for delay-files. This is outpadd to pythonpathut which is needed for 
                   visualization of spikes in Blender using pam_vis
    """
    
    if (len(w_mean) > 1) & (len(w_mean) < len(data['connections'][0])):
        raise Exception("length of weight-vector does not match length of " +
            "connection list")
            
    if len(w_mean) != len(w_sd):
        raise Exception("length of weights-mean and weights-sd-vector do not " +
                        "match")
        
    if (len(d_mean) > 1) & (len(d_mean) < len(data['connections'][0])):
        raise Exception("length of delay-vector does not match length of " +
            "connection list")
            
    if len(d_mean) != len(d_sd):
        raise Exception("length of delay-mean and delay-sd-vector do not " +
                        "match")      
        
    if (len(syn_model) > 1) & (len(syn_model) < len(data['connections'][0])):
        raise Exception("length of syn_model does not match length of " + 
                        "connection list")
    
    if (len(distrib) > 1) & (len(distrib) < len(data['connections'][0])):
        raise Exception("length of distrib does not match length of " + 
                        "connection list")
            
    neurongroups = []
    # first, create the neurongroups
    for ng in data['neurongroups'][0]:
        neurongroups.append(nest.Create(neuron_type, ng[-1]))
        
    if len(delayfile) > 0:
        zf = zipfile.ZipFile(delayfile, 'w', zipfile.ZIP_DEFLATED)
  
    # create connections between neuron groups
    for i, conn in enumerate(data['connections'][0]):
        if (len(w_mean) == 1):
            weight = w_mean[0]
            sd = w_sd[0]
        else:
            weight = w_mean[i]
            sd = w_sd[i]
        
        if (len(d_mean) == 1):
            delay = d_mean[0]
            delay_sd = d_sd[0]
        else:
            delay = d_mean[i]
            delay_sd = d_sd[i]
            
        if len(syn_model) == 1:
            model = syn_model[0]
        else:
            model = syn_model[i]
            
        if len(distrib) == 1:
            distrib_func = distrib[0]
        else:
            distrib_func = distrib[i]
            
        delay_matrix = connectModel(neurongroups[conn[1]], neurongroups[conn[2]], 
                                    data['c'][conn[0]], data['d'][conn[0]], 
                                    weight, sd, delay, delay_sd,
                                    model, distrib_func)
        
        if len(delayfile) > 0:
            csv_write_matrix(zf, 'delay_'+ str(i), delay_matrix)
            
    return neurongroups
    
def csv_write_matrix(file, name, matrix):
    """Write matrix to csv file

    :param file file: open file
    :param str name: name of file
    :param numpy.Array matrix: a matrix

    """
    output = StringIO.StringIO()
    writer = csv.writer(
        output,
        delimiter=";",
        quoting=csv.QUOTE_NONNUMERIC
    )
    writer.writerows(matrix)
    file.writestr("%s.csv" % (name), output.getvalue())        

def import_connections(filepath):
    matrices = copy.deepcopy(SUPPORTED_SUFFIXES)

    with zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED) as file:
        for filename in file.namelist():
            filename_split = os.path.splitext(filename)
            filename_suffix = ''.join(filename_split[:-1]).rsplit("_", 1)[-1]
            filename_extension = filename_split[-1]

            if filename_extension not in SUPPORTED_FILETYPES.keys():
                logger.error("filetype not supported")
                raise Exception("filetype not supported")

            if filename_suffix not in SUPPORTED_SUFFIXES.keys():
                logger.error("unknown suffix")
                raise Exception("unknown suffix")

            data = io.StringIO(unicode(file.read(filename)))
            func = SUPPORTED_FILETYPES[filename_extension]

            matrix = func(data)
            
            if filename_suffix in FULL_INTS.keys():
                matrix = convertToIntFull(matrix)
                
            if filename_suffix in PARTLY_INTS.keys():
                matrix = convertToIntPartly(matrix)                

            matrices[filename_suffix].append(matrix)

    return matrices


def import_UVfactors(filepath):
    result = []
    names = []
    with zipfile.ZipFile(filepath, "r", zipfile.ZIP_DEFLATED) as file:
        for filename in file.namelist():
            filename_split = os.path.splitext(filename)  
            filename_extension = filename_split[-1]
            data = io.StringIO(unicode(file.read(filename)))
            func = SUPPORTED_FILETYPES[filename_extension]
            matrix = func(data)
            result.append(matrix)
            names.append(filename_split[0])
    return result, names
            

def convertToIntFull(matrix):
    return [[int(i) for i in row] for row in matrix]
    
def convertToIntPartly(matrix):
    return [[row[0], row[1], int(row[2])] for row in matrix]    

def csv_read(data):
    reader = csv.reader(
        data,
        delimiter=";",
        quoting=csv.QUOTE_NONNUMERIC
    )
    return [row for row in reader]


SUPPORTED_FILETYPES = {
    ".csv": csv_read
}

FULL_INTS = {
    "c": [],
    "connections": []
}

PARTLY_INTS = {
    "neurongroups": []
}

SUPPORTED_SUFFIXES = {
    "d": [],
    "c": [],
    "connections": [],
    "neurongroups": [],
    "names": []
}



