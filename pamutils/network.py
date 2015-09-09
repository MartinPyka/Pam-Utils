# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 20:54:14 2015

@author: ubuntu
"""

import matplotlib.pyplot as mp
import matplotlib

from nest import *
import nest.voltage_trace
import nest.raster_plot
import io
import csv

import nest_help as nh

import numpy as np
import pickle

import pamutils.pam2nest as pam2nest
import pamutils.nest_vis as nest_vis

#EXPORT_PATH = '/home/ubuntu/ownCloud/work/Projekte/hippocampal_model/results/'
#EXPORT_PATH = '../results/'
DELAY_FACTOR = 4.36

STIM_RATE = 100.


class Network(object):

    
    def __init__(self, filename, inp_group = 'EC_2',out_group = 'EC_5'):
        self.filename = filename
       
        # the main model
        self.m = pam2nest.import_connections(filename)
        self.neurongroupnames = nest_vis.printNeuronGroups(self.m)

        print(nest_vis.printConnections(self.m))
        
       # get input- and outputindex of two neuron groups
        self.inputindex = [i for i, j in enumerate(self.neurongroupnames) if j[0] == inp_group][0]
        self.outputindex = [i for i, j in enumerate(self.neurongroupnames) if j[0] == out_group][0]

       # neuron groups
        self.ngs = []
        
       # order of neuron groups (can be changed into desired order)
        self.order = [i for i,j in enumerate(self.neurongroupnames)]        
        
        self.cue           = []
        self.target           = []
        self.stimulus_list   = []   # list of stimuli-onsets and durations for
                                    # self.noise        
        
        self.dc_1            = []

        # spike detectors
        self.sd_list         = [] # list with spike-detectors
        
        self.sim_time        = 0.   # simulation time
        self.last_sim_time   = 0.   # last duration, when self.simulate was called  
        
        # list of connection weight matrices that should be recorded
        self.tracking        = []
        # list of recorded weight matrices
        self.track_weights   = []
        
        # data structure to store the last spiking-activity to compare with
        # the next one
        self.scattDiff      = [] 
        self.adjustLog      = []    # logs what is happening during adjustWeights()
        
        self.inhLayers      = []    # collection of inhibition layers
        self.noiseLayers    = []    # collection of noise inputs
        
        
    def createNetwork(self, 
                      w_means, w_sds, 
                      d_means, d_sds, 
                      syn_model=['static_synapse'], 
                      neuron_model = 'iaf_psc_delta',
                      distrib = [pam2nest.delayModel_delayDistribNormal],
                      connectModel = pam2nest.Connect,
                      output_prefix = ''):
        ''' Creates a network for a given model and vectors of weights and
        delays plus their standard deviations '''
        self.w_means = w_means
        self.w_sds = w_sds
        self.d_means = d_means
        self.d_sds = d_sds
        self.syn_model = syn_model
        self.neuron_model = neuron_model
        self.distrib = distrib
        self.connectModel = connectModel
        self.output_prefix = output_prefix
        self.initialize()
    
    def initialize(self):
        """ initializes the network based on the specified input given in 
        self.createNetwork()
        """
        self.ngs = pam2nest.CreateNetwork(
            self.m, self.neuron_model,
            self.w_means, self.w_sds, 
            self.d_means, self.d_sds,
            self.syn_model,
            distrib = self.distrib,
            connectModel = self.connectModel,
            delayfile_prefix = self.output_prefix)


        self.inputs = []
        for ng in self.neurongroupnames:
            self.inputs.append(Create("poisson_generator", ng[2]))
        self.cue             = self.inputs[self.inputindex]
        
        self.target          = self.inputs[self.outputindex]
        
        self.dc_1            = nest.Create('dc_generator')
        
               
        # create for each ng a spike-detector
        for ng in self.ngs:
            sd = Create("spike_detector")
            self.sd_list.append(sd)
            ConvergentConnect(ng, sd)
            
        for i, ng in enumerate(self.ngs):
            Connect(self.inputs[i], ng, params={'weight':2000., 'delay': 1.})
            
        #Connect(self.cue, self.ngs[self.inputindex], params={'weight': 2000., 'delay': 1.})
        Connect(self.target, self.ngs[self.outputindex], params={'weight': 2000., 'delay': 1.})

        
    def addNoise(self, i, rate):
        ''' Creates noisy input to a target layer, given by index i. rate 
        determines the amount of noise generated from poisson generators '''
        try:
            noise = Create("poisson_generator", len(self.ngs[i]))
            SetStatus(noise, [{'start':0., 'stop': float('inf'), 'rate': rate}])
            Connect(noise, self.ngs[i], params = {'weight': 2000., 'delay': 1.})
            self.noiseLayers.append([noise, i])
        except IndexError:
            print( 'Warning: No noise added. Index of target layer (i) out of range' )          

       
    def addInhibitionLayer(self, i, percent, 
                             w_exc_inh, w_inh_exc, 
                             s_exc_inh, s_inh_exc):
        ''' Add self-inhibition layer to a given neural layer
        i         : index of neural layer
        percent   : number of inhibitory neurons as percentage of number of 
                    neurons on layer
        w_exc_inh : exc to inh weight
        w_inh_exc : inh to exc weight
        s_exc_inh : number of synapses from excitatory to inhibitory neurons
        s_inh_exc : number of synapses from inhibitory to excitatory neurons
        '''
        inhL = Create(self.neuron_model, np.floor(len(self.ngs[i]) * percent).astype(int))
        self.inhLayers.append([inhL, i])
        self.RandomConnect(self.ngs[i], inhL, w_exc_inh, s_exc_inh)
        self.RandomConnect(inhL, self.ngs[i], w_inh_exc, s_inh_exc)
        
    def RandomConnect(self, ng1, ng2, w, s):
        """ Connects each ng1 neuron with s random ng2 neurons """
        for n in ng1:
            perm = np.random.permutation(len(ng2))
            nest.ConvergentConnect([n],
                                   np.array(ng2)[perm[:s]].tolist(),
                                   weight = w, delay = 1.)

    def setConnectionParams(self, c, params):
        ''' Configures the connection properties of all connections for a given
        connection-index c '''
        pre_ngs = self.ngs[self.m['connections'][0][c][1]]
        post_ngs = self.ngs[self.m['connections'][0][c][2]]
        nest.SetStatus(nest.GetConnections(pre_ngs, post_ngs), params)
        

    def simulate(self, sim_time):
        ''' simulates the network for sim_time ms. If tracking contains
        connection-indices, the weights of the corresponding connections are 
        recorded afterwards '''
        self.last_sim_time = self.sim_time
        self.sim_time = self.sim_time + sim_time
        Simulate(sim_time)
        self.trackWeights()
            
    def trackWeights(self):
        ''' saves the weights for all connctions defined in tracking,
        which is a list of connection-indices '''
        if self.tracking:
            result = [self.getConnInfo(t, 'weight') for t in self.tracking]
            self.track_weights.append(result)
            return result
        
    def addStimulus(self, ng, onset, end, rate=STIM_RATE):
        ''' adds a new stimulus for a given neurongroup,
        onset, end (in ms) and rate
        '''
        self.stimulus_list.append([ng, onset, end, rate])        
        
    def setStimulus(self, area, neurons, start, stop, rate=STIM_RATE):
        ''' Sets start and stop-points for cue-pattern '''
        if len(neurons) == 1:
            cue = [self.inputs[area][neurons]]
        else:
            cue = [self.inputs[area][i] for i in neurons]
        self.stimulus_list.append([cue, start, stop, rate])
        SetStatus(cue, [{'start':start, 'stop': stop, 'rate': rate}])        
        
    def setCue(self, neurons, start, stop, rate=STIM_RATE):
        ''' Sets start and stop-points for cue-pattern '''
        if len(neurons) == 1:
            cue = [self.cue[neurons]]
        else:
            cue = [self.cue[i] for i in neurons]
        self.stimulus_list.append([cue, start, stop, rate])
        SetStatus(cue, [{'start':start, 'stop': stop, 'rate': rate}])
    
    def setTarget(self, neurons, start, stop, rate=100.):
        ''' Sets start and stop-points for target-pattern '''
        if len(neurons) == 1:
            target = [self.target[neurons]]
        else:
            target = [self.target[i] for i in neurons]
        self.stimulus_list.append([target, start, stop, rate])    
        SetStatus(target, [{'start':start, 'stop': stop, 'rate': rate}])
        
    def getDistances(self, index):
        ''' get delays of one connection, given by index '''
        d = self.m['d'][index]
        c = np.array(d)
        c = c.flatten()
        return c
    

    def getPrePostNgs(self, c):
        ''' Returns pre- and post-neurongroup for a given connection
        index '''
        pre_ngs = self.ngs[self.m['connections'][0][c][1]]
        post_ngs = self.ngs[self.m['connections'][0][c][2]]
        return pre_ngs, post_ngs

    def getConnInfo(self, c, info, pre_indices = [], post_indices = []):
        ''' returns the weights as they are returned by GetConnections 
        for a given connection-index c
        if indices is empty, data will be returned for all indices, 
        otherwise only for the requested indices '''
        pre_ngs, post_ngs = self.getPrePostNgs(c)
        
        if pre_indices:
            pre_ngs = [pre_ngs[i] for i in pre_indices]
        if post_indices:
            post_ngs = [post_ngs[i] for i in post_indices]
        
        connections = nest.GetConnections(pre_ngs, post_ngs)
        result = nest.GetStatus(connections, info)
        return result
    
    def getMeanWeights(self):
        ''' Returns the mean weights for every connection-node '''
        data = []
        for c in range(0, len(self.m['connections'][0])):
            data.append(np.mean(self.getConnInfo(c, 'weight')))
        return data
    
    def setWeights(self, c, w_mean, w_sd = 1.0):
        ''' Resets the weights for a connection c given the mean and 
        sd values for the new weights '''
        pre_ngs, post_ngs = self.getPrePostNgs(c)
        connections = nest.GetConnections(pre_ngs, post_ngs);
        weights = np.random.randn(len(connections)) * w_sd + w_mean
        weights[weights < 0.1] = 0.1
        nest.SetStatus(connections, 'weight', weights)
        
    def setInhWeights(self, c, w_mean, w_sd = 1.0):
        ''' Same as setWeights, but for the inhibitory connections '''
        pre_ngs = self.inhLayers[c][0]
        post_ngs = self.ngs[self.inhLayers[c][1]]
        connections = nest.GetConnections(pre_ngs, post_ngs);
        weights = np.random.randn(len(connections)) * w_sd + w_mean
        weights[weights < 0.1] = 0.1
        nest.SetStatus(connections, 'weight', weights)
    
    def getWeightMatrix(self, c):
        ''' returns the weight matrix for a given connection c'''
        # first, get the connection matrix
        pre_num = self.m['neurongroups'][0][self.m['connections'][0][c][1]][2]
        post_num = self.m['neurongroups'][0][self.m['connections'][0][c][2]][2] 
        matrix = np.zeros((pre_num, post_num))
        
        pre_ngs = self.ngs[self.m['connections'][0][c][1]]
        post_ngs = self.ngs[self.m['connections'][0][c][2]]
        
        connections = nest.GetConnections(pre_ngs, post_ngs)
        status = nest.GetStatus(connections)
        
        for s in status:
            matrix[s['source']-pre_ngs[0], s['target']-post_ngs[0]] = s['weight']
        
        return matrix
    
    def getWeightsSpiking(self, sd, post = 4, target = (0,25)):
        ''' Returns the connection weights between neurons of the pre-synaptic
        layer, that fire (given by a spike-detector) and that are connected
        with specific neurons in the post-synaptic layer '''
        # get spiking neurons
        s, t = nh.getEventsFromSpikeDetector(sd)
        # create unique spiking neuron-ids
        su = np.unique(s)
        connections = nest.GetConnections(su.tolist(), self.ngs[post][target[0]:target[1]])
        weights = nest.GetStatus(connections, 'weight')
        targets = nest.GetStatus(connections, 'target')
        return weights, su, targets
    
    def setWeightsSpiking(self, weights, su, post = 4, target = (0, 25)):
        ''' Sets the weights of the connections which can be obtained from 
        getWeightsSpiking '''
        connections = nest.GetConnections(su.tolist(), self.ngs[post][target[0]:target[1]])
        nest.SetStatus(connections, 'weight', weights)    
    
    def plotDelays(self, c):
        ''' plot delays as histogram for one connection (c)'''
        d = self.getConnInfo(c, info = 'delay')
        mp.hist(d, 50, range=(0., 10.))
        mp.title(self.m['neurongroups'][0][self.m['connections'][0][c][1]][0] + ' - ' +
                 self.m['neurongroups'][0][self.m['connections'][0][c][2]][0])
        
    def plotLastSim(self):
        """Plots last simulation"""
        self.plotNetwork(self.last_sim_time, self.sim_time)    
        
        
    def plotNetwork(self, start = 0., end = -1):
        '''Plots network with independant of network size. If end is -1, then end is set to self.sim_time '''
        if end == -1:
            end = self.sim_time
        
        area_size = 0.5   
        mp.figure()
        for i in range(len(self.m['neurongroups'][0])):
            mp.subplot(len(self.m['neurongroups'][0]), 1, (i + 1))
            nh.scatter(self.sd_list[i], area = area_size)
            mp.xlim((start, end))
            mp.ylabel(self.m['neurongroups'][0][i][0])
            
        self.getPOA(True, [start, end]); 
        
        
    def plotWeightHistogram(self):
        ''' Plots all recorded weights '''
        for t in self.track_weights:
            mp.figure()
            rows = len(t)
            for i, h in enumerate(t):
                mp.subplot(rows, 1, (i+1))
                mp.hist(h)
                
    def plotWeights(self):
        ''' assuming that for each item in track_weights the list of 
        weight-lists is the same, plotWeights plots the temporal evolution of
        all weights '''
        
        evo = []
        
        for i, t in enumerate(self.track_weights):
            if i == 0:
                for h in t:
                    evo.append(np.array(h))
            else:
                for j, h in enumerate(t):
                    evo[j] = np.vstack((evo[j], h))
        
        for i, d in enumerate(evo):
            mp.subplot(len(evo), 1, (i+1))
            mp.plot(d)
            

    def scatterSD(self, i, color):
        """ create a scatter plot for a given spikedetector-index 
        for the last simulation. color refers to the color-code """
        s, t = nh.getEventsFromSpikeDetector(self.sd_list[i], [self.last_sim_time, 
                                                               self.sim_time])
        mp.figure()
        mp.scatter(t-self.last_sim_time, s, c=color)
        
    def scatterDiff(self, i, plot = True):
        
        s, t = nh.getEventsFromSpikeDetector(self.sd_list[i], [self.last_sim_time, 
                                                               self.sim_time])
        data = np.array([np.floor(t-self.last_sim_time).astype(int), s])
        data = data.transpose()
        data = data.tolist()
        
        overlap = 0
        newones = 0
        
        for item in data:
            if item in self.scattDiff:
                if plot:
                    mp.scatter(item[0], item[1], c='b')
                overlap += 1
            else:
                if plot:
                    mp.scatter(item[0], item[1], c='r')
                newones += 1

        if plot:
            for item in self.scattDiff:
                if not item in data:
                    mp.scatter(item[0], item[1], c='r')
                    
            mp.xlim(0, 50)                
        
        if len(self.scattDiff)==0:
            self.scattDiff = data
            overlap = len(data)
            newones = 0
        
        return overlap, newones
        
        
    def resetNetwork(self):
        ''' executes nest.ResetNetwork() and changes some network-class related
        variables '''
        self.sim_time = 0     

    def getPOA(self, printit = False, interval = [0, float('inf')]):
        '''Get percentage of activity for each network'''
        POA = [nh.getPOA(self.ngs[i], self.sd_list[i], interval = [0,float('inf')]) for i in range(len(self.m['neurongroups'][0]))]
    
        if printit: 
            for i in range(len(POA)):
                print(self.m['neurongroups'][0][i][0] + " " + str(POA[i]))
        return POA 
    
    def exportSpikes(self, filename):
        """ Exports data from a list of spike detector ids into a given file
        in CSV-format """
        f = open(filename, 'w')
        writer = csv.writer(
            f,
            delimiter=";",
            quoting=csv.QUOTE_NONNUMERIC
        )

        for index, sd in enumerate(self.sd_list):
            sender, times = nh.getEventsFromSpikeDetector(sd)
            for i in range(len(sender)):
                writer.writerow([index, sender[i] - self.ngs[index][0], times[i]])
            
        f.close()        
        
    def stimulusCueTarget(self, isi_interval, c_t_interval, rep, cue = range(0,25), target = range(0,25)):
        ''' Simulates pairing of cue-target stimulation with a given 
        isi_interval         : interstimulus-interval in ms
        c_t_interval         : cue-target-interval in ms
        rep                  : number of repetitions
        '''
        for i in range(0, rep):
            self.setCue(cue, start = self.sim_time, stop= self.sim_time + 10)
            self.setTarget(target, start = self.sim_time+c_t_interval, stop= self.sim_time+c_t_interval+10)
            self.simulate(c_t_interval + isi_interval)
            
            
    def stimulus(self, isi_interval, dur, rep, area, cue = range(0,25), plot = True):
        ''' stimulates neurons in a given area
        isi_interval         : interstimulus-interval in ms
        rep                  : number of repetitions
        '''
        for i in range(0, rep):
            self.setStimulus(area, cue, start = self.sim_time, stop= self.sim_time + dur)
            self.simulate(isi_interval)
            
        if plot:
            self.plotNetwork(self.sim_time - rep * isi_interval, self.sim_time)

            
    def stimulusCue(self, isi_interval, dur, rep, cue = range(0,25), plot = True):
        ''' Simulates pairing of cue-target stimulation with a given 
        isi_interval         : interstimulus-interval in ms
        rep                  : number of repetitions
        '''
        for i in range(0, rep):
            self.setCue(cue, start = self.sim_time, stop= self.sim_time + dur)
            self.simulate(isi_interval)
            
        if plot:
            self.plotNetwork(self.sim_time - rep * isi_interval, self.sim_time)
    

    def getPOAperSegment(self, interval, rep):
        cpoa = []
        for i in range(rep):
            cpoa.append(self.getPOA(False, [self.sim_time - (interval * (i + 1)), self.sim_time - (interval * i)]))
        
        return np.array(cpoa).mean(0)

    def adjustWeights(self, c, poa, start = 0., error = 0.01, max_iter = 50):
        ''' Adjusts the weights for a given connection automatically until a 
        certain percentage of activity (poa) for the target region is reached
        c        : connection index
        poa      : target POA
        start    : start-value
        error    : allowed deviation from the target poa
        max_iter : maximal number of iterations
        '''
        
        interval = 200
        rep = 5
        
        post_sd = self.m['connections'][0][c][2]
        w_mean = start
        self.setWeights(c, w_mean)
        self.stimulusCue(interval, 30, rep, cue = np.arange(0, 25), plot = False)
        
        itera = 0
        current_poa = self.getPOAperSegment(interval, rep)
        while (itera < max_iter) & (abs(poa-current_poa[post_sd]) > error):
            print(str(itera) + ': ' + str(current_poa[post_sd]) + ' w:' + str(w_mean) )
            self.adjustLog.append([c, itera, current_poa[post_sd], w_mean])
            # if there is not enough activity, increase the weight
            if (poa-current_poa[post_sd]) > 0:
                w_mean += 0.2
            if (poa-current_poa[post_sd]) < 0:
                w_mean = max(0., w_mean - 0.2)
                
            self.setWeights(c, w_mean)
            self.stimulusCue(interval, 30, rep, cue = np.arange(0, 25), plot = False)
            current_poa = self.getPOAperSegment(interval, rep)
            itera += 1
            
        if (itera == max_iter):
            print("Aborted because max_iter (" + str(max_iter) + ") has been reached")
        
        return w_mean
    
    def analyseDelayHomogeneity(self, c, neurons, distance=False):
        """ computes for a certain mapping and a specified number of neurons the 
        variance and mean in the actual delays. By progressive shifts of the neuron 
        numbers,     this function helps to check how homogenously the delays are
        distributed.
        If distance == True, then the data from the m-structure will be taken
        """
        start = 0
        end = neurons
        
        means = []
        vars = []
        
        if distance:
            for i in range(len(self.m['d'][c])):
                means.append(np.mean(self.m['d'][0][start:end]))
                vars.append(np.var(self.m['d'][0][start:end]))
                start += 1
                end += 1            
        else:
            pre_ngs, post_ngs = self.getPrePostNgs(c)
            
            for i in range(len(pre_ngs)-neurons):
                print(i)
                conns = GetConnections(pre_ngs[start:end], post_ngs)
                delays = GetStatus(conns, 'delay')
                means.append(np.mean(delays))
                vars.append(np.var(delays))
                start += 1
                end += 1
        
        return means, vars
    
    def getSpikeOnsetTimes(self, spike_take = 4):
        """ returns the occurence of the first spike in each area for the last
        simulation
        """
        data = []
        for o in self.order:
            s, t = nh.getEventsFromSpikeDetector(self.sd_list[o], t_range=[self.last_sim_time,
                                                                           self.sim_time])
            if len(t) >= spike_take:
                data.append(t[spike_take-1])
            else:
                data.append(-1)
        return np.array(data)    
    
    def getSpikeHistogram(self, bins, plot=True, stretch = 3):
        """Creates a histogram of spikes""" 
        data = []
        for o in self.order:
            s, t = nh.getEventsFromSpikeDetector(self.sd_list[o], t_range=[self.last_sim_time,
                                                                           self.sim_time])
            for _ in range(stretch):
                data.append(np.histogram(t, bins, range = [self.last_sim_time, self.sim_time])[0])
    
        if plot:
            mp.figure()             
            mp.imshow(data, vmin = 0, vmax = 50, interpolation='none')
            mp.title('Temporal spike evolution')
            mp.xlabel('ms')        
        return np.array(data)
        
    
    def getSpikeHistogramRange(self, bins, time_range, plot=True, stretch = 3):
        """ Get histogram of spikes for a given time_range [x,y]"""
        data = []
        for o in self.order:
            s, t = nh.getEventsFromSpikeDetector(self.sd_list[o], t_range=time_range)
            for _ in range(stretch):
                data.append(np.histogram(t, bins, range = time_range)[0])
    
        if plot: 
            mp.figure()            
            mp.imshow(data, vmin = 0, vmax = 50, interpolation='none')
            mp.title('Temporal spike evolution')
            mp.xlabel('ms')        
        return np.array(data)
        
        
#
# 
#    def stimulusCueHistogram(self, isi_interval, duration, rep, bins, neurons, area = -1 ):
#        """ Computes a histogram for a given cue pattern based on several 
#        repetitions """
#        if area == -1:
#            area = self.inputindex
#            
#        data = []
#        for i in range(rep):
#            self.stimulus(isi_interval, duration, 1, area, neurons, False)
#            hist_data = np.array(self.getSpikeHistogram(bins, plot = False, stretch = 1))
#            if (i==0):
#                data = hist_data
#            else:
#                data = data + hist_data
#        return data
    
    
 
    def stimulusCueHistogram(self, isi_interval, duration, rep, bins, area = -1 ):
        """ Computes a histogram for a given cue pattern based on several 
        repetitions """
        if area == -1:
            area = self.inputindex
            
        data = []
        for i in range(rep):
            self.stimulus(isi_interval, duration, rep, area, plot = False)
            hist_data = np.array(self.getSpikeHistogram(bins, plot = False, stretch = 1))
            if (i==0):
                data = hist_data
            else:
                data = data + hist_data
        return data
        
        
        
        
    def sampleSpikeHistogram(self, plot=True, title='Temporal spike evolution', rep = 5):
        data = []
    
        duration = 70
        
        stimuli_size = [25]
        sp = len(stimuli_size)
        stretch = 1
        fac = 1.        # factor to reduce the time span for the histogram
        
        for i, stimulus in enumerate(stimuli_size):
            mp.subplot(sp, 1, i+1)
            data = np.zeros((6*stretch,duration/fac))
            for _ in range(rep):
                self.stimulusCue(duration, 30, 1, np.arange(stimulus), False)
                data = data + self.getSpikeHistogram(duration/fac, False, stretch)
            
            mp.imshow(data / rep, vmin = 0, vmax = stimulus, interpolation='none')
            #mp.colorbar()
            if (i == 0):
                mp.title(title)
    
            mp.ylabel('Region')
        
        mp.xlabel('ms')
        
    def sampleOnsetLatency(self, title='Onset latency', rep = 10, spike_take = 4):
        duration = 150
        
        data = []
        for _ in range(rep):
            self.stimulusCue(duration, 30, 1, plot = False)
            data.append(self.getSpikeOnsetTimes(spike_take = spike_take) - self.last_sim_time)
            

        return data
        
    def existConnection(self, i, pre, post):
        return len(np.where(np.array(self.m['c'][i][pre]) == post)[0]) > 0
                     
        
        
class SGNetwork(Network):

            
             
    def initialize(self):
        """ initializes the network based on the specified input given in 
        self.createNetwork()
        """
        self.ngs = pam2nest.CreateNetwork(
            self.m, self.neuron_model,
            self.w_means, self.w_sds, 
            self.d_means, self.d_sds,
            self.syn_model,
            distrib = self.distrib,
            connectModel = self.connectModel,
            delayfile_prefix = self.output_prefix)


        self.inputs = []
        for ng in self.neurongroupnames:
            self.inputs.append(Create("spike_generator", ng[2]))
        
        self.cue = Create("spike_generator", 
                             self.neurongroupnames[self.inputindex][2], 
                             params = {'spike_times': [1.]})
        self.target = Create("spike_generator", 
                                self.neurongroupnames[self.outputindex][2], 
                                params = {'spike_times': [1.]})
        
        self.dc_1            = nest.Create('dc_generator')
        
        # create for each ng a spike-detector
        for ng in self.ngs:
            sd = Create("spike_detector")
            self.sd_list.append(sd)
            ConvergentConnect(ng, sd)
            
        for i, ng in enumerate(self.ngs):
            Connect(self.inputs[i], ng, params={'weight':2000., 'delay': 1.})
            
        #Connect(self.cue, self.ngs[self.inputindex], params={'weight': 2000., 'delay': 1.})
        Connect(self.cue, self.ngs[self.inputindex], [2000.], [1.])
        Connect(self.target, self.ngs[self.outputindex], [2000.], [1.])  
        
    def setCue(self, start, neurons):
        if len(neurons) == 1:
            cue = [self.cue[neurons]]
        else:
            cue = [self.cue[i] for i in neurons]
        SetStatus(cue, [{'spike_times': [start+1]}])
        
    def setTarget(self, start, neurons):
        if len(neurons) == 1:
            target = [self.target[neurons]]
        else:
            target = [self.target[i] for i in neurons]
        SetStatus(target, [{'spike_times': [start+1]}])
                
    def stimulusCueTarget(self, isi_interval, c_t_interval, rep, cue = range(0,25), target = range(0,25)):
        ''' stimulates the input neurons with a spike generator (all at the 
        same time)
        isi_interval : inter-stimulus-interval
        c_t_interval : cue-target-interval
        rep          : repetitions
        '''
        for i in range(0, rep):
            self.setCue(start = self.sim_time, neurons = cue)
            self.setTarget(start = self.sim_time+c_t_interval, neurons = target)
            self.simulate(isi_interval)            
            
    def stimulusCue(self, isi_interval, rep, cue = range(0, 25), plot = True):
        ''' stimulates the input neurons with a spike generator (all at the 
        same time)
        isi_interval : inter-stimulus-interval
        rep          : repetitions
        '''
        for i in range(0, rep):
            self.setCue(start = self.sim_time, neurons = cue)
            self.simulate(isi_interval)
        
        if plot:
            self.plotNetwork(self.sim_time - rep * isi_interval, self.sim_time)
 
           
    def progressiveOverlap(self, neurons, shift, i):
        """ For a given starting sequence, we measure the overlap in region i 
        (based on spike detector) of spikes that remain, when the pattern is 
        progressively shifted towards pattern + shift 
        neurons     : int number of neurons to be activated
        shift       : number of shifts
        i           : index for the spike-detector to be used
        """
        self.scattDiff = []
        pattern = np.arange(neurons)
        oldones = []
        newones = []
        for x in range(shift):
            self.stimulusCue(100, 1, pattern + x, False)
            o, n = self.scatterDiff(i, False)
            oldones.append(o)
            newones.append(n)
        
        return oldones, newones 
        
    def plotFuncProgressiveOverlap(self, o, n):
        mp.figure()
        mp.plot(o, label='overlap')
        mp.plot(n, label='new spikes')
        mp.plot(np.array(o) + np.array(n), label='sum')
        mp.xlabel('Shift of pattern')
        mp.ylabel('number of spikes')
        mp.legend(loc="upper right")
 
    def plotProgressiveOverlap(self, neurons, shift, i):
        """ Plot the result of progressiveOverlap(). See this method
        for argument explanations """
        o, n = self.progressiveOverlap(neurons, shift, i)
        self.plotFuncProgressiveOverlap(o, n)
        
    
    def getSpikeHistogram(self, bins, plot=True, stretch = 3):
        """Creates a histogram of spikes with bins as ms""" 
        data = []
        for o in self.order:
            s, t = nh.getEventsFromSpikeDetector(self.sd_list[o], t_range=[self.last_sim_time,
                                                                           self.sim_time])
            for _ in range(stretch):
                data.append(np.histogram(t, bins, range = [self.last_sim_time, self.sim_time])[0])
    
        if plot:
            mp.figure()             
            mp.imshow(data, vmin = 0, vmax = 50, interpolation='none')
            mp.title('Temporal spike evolution')
            mp.xlabel('ms')        
        return np.array(data)
        
   
#    # does not work yet...     
#    def stimulusCueHistogram(self, isi_interval, duration, rep, bins, neurons, area = -1 ):
#        """ Computes a histogram for a given cue pattern based on several 
#        repetitions """
#        if area == -1:
#            area = self.inputindex
#            
#        data = []
#        for i in range(rep):
#            self.stimulus(isi_interval, duration, area, neurons, False)
#            hist_data = np.array(self.getSpikeHistogram(bins, plot = False, stretch = 1))
#            if (i==0):
#                data = hist_data
#            else:
#                data = data + hist_data
#        return data
        
    # does not work yet...     
    def stimulusCueHistogram(self, isi_interval, duration, rep, bins, neurons, area = -1 ):
        """ Computes a histogram for a given cue pattern based on several 
        repetitions """
        if area == -1:
            area = self.inputindex
            
        data = []
        for i in range(rep):
            self.stimulus(isi_interval, duration, 1, area, neurons, plot = False)
            hist_data = np.array(self.getSpikeHistogram(bins, plot = False, stretch = 1))
            if (i==0):
                data = hist_data
            else:
                data = data + hist_data
        return data
        


    
    def sampleSpikeHistogram(self, plot=True, title='Temporal spike evolution', rep = 5):
        """Computes a network simulation with inter-stimulus-interval of 70ms and creates a spike histogram"""
        data = []
    
        isi_interval = 70
        
        stimuli_size = [25]
        sp = len(stimuli_size)
        stretch = 1
        fac = 1.        # factor to reduce the time span for the histogram
        
        for i, stimulus in enumerate(stimuli_size):
            mp.subplot(sp, 1, i+1)
            data = np.zeros((len(self.m['neurongroups'][0])*stretch,isi_interval/fac))
            for _ in range(rep):
                self.stimulusCue(isi_interval, 1, np.arange(stimulus), False)
                data = data + self.getSpikeHistogram(isi_interval/fac, False, stretch)
            
            mp.imshow(data / rep, vmin = 0, vmax = stimulus, interpolation='none')
            #mp.colorbar()
            if (i == 0):
                mp.title(title)
    
            mp.ylabel('Region')
        
        mp.xlabel('ms')
        
    def sampleOnsetLatency(self, title='Onset latency', rep = 10, spike_take = 4):
        """Generates a sample onset Latency with an inter-stimulus-interval of 150ms and 10 repetitions"""
        isi_interval = 150
        
        data = []
        for _ in range(rep):
            self.stimulusCue(isi_interval, 1, plot = False)
            data.append(self.getSpikeOnsetTimes(spike_take = spike_take) - self.last_sim_time)     
        return data
        
class BalancedNetwork(Network):
    """Createa a random balanced network based on Brunel, 2000. Current version created for NEST version 2.2."""
    
    def createNetwork(self,
                      w_means, w_sds, 
                      d_means, d_sds, 
                      syn_model=['static_synapse'], 
                      neuron_model = 'iaf_psc_delta',
                      distrib = [pam2nest.delayModel_delayDistribNormal],
                      connectModel = pam2nest.Connect,
                      output_prefix = ''):
        ''' Creates a network for a given model and vectors of weights and
        delays plus their standard deviations '''
        self.w_means = w_means
        self.w_sds = w_sds
        self.d_means = d_means
        self.d_sds = d_sds
        self.syn_model = syn_model
        self.neuron_model = neuron_model
        self.distrib = distrib
        self.connectModel = connectModel
        self.output_prefix = output_prefix
        self.initialize()
   
    
    def initialize(self):
        """ initializes the network based on the specified input given in 
        pam2nest.CreateBalancedNetwork() with following parameters taken from Brunel, 2000:
        """
        # Parameters for asynchronous irregular firing of neurons
        self.eta          = 2.0  # external rate relative to threshold
        self.epsilon      = 0.1  # probability for connection
        self.weight_ratio = 5.0  # ratio of inhibitory and excitatory weight
        
        # Amplitudes of post synaptic potentials
        self.post_amp_exc = 0.1  # amplitude of excitatory EPSP in mV
        self.post_amp_inh = -self.weight_ratio*self.post_amp_exc  # amplitude of inhibitory IPSP
        
        # Membrane potential parameters
        self.theta = 20.0 # membrane threshold potential
        self.tauMem = 20.0 # time constant of membrane potential
        
        # Customize connection parameters in NEST 2.2
        self.conn_option_dict = {"allow_autapses": True, "allow_multapses":True} 
        
        # Create customized synapses for exitatory and inhibitory neurons using the 
        CopyModel("static_synapse", "excitatory", {"weight": self.post_amp_exc, "delay":1.})
        CopyModel("static_synapse", "inhibitory", {"weight": self.post_amp_inh, "delay":1.})

        # Create network        
        self.ngs = pam2nest.CreateBalancedNetwork(
            self.m, self.neuron_model,
            self.w_means, self.w_sds, 
            self.d_means, self.d_sds,
            self.syn_model,
            distrib = self.distrib,
            connectModel = self.connectModel,
            delayfile_prefix = self.output_prefix)

        
        self.ConnectBalancedNet()           
       
       
       
    def ConnectBalancedNet(self, delay = 1.5):
        """Creates
        poission generators (inputs and noise),
        connections for excitatory and inhibitory neurons,
        spike detectors for excitatory and inhibitory neurons,
        configures the spikegenerators,
        defines custom synapses,
        and connects the model"""

        self.noise = []        # noise from a poisson generator
        self.inputs = []       # for input 
        
        
        for i, ng in enumerate(self.ngs):
        
            # Create poisson generator for every excitatory neuron as noise
            noise_exc = Create("poisson_generator", len(self.ngs[i]))
            SetStatus(noise_exc, [{'start':0., 'stop': float('inf'), 'rate': 20.}])
         
            # Create and customize spike detectors for excitatory neuron group
            sd_exc = Create("spike_detector")
            SetStatus(sd_exc, [{"label": "balanced_net_exc",
                                "withtime": True,
                                "withgid": True}])           
            
            # create and connect poisson generator for inputs in excitatory neurons
            self.inputs.append({self.inputindex:Create("poisson_generator", len(self.ngs[i]))}) 
            Connect(self.inputs[i][self.inputindex], ng, params={"weight":2000., "delay":1.})
            
            # Connect poisson generators with excitatory neurons (for NEST 2.7 change params to syn_spec)
            Connect(noise_exc, ng, params = {'weight': 2000., 'delay': 1.})
            
            # Connect spike detector with excitatory neurons
            ConvergentConnect(ng, sd_exc)
            
            # Create inhibitory neurons with a ratio of 1:4
            inh_neurons = (Create(self.neuron_model, int(round(0.25*len(ng)))))
            
            # create and customize spike detectors for excitatory neuron group                   
            sd_inh = Create("spike_detector")
            SetStatus(sd_inh, [{"label": "balanced_net_inh",
                                "withtime": True,
                                "withgid": True}])          
                                
            # Create poisson generator for every inhibitory neuron as noise
            noise_inh = Create("poisson_generator", int(round(0.25*len(ng))))
            SetStatus(noise_inh, [{'start':0., 'stop': float('inf'), 'rate': 20.}])

            # Connect poisson generators with inhibitory neurons
            Connect(noise_inh, inh_neurons, params = {'weight': 2000., 'delay': 1.})
            
            # Connect spike detectors with inhibitory neurons
            ConvergentConnect(inh_neurons, sd_inh, model = "excitatory")
            
            # Create a dict with all spike detectors
            self.sd_list.append({'exc':sd_exc, 'inh':sd_inh})
            
            # Create a dict with all noise layers
            self.noiseLayers.append({"exc":noise_exc, "inh":noise_inh})
            
            # Compute number of excitatory and inhibitory Connections depending on connection probability            
            Con_exc = int(self.epsilon*len(ng))  # number of excitatory synapses
            Con_inh = int(self.epsilon*len(inh_neurons))  # number of inhibitory synapses
            
            # NEST 2.2 Version: Fixed indegree (see initialize() for conn_option_dict): Connect all neurons
            RandomDivergentConnect(ng, inh_neurons, Con_exc, weight=1.0, delay=1.0, options = self.conn_option_dict)
            RandomDivergentConnect(inh_neurons, ng, Con_inh, weight=1.0, delay=1.0, options = self.conn_option_dict)
            
                          
            # NEST 2.7 Version: Connect excitatory and inhibitory neuron population to all neurons (NEST 2.7)
            #nest.Connect(ng, ng+inh_neurons, conn_params_exc, model = "excitatory")
            #nest.Connect(inh_neurons, ng+inh_neurons, conn_params_inh, model = "inhibitory")
        
        # Set Cue for inputlayer                  
        self.cue             = self.inputs[self.inputindex][1]
        
        # Set target for outputlayer
        self.target          = self.inputs[self.outputindex][1]

    def plotNetwork(self, start = 0., end = -1):
        '''Plots network independent of network size. If end is -1, then end is set to self.sim_time'''
        if end == -1:
            end = self.sim_time
        
        area_size = 0.5   
        mp.figure()
        for i in range(len(self.m['neurongroups'][0])):
            mp.subplot(len(self.m['neurongroups'][0])*2, 1, (i + 1))
            nh.scatter(self.sd_list[i]['exc'], area = area_size)
            mp.xlim((start, end))
            mp.ylabel(self.m['neurongroups'][0][i][0])
            mp.title('Excitatory neurons')
            
            mp.subplot(len(self.m['neurongroups'][0])*2, 1, (i + 3))
            mp.title('Inhibitory neurons')
            nh.scatter(self.sd_list[i]['inh'], area = area_size)            
            mp.xlim((start, end))
            mp.ylabel(self.m['neurongroups'][0][i][0])

            
        self.getPOA(True, [start, end]);
            
    
    def getPOA(self, printit = False, interval = [0, float('inf')]):
        '''Get percentage of activity for each network'''
        POA_Exc = [nh.getPOA(self.ngs[i], self.sd_list[i]['exc'], interval = [0,float('inf')]) for i in range(len(self.m['neurongroups'][0]))]
        POA_Inh = [nh.getPOA(self.ngs[i], self.sd_list[i]['inh'], interval = [0,float('inf')]) for i in range(len(self.m['neurongroups'][0]))]

    
        if printit:
            print('Percentage of Activity')
            for i in range(len(POA_Exc)):
                print(self.m['neurongroups'][0][i][0] + " " + str(POA_Exc[i]))
                print(self.m['neurongroups'][0][i][0] + " " + str(POA_Inh[i]))
        return POA_Exc, POA_Inh 