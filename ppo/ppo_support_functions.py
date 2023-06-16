# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:06:14 2020

@author: LotteH

This file contains the 'support functions' that support the PPO learning:
    Getting the mean, std, minimum, maximum of an array
    Setting random seeds
    Creating a neural network
    Returning the discounted cumulative sum
    Scaling the input of the neural network
"""

import math
import numpy as np
import random
import time
import torch
import torch.nn as nn
import scipy.signal
import os

def get_dict_from_input(filename):
    parcwd = os.path.realpath('..')
    filepath = os.path.join(parcwd, 'input_files', filename)
    inputfile = open(filepath, 'r')
    input_dict_as_string = inputfile.read()
    input_dict = eval(input_dict_as_string)
    return input_dict

def statistics_scalar(x, with_min_max = False):
    '''
    Get mean and std of array x. 
    If with_min_max = True, also the minimum and maximum are returned
    '''
    x = np.array(x, dtype = np.float32)
    global_sum, global_n = sum(x), len(x)
    mean = global_sum / global_n
    
    global_sum_sq = (np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)
    
    if with_min_max:
        global_min = np.min(x)
        global_max = np.max(x)
        return mean, std, global_min, global_max
    
    return mean, std

def set_seeds(env, seed):
    '''
    Set random seeds for several functions:
        Random
        Numpy
        Torch
        Env action space
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.action_space.seed(seed)

def mlp(sizes, activation, output_activation=nn.Identity):
    '''
    This function creates a neural network with (sizes) layers (1 input, x hidden layers, 1 output layer)
    - sizes:                nr of total layers
    - activation:           activation function of hidden layers
    - output_activation:    activation function of output layer. Default is no activation.
    '''
    layers = []
    for j in range(len(sizes)-1):
        if j < len(sizes) - 2:                  # for all the hidden layers
            act = activation() 
        elif output_activation == nn.Softmax:   # for the output layer of the actor network
            act = output_activation(1)
        else:                                   # for the output layer of the critic network
            act = output_activation()
        layers += [nn.Linear(sizes[j], sizes[j+1]), act]
    return nn.Sequential(*layers)

def discount_cumsum(x, discount_factor, horizon):
    """
    formula from rllab for computing discounted cumulative sums of vectors.
    input:  
        [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """

    #OLD:
    return scipy.signal.lfilter([1], [1, float(-discount_factor)], x[::-1], axis=0)[::-1]



def scale_input(env, state):
    '''
    Scale the input data, such that it is between min and max
    '''
    min = env.case.state_scale_low
    max = env.case.state_scale_high
    scaled_input = ((max - min) *((state - env.observation_space.low)/(env.observation_space.high - env.observation_space.low))) + min
    return scaled_input

def print_weight_bias_grad(ac, msg):
    '''
    This function prints the weights of the neural network, as well as the gradients.
    '''
    weight_idx = [i * 2 for i in range(ac.nr_hidden_layers + 1)]
    print(weight_idx)
    print(msg)
    for i in weight_idx:
        print('layer prob net: ', i, '\n',
              'weight: ', ac.pi.pi_net[i].weight, '\n',
              'bias: ', ac.pi.pi_net[i].bias, '\n', 
              'grad weight: ', ac.pi.pi_net[i].weight.grad, '\n',
              'grad bias: ', ac.pi.pi_net[i].bias.grad)
    for i in weight_idx:
        print('layer value net: ', i, '\n',
              'weight: ', ac.v.v_net[i].weight, '\n',
              'bias: ', ac.v.v_net[i].bias, '\n', 
              'grad weight: ', ac.v.v_net[i].weight.grad, '\n',
              'grad bias: ', ac.v.v_net[i].bias.grad)


def compute_entropy_threshold(n, fraction):
    p = 1.0/n
    entropy = -sum([p * math.log2(p) for _ in range(n)])
    return fraction * entropy

class SimulationBuffer():
    """
    Class for filling a simulation buffer. Has the following functions:
        Init - instantiate, create empty cost, inventory, backorder, setup, demand buffer
        store - save observation to buffer
        next run - increment run index, reset step index
        compute_statistics - returns mean and standard deviation of all runs of buffer
        compute_confidence_interval - returns mean, side and percentage of side, excluding the warmup period
        determine_interval - sets the confidence intervals for all buffers
        fill_buffer - fills the buffer with x runs and y steps
    """
    def __init__(self, simulation_length, simulation_runs, warmup_period):
        self.simulation_length = simulation_length
        self.simulation_runs = simulation_runs
        self.warmup_period = warmup_period
        self.run, self.step = 0, 0
        self.cost_buf = np.zeros((simulation_runs), dtype = np.float32)
        
    def store(self, costs):
        self.cost_buf[self.run] = costs
    
    def next_run(self):
        self.run += 1
        self.step = 0
    
    def compute_statistics(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return mean, std
    
    def compute_confidence_interval(self, data):
        mean, std = self.compute_statistics(data)
        side = 1.96 * std / math.sqrt(self.simulation_runs)
        percentage = (side / mean) * 100
        return np.round(mean, decimals = 2), np.round(side, decimals = 2), np.round(abs(percentage), decimals = 2)
        
    def determine_confidence_intervals(self):
        self.confidence_intervals = {'cost': self.compute_confidence_interval(self.cost_buf)}
        return self.confidence_intervals
    
    def fill_buffer(self, env, model):
        for i in range(self.simulation_runs):
            o = env.reset()
            with torch.no_grad():
                totalreward, totalholdingcosts, totalbackordercosts = 0, 0, 0
                for t in range(self.simulation_length):
                    o = torch.as_tensor(scale_input(env, o), dtype = torch.float32)
                    a, _, _, _, _ = model.step(o, deterministic=True)
                    next_o, r, _, info = env.simulate(a)
                    if t >= self.warmup_period: 
                        totalholdingcosts += info['holding_costs']
                        totalbackordercosts += info['backorder_costs']
                        totalreward += r
                    o = next_o
                self.store(totalreward)
            self.next_run()

def evaluate_policy(ac, env, simulation_runs, simulation_length, warmup_period, no_improvement_count, 
                    best_evaluation_mean, best_evaluation_upperbound):
    '''
    This function simulates the current policy of the neural network for x simulation runs and x simulation length.
    The first x periods (warm up period) are discarded.
    Returns average costs per period with a 95% confidence interval, Boolean of stopping criterion
    '''
    # instantiate simulation buffer, fill it, determine confidence intervals
    simbuffer = SimulationBuffer(simulation_length, simulation_runs, warmup_period)
    simbuffer.fill_buffer(env, ac)
    # mean_costs = simbuffer.cost_buf[]
    evaluation = simbuffer.determine_confidence_intervals()
    mean_costs = evaluation['cost'][0]
    
    # check if the mean is outside the previous confidence interval
    if mean_costs > best_evaluation_upperbound:
        conclusion = 'improvement'
        no_improvement_count = 0
        evaluation_mean = mean_costs
        #KLopt dit als upperbound?
        evaluation_upperbound = mean_costs + evaluation['cost'][1]
    else:
        conclusion = 'no improvement'
        no_improvement_count +=1
        evaluation_mean = best_evaluation_mean
        evaluation_upperbound = best_evaluation_upperbound
    
    print('policy evaluated with {}, costs: {}, upperbound: {}'.format(conclusion, mean_costs, evaluation_upperbound))
    print('consecutive evaluation periods without improvement: {}'.format(no_improvement_count))
    
    return mean_costs, evaluation_mean, evaluation_upperbound, no_improvement_count