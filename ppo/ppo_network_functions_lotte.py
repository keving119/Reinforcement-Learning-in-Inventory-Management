# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:12:28 2020

@author: LotteH

This file contains the functions that are used to create the neural networks in the PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions.categorical import Categorical
import numpy as np
from gym.wrappers import FlattenObservation
from methods.ppo.ppo_support_functions import mlp

class Actor(nn.Module):
    '''
    General actor module
    '''
    def _distribution(self, obs):
        raise NotImplementedError
    
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    
    def forward(self, obs, act = None):
        '''
        Produce action distributions for given observations, and optionally compute the log likelihood of given actions
        under those distributions.
        '''
        pi = self._distribution(obs)
        logp_a = None
        entropy = pi.entropy()
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a, entropy

class MLPDiscreteActor(Actor):
    '''
    Discrete actor module
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        '''
        Initialize the actor with discrete actions. 
        This network is called 'pi_net'
        '''
        super().__init__()
        self.pi_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, output_activation = nn.Softmax)
    
    def _distribution(self, obs):
        '''
        Since it is a discrete distribution, we create a categorical distribution, parameterized by probabilities
        '''
        probs = self.pi_net(obs)
        return Categorical(probs = probs)
    
    def _log_prob_from_distribution(self, pi, act):
        '''
        Returns the log probability of an action based on the distribution
        '''
        return pi.log_prob(act)

class MLPCritic(nn.Module):
    '''
    Critic module for predicting the value of a state
    '''
    def __init__(self, obs_dim, hidden_sizes, activation):
        '''
        Initialize the critic. No output activation function.
        '''
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    
    def forward(self, obs):
        '''
        Returns the output of the critic network with obs as input
        '''
        return torch.squeeze(self.v_net(obs), -1)       # making sure that v has the right shape

class MLPActorCritic(nn.Module):
    '''
    Instantiating the actor critic structures.
    '''
    def __init__(self, env, hidden_sizes = (64, 64), activation = 'tanh'):
        '''
        Initializes the policy and value network. The weights are not shared. 
        If no other sizes or activations are specified, network has size (64, 64), and activation function Tanh.
        If no other values are specified, the initial values for the bias is 0, and we use normal glorot initialization.
        '''
        super().__init__()
        obs_dim = env.observation_space.shape[0]
        if activation == 'tanh':
            network_activation = nn.Tanh
        elif activation == 'relu':
            network_activation = nn.ReLU
        
        # Create networks for discrete actor and critic
        self.pi = MLPDiscreteActor(obs_dim, env.action_space.n, hidden_sizes, network_activation)
        self.v = MLPCritic(obs_dim, hidden_sizes, network_activation)
        self.nr_hidden_layers = len(hidden_sizes)
        
    
    def initialize_weights(self, bias_init = 0.0, init_type = 'glorot-uniform'):
        # Initialize weights, based on activation functions
        for i in [0, 2, 4]:
            init.constant_(self.v.v_net[i].bias, bias_init)
            init.constant_(self.pi.pi_net[i].bias, bias_init)
            activation_v = str(self.v.v_net[i + 1]).lower().replace('()', '')
            activation_pi = str(self.pi.pi_net[i + 1]).lower().replace('()', '')
            if activation_v == 'identity': activation_v = 'linear'
            if activation_pi == 'identity': activation_pi = 'linear' 
            if activation_pi == 'softmax(dim=1)': activation_pi = 'sigmoid'
            if init_type == 'glorot-normal':
                init.xavier_normal_(self.v.v_net[i].weight, gain = init.calculate_gain(activation_v))
                init.xavier_normal_(self.pi.pi_net[i].weight, gain = init.calculate_gain(activation_pi))
            elif init_type == 'glorot-uniform':
                init.xavier_uniform_(self.v.v_net[i].weight, gain = init.calculate_gain(activation_v))
                init.xavier_uniform_(self.pi.pi_net[i].weight, gain = init.calculate_gain(activation_pi))
            elif init_type =='orthogonal':
                init.orthogonal_(self.v.v_net[i].weight, gain = init.calculate_gain(activation_v))
                init.orthogonal_(self.pi.pi_net[i].weight, gain = init.calculate_gain(activation_pi))
            elif init_type == 'he-normal':
                init.kaiming_normal_(self.v.v_net[i].weight, nonlinearity = 'relu')
                init.kaiming_normal_(self.pi.pi_net[i].weight, nonlinearity = 'relu')
            elif init_type == 'he-uniform':
                init.kaiming_uniform_(self.v.v_net[i].weight, nonlinearity = 'relu')
                init.kaiming_uniform_(self.pi.pi_net[i].weight, nonlinearity = 'relu')
    
    def step(self, obs, action = None):
        '''
        Take an action, based on the input observation. This is sampled from the distribution. 
        Returns action, value of state, log probability of action, entropy of network.
        '''
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            entropy = pi.entropy()
            if action != None:
                a = torch.LongTensor([action])
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy(), entropy.numpy()
    
    def get_action_probs(self, obs):
        '''
        Returns a vector of the probability of taking each action
        '''
        with torch.no_grad():
            pi = self.pi._distribution(obs)
        return pi.probs.numpy().squeeze(axis = 0)
    
    def act(self, obs):
        '''
        Returns action taken with this observation
        '''
        return self.step(obs)[0]

#%% The part below is for testing purposes, 2 products
# from env.CLSPEnvironment import CLSP
# from ppo.ppo_support_functions import scale_input

# ENV_products                = 2                     # number of products in environment (int)
# ENV_capacity                = 3                     # hours of production (int)
# ENV_production_rate         = [5, 5]                # production rate per product per hour (int)
# ENV_setup_time              = [0.5, 0.5]            # hours of time to set up production (float)
# ENV_demand_mean             = [5, 3]                # mean demand per product (int)
# ENV_demand_variance         = [5, 3]                # variance of demand (different for normal and uniform)
# ENV_setup_cost              = [40, 40]              # cost of setting up production per setup
# ENV_holding_cost            = [1.0, 1.0]            # daily cost of holding inventory per unit
# ENV_backorder_cost          = [19, 19]              # backorder cost per period
# ENV_initial_inventory       = [0, 0]                # initial inventory if environment is reset
# ENV_inventory_limit         = 15                    # maximum inventory = x * demand
# ENV_backorders_once         = True                  # indicator for how often to count backorders
# RANDOM_SEED                 = 401                   # random seed to ensure reproducability

# env = CLSP(ENV_products, ENV_capacity, ENV_production_rate, ENV_setup_time, ENV_demand_mean, ENV_demand_variance,
#             ENV_setup_cost, ENV_holding_cost, ENV_backorder_cost, ENV_initial_inventory,
#             ENV_inventory_limit, ENV_backorders_once, RANDOM_SEED, hierarchy_actions = True)
# ac = MLPActorCritic(env)
# o = torch.as_tensor([[0.0, 1.0, 0.0, 0.0]], dtype = torch.float32)
# res = ac.step(o)
# #o = torch.as_tensor([[0.25, 0.0, 1.0, 0.0]], dtype = torch.float32)
# feasibility_mask = {'(1,0)': [1, 0, 0, 1, 1, 1, 1, 1, 0, 1]}

# #%% The part below is for testing purposes, 3 products
# from env.CLSPEnvironment import CLSP
# from ppo.ppo_support_functions import scale_input

# ENV_products                = 3                     # number of products in environment (int)
# ENV_capacity                = 3                     # hours of production (int)
# ENV_production_rate         = [5, 5, 5]                # production rate per product per hour (int)
# ENV_setup_time              = [0.5, 0.5, 0.5]            # hours of time to set up production (float)
# ENV_demand_mean             = [5, 3, 3]                # mean demand per product (int)
# ENV_demand_variance         = [5, 3, 3]                # variance of demand (different for normal and uniform)
# ENV_setup_cost              = [40, 40, 40]              # cost of setting up production per setup
# ENV_holding_cost            = [1.0, 1.0, 1.0]            # daily cost of holding inventory per unit
# ENV_backorder_cost          = [19, 19, 19]              # backorder cost per period
# ENV_service_level_target    = [0.98, 0.98, 0.98]          # target service level
# ENV_initial_inventory       = [0, 0, 0]                # initial inventory if environment is reset
# ENV_inventory_limit         = 15                    # maximum inventory = x * demand
# ENV_backorders_once         = True                  # indicator for how often to count backorders
# RANDOM_SEED                 = 401                   # random seed to ensure reproducability

# env = CLSP(ENV_products, ENV_capacity, ENV_production_rate, ENV_setup_time, ENV_demand_mean, ENV_demand_variance,
#             ENV_setup_cost, ENV_holding_cost, ENV_backorder_cost, ENV_service_level_target, ENV_initial_inventory,
#             ENV_inventory_limit, ENV_backorders_once, RANDOM_SEED)
# ac = MLPActorCritic(env.observation_space, env.action_space, env.feasible_actions)
# o = torch.as_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype = torch.float32)
# ac.step(o)
# #o = torch.as_tensor([[0.25, 0.0, 1.0, 0.0]], dtype = torch.float32)
# feasibility_mask = {'(1,0)': [1, 0, 0, 1, 1, 1, 1, 1, 0, 1]}