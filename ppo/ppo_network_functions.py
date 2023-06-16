# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:12:28 2020

@author: LotteH

This file contains the functions that are used to create the neural networks in the PPO algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from gym.spaces import Box, Discrete
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from ppo.ppo_support_functions import mlp

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
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


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
        Adjust function if more than 2 products
        '''
        probs = self.pi_net(obs)
        return Categorical(probs = probs)

    def _log_prob_from_distribution(self, pi, act):
        '''
        Returns the log probability of an action based on the distribution
        '''
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.pi_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.pi_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


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
    def __init__(self, observation_space, action_space, feasible_actions, hidden_sizes = (64, 64), activation = nn.Tanh, 
                 bias_init = 0.0, glorot_type = 'uniform'):
        '''
        Initializes the policy and value network. The weights are not shared. 
        If no other sizes or activations are specified, network has size (64, 64), and activation function Tanh.
        If no other values are specified, the initial values for the bias is 0, and we use normal glorot initialization.
        '''
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPDiscreteActor(obs_dim, action_space.n, hidden_sizes, activation)
        # Create networks for discrete actor and critic
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.nr_hidden_layers = len(hidden_sizes)
        
        # Initialize weights, based on activation functions
        for i in [0, 2, 4]:
            init.constant_(self.v.v_net[i].bias, bias_init)
            init.constant_(self.pi.pi_net[i].bias, bias_init)
            activation_v = str(self.v.v_net[i + 1]).lower().replace('()', '')
            activation_pi = str(self.pi.pi_net[i + 1]).lower().replace('()', '')
            if activation_v == 'identity': activation_v = 'linear'
            if activation_pi == 'identity': activation_pi = 'linear'
            if activation_pi == 'softmax(dim=1)': activation_pi = 'sigmoid'
            if glorot_type == 'normal':
                init.xavier_normal_(self.v.v_net[i].weight, gain = init.calculate_gain(activation_v))
                init.xavier_normal_(self.pi.pi_net[i].weight, gain = init.calculate_gain(activation_pi))
            elif glorot_type == 'uniform':
                init.xavier_uniform_(self.v.v_net[i].weight, gain = init.calculate_gain(activation_v))
                init.xavier_uniform_(self.pi.pi_net[i].weight, gain = init.calculate_gain(activation_pi))
    
    def step(self, obs, deterministic=False):
        '''
        Take an action, based on the input observation. This is sampled from the distribution. 
        Returns action, value of state, log probability of action, entropy of network.
        '''
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            entropy = pi.entropy().mean().item()
            stddev = pi.stddev
            if deterministic:
                a = pi.mean
            else:
                a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy(), entropy, stddev.numpy()
    
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