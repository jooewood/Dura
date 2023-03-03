#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Fernie(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        ## hyperparameters for model architecture
        self.cf = config.cf
        self.h = config.h 
        self.datm = config.datm
        self.damino = config.damino 
        self.dchrg = config.dchrg 
        self.ddist = config.ddist 
        self.A = config.A 
        self.D = config.D
        self.C = config.C 
        self.R = config.R 
        self.kc = config.kc 
        self.kp = config.kp 
        self.max_atom_num = config.max_atom_num
        self.zi = (config.datm + config.dchrg + config.ddist) * (config.kc +\
            config.kp) + config.damino * config.kp
        
        ## model layer hyperparameters
        self.pool_type = config.pool_type
        self.activation = config.activation
        self.dp_rate = config.dp_rate
        self.batchnorm = config.batchnorm

        if self.dp_rate == 0:
            self.dp_rate = False

        if self.activation not in ['relu', 'sigmoid', 'tahn',  'leakyrelu',\
            'gelu', 'elu']:
            self.use_activation = False
        else:
            self.use_activation = True

        self.fc1 = nn.Linear(self.cf, self.h, bias=False)
        self.fc2 = nn.Linear(self.h, 2)
        self.flatten1 = nn.Flatten()

        self.conv1 = nn.Conv2d(1, self.cf, (1, self.zi), 1, bias=False)

        self.embedding_atmo = nn.Embedding(self.A, self.datm)
        self.embedding_chrg = nn.Embedding(self.C, self.dchrg)
        self.embedding_dist = nn.Embedding(self.D, self.ddist)
        self.embedding_amin = nn.Embedding(self.R, self.damino)

        if self.dp_rate:
            self.dp2d = nn.Dropout2d(self.dp_rate)
            self.dp1d = nn.Dropout(self.dp_rate)
            
        if self.use_activation:
            if self.activation=='relu':
                self.activation = nn.ReLU()
            if self.activation=='sigmoid':
                self.activation = nn.Sigmoid()
            if self.activation=='tahn':
                self.activation = nn.Tanh()
            if self.activation=='elu':
                self.activation = nn.ELU()
            if self.activation=='leakyrelu':
                self.activation = nn.LeakyReLU()
            if self.activation=='gelu':
                self.activation = nn.GELU()
                
        if self.batchnorm:
            self.bnm1 = nn.BatchNorm2d(1)
            self.bnm2 = nn.BatchNorm2d(self.cf)
            self.bnm3 = nn.BatchNorm1d(self.cf)
            self.bnm4 = nn.BatchNorm1d(self.h)

        self.convolution1 = nn.Sequential()
        if self.batchnorm:
            self.convolution1.add_module('bnm1', self.bnm1)
        if self.use_activation:
            self.convolution1.add_module('activation1', self.activation)
        if self.dp_rate:
            self.convolution1.add_module('dp2d1', self.dp2d)

        self.convolution1.add_module('conv1', self.conv1)
        
        self.pooling = nn.Sequential()
        if self.batchnorm:
            self.pooling.add_module('bnm2', self.bnm2)
        if self.use_activation:
            self.pooling.add_module('activation2', self.activation)
        if self.dp_rate:
            self.pooling.add_module('dp2d2', self.dp2d)

        if self.pool_type == 'max':
            self.pool1 = nn.MaxPool2d((self.max_atom_num, 1), stride=1)
        elif self.pool_type == 'avg':
            self.pool1 = nn.AvgPool2d((self.max_atom_num, 1), stride=1)
        
        self.pooling.add_module('pool1', self.pool1)


        self.pooling.add_module('flatten1', self.flatten1)
        if self.batchnorm:
            self.pooling.add_module('bnm3', self.bnm3)
        if self.use_activation:
            self.pooling.add_module('activation3', self.activation)
        if self.dp_rate:
            self.pooling.add_module('dp1d1', self.dp1d)
        
        self.fullyconnect1 = nn.Sequential()
        self.fullyconnect1.add_module('fc1', self.fc1)
        # Batchnorm layer
        if self.batchnorm:
            self.fullyconnect1.add_module('bnm4', self.bnm4)
        # Activation layer
        if self.use_activation:
            self.fullyconnect1.add_module('activation4', self.activation)
        # Dropout layer
        if self.dp_rate:
            self.fullyconnect1.add_module('dr1d2', self.dp1d)

    def forward(self, x):
        """
        new version forward
        -----------------------------------------------------------------------
        Embedding layer
        -----------------------------------------------------------------------
        """
        em_atmo = self.embedding_atmo(x[0]).view(-1, self.max_atom_num, 
            (self.kc + self.kp) * self.datm)
        em_chrg = self.embedding_chrg(x[1]).view(-1, self.max_atom_num,
            (self.kc + self.kp) * self.dchrg)
        em_dist = self.embedding_dist(x[2]).view(-1, self.max_atom_num,
            (self.kc + self.kp) * self.ddist)
        em_amin = self.embedding_amin(x[3]).view(-1, self.max_atom_num,
            self.kp * self.damino)

        """
        -----------------------------------------------------------------------
        Cat layer
        -----------------------------------------------------------------------
        """
        out = torch.cat([em_atmo, em_chrg, em_dist, em_amin], 2).view(-1, 1,\
            self.max_atom_num, self.zi)

        """
        -----------------------------------------------------------------------
        First Convoluation layer
        -----------------------------------------------------------------------
        """
        out = self.convolution1(out)
        """
        -----------------------------------------------------------------------
        mask layer
        -----------------------------------------------------------------------
        """
        out = out * x[4]
        """
        -----------------------------------------------------------------------
        Max-pooling layer
        -----------------------------------------------------------------------
        """
        out = self.pooling(out)
        """
        -----------------------------------------------------------------------
        First fully connected layer
        -----------------------------------------------------------------------
        """
        out = self.fullyconnect1(out)
        """
        -----------------------------------------------------------------------
        Output layer (Classifier)
        -----------------------------------------------------------------------
        """
        # fully connected layer
        out = self.fc2(out)
        out = F.log_softmax(out, dim = 1)
        return out