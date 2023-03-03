#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:14:22 2022

@author: zdx
"""

from docking import docking_main, docking_dataset


#docking_dataset('/home/zdx/data/VS_dataset/DUD-E', '3_original_actives')

#docking_dataset('/home/zdx/data/VS_dataset/DUD-E', '4_original_decoys')

docking_dataset('/home/zdx/data/VS_dataset/DUD-E', 'deepcoy_decoys')
