#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 09:28:42 2022

@author: zdx
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

in_dir = '/home/zdx/data/VS_dataset/DUD-E'
target_dir = os.path.join(in_dir, '1_targets')

targets = os.listdir(target_dir)
targets = [x.replace('.pdb', '') for x in targets]

original_actives_dir = os.path.join(in_dir, '3_original_actives')
original_decoys_dir = os.path.join(in_dir, '4_original_decoys')

deepcoy_actives_dir = os.path.join(in_dir, '6_deepcoy_actives')
deepcoy_decoys_dir = os.path.join(in_dir, '5_deepcoy_decoys')

original_active_ns = []
original_decoy_ns = []
deepcoy_active_ns = []
deepcoy_decoy_ns = []

for target in tqdm(targets):
    # target = targets[0]
    original_active_file = os.path.join(original_actives_dir, f'{target}.csv')
    original_active = pd.read_csv(original_active_file)
    original_active_n = len(original_active)
    original_active_ns.append(original_active_n)
    
    original_decoy_file = os.path.join(original_decoys_dir,  f'{target}.csv')
    original_decoy = pd.read_csv(original_decoy_file)
    original_decoy_n = len(original_decoy)
    original_decoy_ns.append(original_decoy_n)
    
    deepcoy_active_file = os.path.join(deepcoy_actives_dir, f'{target}.csv')
    deepcoy_active = pd.read_csv(deepcoy_active_file)
    deepcoy_active_n = len(deepcoy_active)
    deepcoy_active_ns.append(deepcoy_active_n)
    
    deepcoy_decoy_file = os.path.join(deepcoy_decoys_dir, f'{target}.csv')
    deepcoy_decoy = pd.read_csv(deepcoy_decoy_file)
    deepcoy_decoy_n = len(deepcoy_decoy)
    deepcoy_decoy_ns.append(deepcoy_decoy_n)


targets.append('MEAN')
original_active_ns.append(np.mean(original_active_ns))
deepcoy_active_ns.append(np.mean(deepcoy_active_ns))
original_decoy_ns.append(np.mean(original_decoy_ns))
deepcoy_decoy_ns.append(np.mean(deepcoy_decoy_ns))

    
df = pd.DataFrame({
    'target': targets,
    'original_active_n': original_active_ns,
    'deepcoy_active_n': deepcoy_active_ns,
    'original_decoy_n': original_decoy_ns,
    'deepcoy_decoy_n': deepcoy_decoy_ns,
    'original_average_decoy_n': np.array(original_decoy_ns)/np.array(original_active_ns),
    'deepcoy_average_decoy_n': np.array(deepcoy_decoy_ns)/np.array(deepcoy_active_ns)
    })

df[['original_active_n', 'deepcoy_active_n', 
    'original_decoy_n', 'deepcoy_decoy_n',
    'original_average_decoy_n',
    'deepcoy_average_decoy_n']] = \
    df[['original_active_n', 'deepcoy_active_n', 
    'original_decoy_n', 'deepcoy_decoy_n',
    'original_average_decoy_n',
    'deepcoy_average_decoy_n']].applymap(int)

df.to_csv(os.path.join(in_dir, 'summary.csv'), index=False)


in_dir = '/home/zdx/data/VS_dataset/MUV'
target_dir = os.path.join(in_dir, '3_original_actives')

targets = os.listdir(target_dir)
targets = [x.replace('.csv', '') for x in targets]

original_actives_dir = os.path.join(in_dir, '3_original_actives')
original_decoys_dir = os.path.join(in_dir, '4_original_decoys')

original_active_ns = []
original_decoy_ns = []

for target in tqdm(targets):
    # target = targets[0]
    original_active_file = os.path.join(original_actives_dir, f'{target}.csv')
    original_active = pd.read_csv(original_active_file)
    original_active_n = len(original_active)
    original_active_ns.append(original_active_n)
    
    original_decoy_file = os.path.join(original_decoys_dir,  f'{target}.csv')
    original_decoy = pd.read_csv(original_decoy_file)
    original_decoy_n = len(original_decoy)
    original_decoy_ns.append(original_decoy_n)


targets.append('MEAN')
original_active_ns.append(np.mean(original_active_ns))
original_decoy_ns.append(np.mean(original_decoy_ns))
    
df = pd.DataFrame({
    'target': targets,
    'original_active_n': original_active_ns,
    'original_decoy_n': original_decoy_ns,
    'original_average_decoy_n': np.array(original_decoy_ns)/np.array(original_active_ns),
    })

df[['original_active_n', 
    'original_decoy_n',
    'original_average_decoy_n']] = \
    df[['original_active_n', 
    'original_decoy_n',
    'original_average_decoy_n']].applymap(int)

df.to_csv(os.path.join(in_dir, 'summary.csv'), index=False)
