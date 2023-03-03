#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 21:10:14 2022

@author: zdx
"""
import os
import pandas as pd
from glob import glob



# Cross-validation result analysis
data_root = '/home/zdx/project/VS_Benchmark/'
dataset = 'DUD-E'


indir = os.path.join(data_root, dataset)
files = glob(os.path.join(indir, '*.csv'))
eval_files = [x for x in files if '.eval.' in x]

dfs = []
for file in eval_files:
    # file = '/home/zdx/project/VS_Benchmark/DUD-E/ada.original.3-fold.eval.dude.csv'
    # file = '/home/zdx/project/VS_Benchmark/DUD-E/quickscore_v1.top3_undersample_total.top1_pose.original.3-fold.eval.knn.csv'
    df = pd.read_csv(file)
    if not 'quickscore' in file:
        df["model_info"] = df["model_id"].map(str).str.cat([df["feature_type"]], sep='.')
    else:
        df["model_info"] = df["model_id"].map(str).str.cat([df["train_strategy"], df['pred_strategy'], df["feature_type"]], sep='.')
        
    dfs.append(df[['AUC_ROC', 'AUC_PRC', 'EF1', 'EF5', 'EF10', 'target_id', 'decoy_source', 'model_info']])
    
df = pd.concat(dfs)
df_mean = df.groupby('model_info').mean()
df_median = df.groupby('model_info').median()

# Independent test result analysis
data_root = '/home/zdx/project/VS_Benchmark/'
dataset = 'LIT-PCBA'

indir = os.path.join(data_root, dataset)
files = glob(os.path.join(indir, '*.csv'))
eval_files = [x for x in files if '.eval.' in x]

dfs = []
for file in eval_files:
    # file = '/home/zdx/project/VS_Benchmark/LIT-PCBA/ada.DUD-E.original.eval.deepcoy.csv'
    # file = '/home/zdx/project/VS_Benchmark/DUD-E/quickscore_v1.top3_undersample_total.top1_pose.original.3-fold.eval.knn.csv'
    df = pd.read_csv(file)
    if not 'quickscore' in file:
        df["model_info"] = df["model_id"].map(str).str.cat([df["feature_type"]], sep='.')
    else:
        df["model_info"] = df["model_id"].map(str).str.cat([df["train_strategy"], df['pred_strategy'], df["feature_type"]], sep='.')
        
    dfs.append(df[['AUC_ROC', 'AUC_PRC', 'EF1', 'EF5', 'EF10', 'target_id', 'decoy_source', 'train_set', 'model_info']])

df = pd.concat(dfs)
df_mean = df.groupby('model_info').mean()
df_median = df.groupby('model_info').median()