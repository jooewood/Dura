#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:05:39 2022

@author: zdx
"""
import os
import pandas as pd
from tqdm import tqdm
from utils import splitTrainTest

file = '/home/zdx/project/Flora/collect/硝化反应试剂.xlsx'
out_dir = '/home/zdx/src/t5chem/data/USPTO_500_MT/data/USPTO_500_MT/reagent_cls'
label_col='reagent_1_小类并大类'
task='cls'


def ClassDictFromLabel(labels):
    label2class = {}
    for i, l in enumerate(labels):
        label2class[l] = i
    
    class2label = {}
    for i, l in enumerate(labels):
        class2label[i] = l
    return label2class, class2label
        
def excel2rxnsmiles(file, label_col='reagent_1_小类并大类', out_dir=None, task='cls'):
    df = pd.read_excel(file)
    df = df[['substrate_1_smiles', 'product_1_smiles', label_col]]
    df.columns = ['substrate', 'product', 'label']
    df.dropna(inplace=True)
    
    
    labels = list(set(list(df.label.values)))
    label2class, class2label = ClassDictFromLabel(labels)
    
    cls_label = []
    for l in df.label:
        cls_label.append(label2class[l])
        
    df.label = cls_label
    
    print(df.label.value_counts())
    
    if task=='cls':
        labels = set(list(df['label']))
    
    train, val, test = splitTrainTest(df, frac=[0.8, 0.2], mode='cls', 
                                      label_col='label')
    test = val.copy()
    
    for df, set_ in zip([train, val, test], ['train', 'val', 'test']):
        sources = list(df['substrate'].str.cat(df['product'], sep='>>').values)
        targets = list(df['label'].values)
        with open(os.path.join(out_dir, f'{set_}.source'), 'w') as f:
            for s in sources:
                print(s, file=f)
                
        with open(os.path.join(out_dir, f'{set_}.target'), 'w') as f:
            for t in targets:
                print(t, file=f)