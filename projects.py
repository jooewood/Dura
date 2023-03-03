#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 10:08:27 2022

@author: zdx
"""
import pandas as pd
from plot import df_feature_comparison, mol_file_properties_comparison
from utils import sdf2csv, adjust_order


"""
MDZT-1003
"""
## Draw picture
actives = pd.read_csv('/home/zdx/project/MDZT-1003/compute/moe_docking/dock_active225_lowest.csv')
df_right = pd.read_csv('/home/zdx/project/MDZT-1003/compute/moe_docking/dock_1420_lowest.csv')
Fudan = actives.query('source=="Fudan"')
Takeda = actives.query('source=="Takeda"')
# Fudan vs Takeda
df_feature_comparison(
    Fudan,
    Takeda,
    columns=['MW', 'logP', 'logS', 'SA', 'QED', 'TPSA', 'HBA', 'HBD', 
              'NRB', 'rings'],
    title = None,
    output_file = '/home/zdx/project/MDZT-1003/picture/SMS2_Fudan_vs_Takeda.png',
    left_name='SMS2 Fudan actives',
    right_name='SMS2 Takeda actives'
    )

# Fudan vs AI
df_feature_comparison(
    Fudan,
    df_right,
    columns=['MW', 'logP', 'logS', 'SA', 'QED', 'TPSA', 'HBA', 'HBD', 
              'NRB', 'rings'],
    title = None,
    output_file = '/home/zdx/project/MDZT-1003/picture/SMS2_Fudan_vs_AI.png',
    left_name='SMS2 Fudan actives'
    )

# Takeda vs AI
df_feature_comparison(
    Takeda,
    df_right,
    columns=['MW', 'logP', 'logS', 'SA', 'QED', 'TPSA', 'HBA', 'HBD', 
              'NRB', 'rings'],
    title = None,
    output_file = '/home/zdx/project/MDZT-1003/picture/SMS2_Takeda_vs_AI.png',
    left_name='SMS2 Takeda actives'
    )
# All actives vs AI
mol_file_properties_comparison(
    '/home/zdx/project/MDZT-1003/compute/moe_docking/dock_active225_lowest.csv',
    '/home/zdx/project/MDZT-1003/compute/moe_docking/dock_1420_lowest.csv',
    columns=['MW', 'logP', 'logS', 'SA', 'QED', 'TPSA', 'HBA', 'HBD', 
              'NRB', 'rings'],
    title = None,
    output_file = '/home/zdx/project/MDZT-1003/picture/SMS2_all_vs_AI.png',
    left_name='SMS2 all actives'
    )

## Collect Smina and gnina scoring
def get_vs_rank(df, target_id, method_id='gnina'):
    df.sort_values('CNN_VS', inplace=True, ascending=False)
    df.drop_duplicates(subset='ID', inplace=True)
    df[f'{target_id}_{method_id}_rank'] = range(0, len(df))
    return df[['ID', f'{target_id}_{method_id}_rank']]

sms1_gnina_df = sdf2csv('/home/zdx/project/MDZT-1003/SMS1/dock/gnina_size100/docked_default.sdf.gz')
df1 = get_vs_rank(sms1_gnina_df, 'sms1')

smsr_gnina_df = sdf2csv('/home/zdx/project/MDZT-1003/SMSr/dock/gnina_size223/docked_default.sdf.gz')
df2 = get_vs_rank(smsr_gnina_df, 'smsr')

sms2_69_gnina_df = sdf2csv('/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket69/docked_default.sdf.gz')
df3 = get_vs_rank(sms2_69_gnina_df, 'sms2_69')

sms2_106_gnina_df = sdf2csv('/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket106/docked_default.sdf.gz')
df4 = get_vs_rank(sms2_106_gnina_df, 'sms2_106')

dfs = [df3, df4, df1, df2]
from functools import reduce
final_df = reduce(lambda  left,right: pd.merge(left,right,on=['ID'],
                                            how='left'), dfs)


df = sms2_69_gnina_df.merge(final_df, how='left')
df = adjust_order(df, pro=['ID', 'SMILES', 'sms2_69_gnina_rank', 'sms2_106_gnina_rank',
 'sms1_gnina_rank', 'smsr_gnina_rank', 'hERG'])

herg = pd.read_csv('/home/zdx/Downloads/pred_result.csv')

df = df.merge(herg, how='left')
df.to_csv('/home/zdx/project/MDZT-1003/SMS2/SMS2_VS_ranked.csv', index=False)

file = '/home/zdx/project/MDZT-1003/SMS2/out.csv'

rd_filter_df = pd.read_csv(file)
rd_filter_df.rename(columns={'NAME':'ID', 'FILTER':'PAINs'}, inplace=True)
df = pd.read_csv('/home/zdx/project/MDZT-1003/SMS2/SMS2_VS_ranked.csv')
df = df.merge(rd_filter_df[['ID', 'PAINs']], how='left', on='ID')

from utils import add_features

df = add_features(df)
df = adjust_order(df, pro=['ID', 'SMILES', 'sms2_69_gnina_rank', 'sms2_106_gnina_rank',
 'sms1_gnina_rank', 'smsr_gnina_rank', 'hERG','PAINs'])
df.sort_values('sms2_69_gnina_rank', inplace=True)
df.to_csv('/home/zdx/project/MDZT-1003/SMS2/SMS2_VS_ranked.csv', index=False)

"""
Draw structure alerts
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pandas as pd
import numpy as np

df = pd.read_csv('/home/zdx/src/rd_filters/rd_filters/data/alert_collection.csv')

smarts_list = list(df.smarts)
legends = list(df.description)

ms = [Chem.MolFromSmarts(s) for s in smarts_list]
for m in ms: tmp=AllChem.Compute2DCoords(m)

draws = []
failed = []
for m,l,s in zip(ms,legends, smarts_list):
    try:
        img = Draw.MolsToImage([m], subImgSize=(300,300), legends=[l])
        img.save(f'/home/zdx/src/rd_filters/rd_filters/data/structure_alerts/{l}.png')
        draws.append('')
    except:
        failed.append(s)
        draws.append('Draw_failed')


df['Draw'] = draws
df.to_csv('/home/zdx/src/rd_filters/rd_filters/data/alert_collection_draw.csv', index=False)