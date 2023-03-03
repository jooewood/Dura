#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:43:59 2022

@author: zdx
"""

from utils import sdf2csv
import pandas as pd
from utils import GetFileFormat, adjust_order

all_dfs = []

"""
-------------------------------------------------------------------------------
gnina
-------------------------------------------------------------------------------

"""

# score_files = ['/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket69/docked_default_known_actives.sdf.gz',
#                '/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket106/docked_default_known_actives.sdf.gz']
score_files = ['/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket69/docked_default_known_actives_new_0.sdf.gz',
               '/home/zdx/project/MDZT-1003/SMS2/dock/gnina_pocket106/docked_default_known_actives_new_0.sdf.gz']

actives_file = '/home/zdx/project/MDZT-1003/SMS2/inhibitors/known_actives.csv'
id_col='ID'
activity_col='IC50_um'
sort_col='CNNaffinity'
ascending=False
cut=False
pocketsizes = [69, 106]
sort_cols = ['minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS']
method = 'spearman'

total_corr = []
takeda_corr = []
fudan_corr = []

df_corrs = []
for method in ['pearson', 'kendall', 'spearman']:
    for sort_col in sort_cols:
    
        for cut in [True, False]:
            for score_file,pocketsize in zip(score_files, pocketsizes):    
                inScoreFileFormat = GetFileFormat(score_file)
                if inScoreFileFormat == 'sdf':
                    df_score = sdf2csv(score_file)
                elif inScoreFileFormat == 'csv':
                    df_score = pd.read_csv(score_file)
                    
                inKnownActivesFileFormat = GetFileFormat(actives_file)
                if inKnownActivesFileFormat == 'csv':
                    df_active = pd.read_csv(actives_file)
                if inKnownActivesFileFormat == 'sdf':
                    df_active = sdf2csv(actives_file)
                    
                df_score[id_col] = df_score[id_col].apply(str)
                df_active[id_col] = df_active[id_col].apply(str)
                
                intersection = set((df_score.columns).intersection(set(df_active.columns)))
                intersection.remove(id_col)
                
                df_score = df_score.merge(df_active.drop(columns=list(intersection)), 
                                          how='left', on=id_col)
                    
                
                for col in ['minimizedAffinity', 'CNNscore', 'CNNaffinity', 'CNN_VS',
                       'CNNaffinity_variance', 'IC50_um']:
                    df_score[col] = pd.to_numeric(df_score[col])
                    
                df_score.sort_values(sort_col, ascending=ascending, inplace=True)
                df_score.drop_duplicates(subset=id_col, inplace=True)
                
                if cut:
                    df_score = df_score.query("IC50_um<10")
        
                df_takeda = df_score.query("source=='w'")
                df_fudan = df_score.query("source=='f'")
                
                corrs = []
                
                corrs.append(df_score[['minimizedAffinity', 'CNNaffinity', 'CNN_VS', 'CNNscore',
                      'IC50_um']].corr(method).loc['IC50_um', ['minimizedAffinity', 'CNNaffinity', 'CNN_VS']])
                corrs.append(df_takeda[['minimizedAffinity', 'CNNaffinity', 'CNN_VS', 'CNNscore',
                    'IC50_um']].corr(method).loc['IC50_um', ['minimizedAffinity', 'CNNaffinity', 'CNN_VS']])
                corrs.append(df_fudan[['minimizedAffinity', 'CNNaffinity', 'CNN_VS', 'CNNscore',
                    'IC50_um']].corr(method).loc['IC50_um', ['minimizedAffinity', 'CNNaffinity', 'CNN_VS']])
        
                df_corr = pd.concat(corrs, axis=1)
                df_corr.columns = ['Total', 'Takeda', 'Fudan']
                df_corr = df_corr.round(3)
                
                df_corr = df_corr.applymap(abs)
                
                df_corr['pocket_size'] = pocketsize
                df_corr['sorted_by'] = sort_col
                df_corr['corr_method'] = method
                df_corr['cut'] = cut
                df_corrs.append(df_corr)
            
df = pd.concat(df_corrs)
df.reset_index(inplace=True)
df.rename(columns={'index':'Score_func'}, inplace=True)
df = adjust_order(df, pro=['pocket_size', 'corr_method', 'Total', 'Takeda', 'Fudan'])
df.sort_values(['pocket_size', 'corr_method'], inplace=True)
all_dfs.append(df)
df.to_excel(f'/home/zdx/project/MDZT-1003/result/gnina_SMS2_actives_corr_new_0.xlsx', index=False)

"""
MOE
"""

files = ['/home/zdx/project/MDZT-1003/SMS2/dock/moe/Q8NHU3_dock_pocket_size69.csv',
         '/home/zdx/project/MDZT-1003/SMS2/dock/moe/SMS2_docking_pocket_size106.csv'
         ]
cut=False
pocketsizes = [69, 106]

a = []
w = []
f = []
methods = []
pocket_sizes = []
cuts = []
for cut in [True, False]:
    for method in ['pearson', 'kendall', 'spearman']:
        for file,pocketsize in zip(files, pocketsizes):
            df = pd.read_csv(file)
            df.rename(columns=({'IC50(um)':'IC50'}), inplace=True)
            m = df.groupby(['ID'])['S'].max()
            df.drop_duplicates('ID', inplace=True)
            df.sort_values('ID', inplace=True)
            df.S = m.values
            
            if cut:
                df = df.query("IC50<10")
            
            df_wutian = df.query("source=='w'")
            df_fudan = df.query("source=='f'")
            # import seaborn as sns
            # sns.scatterplot(x="IC50", y="S", data=df);
            # sns.scatterplot(x="IC50", y="S", data=df_wutian);
            # sns.scatterplot(x="IC50", y="S", data=df_fudan);
            
            a.append(df[['IC50', 'S']].corr(method).iloc[1,0])
            w.append(df_wutian[['IC50', 'S']].corr(method).iloc[1,0])
            f.append(df_fudan[['IC50', 'S']].corr(method).iloc[1,0])
            pocket_sizes.append(pocketsize)
            methods.append(method)
            cuts.append(cut)
        
        
df = pd.DataFrame({
    'pocket_size':pocket_sizes,
    'corr_method': methods,
    'Total':a,
    'Takeda':w,
    'Fudan':f,
    'Score_func':'MOE',
    'sorted_by': 'S',
    'cut':cuts
    })
all_dfs.append(df)
df.to_csv('/home/zdx/project/MDZT-1003/result/moe_SMS2_actives_corr_max.csv', index=False)

"""
Smina 
"""
files = ['/home/zdx/project/MDZT-1003/SMS2/dock/Smina_pocket69_known_actives/scores.csv',
         '/home/zdx/project/MDZT-1003/SMS2/dock/Smina_pocket106_known_actives/scores.csv'
         ]
cut=False
pocketsizes = [69, 106]

a = []
w = []
f = []
methods = []
pocket_sizes = []
cuts = []
for cut in [True, False]:
    for method in ['pearson', 'kendall', 'spearman']:
        for file,pocketsize in zip(files, pocketsizes):
            df = pd.read_csv(file)
            df.rename(columns=({'IC50(um)':'IC50'}), inplace=True)
            m = df.groupby(['ID'])['S'].median()
            df.drop_duplicates('ID', inplace=True)
            df.sort_values('ID', inplace=True)
            df.S = m.values
            
            if cut:
                df = df.query("IC50<10")
            
            df_wutian = df.query("source=='w'")
            df_fudan = df.query("source=='f'")
            # import seaborn as sns
            # sns.scatterplot(x="IC50", y="S", data=df);
            # sns.scatterplot(x="IC50", y="S", data=df_wutian);
            # sns.scatterplot(x="IC50", y="S", data=df_fudan);
            
            a.append(df[['IC50', 'S']].corr(method).iloc[1,0])
            w.append(df_wutian[['IC50', 'S']].corr(method).iloc[1,0])
            f.append(df_fudan[['IC50', 'S']].corr(method).iloc[1,0])
            pocket_sizes.append(pocketsize)
            methods.append(method)
            cuts.append(cut)
        
        
df = pd.DataFrame({
    'pocket_size':pocket_sizes,
    'corr_method': methods,
    'Total':a,
    'Takeda':w,
    'Fudan':f,
    'cut':cuts
    })
df.to_csv('/home/zdx/project/MDZT-1003/result/moe_SMS2_actives_corr.csv', index=False)



