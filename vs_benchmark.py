#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 17:50:18 2022

@author: zdx
"""
import os
import time
import pandas as pd
from vs_lb_ml import vscv, mlcv, mlholdout, vsit, ML_names
from vs_lb_dl import deeppurposeholdout, DL_names
from utils import build_new_folder, get_time_string
from plot import vioCompare


model_list = ML_names + DL_names


def VSbenchmark(data_root, outdir, train_set, model_id, 
    decoy_source, active_folder='original_actives', mode='cls',
    
    test_sets=['MUV', 'LIT-PCBA'], 
    
    holdoutfunc=mlholdout, feature_type=None, 
    target_id_col='target_id',
    model_id_col = 'model_id', decoy_source_col = 'decoy_source', 
    cvfunc=mlcv, add_mol_descriptors=False,
    feature_cols=None, label_col='label', smiles_col="SMILES",
    downsample=False, overwrite=False):
    """
    feature_type: dude, deepcoy, descriptors, smiles

    data_root = '/home/zdx/data/VS_dataset/'
    outdir = '/home/zdx/project/VS_Benchmark'
    train_set = 'DUD-E'
    
    model_id='lr'
    target_id_col = 'target_id'
    cvfunc=mlcv
    model_id_col='model_id'
    decoy_source_col='decoy_source'
    decoy_folder = '4_original_decoys'
    active_folder='3_original_actives'
    feature_type='dude'
    feature_cols=None
    label_col='label'
    mode='cls'
    smiles_col="SMILES"
    downsample=False
    """

    if model_id in ['lr', 'lightgbm', 'catboost', 'nb', 'dt', 'svm', 'rf',
                    'rbfsvm', 'mlp', 'ridge', 'knn', 'gpc', 'qda', 'ada',
                    'gbc', 'lda', 'et', 'xgboost']:
        holdoutfunc = mlholdout
    elif model_id in DL_names:
        feature_type = 'smiles'
        add_mol_descriptors = False
        holdoutfunc = deeppurposeholdout
        feature_cols = ['SMILES']

    if feature_type == 'dude':
        feature_cols = ['MW', 'logP', 'HBA', 'HBD', 'NRB', 'formal_net_charge']
    elif feature_type == 'deepcoy':
        feature_cols = ['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
             'nitrogen_n', 'oxygen_n', 'fluorine_n', # 7
             'phosphorus_n', 'sulfur_n', 'chlorine_n', # 10
             'bromine_n', 'iodine_n', 'logP', 'HBA', # 14
             'HBD', 'rings', 'stereo_centers', 'MW', # 18
             'aromatic_rings', 'NRB', 'pos_charge', # 21
             'neg_charge', 'formal_net_charge', 'TPSA', # 24
             'SA', 'QED' # 26
             ]  
    elif feature_type == 'muv':
        feature_cols = ['atom_n', 'heavy_n', 'boron_n', 'carbon_n',# 4
             'nitrogen_n', 'oxygen_n', 'fluorine_n', # 7
             'phosphorus_n', 'sulfur_n', 'chlorine_n', # 10
             'bromine_n', 'iodine_n', 'logP', 'HBA', # 14
             'HBD', 'rings', 'stereo_centers' # 17 
             ]
    
    decoy_folder = f'{decoy_source}_decoys'

    ## target-specific on training set
    # sub_outdir = os.path.join(outdir, train_set, 'ts')
    # build_new_folder(sub_outdir)
    # outfile = os.path.join(sub_outdir, 
    #     f'{model_id}.{decoy_source}.{feature_type}.eval.csv')
    # eval_ts = btscv(
    #     data_root = data_root,
    #     dataset_name = train_set,
    #     decoy_folder=decoy_folder,
    #     active_folder=active_folder,
    #     feature_cols=feature_cols,
    #     decoy_source=decoy_source,
    #     downsample=downsample,
    #     label_col=label_col,
    #     mode=mode,
    #     compare=False,
    #     model_id=model_id,
    #     outfile=outfile,
    #     overwrite=overwrite
    #     )
    
    # Clustered 3-fold cross validation
    vscv(
        data_root=data_root,
        dataset_name=train_set,
        model_id=model_id,
        active_folder=active_folder,
        decoy_folder=decoy_folder,
        decoy_source=decoy_source,

        cvfunc=cvfunc,
        feature_type=feature_type,
        feature_cols=feature_cols,
        holdoutfunc=holdoutfunc,

        outdir=os.path.join(outdir, train_set),
        mode=mode,
        overwrite=overwrite,
        add_mol_descriptors=add_mol_descriptors
        )
    
    # Independent Test
    for test_set in test_sets:
        vsit(test_set=test_set, data_root=data_root, model_id=model_id,
            train_set=train_set, decoy_source=decoy_source, 
            feature_type=feature_type, holdoutfunc=holdoutfunc, 
            add_mol_descriptors=add_mol_descriptors, mode=mode,
            overwrite=overwrite,
            active_folder=active_folder,
            decoy_folder=decoy_folder,
            outdir_root=outdir,
            feature_cols=feature_cols
            )

def cvbenchmarkDecoy(indir, outdir, model_ids, cv_dataset, feature_type,
    decoy_sources=['original', 
    'deepcoy'], fold_n=3, figsize=(20,8), dpi=300, outfile=None, 
    model_id_col="model_id", hue="decoy_source", palette="muted",
    metrics=["AUC_ROC", "AUC_PRC", "EF1", "EF5", "EF10"],
    ylims=None):
    
    dfs_eval = []
    for model_id in model_ids:
        for decoy_source in decoy_sources:    
            eval_file = os.path.join(indir, cv_dataset, 
                f'{model_id}.{decoy_source}.{fold_n}-fold.eval.'
                f'{feature_type}.csv')
            dfs_eval.append(pd.read_csv(eval_file))
    df_eval = pd.concat(dfs_eval)
        
    if ylims is None:
        ylims = [None] * len(metrics)
        
    for m,ylim in zip(metrics, ylims):
        build_new_folder(outdir)
        outfile = os.path.join(outdir, f'{cv_dataset}.{m}.{feature_type}.png')
        vioCompare(df=df_eval, x_col=model_id_col, y_col=m, hue=hue,
                   palette=palette, figsize=figsize, dpi=dpi, outfile=outfile,
                   ylim=ylim)

def itbenchmarkDecoy(indir, outdir, model_ids, test_set, feature_type,train_set,
    decoy_sources=['original', 'deepcoy'], 
    
    figsize=(20,8), dpi=300, outfile=None, 
    model_id_col="model_id", hue="decoy_source", palette="muted",
    metrics=["AUC_ROC", "AUC_PRC", "EF1", "EF5", "EF10"],
    ylims=None):
    
    dfs_eval = []
    for model_id in model_ids:
        for decoy_source in decoy_sources:    
            eval_file = os.path.join(indir, test_set, 
                f'{model_id}.{train_set}.{decoy_source}.'
                f'{feature_type}.csv')
            dfs_eval.append(pd.read_csv(eval_file))
    df_eval = pd.concat(dfs_eval)
        
    if ylims is None:
        ylims = [None] * len(metrics)
    
    for m,ylim in zip(metrics, ylims):
        build_new_folder(outdir)
        outfile = os.path.join(outdir, f'{test_set}.{m}.{feature_type}.png')
        vioCompare(df=df_eval, x_col=model_id_col, y_col=m, hue=hue,
                   palette=palette, figsize=figsize, dpi=dpi, outfile=outfile,
                   ylim=ylim)
    
"""
--data_root
-o
--feature_type
--active_folder
--decoy_folder
--train_set
-m --model_id
--model_ids
--overwrite
--all

Examples:
./vs_benchmark.py -m DGL_AttentiveFP -f dude -t DUD-E -i /home/zdx/data/VS_dataset/ -o /home/zdx/project/VS_Benchmark
./vs_benchmark.py --all -f dude -t DUD-E -i /home/zdx/data/VS_dataset/ -o /home/zdx/project/VS_Benchmark
"""

if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('-m', '--model_id', default='lr')
    ap.add_argument('-f', '--feature_type', choices=['dude', 'muv', 'deepcoy', 
        'smiles', None], default=None)
    ap.add_argument('-i', '--data_root', help="Virtual screening data dir.")
    ap.add_argument('-d', '--decoy_source', default='original')
    ap.add_argument('-o', '--outdir')
    ap.add_argument('-t', '--train_set', default='DUD-E')
    ap.add_argument(('--test_sets'), default=['MUV', 'LIT-PCBA'])
    ap.add_argument('--compute', default=False, action='store_true',
        help="compute molecule descriptors.")
    ap.add_argument('--active_folder', choices=['original_actives', 
                                                'deepcoy_actives'],
        default='original_actives')
    ap.add_argument('--decoy_sources', default=['original', 'deepcoy'], 
                    nargs='+')
    ap.add_argument('--overwrite', action='store_true', default=False)
    ap.add_argument('--all', action='store_true', default=False)
    ap.add_argument('--model_ids', nargs='*', default=model_list)
    ap.add_argument('--mode', default='cls', choices=['reg', 'cls'])
    ap.add_argument('--downsample', action='store_true', default=False)
    
    args = ap.parse_args()
    
    if args.all:
        time_string = get_time_string()
        models = '.'.join(args.model_ids)
        logfile = os.path.join(args.outdir, f'{time_string}.{models}.log.'
            f'{args.feature_type}.txt')

        build_new_folder(args.outdir)
        with open(logfile, 'w') as f:
            for model_id in args.model_ids:
                print(model_id)
                f.write(model_id+' ')
                # try:
                for decoy_source in args.decoy_sources:
                    s = time.time()
                    VSbenchmark(
                        data_root=args.data_root, 
                        outdir=args.outdir, 
                        train_set=args.train_set,
                        test_sets=args.test_sets,
                        model_id=model_id, 
                        feature_type=args.feature_type,
                        decoy_source = args.decoy_source,
                        active_folder = args.active_folder,
                        add_mol_descriptors=args.compute,
                        overwrite=args.overwrite)
                    e = time.time()
                    time_cost = e-s
                    f.write(str(time_cost)+'s\n')
                # except:
                #     print("failed.")
                #     f.write('Failed\n')
                #     args.model_ids.remove(model_id)
                #     continue

        # benchmark_dir = os.path.join(args.outdir, 'Benchmark_pictures')
        # build_new_folder(benchmark_dir)
        
        # # Draw multiple models benchmark pictures
        # cvbenchmarkDecoy(indir=args.outdir, model_ids=args.model_ids, 
        #     feature_type=args.feature_type, cv_dataset=args.train_set,
        #     figsize=(len(args.model_ids), 8), outdir=benchmark_dir)
        
        # for test_set in args.test_sets:
        #     itbenchmarkDecoy(indir=args.outdir, test_set=test_set, 
        #         model_ids=args.model_ids, train_set = args.train_set, 
        #         feature_type=args.feature_type, figsize=(len(args.model_ids), 8),
        #         outdir=benchmark_dir)
    else:
        # single model run all benchmark experiments
        s = time.time()
        VSbenchmark(
            data_root=args.data_root, 
            outdir=args.outdir, 
            train_set=args.train_set,
            test_sets=args.test_sets,
            model_id=args.model_id,
            feature_type = args.feature_type,
            decoy_source = args.decoy_source,
            active_folder = args.active_folder,
            add_mol_descriptors=args.compute,
            overwrite = args.overwrite)
        e = time.time()
        print('Time cost:', round(e-s, 4), 'S')
