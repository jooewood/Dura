#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:22:51 2022

@author: zdx
"""

import os
import re
import pandas as pd
from tqdm import tqdm
import glob
from quickscore_v1.main import main as quickscore_v1_main
from utils import GetFileName, get_time_string, delete_a_folder, readJson,\
    build_new_folder, eval_one_target, delete_a_file, get_dataset_target_ids
from plot import DataFrameDistribution

def quickscore_v1_holdout(test=None, train=None, mode='cls', pred_save=None,
    overwrite=False, result_dir=None, train_strategy=None, gpu_device=0,
    remove_checkpoint = False):
    """
    test: test pickle files: test1.pickle, test2.pickle, test3.pickle
    train: train pickle files: train1.pickle, train2.pickle, train3.pickle
    result_dir: model output dir
    
    result_dir = '/home/zdx/src/dura/dura/quickscore_v1.2022.08.31_11.01.55'
    """
    # train

    if result_dir is not None and pred_save is None:
        pred_save = os.path.join(result_dir, 'out', 'pred.csv')

    if os.path.exists(pred_save):
        return pd.read_csv(pred_save)
    
    if train is not None:
        if result_dir is None:
            remove_checkpoint = True
            current = os.path.dirname(os.path.realpath(__file__))
            result_dir = os.path.join(current, 
                                      f'quickscore_v1.{get_time_string()}')
        result_dir = quickscore_v1_main('train', 
                                        train_strategy=train_strategy,
                                        train_file=train,
                                        result_dir=result_dir,
                                        gpu_device=gpu_device)
    if test is not None:
        if isinstance(test, str):
            pred = quickscore_v1_main('predict', result_dir=result_dir,
                pred_file=test)
        elif isinstance(test, list):
            pred_dfs = []
            for t in test:
                pred_sub = quickscore_v1_main('predict', result_dir=result_dir,
                    pred_file=t, gpu_device=gpu_device)
                pred_dfs.append(pred_sub)
            pred = pd.concat(pred_dfs)
            
        if remove_checkpoint:
            delete_a_folder(result_dir)
        if pred_save is not None:
            pred.to_csv(pred_save, index=False)
        return pred
    
def get_feature_files(target_ids, data_root, dataset_name, model_id, 
                      decoy_source,
                      feature_type='knn', docktool='smina', suffix='pickle'):
    
    if 'quickscore_v1' in model_id:
        feature_type = 'knn'
        suffix = 'pickle'
        
    files = []
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    for target_id in target_ids:
        # target_id = 'aa2ar'
        active_file = os.path.join(data_root, dataset_name, 
            'original_actives_docked_feature',
            f'{target_id}.{docktool}.{feature_type}.{suffix}')
        if os.path.exists(active_file):
            files.append(active_file)
        decoy_file = os.path.join(data_root, dataset_name, 
            f'{decoy_source}_decoys_docked_feature',
            f'{target_id}.{docktool}.{feature_type}.{suffix}')
        if os.path.exists(decoy_file):
            files.append(decoy_file)
    return files

def quickscore_v1_cv(data_root, dataset_name, decoy_source, split_dict_file, 
    model_id, feature_type, docktool, suffix, mode, outdir, train_strategy,
    gpu_device):
    
    dataset_dir = os.path.join(data_root, dataset_name)
    split_dict_file = os.path.join(dataset_dir, split_dict_file)
    split_dict = readJson(split_dict_file)
    fold_n = len(split_dict)    
    
    dfs_pred = []
    targets = []
    for i in range(fold_n):
        # i = 0
        print(f'{str(i)} Fold')
        tmp_outdir = os.path.join(outdir, str(i))
        
        pred_save = os.path.join(tmp_outdir, 'out', 'pred.csv')
        if not os.path.exists(pred_save):
            print('Prediction is not exists.')
            targets += split_dict[str(i)]
            test_files = get_feature_files(target_ids=split_dict[str(i)], 
                data_root=data_root, dataset_name=dataset_name, model_id=model_id, 
                decoy_source=decoy_source, feature_type=feature_type,
                docktool=docktool, suffix=suffix)
            target_ids = list(range(fold_n)); target_ids.remove(i)
            train_targets = []
            for id_ in target_ids:
                train_targets += split_dict[str(id_)]
            train_files = get_feature_files(target_ids=train_targets, 
                data_root=data_root, dataset_name=dataset_name, model_id=model_id, 
                decoy_source=decoy_source, feature_type=feature_type,
                docktool=docktool, suffix=suffix)
            pred = quickscore_v1_holdout(test=test_files, train=train_files, 
                result_dir=tmp_outdir, mode=mode, train_strategy=train_strategy,
                gpu_device=gpu_device, pred_save=pred_save)
        else:
            pred = pd.read_csv(pred_save)
            print('Prediction exists.')
        dfs_pred.append(pred)
    return pd.concat(dfs_pred)

def select_pose(df, pred_strategy, mol_id_col='mol_id',
                target_id_col='target_id', pose_id_col='pose_id',
                label_col='label', pred_col='score', ascending=False):
    """
    pred_strategy: topN_pose_mean, topN_score_mean (mean, median, max, min)
    """
    print("\nSelecting pose...")
    print(f'Strategy: {pred_strategy}')
    compound_ids = [x.split('_mode_')[0] for x in df[mol_id_col] ]
    pose_ids = [int(x.split('_mode_')[1]) for x in df[mol_id_col] ]
    df[mol_id_col] = compound_ids
    df[pose_id_col]= pose_ids
    
    if 'pose' in pred_strategy:
        # select top N pose's result
        pose = list(range(int(re.findall('\d', pred_strategy)[0])))
        df = df.loc[df[pose_id_col].isin(pose)]
    
    target_ids = list(set(df[target_id_col]))
    
    df_preds = []
    for target_id in tqdm(target_ids):
        # target_id = target_ids[0]
        df_sub = df[df[target_id_col]==target_id]
        compound_ids = list(set(list(df_sub[mol_id_col].values)))
        
        scores = []
        labels = []
        for compound_id in compound_ids:
            # compound_id = compound_ids[0]
            df_compound = df_sub[df_sub[mol_id_col]==compound_id]
            if len(set(df_compound[label_col])) > 1:
                df_compound = df_compound[df_compound[label_col]==1]
            label = list(df_compound[label_col].values)[0]
            labels.append(label)
            
            if len(df_compound) == 1:
                score = list(df_compound[pred_col].values)[0]
            elif 'mean' in pred_strategy:
                score = df_compound[pred_col].mean()
            elif 'median' in pred_strategy:
                score = df_compound[pred_col].median()
            elif 'max' in pred_strategy:
                score = df_compound[pred_col].max()
            elif 'min' in pred_strategy:
                score = df_compound[pred_col].min()
                
            scores.append(round(score, 4))
        df_pred = pd.DataFrame({
            mol_id_col: compound_ids,
            target_id_col: target_id,
            label_col: labels,
            pred_col: scores
            })
        df_pred.sort_values(pred_col, ascending=ascending, inplace=True)
        df_preds.append(df_pred)
    return pd.concat(df_preds)

def sbvscv(data_root, dataset_name, model_id, decoy_source, outdir_root, 
           cvfunc=None, feature_type='knn', holdoutfunc=None, 
                      
           train_strategy = 'top1_undersample_per', pred_strategy = 'top1_pose', 
           train_strategy_col = 'train_strategy', 
           pred_strategy_col = 'pred_strategy',
           
           target_id_col='target_id', model_id_col='model_id', 
           mol_id_col='mol_id', pose_id_col='pose_id',
           split_dict_file='3_fold.json', cv=True, mode='cls',
           pred_col='score', label_col='label', other_cols=None,
           fold_id_col='fold_id', smiles_col='SMILES', ascending=False, 
           ef_list=[0.01, 0.05, 0.1], feature_type_col='feature_type',
           overwrite=False, decoy_source_col='decoy_source', docktool='smina',
           docktool_col = 'docktool',
           suffix='pickle', gpu_device=0
           ):
    
    print()
    print(f'Predict: {train_strategy}')
    dataset_dir = os.path.join(data_root, dataset_name)
    split_dict_file = os.path.join(dataset_dir, split_dict_file)
    split_dict = readJson(split_dict_file)
    fold_n = len(split_dict)
    
    outdir = os.path.join(outdir_root, dataset_name, 'cv')

    if outdir is not None:
        build_new_folder(outdir)
        pre = f'{model_id}.{train_strategy}.{pred_strategy}.{decoy_source}.'\
            f'{docktool}.{fold_n}-fold'
        fig_save = os.path.join(outdir, f'{pre}.eval.{feature_type}.png')
        pred_save = os.path.join(outdir, f'{pre}.pred.{feature_type}.csv')
        eval_save = os.path.join(outdir,  f'{pre}.eval.{feature_type}.csv')
        raw_result = os.path.join(outdir,  f'{model_id}.{train_strategy}.'
                                           f'{decoy_source}.{fold_n}-fold')
        build_new_folder(raw_result, overwrite=overwrite)
        if overwrite:
            print("\nOverwrite...")
            delete_a_file(fig_save)
            delete_a_file(pred_save)
            delete_a_file(eval_save)

    if 'quickscore_v1' in model_id:
        cvfunc = quickscore_v1_cv
    
    # Cross validation
    print(f'\nCross validation on {dataset_name}...')
    if not os.path.exists(pred_save) or overwrite:
        pred = cvfunc(data_root=data_root, dataset_name=dataset_name, 
            decoy_source=decoy_source, split_dict_file=split_dict_file,
            model_id=model_id, feature_type=feature_type, 
            train_strategy=train_strategy, outdir=raw_result,
            docktool=docktool, suffix=suffix, mode=mode, gpu_device=gpu_device)
    else:
        print('Result already exists.')
        pred = pd.read_csv(pred_save)
    
    print('\nComputing cross validation performance...')
    if not os.path.exists(eval_save) or overwrite:
        pred = select_pose(pred, pred_strategy=pred_strategy, 
            mol_id_col=mol_id_col, target_id_col=target_id_col, 
            pose_id_col=pose_id_col, label_col=label_col, pred_col=pred_col)
            
        targets = list(set(pred[target_id_col]))
        evals = []
        for target in tqdm(targets):
            df_target_pred = pred[pred[target_id_col]==target]
            target_eval = eval_one_target(df_target_pred, pred_col=pred_col, 
                label_col=label_col, sort=True, ascending=ascending, 
                ef_list=ef_list)
            evals.append(target_eval)
            
        df_cv_eval = pd.DataFrame(evals)
        if outdir is not None:
            DataFrameDistribution(df_cv_eval, outfile=fig_save)
        
        df_cv_eval[target_id_col] = targets
        if decoy_source is not None:
            df_cv_eval[decoy_source_col] = decoy_source
        if model_id is not None:
            df_cv_eval[model_id_col] = model_id
        if feature_type is not None:
            df_cv_eval[feature_type_col] = feature_type
        if train_strategy is not None:
            df_cv_eval[train_strategy_col] = train_strategy
        if pred_strategy is not None:
            df_cv_eval[pred_strategy_col] = pred_strategy
        df_cv_eval[docktool_col] = docktool
        if outdir is not None:
            df_cv_eval.to_csv(eval_save, index=False)

    else:
        print('Result already exists.')
        df_cv_eval = pd.read_csv(eval_save)
    return df_cv_eval

def sbvsit(data_root, train_set, test_set, model_id, decoy_source, outdir_root, 
           feature_type='knn', holdoutfunc=None, 
                      
           train_strategy = 'top1_undersample_per', pred_strategy = 'top1_pose', 
           train_strategy_col = 'train_strategy', 
           pred_strategy_col = 'pred_strategy',
           
           target_id_col='target_id', model_id_col='model_id', 
           mol_id_col='mol_id', pose_id_col='pose_id',
           split_dict_file='3_fold.json', cv=True, mode='cls',
           pred_col='score', label_col='label', other_cols=None,
           fold_id_col='fold_id', smiles_col='SMILES', ascending=False, 
           ef_list=[0.01, 0.05, 0.1], feature_type_col='feature_type',
           overwrite=False, decoy_source_col='decoy_source', docktool='smina',
           docktool_col = 'docktool',
           suffix='pickle', gpu_device=0):

    """
    feature_type='knn'
    data_root = '/home/zdx/data/VS_dataset'
    
    train_set = 'DUD-E'
    test_set = 'LIT-PCBA'
    model_id = 'quickscore_v1'
    decoy_source = 'original'
    outdir_root = '/home/zdx/project/VS_Benchmark'
    
    train_strategy = 'top1_undersample_per'
    pred_strategy = 'top1_pose'
    
    train_strategy_col = 'train_strategy'
    pred_strategy_col = 'pred_strategy'
    
    target_id_col='target_id'
    model_id_col='model_id'
    mol_id_col='mol_id'
    pose_id_col='pose_id'
    split_dict_file='3_fold.json'
    cv=True
    mode='cls'
    pred_col='score'
    label_col='label'
    other_cols=None
    fold_id_col='fold_id'
    smiles_col='SMILES'
    ascending=False
    ef_list=[0.01, 0.05, 0.1]
    feature_type_col='feature_type'
    overwrite=False
    decoy_source_col='decoy_source'
    docktool='smina'
    suffix='pickle'
    gpu_device=0
    
    """
    print(f"SBVS independent test on {test_set}, trained by {train_set}")

    if outdir_root is not None:
        # Save model path
        model_dir = os.path.join(outdir_root, 'Trained_models')
        build_new_folder(model_dir)
        checkpoint_path = os.path.join(model_dir, 
            f'{model_id}.{train_strategy}.{train_set}.{decoy_source}.'
            f'{docktool}.{feature_type}')
        # Save prediction and evaluation path
        ourdir = os.path.join(outdir_root, test_set)
        build_new_folder(ourdir)
        eval_file = os.path.join(ourdir, 
            f'{model_id}.{train_strategy}.{pred_strategy}.{train_set}'
            f'.{decoy_source}.{docktool}.{feature_type}.eval.csv')
        pred_file = os.path.join(ourdir, 
            f'{model_id}.{train_strategy}.{pred_strategy}.{train_set}'
            f'.{decoy_source}.{docktool}.{feature_type}.pred.csv')
        if os.path.exists(eval_file) and not overwrite:
            eval_test = pd.read_csv(eval_file)
            return eval_test
    else:
        checkpoint_path = None
        outfile = None

    train_targets = get_dataset_target_ids(data_root=data_root, 
                                           dataset_name=train_set)
    train_files = get_feature_files(train_targets, data_root=data_root,
        dataset_name=train_set, model_id=model_id, decoy_source=decoy_source,
        feature_type=feature_type, docktool=docktool, suffix=suffix)

    test_targets =  get_dataset_target_ids(data_root=data_root, 
                                           dataset_name=test_set)
    """
    ---------------------------------------------------------------------------
    ESR1_ant TP53 ESR1_ago
    ---------------------------------------------------------------------------
    """
    if test_set == 'LIT-PCBA':
        test_targets = [x for x in test_targets if not 'ESR1_ant' in x and not 'TP53' in x and not 'ESR1_ago' in x]
    
    test_files = get_feature_files(test_targets, data_root=data_root,
        dataset_name=test_set, model_id=model_id, decoy_source='original',
        feature_type=feature_type, docktool=docktool, suffix=suffix)

    if not os.path.exists(pred_file):
        pred = quickscore_v1_holdout(test=test_files, train=train_files, 
            pred_save=pred_file, overwrite=overwrite, result_dir=checkpoint_path,
            train_strategy=train_strategy, gpu_device=gpu_device)
    else:
        pred = pd.read_csv(pred_file)
    
    pred = select_pose(pred, pred_strategy=pred_strategy, mol_id_col=mol_id_col,
        target_id_col=target_id_col, pose_id_col=pose_id_col,
        label_col=label_col, pred_col=pred_col)
    
    evals = []
    for target in tqdm(test_targets):
        df_target_pred = pred[pred[target_id_col]==target]
        target_eval = eval_one_target(df_target_pred, pred_col=pred_col, 
            label_col=label_col, sort=True, ascending=ascending, 
            ef_list=ef_list)
        evals.append(target_eval)
        
    df_eval = pd.DataFrame(evals)
    df_eval[target_id_col] = test_targets
    if decoy_source is not None:
        df_eval[decoy_source_col] = decoy_source
    if model_id is not None:
        df_eval[model_id_col] = model_id
    if feature_type is not None:
        df_eval[feature_type_col] = feature_type
    if train_strategy is not None:
        df_eval[train_strategy_col] = train_strategy
    if pred_strategy is not None:
        df_eval[pred_strategy_col] = pred_strategy
    df_eval[docktool_col] = docktool
    df_eval.to_csv(eval_file, index=False)

# sbvsit(    feature_type='knn',
#     data_root = '/home/zdx/data/VS_dataset',
    
#     train_set = 'DUD-E',
#     test_set = 'LIT-PCBA',
#     model_id = 'quickscore_v1',
#     decoy_source = 'original',
#     outdir_root = '/home/zdx/project/VS_Benchmark',
    
#     train_strategy = 'top1_undersample_per',
#     pred_strategy = 'top1_pose',
    
#     train_strategy_col = 'train_strategy',
#     pred_strategy_col = 'pred_strategy',
    
#     target_id_col='target_id',
#     model_id_col='model_id',
#     mol_id_col='mol_id',
#     pose_id_col='pose_id',
#     split_dict_file='3_fold.json',
#     cv=True,
#     mode='cls',
#     pred_col='score',
#     label_col='label',
#     other_cols=None,
#     fold_id_col='fold_id',
#     smiles_col='SMILES',
#     ascending=False,
#     ef_list=[0.01, 0.05, 0.1],
#     feature_type_col='feature_type',
#     overwrite=False,
#     decoy_source_col='decoy_source',
#     docktool='smina',
#     suffix='pickle',
#     gpu_device=0)

if __name__ == "__main__":
    from argparse import ArgumentParser

    ap = ArgumentParser()
    ap.add_argument('-i', '--data_root', default='/home/zdx/data/VS_dataset')
    ap.add_argument('-o', '--outdir_root', default='/home/zdx/project/VS_Benchmark')
    ap.add_argument('-d', '--dataset_name')
    ap.add_argument('--docktool', default='smina')
    ap.add_argument('-m', '--model_id', default='quickscore_v1')
    ap.add_argument('--decoy_source', default='original')
    ap.add_argument('-g', '--gpu_device', default=0, type=int)
    ap.add_argument('--overwrite', default=False, action='store_true')
    ap.add_argument('-t', '--train_strategy', default='top1_undersample_per',
        help='topN_undersample_per/total')
    ap.add_argument('-p', '--pred_strategy', default='top1_pose',
        help='topN_pose_mean/median/max/min')

    args = ap.parse_args()
    # Cross Validation
    sbvscv(data_root=args.data_root, 
           dataset_name=args.dataset_name, 
           model_id=args.model_id, 
           decoy_source=args.decoy_source, 
           outdir_root=args.outdir_root, 
           train_strategy=args.train_strategy, 
           gpu_device=args.gpu_device,
           overwrite=args.overwrite
           docktool=args.docktool)

"""
./vs_sb_dl.py -d DUD-E

./vs_sb_dl.py -d DUD-E -g 3 -t top1_undersample_per --overwrite  running
./vs_sb_dl.py -d DUD-E -g 3 -t top3_undersample_per --overwrite  running

./vs_sb_dl.py -d DUD-E -g 2 -t top1_undersample_total --overwrite  running
./vs_sb_dl.py -d DUD-E -g 2 -t top3_undersample_total --overwrite  running

./vs_sb_dl.py -d DUD-E -g 1 -t top1 --overwrite  running
./vs_sb_dl.py -d DUD-E -g 1 -t top3 --overwrite  running

"""
