#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:57:04 2022

@author: zdx
"""

import os
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"
from functools import partial
import pandas as pd
import numpy as np
from tqdm import tqdm
# from rdkit.Chem import AllChem, PandasTools
# from utils import drop_duplicates_between_two_dataframes, file_name_format
# Classification
from pycaret.classification import setup as cls_setup
from pycaret.classification import compare_models as cls_compare_models
from pycaret.classification import tune_model as cls_tune_model
from pycaret.classification import finalize_model as cls_finalize_model
from pycaret.classification import predict_model as cls_predict_model
from pycaret.classification import save_model as cls_save_model
from pycaret.classification import load_model as cls_load_model
from pycaret.classification import create_model as cls_create_model
from pycaret.classification import pull as cls_pull
from pycaret.classification import get_metrics as cls_get_metrics
from pycaret.classification import add_metric as cls_add_metric
# Regression
from pycaret.regression import setup as reg_setup
from pycaret.regression import compare_models as reg_compare_models
from pycaret.regression import tune_model as reg_tune_model
from pycaret.regression import finalize_model as reg_finalize_model
from pycaret.regression import predict_model as reg_predict_model
from pycaret.regression import save_model as reg_save_model
from pycaret.regression import load_model as reg_load_model
from pycaret.regression import create_model as reg_create_model
from pycaret.regression import pull as reg_pull
from pycaret.regression import get_metrics as reg_get_metrics
from pycaret.regression import add_metric as reg_add_metric

from utils import eval_one_target, readJson, add_features, build_new_folder,\
    eval_one_dataset, get_dataset_target_ids
from plot import DataFrameDistribution

ML_names = [
        'lr',
        'lightgbm',
        'catboost',
        'nb',
        'dt',
        # 'svm',
        'rf',
        # 'rbfsvm',
        'mlp',
        # 'ridge',
        'knn',
        #'gpc',
        'qda',
        'ada',
        'gbc',
        'lda',
        'et',
        'xgboost'
    ]

# def get_fingerprint(df, bits=2048, r=2):
#     print('Getting fingerprint ...')
#     PandasTools.AddMoleculeColumnToFrame(df, "SMILES")
#     fps = []
#     for mol in df.ROMol:
#         fp = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, r, bits)]
#         fps.append(fp)
#     fps = np.array(fps)
#     X = pd.DataFrame(fps)
#     return X

def concatPosNeg(pos_data, neg_data, feature_cols=None, label_col='label', 
                 downsample=False, add_label=False, add_mol_descriptors=False, 
                 smiles_col='SMILES'):
    if isinstance(pos_data, str):
        pos_data = pd.read_csv(pos_data)

    if not isinstance(pos_data, pd.DataFrame):
        print('Please check pos_data format, it should be dataframe or file'
              ' path.')
        return
    
    if add_mol_descriptors and feature_cols is not None:
        pos_data = add_features(pos_data, order=False, sort=False, 
            remove_na=True, pro=feature_cols, smiles_col=smiles_col)

    if isinstance(neg_data, str):
        neg_data = pd.read_csv(neg_data)
    
    if not isinstance(neg_data, pd.DataFrame):
        print('Please check neg_data format, it should be dataframe or file'
              ' path.')
        return

    if add_mol_descriptors and feature_cols is not None:
        neg_data = add_features(neg_data, order=False, sort=False, 
            remove_na=True, pro=feature_cols, smiles_col=smiles_col)

    if label_col not in pos_data.columns:
        pos_data[label_col] = 1
    if label_col not in neg_data.columns:
        neg_data[label_col] = 0
    
    if add_label:
        pos_data[label_col] = 1
        neg_data[label_col] = 0
        
    if feature_cols is not None and isinstance(feature_cols, list):
        if label_col not in feature_cols:
            feature_cols += [label_col]
        pos_data = pos_data[feature_cols]
        neg_data = neg_data[feature_cols]
        
    if downsample:
        if len(neg_data)>len(pos_data):
            neg_data = neg_data.sample(n=len(pos_data))
        elif len(neg_data)<len(pos_data):
            pos_data = pos_data.sample(n=len(neg_data))
        else:
            pass
        
    df = pd.concat([pos_data, neg_data])
    return df

class ClassicML:
    def __init__(self, mode='cls', checkpoint_path=None):
        """
        train_file, val_file, test_file should all
        
        ---------------------
        Classification Model:
        ---------------------
        'lr' - Logistic Regression
        'knn' - K Neighbors Classifier
        'nb' - Naive Bayes
        'dt' - Decision Tree Classifier
        'svm' - SVM - Linear Kernel
        'rbfsvm' - SVM - Radial Kernel
        'gpc' - Gaussian Process Classifier
        'mlp' - MLP Classifier
        'ridge' - Ridge Classifier
        'rf' - Random Forest Classifier
        'qda' - Quadratic Discriminant Analysis
        'ada' - Ada Boost Classifier
        'gbc' - Gradient Boosting Classifier
        'lda' - Linear Discriminant Analysis
        'et' - Extra Trees Classifier
        'xgboost' - Extreme Gradient Boosting
        'lightgbm' - Light Gradient Boosting Machine
        'catboost' - CatBoost Classifier
        """
        self.mode = mode
        if mode == 'cls':
            self.setup = partial(cls_setup, html=False, silent=True, 
                                 verbose=False, preprocess=True) # 
            self.compare_models = partial(cls_compare_models, verbose=False)
            self.tune_model = cls_tune_model
            self.finalize_model = cls_finalize_model
            self.predict_model = cls_predict_model
            self.save_model = partial(cls_save_model, verbose=False)
            self.load_model = partial(cls_load_model, verbose=False)
            self.create_model = partial(cls_create_model, verbose=False)
            self.get_metrics = cls_get_metrics
            self.add_metric = cls_add_metric
            self.pull = cls_pull
        elif mode == 'reg':
            self.setup = partial(reg_setup, html=False, silent=True, 
                                 verbose=False) # , preprocess=False
            self.compare_models = partial(reg_compare_models, verbose=False)
            self.tune_model = reg_tune_model
            self.finalize_model = reg_finalize_model
            self.predict_model = reg_predict_model
            self.save_model = partial(reg_save_model, verbose=False)
            self.load_model = partial(reg_load_model, verbose=False)
            self.create_model = partial(reg_create_model, verbose=False)
            self.get_metrics = reg_get_metrics
            self.add_metric = reg_add_metric
            self.pull = reg_pull

        self.model = None
            
        if checkpoint_path is not None:
            if '.pkl' in checkpoint_path:
                checkpoint_path = checkpoint_path.replace('.pkl', '')
            self.checkpoint_path = checkpoint_path
            self.model = self.load_model(checkpoint_path)

    def train(self, train_data, feature_cols=None, model_id = None, 
              checkpoint_path=None,
              # pycaret parameter
              session_id=0, label_col='label', fold_n=3, 
              # custom
              compare=False, tune=False, save=False, finalize=False,
              cross_validation=False, smiles_col='SMILES', overwrite=False,
              add_mol_descriptors=False
              ):

        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
            if os.path.exists(checkpoint_path+'.pkl') and not overwrite:
                if '.pkl' in checkpoint_path:
                    checkpoint_path = checkpoint_path.replace('.pkl', '')
                self.model = self.load_model(checkpoint_path)
                # print("Loaded existed checkpoint.")
                return self.model, None
        
        if isinstance(train_data, str):
            train_data = pd.read_csv(train_data)
        
        if add_mol_descriptors:
            train_data = add_features(train_data, order=False, sort=False, 
                remove_na=True, pro=feature_cols, smiles_col=smiles_col)
        
        if feature_cols is not None:
            self.feature_cols = feature_cols

        if self.feature_cols is not None:
            if label_col not in self.feature_cols:
                self.feature_cols += [label_col]
            train_data = train_data[self.feature_cols]
        
        # train_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # train_data.dropna(inplace=True)
        
        exp = self.setup(data=train_data, target=label_col,
                    session_id=session_id, fold=fold_n)

        if compare:
            self.model = self.compare_models()
        else:
            self.model = self.create_model(model_id,
                                           cross_validation=cross_validation)
        scores = self.pull()
        
        if finalize:
            if tune:
                print('Tuning...')
                self.model = self.tune_model(self.model)
            
            print('Finalizing...')
            self.model = self.finalize_model(self.model)
            
        if checkpoint_path is not None:
            if not os.path.exists(os.path.dirname(checkpoint_path)):
                os.mkdirs(os.path.dirname(checkpoint_path))
            self.save_model(self.model, checkpoint_path)
        
        if compare:
            scores.reset_index(inplace=True)
            scores.rename(columns={'index':'model_id'}, inplace=True)
            
        del exp
        
        return self.model, scores
            
    def predict(self, test_data, checkpoint_path=None, feature_cols=None,
                pred_col='score', outfile=None, score_col='Score', 
                smiles_col='SMILES', add_mol_descriptors=False):

        if feature_cols is not None:
            self.feature_cols = feature_cols

        if isinstance(test_data, str):
            test_data = pd.read_csv(test_data)
        
        if add_mol_descriptors:
            test_data = add_features(test_data, order=False, sort=False, 
                remove_na=True, pro=self.feature_cols, smiles_col=smiles_col)
        
        # test_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # test_data.dropna(inplace=True)

        org_data = test_data.copy()
        
        if not isinstance(test_data, pd.DataFrame):
            print('Please your input.')
            return
            
        if self.model is None:
            if checkpoint_path is not None:
                if '.pkl' in checkpoint_path:
                    checkpoint_path = checkpoint_path.replace('.pkl', '')
                self.model = self.load_model(checkpoint_path)
            else:
                print('You should give checkpoint_path=.')
                return

        prediction = self.predict_model(self.model, 
            data=test_data[self.feature_cols])
        org_data[pred_col] = list(prediction[score_col].values)
        
        if outfile is not None:
            org_data.to_csv(outfile, index=False)
        
        return org_data
    
def tscv(active_data, decoy_data=None, target_id=None, use_ml=True,
    feature_cols=None, target_id_col='target_id', model_id='lr', 
    checkpoint_path=None, downsample=False, compare=True, label_col='label', 
    mode='cls', smiles_col="SMILES", add_mol_descriptors=False):
    """ Target-specific virtual screening cross validation
    """

    if decoy_data is not None and mode=='cls':
        df = concatPosNeg(active_data, decoy_data, feature_cols=feature_cols,
                          label_col=label_col, downsample=downsample)
        if add_mol_descriptors:
            df = add_features(df, order=False, sort=False, remove_na=True,
                              pro=feature_cols, smiles_col=smiles_col)
    
    if use_ml:
        ml_model = ClassicML(mode)
        model, scores = ml_model.train(train_data=df, 
            model_id=model_id, checkpoint_path=checkpoint_path,
            feature_cols=feature_cols, compare=compare)
        
    if target_id is not None:
        scores[target_id_col] = target_id

    return scores

def btscv(data_root, dataset_name, decoy_folder, model_id, 
    outfile=None, use_ml=True, active_folder='3_original_actives',
    summary_file='summary.csv', model_id_col='model_id', 
    decoy_source=None, decoy_source_col='decoy_source',
    feature_cols=None, target_id_col='target_id',
    checkpoint_path=None, downsample=False, compare=True,
    label_col='label', mode='cls', smiles_col="SMILES", overwrite=False):
    """ Batch target-specific virtual screening cross validation
    """
    
    print('Batch target-specific virtual screening cross validation')
    
    if outfile is not None and not overwrite:
        if os.path.exists(outfile):
            # print('Result already exists.')
            return pd.read_csv(outfile)
    
    dataset_dir = os.path.join(data_root, dataset_name)
    
    targets = get_dataset_target_ids(data_root=data_root, 
        dataset_name=dataset_name)

    dfs_scores = []
    for target in tqdm(targets):
        active_data = os.path.join(dataset_dir, active_folder, f'{target}.csv')
        decoy_data = os.path.join(dataset_dir, decoy_folder, f'{target}.csv')
        scores = tscv(active_data=active_data,
            decoy_data=decoy_data, 
            target_id=target,
            feature_cols=feature_cols, 
            target_id_col=target_id_col,
            model_id=model_id, 
            compare=compare, 
            use_ml=use_ml,
            label_col=label_col, 
            mode=mode, 
            smiles_col=smiles_col)
        dfs_scores.append(scores)
    df = pd.concat(dfs_scores)
    if decoy_source is not None:
        df[decoy_source_col] = decoy_source
    if model_id is not None:
        df[model_id_col] = model_id
    if outfile is not None:
        df.to_csv(outfile, index=False)
    return df

def splitCV():
    pass

def mlholdout(model_id, test=None, train=None, mode='cls', 
    feature_cols=None, checkpoint_path=None, pred_col='score', 
    pred_save=None, overwrite=False):
    """ Machine learning independent test or hold out
    """
    ml_model = ClassicML(mode)
    ml_model.train(train_data=train, 
        model_id=model_id,
        cross_validation=False,
        checkpoint_path=checkpoint_path,
        feature_cols=feature_cols, compare=False, overwrite=overwrite)

    if test is not None:
        if isinstance(test, list):
            pred = []
            for t in test:
                p = ml_model.predict(t, feature_cols=feature_cols,
                            pred_col=pred_col)
                pred.append(p)
        else: 
            pred = ml_model.predict(test, feature_cols=feature_cols,
                        pred_col=pred_col, outfile=pred_save)
    else:
        pred = None

    return pred

def get_vs_dataset(data_root, dataset_name, active_folder, decoy_folder,
                feature_cols=None, downsample=False, add_mol_descriptors=False,
                summary_file='summary.csv', target_id_col='target_id'):
    dfs = []
    active_dir = os.path.join(data_root, dataset_name, active_folder)
    decoy_dir = os.path.join(data_root, dataset_name, decoy_folder)
    if not dataset_name == 'MUV':
        targets = get_dataset_target_ids(data_root=data_root, 
            dataset_name=dataset_name)
    else:
        targets = list(pd.read_csv(os.path.join(data_root, dataset_name, 
                    summary_file))[target_id_col].values)
    dfs = []
    for target in targets:
        if dataset_name == 'LIT-PCBA':
            active_data = os.path.join(active_dir, f'{target}.full.csv')
            decoy_data = os.path.join(decoy_dir, f'{target}.full.csv')
        else:
            active_data = os.path.join(active_dir, f'{target}.csv')
            decoy_data = os.path.join(decoy_dir, f'{target}.csv')

        df = concatPosNeg(pos_data=active_data, neg_data=decoy_data,
                     feature_cols=feature_cols, downsample=downsample,
                     add_mol_descriptors=add_mol_descriptors)
        df[target_id_col] = target
        dfs.append(df)
    return pd.concat(dfs)

def vsit(test_set, data_root, model_id, train_set, decoy_source, decoy_folder,
    feature_type, feature_cols, feature_type_col='feature_type', 
    holdoutfunc=mlholdout, add_mol_descriptors=False, outdir_root=None, 
    active_folder='original_actives',
    mode='cls', overwrite=False):
    """ Virtual screening independent test
    
    test_set='MUV'
    data_root = '/home/zdx/data/VS_dataset'
    model_id='CNN'
    train_set='DUD-E'
    decoy_source='original'
    feature_type = 'seq'
    feature_type_col='feature_type'
    holdoutfunc=deeppurposeholdout
    mode='cls'
    overwrite=False
    checkpoint_path='/home/zdx/Downloads/test'
    outdir_root=None
    feature_cols= ['SMILES']
    add_mol_descriptors=False
    """

    print(f'LBVS independent test on {test_set},'
          f' trained by {train_set}')

    if outdir_root is not None:
        # Save model
        model_dir = os.path.join(outdir_root, 'Trained_models')
        build_new_folder(model_dir)
        checkpoint_path = os.path.join(model_dir, 
            f'{model_id}.{train_set}.{decoy_source}.{feature_type}')
            
        # Save prediction evaluation
        ourdir = os.path.join(outdir_root, test_set)
        build_new_folder(ourdir)
        outfile = os.path.join(ourdir, 
            f'{model_id}.{train_set}.{decoy_source}.eval.{feature_type}.csv')

        if os.path.exists(outfile) and not overwrite:
            eval_test = pd.read_csv(outfile)
            return eval_test
    else:
        checkpoint_path = None
        outfile = None

    data_train = get_vs_dataset(data_root=data_root, dataset_name=train_set, 
                   active_folder=active_folder, 
                   decoy_folder=decoy_folder, 
                   feature_cols=feature_cols,
                   add_mol_descriptors=add_mol_descriptors)

    data_test = get_vs_dataset(data_root=data_root, dataset_name=test_set, 
                   active_folder='original_actives', 
                   decoy_folder='original_decoys', 
                   feature_cols=feature_cols,
                   add_mol_descriptors=add_mol_descriptors)
    
    test_pred = holdoutfunc(test=data_test, train=data_train, 
        model_id=model_id, checkpoint_path=checkpoint_path, mode=mode,
        feature_cols=feature_cols, overwrite=overwrite)

    eval_test = eval_one_dataset(data=test_pred, model_id=model_id, 
        decoy_source=decoy_source, train_set=train_set)

    eval_test[feature_type_col] = feature_type

    if outdir_root is not None:
        eval_test.to_csv(outfile, index=False)
    return eval_test


def mlcv(data, holdoutfunc=mlholdout, fold_n=3, feature_cols=None, pred_col='score',
         label_col='label', fold_id_col='fold_id', mode='cls', model_id='lr'):
    """ Machine learning cross validation
    -----------------
    data: pd.DataFrame, csv file path 
    feature1, feature2, ... label, fold_id
    0.1       1.2           0      0
    0.2       1.3           1      1
    -----------------
    fold_n: None or int
    
    """
    if isinstance(data, str):
        data = splitCV(data, fold_n=fold_n, feature_cols=feature_cols, 
                       label_col=label_col) # ToDo
        
    if not isinstance(data, pd.DataFrame):
        print('Please check your input.')
        return
    
    if fold_id_col not in data.columns:
        data = splitCV(data, fold_n=fold_n, feature_cols=feature_cols, 
                       label_col=label_col) # ToDo
        
    fold_numbers = set(data[fold_id_col])
    fold_n = len(fold_numbers)
    print('%d Fold Cross Valiation.' % fold_n)
    
    dfs_pred = []
    for i in tqdm(fold_numbers):
        # fold_id_col = 'fold_id'
        # i = 0
        test = data[data[fold_id_col]==i]
        del test[fold_id_col]
        train = data[data[fold_id_col]!=i]
        del train[fold_id_col]
        res = holdoutfunc(test=test, train=train, model_id=model_id, mode=mode,
                         feature_cols=feature_cols)

        dfs_pred.append(res)
    return pd.concat(dfs_pred)

def vscv(data_root, dataset_name, model_id, active_folder, decoy_folder, 
    decoy_source, cvfunc, feature_type, feature_cols, holdoutfunc,

    outdir=None,
    target_id_col='target_id',
    model_id_col='model_id', 
    split_dict_file='3_fold.json',
    cv=True,
    mode='cls',
    pred_col='score',
    label_col='label', other_cols=None,
    fold_id_col='fold_id', smiles_col='SMILES', ascending=False, 
    ef_list=[0.01, 0.05, 0.1], 
    
    # DeepPurpose
    protein_col='protein_seq',
    add_protein_seq = False,

    # General
    add_mol_descriptors=False,
    feature_type_col='feature_type',
    overwrite=False,
    decoy_source_col='decoy_source'):
    """
    Virtual screening cross validation.
    
    feature_cols: dataframe columns as feature
    feature_type: 'deepcoy', 'dude', 'muv', 'ecfp4'
    
    data_root = '/home/zdx/data/VS_dataset'
    dataset_name = 'DUD-E'
    model_id = 'CNN'
    cvfunc = mlcv
    active_folder = 'original_actives'
    decoy_folder = 'original_decoys'
    feature_type = 'smiles'
    feature_cols = ['SMILES']
    outdir = None
    
    protein_col='protein_seq'
    add_mol_descriptors = False
    add_protein_seq = False
    decoy_source='original'
    holdoutfunc=deeppurposeholdout
    feature_type_col='feature_type'
    overwrite=False
    decoy_source_col='decoy_source'
    target_id_col='target_id'
    model_id_col='model_id'
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
    """       
    print(f"LBVS cross validation on {dataset_name}")
    
    if add_protein_seq:
        feature_cols.append(protein_col)
    
    dataset_dir = os.path.join(data_root, dataset_name)
    split_dict_file = os.path.join(dataset_dir, split_dict_file)
    split_dict = readJson(split_dict_file)
    fold_n = len(split_dict)
    
    if outdir is not None:
        build_new_folder(outdir)
        pre = f'{model_id}.{decoy_source}.{fold_n}-fold'
        fig_save = os.path.join(outdir, f'{pre}.eval.{feature_type}.png')
        pred_save = os.path.join(outdir, f'{pre}.pred.{feature_type}.csv')
        eval_save = os.path.join(outdir,  f'{pre}.eval.{feature_type}.csv')
    
    other_cols = [target_id_col, label_col, fold_id_col]
    
    active_dir = os.path.join(dataset_dir, active_folder)
    decoy_dir = os.path.join(dataset_dir, decoy_folder)

    if not os.path.exists(pred_save) or overwrite:
        targets = []
        dfs_cv = [] 
        for i, (k, v) in enumerate(split_dict.items()):
            for target in v:
                targets.append(target)
                # Active
                active_data = os.path.join(active_dir, f'{target}.csv')
                active_data = pd.read_csv(active_data)
                
                if add_mol_descriptors:
                    active_data = add_features(active_data, order=False, 
                                            sort=False, remove_na=True,
                                    pro=feature_cols, smiles_col=smiles_col)
                
                if label_col not in active_data.columns:
                    active_data[label_col] = 1
                active_data[target_id_col] = target
                active_data[fold_id_col] = i
                active_data = active_data[feature_cols+other_cols]
                
                dfs_cv.append(active_data)
                # Decoy
                decoy_data = os.path.join(decoy_dir, f'{target}.csv')
                decoy_data = pd.read_csv(decoy_data)
                if add_mol_descriptors:
                    decoy_data = add_features(decoy_data, order=False, sort=False, 
                                    remove_na=True, pro=feature_cols, 
                                    smiles_col=smiles_col)
                
                if label_col not in decoy_data.columns:
                    decoy_data[label_col] = 0
                decoy_data[target_id_col] = target
                decoy_data[fold_id_col] = i
                decoy_data = decoy_data[feature_cols+other_cols]
                dfs_cv.append(decoy_data)
        df = pd.concat(dfs_cv)
        # print("get cv data.")
        if not cv:
            return df
        # Cross validation
        pred = cvfunc(df,  mode=mode, model_id=model_id, 
            feature_cols=feature_cols, holdoutfunc=holdoutfunc)
        if outdir is not None:
            pred.to_csv(pred_save,index=False)
    else:
        pred = pd.read_csv(pred_save)
    
    if not os.path.exists(eval_save) or overwrite:
        evals = []
        for target in tqdm(targets):
            # target = targets[0]
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
        if outdir is not None:
            df_cv_eval.to_csv(eval_save, index=False)
    else:
        df_cv_eval = pd.read_csv(eval_save)
    return df_cv_eval
