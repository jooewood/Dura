#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
conda install pytorch=1.8.2=py3.7_cuda11.1_cudnn8.0.5_0 torchvision=0.2.1=py37_0 torchaudio==0.8.2 cudatoolkit=11.1 -c pytorch-lts -c nvidia -y
"""
import os
from DeepPurpose import CompoundPred as models
from DeepPurpose import utils
import pandas as pd
from utils import splitTrainTest, get_time_string, delete_a_folder
import multiprocessing


DL_names = [
    'Morgan',
    # 'Pubchem',
    'Daylight',
    'rdkit_2d_normalized',
    'ESPF',
    'ErG',
    'CNN',
    'CNN_RNN',
    'Transformer',
    # 'MPNN',
    # 'DGL_GCN',
    # 'DGL_NeuralFP',
    # 'DGL_GIN_AttrMasking',
    # 'DGL_GIN_ContextPred',
    # 'DGL_AttentiveFP'
]

class deeppurposeDL:
    
    def __init__(self):
        pass
    
    def encodeInput(self, data, drug_encoding, drug_col='SMILES', 
                    protein_col='protein_seq', target_encoding=None,
                    split_method='no_split', ycol='label', 
                    frac = [0.7, 0.1, 0.2], random_seed = '1234'
                    ):
        """ Encode input data using DeepPurpose's methods.
        """
        
        if isinstance(data, str):
            data = pd.read_csv(data)
        
        X_drug =  data[drug_col].values
        if target_encoding is not None and protein_col in data.columns:
            X_target = data[protein_col].values
        else:
            X_target = None
        
        y = data[ycol].values
        
        if split_method == 'no_split':
            if X_target is not None:
                data_encoded = utils.data_process(
                    X_drug = X_drug, X_target = X_target, y=y, 
                    drug_encoding=drug_encoding, target_encoding=target_encoding, 
                	split_method = split_method)
            else:
                data_encoded = utils.data_process(
                    X_drug = X_drug, y=y, 
                    drug_encoding=drug_encoding,
                	split_method = split_method)
            return data_encoded
        else:
            if X_target is not None:
                train, val, test = utils.data_process(
                    X_drug = X_drug, X_target = X_target, y=y, frac=frac,
                    drug_encoding=drug_encoding, 
                    target_encoding=target_encoding, 
                	split_method = split_method,
                    random_seed=random_seed)
            else:
                train, val, test = utils.data_process(
                    X_drug = X_drug, y=y, frac=frac,
                    drug_encoding=drug_encoding,
                	split_method = split_method,
                    random_seed=random_seed)
            return train, val, test

    def train(self, train_data, drug_encoding, checkpoint_path=None,
              
              target_encoding=None, feature_cols=None, pred_save=None,
              test_data=None, pred_col='score', label_col='label',
              val_data=None, drug_col='SMILES', protein_col='protein_seq',
              frac=[0.9, 0.1], random_seed=1234, overwrite=False,
              # Train hyperparameters
              cuda_id=0, num_workers=-1, LR=0.001, mode='cls',
              batch_size=128, train_epoch=20, **kwargs):

        if checkpoint_path is None:
            rm_outdir = True
            checkpoint_path = os.path.join('./tmp_model', get_time_string())
        else:
            rm_outdir = False

        model_save = os.path.join(checkpoint_path, 'model.pt')
        config_path = os.path.join(checkpoint_path, 'config.pkl')

        if not os.path.exists(model_save) and not\
            os.path.exists(config_path) or overwrite:
            if isinstance(train_data, str):
                train_data = pd.read_csv(train_data)

            if feature_cols is None:
                feature_cols = [drug_col, protein_col]
                
            if target_encoding is None and protein_col in feature_cols:
                feature_cols.remove(protein_col)
            
            if val_data is None:
                train_data, val_data, _ = splitTrainTest(train_data, frac=frac,
                    random_seed=random_seed, mode=mode)
                
            train_data = self.encodeInput(train_data, drug_encoding=drug_encoding,
                                        target_encoding=target_encoding)
            val_data = self.encodeInput(val_data, drug_encoding=drug_encoding,
                                        target_encoding=target_encoding)
                
            if num_workers == -1:
                num_workers = multiprocessing.cpu_count() - 1
            
            self.config = utils.generate_config(drug_encoding=drug_encoding, 
                target_encoding=target_encoding,
                result_folder = checkpoint_path,
                cuda_id = cuda_id,
                num_workers = num_workers,
                train_epoch = train_epoch,
                LR=LR)
            
            self.model = models.model_initialize(**self.config)
            self.model.train(train_data, val_data)
            if checkpoint_path[-1] == '/':
                checkpoint_path[-1]  = checkpoint_path[:-1]
            self.model.save_model(checkpoint_path)
            
            if rm_outdir:
                delete_a_folder(checkpoint_path)
        else:
            self.config = utils.load_dict(checkpoint_path)
            self.model = models.model_initialize(**self.config)
            self.model.load_pretrained(model_save)

        if test_data is not None:
            if isinstance(test_data, str):
                test_data = pd.read_csv(test_data)
            
            test_data_encoded = self.encodeInput(test_data, 
                 drug_encoding=drug_encoding,
                 target_encoding=target_encoding)
            y_pred = self.model.predict(test_data_encoded)
            test_data[pred_col] = y_pred
            if pred_save is not None:
                test_data.to_csv(pred_save, index=False)
            return test_data
        else:
            return None
        
    def predict(self, test_data, checkpoint_path=None, pred_col='score'):
        if isinstance(test_data, str):
            test_data = pd.read_csv(test_data)
        
        if checkpoint_path is not None:
            self.config = utils.load_dict(checkpoint_path)
            checkpoint_path = os.path.join(checkpoint_path, 'model.pt')
            self.model = models.model_initialize(**self.config)
            self.model.load_pretrained(checkpoint_path)
        
        test_data_encoded = self.encodeInput(test_data, 
             drug_encoding=self.config.drug_encoding,
             target_encoding=self.config.target_encoding)

        y_pred = self.model.predict(test_data_encoded)
        test_data[pred_col] = y_pred
        return test_data

def deeppurposeholdout(model_id, test=None, train=None, mode='cls',
    label_col='label', feature_cols=['SMILES'], 
    pred_col='score', pred_save=None, overwrite=False, checkpoint_path=None,
    protein_col='protein_seq'):
    
    if '.' in model_id:
        drug_encoding, target_encoding = model_id.split('.')
    else:
        drug_encoding = model_id
        target_encoding = None
        
    if target_encoding is None and protein_col in feature_cols:
        feature_cols.remove(protein_col)

    model = deeppurposeDL()
    pred = model.train(train_data=train, test_data=test, 
               feature_cols=feature_cols,
               drug_encoding=drug_encoding,
               target_encoding=target_encoding,
               checkpoint_path=checkpoint_path,
               overwrite=overwrite,
               mode=mode)
    return pred
        
        
        
        
        
        
        
        
        
        