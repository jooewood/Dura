#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
import os
import pickle
import numpy as np

import torch
from torch import tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split,\
    RandomSampler
from sklearn.model_selection import KFold

def collater(data):
        name = []
        atom_type = []
        charge = []
        distance = []
        amino_acid  = []
        mask_vector = []
        label = []
        if 'label' in data[0].keys():
            for unit in data:
                atom_type.append(unit['atom_type'])
                charge.append(unit['charge'])
                distance.append(unit['distance'])
                amino_acid.append(unit['amino_acid'])
                mask_vector.append(unit['mask_vector'])
                label.append(unit['label'])

            return [torch.tensor(np.array(atom_type)),
                    torch.tensor(np.array(charge)),
                    torch.tensor(np.array(distance)),
                    torch.tensor(np.array(amino_acid)),
                    torch.tensor(np.array(mask_vector))], torch.tensor(np.array(label))
        else:
            for unit in data:
                atom_type.append(unit['atom_type'])
                charge.append(unit['charge'])
                distance.append(unit['distance'])
                amino_acid.append(unit['amino_acid'])
                mask_vector.append(unit['mask_vector'])
            return [torch.tensor(np.array(atom_type)), 
                    torch.tensor(np.array(charge)),
                    torch.tensor(np.array(distance)),
                    torch.tensor(np.array(amino_acid)),
                    torch.tensor(np.array(mask_vector))]

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        if 'label' in sample.keys():
            return {
            "atom_type": tensor(sample["atom_type"], dtype=torch.int64),
            "charge": tensor(sample["charge"], dtype=torch.int64),
            "distance": tensor(sample["distance"], dtype=torch.int64),
            "amino_acid": tensor(sample["amino_acid"], dtype=torch.int64),
            "mask_vector": tensor(sample["mask_vector"], dtype=torch.float32),
            "label": tensor(sample["label"], dtype=torch.int64)
            }
        else:
            return {
            "atom_type": tensor(sample["atom_type"], dtype=torch.int64),
            "charge": tensor(sample["charge"], dtype=torch.int64),
            "distance": tensor(sample["distance"], dtype=torch.int64),
            "amino_acid": tensor(sample["amino_acid"], dtype=torch.int64),
            "mask_vector": tensor(sample["mask_vector"], dtype=torch.float32)
            }


class FernieDataset(Dataset):
    def __init__(self, config, pickle_path):
                
        with open(pickle_path, 'rb') as f:
            features = pickle.load(f)

        self.config = config
        self.cf = config.cf
        self.max_atom_num = config.max_atom_num
        self.debug = config.debug
        self.batch_size = config.batch_size
        self.pose = config.pose

        try:
            x = features[7][:,1]
            self.label = True
        except:
            self.label = False

        self.names = []
        self.atom_type = []
        self.charge = []
        self.distance = []
        self.amino_acid = []
        self.mask_vector = []
        if self.label:
            self.labels = []

        if self.config.working_mode == 'train':
            try:
                for i, name in enumerate(features[0]):
                    mode_no = int(name.split('_mode_')[1])
                    if mode_no in self.pose:
                        self.names.append(features[0][i])
                        self.atom_type.append(features[1][i].astype(np.int64))
                        self.charge.append(features[2][i].astype(np.int64))
                        self.distance.append(features[3][i].astype(np.int64))
                        self.amino_acid.append(features[4][i].astype(np.int64))
                        self.mask_vector.append(features[5][i])
                        if self.label:
                            self.labels.append(features[7][i,1].astype(np.int64))
                if not self.label:
                    self.labels = None
            except:
                print("Molecule name doesen't contain _mode_.")
                self.names = features[0]
                self.atom_type = features[1].astype(np.int64)
                self.charge = features[2].astype(np.int64)
                self.distance = features[3].astype(np.int64)
                self.amino_acid = features[4].astype(np.int64)
                self.mask_vector = features[5]
                if self.label:
                    self.labels = features[7][:,1].astype(np.int64)
                else:
                    self.labels = None

        elif self.config.working_mode == 'predict':
            self.names = features[0]
            self.atom_type = features[1].astype(np.int64)
            self.charge = features[2].astype(np.int64)
            self.distance = features[3].astype(np.int64)
            self.amino_acid = features[4].astype(np.int64)
            self.mask_vector = features[5]
            if self.label:
                self.labels = features[7][:,1].astype(np.int64)
            else:
                self.labels = None

        if self.debug:
            self.names = self.names[:self.batch_size]
            self.atom_type = self.atom_type[:self.batch_size]
            self.charge = self.charge[:self.batch_size]
            self.distance = self.distance[:self.batch_size]
            self.amino_acid = self.amino_acid[:self.batch_size]
            self.mask_vector = self.mask_vector[:self.batch_size]
            if self.label:
                self.labels = self.labels[:self.batch_size]

    def get_mask_mat(self, i):
        return np.hstack((
                         np.ones((self.cf, i, 1)), 
                         np.zeros((self.cf, self.max_atom_num - i, 1))
                         )).astype(np.float32)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.label:
            sample = {
                "atom_type": self.atom_type[idx],
                "charge": self.charge[idx],
                "distance": self.distance[idx],
                "amino_acid": self.amino_acid[idx],
                "mask_vector": self.get_mask_mat(self.mask_vector[idx]),
                "label": self.labels[idx]
                }
        else:
                sample = {
                "atom_type": self.atom_type[idx],
                "charge": self.charge[idx],
                "distance": self.distance[idx],
                "amino_acid": self.amino_acid[idx],
                "mask_vector": self.get_mask_mat(self.mask_vector[idx])
                }
        
        return sample

class FernieDataLoader:
    def __init__(self, config):
        
        self.config = config
        self.undersample = config.undersample
        self.undersample_type = config.undersample_type
        self.seed = config.seed
        self.working_mode = config.working_mode
        self.n_splits = config.n_splits

        self.validation_split = config.validation_split
        self.cf = config.cf
        self.max_atom_num = config.max_atom_num
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.debug = config.debug

        ## input
        try:
            self.train_file = config.train_file
        except:
            self.train_file = None
        try:
            self.valid_file = config.valid_file
        except:
            self.valid_file = None
        try:
            self.test_file = config.test_file
        except:
            self.test_file = None
        try:
            self.pred_file = config.pred_file
        except:
            self.pred_file = None

        if self.working_mode == 'train' or self.working_mode == 'random':
            if self.valid_file is None:
                print('Do not detect a valid file, program will split '
                      'validation file by itself.')
                self.split_train_valid(self.train_file, seed=self.seed, 
                    validation_split=self.validation_split)
            else:
                self.dataset_init('train', self.train_file) 
                self.dataset_init('valid', self.valid_file)
            if self.pred_file is not None:
                self.dataset_init('pred', self.pred_file)
                    
            self.train_loader = DataLoader(self.train_set,
                                           batch_size=self.batch_size,
                                           shuffle=True, 
                                           pin_memory=False,
                                           num_workers=self.num_workers,
                                           collate_fn=collater,
                                           drop_last=True
                                           )
            self.train_iterations = len(self.train_loader)
            
            self.valid_loader = DataLoader(self.valid_set,
                                           batch_size=self.batch_size,
                                           shuffle=False, 
                                           pin_memory=False,
                                           num_workers=self.num_workers,
                                           collate_fn=collater
                                           )
            self.valid_iterations = len(self.valid_loader)
            
        elif self.working_mode == 'test' or self.working_mode == 'random':
            self.dataset_init('test', self.test_file)
            self.test_loader = DataLoader(self.test_set, 
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          pin_memory=False,
                                          num_workers=self.num_workers,
                                          collate_fn=collater
                        )
            self.test_iterations = len(self.test_loader)

        elif self.working_mode == 'predict' or self.working_mode == 'random':
            self.dataset_init('pred', self.pred_file)
            self.pred_loader = DataLoader(self.pred_set,
                                          batch_size=self.batch_size,
                                          shuffle=False, 
                                          num_workers=self.num_workers,
                                          collate_fn=collater
                                          )
            self.pred_iterations = len(self.pred_loader)

    def read_dataset(self, file=None):
        if '.pt' == os.path.splitext(os.path.basename(file))[1]:
            return torch.load(file)
        elif '.pickle' == os.path.splitext(os.path.basename(file))[1]:
            return FernieDataset(self.config, file)

    def split_train_valid(self, path, seed, validation_split):
        """
        undersample_type: 'per' or 'total'
        """
        if isinstance(path, str):
            whole_dataset = self.read_dataset(path)
            valid_set_size = int(validation_split * len(whole_dataset))
            train_set_size = len(whole_dataset) - valid_set_size

            self.train_set, self.valid_set = random_split(whole_dataset, 
                [train_set_size, valid_set_size],  
                generator=torch.Generator().manual_seed(seed))
        else:
            active_datasets = []
            decoy_datasets = []
            for file in path:
                """
                When training model, the file list must be like:
                target1.active.pickle, target1.decoy.pickle, target2.active.pickle, target2.decoy.pickle,...
                """
                if 'active' in file:
                    tmp_active_set = self.read_dataset(file)
                    active_datasets.append(tmp_active_set)
                    if self.undersample and self.undersample_type=='per':
                        active_n = len(tmp_active_set)
                if 'decoy' in file:
                    tmp_decoy_set = self.read_dataset(file)
                    if self.undersample and self.undersample_type=='per':
                        try:
                            tmp_decoy_set, _ = random_split(tmp_decoy_set, 
                                [active_n, len(tmp_decoy_set)-active_n],  
                                generator=torch.Generator().manual_seed(seed))
                        except:
                            self.undersample_type = 'total'
                    decoy_datasets.append(tmp_decoy_set)

            active_dataset = ConcatDataset(active_datasets)
            decoy_dataset = ConcatDataset(decoy_datasets)

            active_valid_set_size = int(validation_split * len(active_dataset))
            active_train_set_size = len(active_dataset) - active_valid_set_size
            active_train_set, active_valid_set = random_split(active_dataset, 
                [active_train_set_size, active_valid_set_size],  
                generator=torch.Generator().manual_seed(seed))

            if self.undersample and self.undersample_type=='total':
                active_n = len(active_dataset)
                decoy_dataset, _ = random_split(decoy_dataset, 
                                    [active_n, len(decoy_dataset)-active_n],  
                                    generator=torch.Generator().manual_seed(seed))

            decoy_valid_set_size = int(validation_split * len(decoy_dataset))
            decoy_train_set_size = len(decoy_dataset) - decoy_valid_set_size
            decoy_train_set, decoy_valid_set = random_split(decoy_dataset, 
                [decoy_train_set_size, decoy_valid_set_size],  
                generator=torch.Generator().manual_seed(seed))

            self.train_set = ConcatDataset([active_train_set, decoy_train_set])
            self.valid_set = ConcatDataset([active_valid_set, decoy_valid_set])

    def dataset_init(self, mode, files=None):      
        assert mode in ['train', 'valid', 'test', 'pred']

        if isinstance(files, str):
            dataset_ = self.read_dataset(files)
        else:
            datasets = []
            for file in files:
                datasets.append(self.read_dataset(file))
            dataset_ = ConcatDataset(datasets)

        if mode == 'train':
            self.train_set = dataset_
        elif mode == 'valid':
            self.valid_set = dataset_
        elif mode == 'test':
            self.test_set = dataset_
        elif mode == 'pred':
            self.pred_set = dataset_


    def finalize(self):
        pass