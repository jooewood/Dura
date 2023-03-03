#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import sys
import pickle
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial

from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy

from .base import BaseAgent
# directory reach
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
# setting path
sys.path.append(parent)
try:
    from graphs.models.fernie import Fernie
    from datasets.fernie import FernieDataLoader
    from utils.misc import print_cuda_statistics, adjust_order
    from utils.metrics import AverageMeter, cls_accuracy
except:
    from quickscore_v1.graphs.models.fernie import Fernie
    from quickscore_v1.datasets.fernie import FernieDataLoader
    from quickscore_v1.utils.misc import print_cuda_statistics, adjust_order
    from quickscore_v1.utils.metrics import AverageMeter, cls_accuracy

class FernieAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

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
        try:
            self.pred_result = config.pred_result
        except:
            self.pred_result = None

        self.debug = config.debug
        self.working_mode = config.working_mode
        # self.async_loading = config.async_loading

        # -------------
        # training set
        # -------------
        self.use_scheduler = config.use_scheduler
        self.seed = config.seed
        self.gpu_device = config.gpu_device
        self.cuda = config.cuda
        self.max_epoch = config.max_epoch

        if self.debug:
            self.max_epoch = 1

        # ----------
        # Optimizer
        # ----------
        self.optimizer_name = config.optimizer_name
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.weighted_loss = config.weighted_loss
        self.pos_weight = float(config.pos_weight)
        self.neg_weight = float(config.neg_weight)

        # -----------
        # Path
        # --------  
        try:
            self.checkpoint_dir = config.checkpoint_dir
        except:
            self.checkpoint_dir = None
        self.checkpoint_file = config.checkpoint_file
        try:
            self.summary_dir = config.summary_dir
        except:
            self.summary_dir = None


        # define models
        self.model = Fernie(self.config)
        
        # define data_loader
        self.data_loader = FernieDataLoader(self.config)

        if self.weighted_loss:
            weight = torch.tensor([self.neg_weight, self.pos_weight])
            weight = weight.cuda()
            self.loss = partial(F.nll_loss, weight=weight)
        else:
            self.loss = F.nll_loss
        self.acc = accuracy

        # define optimizer
        if self.optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                             lr=self.lr)
        if self.optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        if self.optimizer_name == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.model.parameters())

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        if self.is_cuda and not self.cuda:
            self.logger.info("WARNING: You have a CUDA device, "
                             "so you should probably enable CUDA")
        
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.cuda

        # set the manual seed for torch
        self.manual_seed = self.seed
        if self.cuda:
            torch.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.gpu_device)
            self.model = self.model.to(self.device)
            # self.loss = self.loss.to(self.device)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")
            
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.checkpoint_file)
        # Summary Writer
        if self.working_mode == 'train':
            self.summary_writer = SummaryWriter(log_dir=self.summary_dir, 
                                                comment='Fernie')
        # scheduler for the optimizer
        if self.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                'min', patience=self.learning_rate_patience,
                 min_lr=1e-10, verbose=True)

        
    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        """
        if self.checkpoint_dir is not None:
            filename = os.path.join(self.checkpoint_dir, filename)

        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(filename))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best:  flag is it is the best model
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        if self.checkpoint_dir is not None:
            torch.save(state, os.path.join(self.checkpoint_dir, filename))
            # If it is the best copy it to another file 'model_best.pth.tar'
            if is_best:
                shutil.copyfile(os.path.join(self.checkpoint_dir, filename),
                                os.path.join(self.checkpoint_dir , 
                                             'model_best.pth.tar'))

    def run(self):
        """
        This function will the operator
        """
        assert self.working_mode in ['train', 'valid', 'test', 'predict', 'random']
        try:
            if self.working_mode == 'valid':
                print('Start to valid.')
                self.validate()
            elif self.working_mode == 'train':
                print('Start to train.')
                res = self.train()
            elif self.working_mode == 'test':
                print('Start to test.')
                pass
            elif self.working_mode == 'predict':
                res = self.predict()
            return res

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.max_epoch):
            print('Training at epoch', epoch)
            self.current_epoch = epoch
            self.train_one_epoch()

            print('Valid after epoch', epoch)
            valid_acc = self.validate()
            is_best = valid_acc > self.best_valid_acc
            if is_best:
                self.best_valid_acc = valid_acc
            print('Save the checkpoint of epoch', epoch)
            self.save_checkpoint(is_best=is_best)
        return self.config.result_dir

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm

        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()

        current_batch = 0
        for x, y in tqdm(self.data_loader.train_loader):
            x = [i.to(self.device) for i in x]
            y = y.to(self.device)

            # model

            logits = self.model(x)
            pred = torch.exp(logits)

            # loss
            cur_loss = self.loss(logits, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
                
            # optimizer
            self.optimizer.zero_grad()
            cur_loss.backward()
            epoch_loss.update(cur_loss.item())
            self.optimizer.step()

            top1 = cls_accuracy(pred.data, y.data)
            top1_acc.update(top1[0].item(), y.size(0))

            self.current_iteration += 1
            current_batch += 1

            self.summary_writer.add_scalar("epoch/loss", epoch_loss.val, self.current_iteration)
            self.summary_writer.add_scalar("epoch/accuracy", top1_acc.val, self.current_iteration)

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Acc: " + str(top1_acc.val) )


    def validate(self):
        """
        One epoch validation
        :return:
        """

        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        top1_acc = AverageMeter()

        for x, y in tqdm(self.data_loader.valid_loader):
            x = [i.to(self.device) for i in x]
            y = y.to(self.device)
            
            # model
            logits = self.model(x)
            pred = torch.exp(logits)
            # loss
            cur_loss = self.loss(logits, y)
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')

            top1 = cls_accuracy(pred.data, y.data)
            epoch_loss.update(cur_loss.item())
            top1_acc.update(top1[0].item(), y.size(0))

        self.logger.info("Validation results at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.avg) + "- Acc: " + str(top1_acc.val))

        return top1_acc.avg

    def pickle_data2df(self, file, debug=False, batch_size=16,
        mol_id_col='mol_id', pred_col='score', label_col='label', 
        target_id_col='target_id'):
        with open(file, 'rb') as f:
            features = pickle.load(f)
            if debug:
                features[0] = features[0][:batch_size]
            try:
                if debug:
                    features[6] = features[6][:batch_size]
                    features[7] = features[7][:batch_size]
                return pd.DataFrame({
                    target_id_col: features[6],
                    mol_id_col: features[0],
                    pred_col: None,
                    label_col : features[7][:,1]
                    })
            except:
                return pd.DataFrame({
                    target_id_col: features[6],
                    mol_id_col: features[0],
                    pred_col : None,
                    label_col: None
                    })

    def predict(self, pred_col='score', target_id_col='target_id',
        label_col='label', mol_id_col='mol_id'):

        print(f"Start to predict on {self.pred_file}")
        self.model.eval()
        y_pred = None
        with torch.no_grad():
            try:
                for x in tqdm(self.data_loader.pred_loader):
                    x = [i.to(self.device) for i in x]
                    # model
                    logits = self.model(x)
                    pred = torch.exp(logits)[:,1]

                    if y_pred is not None:
                        y_pred = np.concatenate([y_pred, 
                            pred.cpu().detach().numpy()], axis=0)
                    else:
                        y_pred = pred.cpu().detach().numpy()
            except:
                for x, y in tqdm(self.data_loader.pred_loader):
                    x = [i.to(self.device) for i in x]
                    # model
                    logits = self.model(x)
                    pred = torch.exp(logits)[:,1]

                    if y_pred is not None:
                        y_pred = np.concatenate([y_pred, 
                            pred.cpu().detach().numpy()], axis=0)
                    else:
                        y_pred = pred.cpu().detach().numpy()

        result = self.pickle_data2df(self.pred_file, debug=self.config.debug,
            batch_size=self.config.batch_size, target_id_col=target_id_col)
        result[pred_col] = y_pred
        try:
            if self.config.target_id is not None:
                result[target_id_col] = self.config.target_id
        except:
            pass
        if result[label_col][0] is None:
            del result[label_col]

        result.sort_values(by=pred_col, ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)
        result = adjust_order(result, [mol_id_col, target_id_col, label_col,
            pred_col])

        if self.pred_result is not None:
            result.to_csv(self.pred_result, index=False)
            print(f"Succeed to save prediction into {self.pred_result}")

        return result

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the 
            operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        if self.working_mode == 'train':
            self.save_checkpoint()
            self.summary_writer.export_scalars_to_json(
                "{}all_scalars.json".format(self.summary_dir))
            self.summary_writer.close()
        self.data_loader.finalize()