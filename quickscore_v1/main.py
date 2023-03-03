#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""

import os
import re
import sys
import pandas as pd
from pprint import pprint
from  argparse import ArgumentParser

try:
    from utils.config import process_config
    from agents.fernie import FernieAgent
except:
    from .utils.config import process_config
    from .agents.fernie import FernieAgent

def main(working_mode='train', train_file=None, result_dir=None, 
    train_strategy=None, gpu_device=None,
    batch_size=None, checkpoint_file=None, target_id=None, pred_file=None, 
    pred_result=None, debug=False, config_file=None, undersample=None,
    undersample_type=None, pose=None,
    command_line=False, max_epoch=None):
    # parse the path of the json config file

    if command_line:
        ap = ArgumentParser()
        ap.add_argument('-w', '--working_mode', choices=['train', 'valid', 'test', 
                                                    'predict', 'random'],
                        default=None)
        ap.add_argument('-r', '--result_dir', default=None)
        ap.add_argument('-m', '--checkpoint_file', default=None)
        ap.add_argument('-t', '--target_id', default=None)
        ap.add_argument('-p', '--pred_file', default=None)
        ap.add_argument('-o', '--pred_result', default=None)
        ap.add_argument('-d', '--debug', default=False, action='store_true')
        ap.add_argument('-b', '--batch_size', default=None, type=int)
        ap.add_argument('-c', '--config_file',
            metavar='config_json_file',
            default='./configs/train_test.json',
            help='The Configuration file in json format')
        args = ap.parse_args()
        # parse the config json file
        config = process_config(args.config_file)

        if args.batch_size is not None:
            config.batch_size = args.batch_size
        if args.target_id is not None:
            config.target_id = args.target_id
        if args.debug:
            config.debug = args.debug
        if args.working_mode is not None:
            config.working_mode = args.working_mode

        if working_mode == 'train':
            if args.result_dir is not None:
                config.result_dir = args.result_dir

        if working_mode == 'predict':
            if args.checkpoint_file is not None:
                config.checkpoint_file = args.checkpoint_file

            if args.result_dir is not None:
                config.checkpoint_file = os.path.join(args.result_dir, 
                    'checkpoints', 'model_best.pth.tar')
            else:
                 config.checkpoint_file = args.checkpoint_file
            if args.pred_result is not None:
                config.pred_result = args.pred_result
            if args.pred_file is not None:
                config.pred_file = args.pred_file
        config = process_config(config)

    else:
        if config_file is None:
            current = os.path.dirname(os.path.realpath(__file__))
            config_file = os.path.join(current, 'configs/train.json')
        
        config = process_config(config_file, result_dir)

        if train_strategy is not None:
            pose = list(range(int(re.findall('\d', train_strategy)[0])))
            
            if 'undersample' in train_strategy:
                undersample = True
            if 'per' in train_strategy:
                undersample_type = 'per'
            elif 'total' in train_strategy:
                undersample_type = 'total'
    
        if gpu_device is not None:
            config.gpu_device = gpu_device
        if pose is not None:
            config.pose = pose
        if undersample is not None:
            config.undersample = undersample
        if undersample_type is not None:
            config.undersample_type = undersample_type
        if max_epoch is not None:
            config.max_epoch = max_epoch
        if batch_size is not None:
            config.batch_size = batch_size
        if target_id is not None:
            config.target_id = target_id

        if working_mode == 'train':
            config.train_file = train_file

        elif working_mode == 'predict':
            config.pred_file = pred_file
            if result_dir is not None:
                config.checkpoint_file = os.path.join(result_dir, 'checkpoints', 
                'model_best.pth.tar')
            else:
                 config.checkpoint_file = checkpoint_file
            config.pred_result = pred_result
        
        config.working_mode = working_mode
        config.debug = debug

    # Create the Agent and pass all the configuration to it then run it..
    # agent_class = globals()[config.agent]
    # agent = agent_class(config)

    print(" THE Configuration of your experiment ..")
    pprint(config)

    print('Configing Agent ...')
    agent = FernieAgent(config)
    print('Runing Agent ...')
    res = agent.run()
    print('Finishing ...')
    agent.finalize()

    if working_mode == 'train' and not command_line:
        return res
    elif working_mode == 'predict' and not command_line:
        return res

if __name__ == '__main__':
    main(command_line=True)