import os

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

import json
from easydict import EasyDict
from pprint import pprint

from .dirs import create_dirs


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def process_config(config, result_dir=None):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    if isinstance(config, str):
        config, _ = get_config_from_json(config)

    # making sure that you have provided the exp_name.

    if result_dir is not None:
        config.result_dir = result_dir

    try:
        if config.result_dir is not None:        
        # create some important directories to be used for that experiment.
            config.summary_dir = os.path.join(config.result_dir, "summaries/")
            config.checkpoint_dir = os.path.join(config.result_dir, "checkpoints/")
            config.out_dir = os.path.join(config.result_dir, "out/")
            config.log_dir = os.path.join(config.result_dir, "logs/")
            create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, 
                config.log_dir])
            # setup logging in the project
            setup_logging(config.log_dir)

            logging.getLogger().info("Hi, This is QuickScore V1.")
            logging.getLogger().info("After the configurations are successfully"
                                    "processed and dirs are created.")
            logging.getLogger().info("The pipeline of the project will begin now.")
        else:
            config.summary_dir = None
            config.checkpoint_dir = None
            config.out_dir = None
            config.log_dir = None
    except:
        config.summary_dir = None
        config.checkpoint_dir = None
        config.out_dir = None
        config.log_dir = None

    return config
