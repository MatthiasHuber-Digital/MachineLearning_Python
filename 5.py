# train_optimizer
import config
import torch.nn as nn
import torch
import pickle
import numpy as np
import optuna
from datetime import datetime
from pathlib import Path
import json

from torch import optim
from tqdm import tqdm

from methods.modeling.utils import *
from methods.modeling.config import gen_config, DEFAULT_PARAMS
from train import train

def create_trial_config(trial: optuna.Trial, opt_config : dict):
    opt_name = opt_config['MODEL_NAME']
    opt_characters = opt_config['MODEL_CHARACTERS'][opt_name]
    model_name = None
    model_characters = None

    if opt_name == 'resnet_optimizer':
        model_characters = {
            'D_IN'           : opt_characters['D_IN'],
            'D_OUT'          : opt_characters['D_OUT'],
            'N_BLOCKS'       : trial.suggest_int('n_blocks', opt_characters['N_BLOCKS'][0], opt_characters['N_BLOCKS'][1]),
            'D_MAIN'         : opt_characters['D_MAIN'],
            'D_HIDDEN'       : opt_characters['D_HIDDEN'],
            'DROPOUT_FIRST'  : trial.suggest_float('dropout_first', opt_characters['DROPOUT_FIRST'][0], opt_characters['DROPOUT_FIRST'][1], step=0.1),
            'DROPOUT_SECOND' : trial.suggest_float('dropout_second', opt_characters['DROPOUT_SECOND'][0], opt_characters['DROPOUT_SECOND'][1], step=0.1)
        }
        model_name = 'resnet'

    elif opt_name == 'fttransformer_optimizer':
        model_characters = {
            'N_NUM_FEATURES'           : opt_characters['N_NUM_FEATURES'],
            'CAT_CARDINALITIES'        : opt_characters['CAT_CARDINALITIES'],
            'D_TOKEN'                  : opt_characters['D_TOKEN'],
            'N_BLOCKS'                 : trial.suggest_int('n_blocks', opt_characters['N_BLOCKS'][0], opt_characters['N_BLOCKS'][1], step=1),
            'ATTENTION_N_HEADS'        : trial.suggest_int('attention_n_heads', opt_characters['ATTENTION_N_HEADS'][0], opt_characters['ATTENTION_N_HEADS'][1], step=1),
            'ATTENTION_DROPOUT'        : opt_characters['ATTENTION_DROPOUT'],
            'ATTENTION_INITIALIZATION' : opt_characters['ATTENTION_INITIALIZATION'],
            'FFN_D_HIDDEN'             : opt_characters['FFN_D_HIDDEN'],
            'FFN_DROPOUT'              : opt_characters['FFN_DROPOUT'],
            'RESIDUAL_DROPOUT'         : opt_characters['RESIDUAL_DROPOUT'],
            'D_OUT'                    : opt_characters['D_OUT']
        }
        model_name = 'fttransformer_custom'

    else:
        raise Exception('uknown model name: {}'.format(opt_name))

    trial_config = {k : opt_config[k] for k in DEFAULT_PARAMS}
    trial_config['usecase_name']     = opt_config['USECASE']
    trial_config['model_name']       = model_name
    trial_config['model_characters'] = model_characters
    trial_config['result_path']      = Path(opt_config['RESULT_PATH']) / str(trial.number)
    trial_config['dataset_path']     = opt_config['DATA_PATH']

    return trial_config

def run_trial(trial, opt_config):
    trial_config = create_trial_config(trial, opt_config)
    print("NUM_EPOCHS: {}".format(trial_config['NUM_EPOCHS']))
    val_accuracy = train(trial_config, trial)

    return val_accuracy

class Objective(object):
    def __init__(self, opt_config):
        self.opt_config = opt_config

    def __call__(self, trial):
        return run_trial(trial, self.opt_config)

def main(opt_config : dict):
    opt_config = gen_config(**opt_config)
    trials_number = opt_config['MODEL_CHARACTERS'][opt_config['MODEL_NAME']]['TRIALS']

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(), direction="minimize")
    objective = Objective(opt_config)
    study.optimize(objective, n_trials=trials_number)

    with open(opt_config['RESULT_PATH'] / f"{datetime.now().strftime('%d.%m.%H.%M')}.txt", "a") as f:
        f.write(f"Number of finished trials: {len(study.trials)}\n")

        f.write("Best trial:\n")
        trial = study.best_trial

        f.write(f"\tAccuracy: {study.best_trial.value}\n")

        f.write("\tParams: \n")
        for key, value in study.best_trial.params.items():
            f.write(f"\t\t{key}: {value}\n")

        f.write("\n\n\n\nOther trials:")
        for t in study.trials:
            f.write(f"\n\t{str(t.params)}, Accuracy: {t.value}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="a config json file", required=True)

    args = parser.parse_args()

    path = Path(args.config)

    if not path.exists():
        raise Exception('config file does not exist: {}'.format(str(path)))
    if path.suffix != '.json':
        raise Exception('config should be a json file: {}'.format(str(path)))
    
    with open(path, 'r') as f:
        text = f.read()
        config_args = json.loads(text)

    main(opt_config=config_args)

