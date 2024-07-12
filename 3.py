# train
from methods.modeling.config import gen_config, gen_default_config, create_folders
import torch.nn as nn
import torch
import pickle
import numpy as np
import json
from pathlib import Path

from torch import optim
from methods.modeling.utils import *
from tqdm import tqdm
def train_fn(loader, model, optimizer, loss_fn, basis, logs, config, mean=None):
    """
    This function executes the trainig loop based on model, data loader, optimizer, loss function, scaler choice.

    Args:
        loader (torch.utils.data.DataLoader): Corresponding data loader, can be validation or training.
        model: Chosen model FTTransformer or ResNetModel.
        optimizer: Optimizer of a certain type. Change only if you are proficient in machine learning.
        loss_fn: Loss function of a certain type. Change only if you are proficient in machine learning.       
        logs (dict): Logs dictionary.
        basis: Normalized basis of vectors, coming from the dimension reduction.
        mean: Mean of the dimension reduction procedure. Defaults to None.
    """
    basis = basis.to(device=config['DEVICE'])
    if mean is not None:
        mean = mean.to(device=config['DEVICE'])

    loop = tqdm(loader)
    accumulated_loss = []
    full_data_size = 0
    for _, (inp, coef, out) in enumerate(loop):
        #print("train_fn(): inp.shape: {}, coef.shape: {}, out.shape: {}".format(inp.shape, coef.shape, out.shape))
        data = inp.to(device=config['DEVICE'])
        coef = coef.to(device=config['DEVICE'])
        output = out.to(device=config['DEVICE'])        

        # Forward propagation - compute predictions of coefficients, i.e. model output:
        predictions_coef = model(data).to(device=config['DEVICE'])
        
        # Computing the entire prediction result, depending on if there is a mean or not:
        if mean is not None:
            predictions_output = torch.matmul(predictions_coef, basis) + mean
        else:
            predictions_output = torch.matmul(predictions_coef, basis)

        #loss_1 = loss_fn(predictions_coef, coef)
        loss_2 = loss_fn(predictions_output, output)

        loss = loss_2

        accumulated_loss.append(loss.item() * len(data))
        full_data_size += len(data)

        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update tqdm loop
        # loop.set_postfix(loss=loss.item())
    logs['loss'].append(loss.item())

    return np.sum(accumulated_loss)/full_data_size

def train(config : dict = None, optuna_trial = None):
    if not config:
        config = gen_default_config()
    else:
        config = gen_config(**config)
    create_folders(config)

    model, best_weights, last_weights = choose_model_and_weights(
        config['MODEL_NAME'], 
        config['MODEL_CHARACTERS'],
        config['DEVICE']
        )
    
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=0)
    logs = {'loss_test': [], 'loss_train': [], 'loss': []}

    BASIS, MEAN = load_companents_to_restore(config['PATH_TO_BASIS'], config['PATH_TO_MEAN'])
    #print(f'Shapes: basis={BASIS.shape}, mean={MEAN.shape}')

    train_loader, val_loader = get_loaders(
        config['PATH_INP_DATA'],
        config['PATH_COEF_DATA'],
        config['PATH_OUTPUT_DATA'],
        config['TRAIN'],
        config['TEST'],
        config['BATCH_SIZE'],
        config['NUM_WORKERS'],
        config['PIN_MEMORY']
    )

    #load checkpoint and Logs
    if config['LOAD_MODEL']:
        load_checkpoint(torch.load(best_weights), model)

    if config['LOAD_LOGS']:
        with open(config.LOGS, 'rb') as f:
            logs = pickle.load(f)

    best_accuracy = check_accuracy(val_loader, model, BASIS, device=config['DEVICE'], mean=MEAN)
    print('Start accuracy:', best_accuracy)

    for step in range(config['NUM_EPOCHS']):
        print(f"Epoch {step}")
        full_epoch_loss = train_fn(train_loader, model, optimizer, loss_fn, BASIS, logs, config, mean=MEAN)

        # save model
        if config['SAVE_MODEL']:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
            
            #print(f'Shape check, val_loader: {iter(val_loader).next()[0].shape}, train_loader: {iter(train_loader).next()[0].shape}')
            
            # check accuracy
            step_accuracy_test = check_accuracy(val_loader, model, BASIS, device=config['DEVICE'], mean=MEAN)
            step_accuracy_train = check_accuracy(train_loader, model, BASIS, device=config['DEVICE'], mean=MEAN)
            print(f'=> step_mse_train: {step_accuracy_train}, step_mse_test: {step_accuracy_test}')
            best_accuracy = save_checkpoint(
                checkpoint, 
                step_accuracy_test, 
                best_accuracy, 
                best_weights,
                last_weights)

            #Save Logs
            logs['loss_test'].append(step_accuracy_test)
            logs['loss_train'].append(step_accuracy_train)
            with open(config['LOGS'], 'wb') as f:
                pickle.dump(logs, f)

        if optuna_trial:
            import optuna

            optuna_trial.report(full_epoch_loss, step)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

    return best_accuracy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="a config json file")

    config_args = None

    args = parser.parse_args()
    if args.config:
        path = Path(args.config)

        if not path.exists():
            raise Exception('config file does not exist: {}'.format(str(path)))
        if path.suffix != '.json':
            raise Exception('config should be a json file: {}'.format(str(path)))
        
        with open(path, 'r') as f:
            text = f.read()
            config_args = json.loads(text)

    train(config=config_args)

