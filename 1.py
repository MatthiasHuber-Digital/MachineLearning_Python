# visualization
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np

from models.transformer import FTTransformer
from models.resnet import ResNetModel
from models.mlp import MlpModel

from dataset import VectorDataset
from torch.utils.data import DataLoader
from utils import load_checkpoint

from sklearn import metrics

import config

with open('data/data_pca/compressed_basis.pickle', 'rb') as handle:
    compressed_basis = pickle.load(handle)

with open('data/data_pca/output_data.pickle', 'rb') as handle:
    output_data = pickle.load(handle)  

with open('data/data_pca/nominal_curve.pickle', 'rb') as handle:
    nominal_curve = pickle.load(handle)      
    
with open('data/data_pca/mean_to_restore.pickle', 'rb') as handle:
    mean_to_restore = pickle.load(handle) 

# with open('data/data_svm/compressed_basis.pickle', 'rb') as handle:
#     compressed_basis = pickle.load(handle)

# with open('data/data_svm/output_data.pickle', 'rb') as handle:
#     output_data = pickle.load(handle)  

# with open('data/data_svm/nominal_curve.pickle', 'rb') as handle:
#     nominal_curve = pickle.load(handle)      

def vis_logs(path_to_pkl):
    with open(path_to_pkl, 'rb') as f:
        logs = pickle.load(f)

    # plotting
    plt.title('R2_score Graph')
    plt.xlabel("R2_score")
    plt.ylabel("Epochs")
    plt.plot(logs['acc_test'], color ="red")
    plt.plot(logs['acc_train'], color ="Green")
    plt.legend(['Test', 'Train'])
    plt.show();

    plt.title('Loss Graph')
    plt.xlabel("Loss")
    plt.ylabel("Epochs")
    plt.plot(logs['loss'], color ="blue")
    plt.show();

def plot_2_graph(curve1, curve2):
     
    plt.figure(figsize=(15,8))

    plt.plot(curve1)
    plt.plot(curve2)
    plt.ylabel('Features')
    plt.xlabel('Time points')
    plt.legend(['Predict', 'Target'])
     
    plt.show();

def score_metrics(surrogate, simulation):
    max_simulation, rms_simulation = diff_btw_nominal_simulation(nominal_curve, output_data)
    list_max_relative, list_rms_relative = [], []
    for idx in range(surrogate.shape[0]):
        max_relative = metrics.max_error(surrogate[idx], simulation[idx]) / max_simulation
        rms_relative = metrics.mean_squared_error(surrogate[idx], simulation[idx], squared=False) / rms_simulation

        list_max_relative.append(max_relative)
        list_rms_relative.append(rms_relative)

        print('max_error', max_relative)
        print('mse_error', rms_relative)
        print()
        plot_2_graph(surrogate[idx], simulation[idx])
    
    print('Avg_max_relative', np.mean(list_max_relative))
    print('Avg_rms_relative', np.mean(list_rms_relative))

def diff_btw_nominal_simulation(nominal_curve, simulation_curve):
    max_simulation, rms_simulation = 0, 0
    for idx in range(simulation_curve.shape[0]):
        step_max_simulation = metrics.max_error(nominal_curve, simulation_curve[idx])
        step_rms_simulation = metrics.mean_squared_error(nominal_curve, simulation_curve[idx], squared=False)
        max_simulation = max(max_simulation, step_max_simulation)
        rms_simulation = max(rms_simulation, step_rms_simulation)

    print('Max_simulation', max_simulation)
    print('Rms_simulation', rms_simulation)
    print()
    return max_simulation, rms_simulation

def load_model(model, weights):

    load_checkpoint(torch.load(weights), model)

    return model


def check_metrics(model, weights):

    load_checkpoint(torch.load(weights), model)

    test_ds = VectorDataset(
        path_inp_data = config.PATH_INP_DATA,
        path_out_data = config.PATH_OUTPUT_DATA,
        type_vector = config.TEST,
    )

    # make dataloader
    test_loader = DataLoader(
        test_ds,
        batch_size=1000,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    model.eval()
    with torch.no_grad():
        for inp, out in test_loader:
            predict = model(inp).to(device=config.DEVICE).cpu().detach().numpy()
            targets = out.to(device=config.DEVICE).cpu().detach().numpy()
            predict_restore = np.dot(predict, compressed_basis) + mean_to_restore
            score_metrics(predict_restore, targets)

if __name__ == "__main__":

    # check_metrics(MlpModel(
    #         size=config.MLP_CHAR['SIZE'],
    #         dropout=config.MLP_CHAR['DROPOUT']
    #         ).to(config.DEVICE).double(), config.MLP_CHAR['BEST_WEIGHTS'])
    
    check_metrics(ResNetModel(
            d_in=config.RESNET_CHAR['D_IN'],
            d_out=config.RESNET_CHAR['D_OUT'],
            n_blocks=config.RESNET_CHAR['N_BLOCKS'],
            d_main=config.RESNET_CHAR['D_MAIN'],
            d_hidden=config.RESNET_CHAR['D_HIDDEN'],
            dropout_first=config.RESNET_CHAR['DROPOUT_FIRST'],
            dropout_second=config.RESNET_CHAR['DROPOUT_SECOND']
            ).to(config.DEVICE).double(), config.RESNET_CHAR['BEST_WEIGHTS'])
    
    # check_metrics(FTTransformer.default(
    #         n_num_features=config.FTTransformer_char['N_NUM_FEATURES'],
    #         cat_cardinalities=config.FTTransformer_char['CAT_CARDINALITIES']
    #         ).to(config.DEVICE).double(), config.FTTransformer_char['BEST_WEIGHTS'])

    # check_metrics(FTTransformer.default(
    #         n_num_features=config.FTTransformer_char['N_NUM_FEATURES'],
    #         cat_cardinalities=config.FTTransformer_char['CAT_CARDINALITIES'],
    #         n_out_features=config.FTTransformer_char['D_OUT'],
    #         ).to(config.DEVICE).double(), 'weights/best_w_transformer.tar')
    
    #vis_logs(config.LOGS)
