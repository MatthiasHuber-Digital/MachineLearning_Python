# utils
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sklearn
from sklearn import metrics

from methods.modeling.dataset import VectorDataset
from torch.utils.data import DataLoader

from methods.error_metrics.generic_error_metrics import (
    compute_relative_variational_RMSE_of_dataset, 
    compute_RMSEs_of_all_samples,
    compute_relative_variational_RMSEs_of_all_samples
    )

from methods.modeling.models.autoencoder import Autoencoder
from methods.modeling.models.autoencoder_conv import Autoencoder_conv
from methods.modeling.models.transformer import FTTransformer
from methods.modeling.models.resnet import ResNetModel

def save_checkpoint(state, step_accuracy, best_accuracy, best, last):
    #Put > in you use R_2 score. And Put < if you use something like MSE
    if step_accuracy < best_accuracy:
        best_accuracy = step_accuracy
        print("=> Saving best weights, validation MSE = {}".format(step_accuracy))
        torch.save(state, best)
        
        return best_accuracy

    #print("=> Saving last weights, validation MSE = {}".format(step_accuracy))
    #torch.save(state, last)
    return best_accuracy

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def choose_model_and_weights(model_name, model_char, device):

    if model_name == 'resnet':
        return ResNetModel(
            d_in=model_char[model_name]['D_IN'],
            d_out=model_char[model_name]['D_OUT'],
            n_blocks=model_char[model_name]['N_BLOCKS'],
            d_main=model_char[model_name]['D_MAIN'],
            d_hidden=model_char[model_name]['D_HIDDEN'],
            dropout_first=model_char[model_name]['DROPOUT_FIRST'],
            dropout_second=model_char[model_name]['DROPOUT_SECOND']
            ).to(device).double(), model_char[model_name]['BEST_WEIGHTS'], model_char[model_name]['LAST_WEIGHTS']
        
    if model_name == 'fttransformer_default':
        return FTTransformer.default(
            n_num_features=model_char[model_name]['N_NUM_FEATURES'],
            cat_cardinalities=model_char[model_name]['CAT_CARDINALITIES'],
            d_out=model_char[model_name]['D_OUT']
            ).to(device).double(), model_char[model_name]['BEST_WEIGHTS'], model_char[model_name]['LAST_WEIGHTS']

    if model_name == 'fttransformer_custom':
        return FTTransformer.custom(
            n_num_features=model_char[model_name]['N_NUM_FEATURES'],
            cat_cardinalities=model_char[model_name]['CAT_CARDINALITIES'],
            d_token=model_char[model_name]['D_TOKEN'],
            n_blocks=model_char[model_name]['N_BLOCKS'],
            attention_n_heads=model_char[model_name]['ATTENTION_N_HEADS'],
            attention_dropout=model_char[model_name]['ATTENTION_DROPOUT'],
            attention_initialization = model_char[model_name]['ATTENTION_INITIALIZATION'],
            ffn_d_hidden = model_char[model_name]['FFN_D_HIDDEN'],
            ffn_dropout = model_char[model_name]['FFN_DROPOUT'],
            residual_dropout = model_char[model_name]['RESIDUAL_DROPOUT'],
            d_out=model_char[model_name]['D_OUT']
        ).to(device).double(), model_char[model_name]['BEST_WEIGHTS'], model_char[model_name]['LAST_WEIGHTS']
        
    if model_name == 'autoencoder':
        return Autoencoder(
            size=model_char[model_name]['SIZE'], 
            dropout=model_char[model_name]['DROPOUT'],
            ).to(device).double(), model_char[model_name]['BEST_WEIGHTS'], model_char[model_name]['LAST_WEIGHTS']
    
    elif model_name == 'autoencoder_conv':
        return Autoencoder_conv(
            size=model_char[model_name]['SIZE'], 
            kernel_size = model_char[model_name]['KERNEL_SIZE'],
            dropout=model_char[model_name]['DROPOUT'],
            ).to(device).double(), model_char[model_name]['BEST_WEIGHTS'], model_char[model_name]['LAST_WEIGHTS']
    
def get_loaders(
    path_inp_data,
    path_coef_data,
    path_out_data,
    train,
    val,
    batch_size,
    num_workers,
    pin_memory,
):

    train_ds = VectorDataset(
        path_inp_data = path_inp_data,
        path_coef_data = path_coef_data,
        path_out_data = path_out_data,
        type_vector = train,
    )

    # make dataloader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    val_ds = VectorDataset(
        path_inp_data = path_inp_data,
        path_coef_data = path_coef_data,
        path_out_data = path_out_data,
        type_vector = val,
    )

    # make dataloader
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    return train_loader, val_loader

def load_companents_to_restore(path_to_basis, path_to_mean=None):

    with open(path_to_basis, 'rb') as handle:
        basis = pickle.load(handle)

    if path_to_mean:
        with open(path_to_mean, 'rb') as handle:
            mean = pickle.load(handle)
        
        return torch.from_numpy(basis), torch.from_numpy(mean)

    return torch.from_numpy(basis), None

def check_accuracy(loader, model: torch.nn.Module, basis, device='cpu', mean=None):
    
    model = model.to(device=device)
    basis = basis.to(device=device)
    if mean is not None:
        mean = mean.to(device=device)

    model.eval()
    with torch.no_grad():
        accuracy = 0
        step=0
        basis = basis.to(device)
        mean = mean.to(device)
        for (inp, _, out) in loader:
            inp = inp.to(device=device)
            out = out.to(device=device)
            
            outputs_predict = model(inp)
            
            if (out.shape == inp.shape):
                #if we use conv1d
                outputs_predict = outputs_predict.squeeze(1)
            else:
                if mean is not None:
                    outputs_predict = torch.matmul(outputs_predict, basis) + mean
                else:
                    outputs_predict = torch.matmul(outputs_predict, basis)

            outputs_target = out.to(device='cpu')
            outputs_predict = outputs_predict.to(device='cpu')
            acc_out = metrics.mean_squared_error(outputs_predict, outputs_target)

            accuracy += acc_out
            step += 1
        final_accuracy = accuracy/step
        
    model.train()
    return final_accuracy
    
def test_model(
    model, 
    time,
    input_data, 
    output_data, 
    nominal_curve, 
    full_output_data,
    name=None,
    basis=None, 
    mean=None,
    model_type='AI',
    subspace='pca',
    device='cpu',
    show_graphs=False,
):
    
    inp_data = input_data.copy()
    
    if model_type == 'ML':
        predict = model.predict(inp_data)
    elif model_type == 'AI':
        model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(inp_data)
            predict = model(inp).to(device=device).cpu().detach().numpy()
    
    if subspace=='pca':
        X_restore = np.dot(predict, basis) + mean
    elif subspace=='svm':
        X_restore = np.dot(predict, basis)

    #metrics = score_metrics(X_restore, output_data, nominal_curve, full_output_data, time, name, show_graphs)
    metrics = compute_relative_variational_RMSE_of_dataset(X_restore, output_data)
    print('Rel Var RMSE of Dataset:', metrics)
    
    return X_restore, metrics

def compute_FDS(surrogate, simulate):
    curve_len = len(surrogate[0])
    
    FDS_List=[]
    for i in range(len(surrogate)):
        n = curve_len - 200
        values = np.empty(n)
        for j,f in enumerate(np.arange(200,curve_len)):
            G_SMO=calculate_G_SMO(f, curve_len)
            values[j]=(np.sum((surrogate[i]*G_SMO)**2/f**2)**5/
                      (np.sum((simulate[i]*G_SMO)**2/f**2)**5)) 
        FDS_List.append(values)
        
    return FDS_List

def calculate_G_SMO(w_0, curve_len):  
    f = np.arange(100, curve_len+100, 1, dtype=float)  
  
    a = 0.205
    beta = a* 200 /w_0 
    G_SMO = np.zeros_like(f)  
    for i,l in enumerate(f):  
        G_SMO[i] = w_0**2/(f[i]* np.sqrt((1/f[i]**2)*(f[i]**2-w_0**2)**2+(2*beta*w_0)**2))  
      
    return G_SMO

def plot_FDS(fds_list, indices, log=True, legend=True):
    y = np.arange(200, len(fds_list[0])+200) 
    max_value_index = np.argmax(np.max(fds_list, axis=1))
    
    if log:
        plt.yscale("log")
        max_value_index = np.argmax(np.max(np.abs(np.log(fds_list)-1), axis=1))
        
    plt.title(f'FDS plot.\nMax deviation: test curve #{indices[max_value_index]}.')
    for i, f in enumerate(indices):
        plt.plot(y, fds_list[i], label=f"run number {f}") 

    if legend:
        plt.legend()

def calculate_metrics_autoencoder(
    cb_model,
    autoencoder_model,
    input_data,
    simulate_data,
    show_FDS_plot=True,
    test_indices=None,
):
    
    inp_data = input_data.copy()
    
    predict = cb_model.predict(inp_data)

    surrogate_data = autoencoder_model.decoder(torch.from_numpy(predict)).detach().numpy()
        
    avg_rms_metrics = compute_relative_variational_RMSE_of_dataset(surrogate_data, simulate_data)
    print('Rel Var RMSE of Dataset:', avg_rms_metrics)
    
    if show_FDS_plot:
        fds_list = compute_FDS(surrogate_data, simulate_data)
        plot_FDS(fds_list, test_indices)
        
    return avg_rms_metrics
    
def calculate_metrics(
    model,
    input_data,
    simulate_data,
    show_FDS_plot=True,
    test_indices=None,
):
    
    inp_data = input_data.copy()
    
    predict = model.predict(inp_data)

    surrogate_data = autoencoder_model.decoder(torch.from_numpy(predict)).detach().numpy()
        
    avg_rms_metrics = compute_relative_variational_RMSE_of_dataset(surrogate_data, simulate_data)
    print('Rel Var RMSE of Dataset:', avg_rms_metrics)
    
    if show_FDS_plot:
        fds_list = compute_FDS(surrogate_data, simulate_data)
        plot_FDS(fds_list, test_indices)
        
    return avg_rms_metrics

def best_case(list_w_rms_metrics, surrogate_data, simulate_data, test_indices, time):
    
    min_value = min(list_w_rms_metrics)
    min_index = list_w_rms_metrics.index(min_value)
    best_curve = test_indices[min_index]

    print('Best relative variational RMSE value:', min_value)
    print('Best curve Index:', best_curve)
    
    title = str(f'Best case')
    
    plot_curve(simulate_data[min_index], surrogate_data[min_index], best_curve, time, title)
    
def worst_case(list_w_rms_metrics, surrogate_data, simulate_data, test_indices, time):

    max_value = max(list_w_rms_metrics)
    max_index = list_w_rms_metrics.index(max_value)
    worst_curve = test_indices[max_index]

    print('Worst relative variational RMSE value:', max_value)
    print('Worst curve Index:', worst_curve)
    
    title = str(f'Worst case')

    plot_curve(simulate_data[max_index], surrogate_data[max_index], worst_curve, time, title)
    
def median_case(list_w_rms_metrics, surrogate_data, simulate_data, test_indices, time):

    value, median_index = find_median_index(list_w_rms_metrics)
    median_curve = test_indices[median_index]

    print('Median relative variational RMSE value:', value)
    print('Median curve Index:', median_curve)
    
    title = str(f'Median case')

    plot_curve(simulate_data[median_index], surrogate_data[median_index], median_curve, time, title)

def find_median_index(lst):

    sorted_lst = sorted(lst)
    n = len(sorted_lst)

    median = sorted_lst[n // 2]
    median_index = lst.index(median)

    return median, median_index
    
def plot_curve(simulate_data, surrogate_data, index, time, title):

    if title:
        plt.title(title)
        
    plt.ylabel('FRF')
    plt.xlabel('Frequency (Hz)')
    
    plt.plot(time, surrogate_data, 'r--', label="Surrogate")
    plt.plot(time, simulate_data,  'k', label=f"Validation curve {index}")
    
    error = np.abs(surrogate_data-simulate_data)
    plt.plot(time, error, 'b', label="Abs. errror ")
    
    xlim = [time[0], time[-1]]
    ylim=[0, np.max([np.max(surrogate_data), np.max(simulate_data)])]
    
    plt.ylim(ylim[0],ylim[1])
    plt.xlim(xlim[0],xlim[1])
    
    plt.xticks([xlim[0], 5*round(1*(xlim[1]-xlim[0])/15), 5*round(2*(xlim[1]-xlim[0])/15), xlim[1]])
    plt.yticks([ylim[0], 5*round(1*(ylim[1]-ylim[0])/15), 5*round(2*(ylim[1]-ylim[0])/15), 5*round(1.15*ylim[1]/5)])
    
    plt.grid()
    plt.legend()
    plt.show()

def best_worst_cases(
    predict,
    simulate_data,
    test_indices,
    time
):
    lst_rms_metrics = compute_relative_variational_RMSEs_of_all_samples(predict, simulate_data)["list_samples"].tolist()

    best_case(lst_rms_metrics, predict, simulate_data, test_indices, time)
    median_case(lst_rms_metrics, predict, simulate_data, test_indices, time)
    worst_case(lst_rms_metrics, predict, simulate_data, test_indices, time)

def restore_PCA(coef_PCA, basis_PCA, mean_PCA, data, silent=True):
    X_restore = np.dot(coef_PCA, basis_PCA) + mean_PCA
    if len(data.shape)==1:
        accuracy = sklearn.metrics.r2_score(data, X_restore)
        if silent==False:
            print('Restored curve with R2 score accuracy:', accuracy)
    else:
        accuracy = [sklearn.metrics.r2_score(data[idx], X_restore[idx]) for idx in range(len(data))]
        if silent==False:
            print('Restored curves with mean R2 score accuracy:', np.mean(accuracy))
    return X_restore

