# train_romnet
import pickle
import torch
from torch import nn
from methods.modeling.models.romnet.encoder import Encoder
from methods.modeling.models.romnet.decoder import Decoder
from methods.modeling.models.romnet.ffnet import FfNet

from methods.modeling.config_romnet import config

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader

def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

device = get_device()

def save_checkpoints(config, checkpoint_enc, checkpoint_ffnet, checkpoint_dec):
    torch.save(checkpoint_enc,  config['weights_enc'])
    torch.save(checkpoint_ffnet,config['weights_ffn'])
    torch.save(checkpoint_dec,  config['weights_dec'])


def loss_fn(config, inp, enc_out, dec_out, ff_out):
    omega_h = config['omega_h']
    omega_n = config['omega_n']

    loss1 = omega_h * nn.functional.mse_loss(enc_out, ff_out, reduction='sum')
    loss2 = omega_n * nn.functional.mse_loss(dec_out, inp, reduction='sum')

    loss = loss1 + loss2
    return loss


def train(dataloader, epoch, config, optimizer, encoder, decoder, ffnet):

    encoder.train()
    ffnet.train()
    decoder.train()

    accumulated_loss = []
    full_data_size = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device).float(), y.to(device).float()

        enc_out = encoder(y)
        ff_out = ffnet(X)
        dec_out = decoder(ff_out)

        loss = loss_fn(config, y, enc_out, dec_out, ff_out)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 20 == 0:
            plt.plot(y[0].cpu().numpy())
            plt.plot(dec_out[0].detach().cpu().numpy())
            plt.savefig(f'plot_train_last_batch.png')
            plt.clf()
            
def train_all(config, dataloader, val_loader):

    encoder = Encoder(config).to(device)
    convs_out_features = encoder.convs_out_features
    conv_dims = encoder.conv_dims
    config['conv_out_features'] = convs_out_features
    config['conv_dims'] = conv_dims

    ffnet = FfNet(config).to(device)

    decoder = Decoder(config).to(device
                                     )
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(ffnet.parameters()) + list(decoder.parameters()), config['lr'])

    best_accuracy = check_accuracy(val_loader, ffnet, decoder)

    no_improvement_count = 0
    for epoch in range(config['n_epochs']):
        train(dataloader, epoch, config, optimizer, encoder, decoder, ffnet)

        val_accuracy = check_accuracy(val_loader, ffnet, decoder)
        if val_accuracy < best_accuracy:
            best_accuracy = val_accuracy 
            print(f"=> Saving best weights. Epoch {epoch}, val accuracy {val_accuracy}")

            checkpoint_enc = {"state_enc": encoder.state_dict(), "config": config}
            checkpoint_ffnet = {"state_ffnet": ffnet.state_dict(), "config": config}
            checkpoint_dec = {"state_dec": decoder.state_dict(), "config": config}
            save_checkpoints(config, checkpoint_enc, checkpoint_ffnet, checkpoint_dec)

            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= config['cut_epoch']:
            print(f'Breaking training at epoch {epoch} due to loss stagnation.')
            break


def check_accuracy(loader, fcn, dec):
    fcn.eval()
    dec.eval()

    with torch.no_grad():
        accuracy = 0
        step=0
        
        for (X, y) in loader:
            X, y = X.to(device).float(), y.to(device).float()

            ff_out = fcn(X)
            dec_out = dec(ff_out)

            acc_out = nn.functional.mse_loss(dec_out, y, reduction='sum')

            accuracy += acc_out
            step += len(X)

            plt.plot(y[0].cpu().numpy())
            plt.plot(dec_out[0].detach().cpu().numpy())
            plt.savefig(f'plot_val.png')
            plt.clf()

        final_accuracy = accuracy/step
        
    fcn.train()
    dec.train()
    return final_accuracy

def main():

    train_loader, val_loader = get_loaders(
        config['PATH_INP_DATA'],
        config['PATH_OUTPUT_DATA'],
        config['TRAIN'],
        config['VAL'],
        config['batch_size'],
        config['num_workers'],
        config['pin_memory']
    )
    
    train_all(config, train_loader, val_loader)


class VectorDataset(Dataset):
    def __init__(
            self, 
            path_inp_data,  
            path_out_data, 
            type_vector, 
            config
            ):
        super(VectorDataset, self).__init__()

        with open(path_inp_data, 'rb') as handle:
            self.inp_data = pickle.load(handle)

        with open(path_out_data, 'rb') as handle:
            self.out_data = pickle.load(handle)

  

        if self.type_vector == 'train':
            calculate_min_max_mean(self.inp_data['train'], config)
            
            config['N_h'] = self.out_data['train'].shape[1]
            config['n_params'] = self.inp_data['train'].shape[1]

            print(f'\tSaved size of input data: {config["n_params"]},  output data: {config["N_h"]}  to config.')

    def __len__(self):
        return self.inp_data[self.type_vector].shape[0]
    
    def __getitem__(self, index):
        input_vector = self.inp_data[self.type_vector][index]
        output_vector = self.out_data[self.type_vector][index]

        return input_vector, output_vector


def get_loaders(path_inp_data, path_out_data, train, val, batch_size, num_workers, pin_memory):
    print('\nLoading data...')
    train_ds = VectorDataset(
        path_inp_data = path_inp_data,
        path_out_data = path_out_data,
        type_vector = train,
        config = config
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
        path_out_data = path_out_data,
        type_vector = val,
        config = config
    )

    print(f'\tTraining dataset: {len(train_ds)}, validation dataset : {len(val_ds)}.')
    
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

def calculate_min_max_mean(dataset, config):
    mins = []
    means = []
    dif = []
    deviations = []
    for i in range(dataset.shape[1]):
        tmp = []
        tmp = [x[i] for x in dataset]
        mins.append(float(np.min(tmp)))
        means.append(float(np.mean(tmp)))
        dif.append(float(np.max(tmp)) - float(np.min(tmp)))
        deviations.append(float(np.array(tmp).std()))
    
    config['mins'] = mins
    config['means'] = means 
    config['dif'] = dif 
    config['deviations'] = deviations

if __name__ == "__main__":
    main()

