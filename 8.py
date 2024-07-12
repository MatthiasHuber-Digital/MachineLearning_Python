# config_romnet
config = {}

config['n_epochs'] = 10000 # maximum number of epochs

config['PATH_INP_DATA'] = '../../data_LFS/CC_IPB/data/input_data.pickle'
config['PATH_OUTPUT_DATA'] = '../../data_LFS/CC_IPB/data/output_data_FA.pickle'

config['TRAIN'] = 'train'
config['VAL'] = 'val'

config['weights_enc'] = '../../usecases/CC_IPB/weights_romnet/upsampling_nearest_0505_w_enc.tar' 
config['weights_ffn'] = '../../usecases/CC_IPB/weights_romnet/upsampling_nearest_0505_w_ffn.tar' 
config['weights_dec'] = '../../usecases/CC_IPB/weights_romnet/upsampling_nearest_0505_w_dec.tar'

config['scaling'] = 'minmax' # batchnorm, minmax, standard  -> normalization of parameters at input of FCN

config['decoder'] = 'upsampling'  # conv, upsampling
config['upsampling_mode'] = 'nearest' # For 2D: 'nearest', 'linear', for 3D: 'bilinear', 'bicubic', for 5D: 'trilinear'

config['n'] = 20 # Output size for entire encoder (after linear layers)
config['n_layers'] = 6 # Number of layers in FCN
config['n_neurons'] = 500 # Size for linear layers in FCN
config['size'] = 7 # Convolutional kernel size 
config['stride'] = 4 # Stride of CNN
config['lin_size'] = 256 # Size of intermediate liner layer in encoder, decoder

config['conv_spec'] = [
    {
        'ch_in': 1,
        'ch_out': 8,
        'stride': 1,
    },
    {
        'ch_in': 8,
        'ch_out': 16,
        'stride': config['stride'] 
    },
    {

        'ch_in': 16,
        'ch_out': 32,
        'stride': config['stride']
    },
    {
        'ch_in': 32,
        'ch_out': 64,
        'stride': config['stride']
    },
    {
        'ch_in': 64,
        'ch_out':128,
        'stride': config['stride'],
    },
    ]

config['cut_epoch'] = 500 # stop training after no improvement has been reached for this number of epochs

config['lr'] = 0.0001 # learning rate
config['omega_h'] = 0.5 # Relative weight of Encoder-FCN loss
config['omega_n'] = 0.5 # Relative weight of Decoder-val loss

config['batch_size'] = 10

config['num_workers'] = 0
config['pin_memory'] = False

