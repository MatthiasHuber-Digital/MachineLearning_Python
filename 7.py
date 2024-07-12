# config
from pathlib import Path

USECASES = {'PS_PAG', 'PS_DAI', 'CC_IPB'}

DATA_PATHS = {
    'PS_PAG': '../../data_LFS/PS_PAG/data/data_pca',
    'PS_DAI': '../../data_LFS/PS_DAI/data/data_process', 
    'CC_IPB_FA': '../../data_LFS/CC_IPB/data/data_pca_FA',
    'CC_IPB_RA': '../../data_LFS/CC_IPB/data/data_pca_RA'
}

MODEL_CHARACTERS = {
    'resnet_optimizer': {
        'D_IN': 8,
        'D_OUT': 25,
        'D_MAIN': 128,
        'D_HIDDEN': 256,
        'N_BLOCKS': [2, 6],
        'DROPOUT_FIRST': [0.2, 0.5],
        'DROPOUT_SECOND': [0.1, 0.3],
        'EPOCHS': [5000, 5001],
        'TRIALS': 50
    },

    'resnet': {
    'D_IN': 5,  #5, 8, 13 in different usecases
    'D_OUT': 25,
    'D_MAIN': 128,
    'D_HIDDEN': 256,
    'N_BLOCKS': 6,
    'DROPOUT_FIRST': 0.2,
    'DROPOUT_SECOND': 0.1,
    'BEST_WEIGHTS': "best_w_resnet.tar",
    'LAST_WEIGHTS': "last_w_resnet.tar",
    },

    'fttransformer_default': {
    'N_NUM_FEATURES': 5,
    'CAT_CARDINALITIES': [],
    'D_OUT': 25,
    'BEST_WEIGHTS': "best_w_transformer_default.tar",
    'LAST_WEIGHTS': "last_w_transformer_default.tar",
    },

    'fttransformer_custom': {
        'N_NUM_FEATURES': 13,
        'CAT_CARDINALITIES': [],
        'D_TOKEN': 192,
        'N_BLOCKS': 3,
        'ATTENTION_N_HEADS': 8,
        'ATTENTION_DROPOUT': 0.2,
        'ATTENTION_INITIALIZATION': 'xavier',
        'FFN_D_HIDDEN': 768,
        'FFN_DROPOUT': 0.1,
        'RESIDUAL_DROPOUT': 0,
        'D_OUT': 25,
        'BEST_WEIGHTS': "best_w_transformer_custom.tar",
        'LAST_WEIGHTS': "last_w_transformer_custom.tar",
    },

    'fttransformer_optimizer': {
        'EPOCHS': [5000, 10000],
        'N_NUM_FEATURES': 13,
        'CAT_CARDINALITIES': [],
        'D_TOKEN': 192,
        'N_BLOCKS': [3, 6],
        'ATTENTION_N_HEADS': [8, 8],
        'ATTENTION_DROPOUT': 0.2,
        'ATTENTION_INITIALIZATION': 'xavier',
        'FFN_D_HIDDEN': 768,
        'FFN_DROPOUT': 0.1,
        'RESIDUAL_DROPOUT': 0,
        'D_OUT': 25,
        'TRIALS': 100,
    }
}

DEFAULT_PARAMS = {
    # column labels for train-val-test split
    'TRAIN': 'train',
    'VAL' : 'val',
    'TEST' : 'test',

    # GPU or CPU processing
    'DEVICE' : 'cpu',

    # learning hyperparams
    'LEARNING_RATE' : 0.0002,
    'BATCH_SIZE' : 10,
    'NUM_EPOCHS' : 3000,

    # data loading params
    'NUM_WORKERS' : 0,
    'PIN_MEMORY' : False,

    # checkpoint preferences
    'SAVE_MODEL' : True,
    'LOAD_MODEL' : False,
    'LOAD_LOGS' : False
}

def gen_config(
    usecase_name, 
    model_name,
    model_characters = None,
    experiment_name = None,
    dataset_path = None,
    result_path = None,
    **kwargs):

    # check if user has specified unknown config parameters
    args = set(kwargs.keys())
    default_args = set(DEFAULT_PARAMS.keys())
    unknown = args - default_args
    if len(unknown) > 0:
        raise Exception('unknown config arguments: {}, \n possible ones: {}'.format(
            str(list(unknown)), str(list(default_args))))

    config = DEFAULT_PARAMS | kwargs

    if not (usecase_name in USECASES):
        raise Exception(
            "The usecase name is {}, but only these are known {}".format(
                usecase_name, str(list(PATHS.keys()))
            )
            )
    
    # Decide on input data path
    config['USECASE'] = usecase_name
    if dataset_path is None:
        k = config['USECASE']
        if not (experiment_name is None):
            k = k + '_' + experiment_name

        config['DATA_PATH'] = DATA_PATHS[k]
    else:
        config['DATA_PATH'] = dataset_path
    config['DATA_PATH'] = Path(config['DATA_PATH'])

    config['PATH_INP_DATA']    = config['DATA_PATH'] / 'tts_input_data.pickle'
    config['PATH_OUTPUT_DATA'] = config['DATA_PATH'] / 'tts_output_data.pickle'
    config['PATH_COEF_DATA']   = config['DATA_PATH'] / 'PCA_coef.pickle'
    config['PATH_TO_BASIS']    = config['DATA_PATH'] / 'PCA_basis.pickle'
    config['PATH_TO_MEAN']     = config['DATA_PATH'] / 'PCA_mean.pickle'

    # Decide on output data path
    if result_path is None:
        result_path = Path('../../usecases/') / config['USECASE']

    result_path = Path(result_path)
    if not result_path.exists():
        print('gen_config: the result path {} does not exist'.format(str(result_path)))
    config['RESULT_PATH'] = result_path 

    config['PATH_TO_WEIGHTS'] = result_path / 'weights'
    if not (experiment_name is None):
        config['PATH_TO_WEIGHTS'] = config['PATH_TO_WEIGHTS'] / experiment_name

    # Generate a path to save the logs
    config['LOGS'] = result_path / 'logs'
    if not (experiment_name is None):
        config['LOGS'] = config['LOGS'] / experiment_name
    config['LOGS'] = config['LOGS'] / 'logs.pickle'

    # Model related settings
    if not (model_name in MODEL_CHARACTERS):
        raise Exception(
            "unknown model type {}, supported models: {}".format(
                model_name, 
                str(list(MODEL_CHARACTERS.keys()))
            ))
    config['MODEL_NAME'] = model_name
    config['MODEL_CHARACTERS'] = MODEL_CHARACTERS[config['MODEL_NAME']].copy()
    if model_characters is not None:
        diff = set(model_characters.keys()) - set(config['MODEL_CHARACTERS'].keys())
        if len(diff) > 0:
            raise Exception('unknown model characters: {}, possible ones: {}'.format(
                str(diff), str(set(config['MODEL_CHARACTERS'].keys()))
            ))
        
        config['MODEL_CHARACTERS'] = config['MODEL_CHARACTERS'] | model_characters

    if 'BEST_WEIGHTS' in config['MODEL_CHARACTERS']:
        config['MODEL_CHARACTERS']['BEST_WEIGHTS'] = config['PATH_TO_WEIGHTS'] / config['MODEL_CHARACTERS']['BEST_WEIGHTS']

    if 'LAST_WEIGHTS' in config['MODEL_CHARACTERS']:
        config['MODEL_CHARACTERS']['LAST_WEIGHTS'] = config['PATH_TO_WEIGHTS'] / config['MODEL_CHARACTERS']['LAST_WEIGHTS']

    # for compatibility with utils::choose_model_and_weights
    config['MODEL_CHARACTERS'] = {model_name : config['MODEL_CHARACTERS']}

    return config

# Sample configs
def cc_ipb_fa_usecase_config():
    return gen_config(usecase_name='CC_IPB', model_name='fttransformer_default', experiment_name='FA')

def cc_ipb_ra_usecase_config():
    return gen_config(usecase_name='CC_IPB', model_name='fttransformer_default', experiment_name='RA')

def ps_dai_usecase_config():
    return gen_config(usecase_name='PS_DAI', model_name='fttransformer_default')

def cc_ipb_usecase_debug():
    return gen_config(
        usecase_name='CC_IPB', 
        model_name='fttransformer_default', 
        dataset_path='C:/!Users/projects/mso-045/sw/abel_cc_ipb/MSO-045/data_research/CC_IPB/new_data/FA/04_new_train_old_test',
        result_path='C:/!Users/projects/mso-045/sw/abel_cc_ipb/MSO-045/data_research/CC_IPB/new_data/FA/04_new_train_old_test/weights')

# The config that is used by default
def gen_default_config():
    return cc_ipb_usecase_debug()

def create_folders(config):
    model_name = config['MODEL_NAME']
    Path(config['MODEL_CHARACTERS'][model_name]['BEST_WEIGHTS']).parent.mkdir(exist_ok=True, parents=True)
    Path(config['MODEL_CHARACTERS'][model_name]['LAST_WEIGHTS']).parent.mkdir(exist_ok=True, parents=True)
    Path(config['LOGS']).parent.mkdir(exist_ok=True, parents=True)
    Path(config['PATH_TO_WEIGHTS']).mkdir(exist_ok=True, parents=True)

