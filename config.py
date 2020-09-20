import torch
class Config:
    '''
    Model configuration
    '''
    status = 'train'
    DATA_ROOT = '/mnt/d/multi-task/GOKO/'
    MAXLEN = 1013
    short_AMINO='ARNDCQEGHILKMFPOSUTWYV'
    train_val_split = 0.8
#     FUNCTION = 'mf'
    #save_model_path = 'save_model_path/'
    save_model_path = 'saved_model/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ResNet_out_channels = 32
    ResNet_kernel_size = 3
    ResNet_n_layers = 1
    LSTM_n_hidden_state = 32
    use_gru = True
    lstm_dropout=0
    n_lstm_layers=1
    activation='sigmoid'
    
    epochs = 10
    batch_size = 24
    workers = 1
    lr = 1e-4
    weight_decay = 1e-5
    decay_epoch = 5
    improvement_epoch = 5
    print_freq = 100
    checkpoint = None
    best_model = None
    grad_clip = True
