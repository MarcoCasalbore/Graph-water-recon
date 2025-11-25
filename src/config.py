import os
from time import strftime


class Parameters:
    # Trainer parameters
    accelerator = 'gpu'
    log_every_n_steps = 300
    max_epochs = 100
    enable_progress_bar = True
    check_val_every_n_epoch = 3
    node_features = 23
    time_series_step = 4

    # Datasets
    dataset_name = 'METR-LA'  #  # ['PV', 'wind', 'METR-LA', 'Electricity', 'solar']
    dataset_path = '../../data/PV/Generated_time_series_output_31_with_weigth_multivariate_and_time.json'
    # fold = '0'

    # Model parameters
    conv_layers = [[32, 32], [32, 16], [16, 8]]
    dgm_layers = [[32, 16, 4], [36, 16, 4], []]
    fc_layers = [8, 8, 3]
    pre_fc = [-1, 32]
    emb_dim = 64
    gfun = 'gcn'
    ffun = 'gcn'  # 'gcn', 'gat', None, 'mlp', 'knn'
    k = 5
    pooling = 'add'
    distance = 'euclidean'

    dropout = 0.0
    lr = 1e-3
    test_eval = 10
    model = 'dgm_gcn'  # 'dgm_gcn', 'dgm_gat', 'gcn', 'gat', 'mlp'

    early_stop_callback_flag = False

    lags = 24
    prediction_window = 24
    batch_size = 128
    num_nodes = 31
    logging = False
    save_ckpts = False
    save_logs = True
    chkpt_dir = ''
    reproducible = True
    seed = 42

    def __init__(self, params=None):
        # Show parser args
        if params is not None and params.verbose:
            print("Parameters:")
            for name, value in vars(params).items():
                print(f"  {name}: {value}")

        # Copy all default class attributes as istance attribute
        for attr, value in type(self).__dict__.items():
            if not attr.startswith('__') and not callable(value):
                setattr(self, attr, value)

        # If a Namespace is passed, overwrite the values
        if params is not None:
            # If Namespace argparse, convert in dict
            items = (vars(params).items()
                     if not isinstance(params, dict)
                     else params.items())
            for name, val in items:
                # Assign only if attribute exists and val is not None
                if hasattr(self, name) and val is not None:
                    setattr(self, name, val)

        # Actions to execute when instance is created


        # Define paths and directories