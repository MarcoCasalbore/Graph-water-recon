import pandas as pd
import torch

from tsl.data import SpatioTemporalDataset, TemporalSplitter, SpatioTemporalDataModule
from tsl.data.preprocessing import MinMaxScaler
from tsl.datasets import MetrLA, Elergone

from src.dataset.dataset import MyDataModule
from src.dataset.dataset_Solar import SolarCustom
from src.model.dgm import DGMModel
from src.train import train_and_test_baseline, BaselineModelPV
from src.train_dgm import train_and_test_dgm


def get_model(run_params):
    if 'dgm' in run_params.model:
        model = DGMModel(run_params)
    elif run_params.model in ['gcn', 'gat', 'mlp', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU', 'DCRNN', 'GraphWavenet', 'AGCRNModel']:
        model = BaselineModelPV(run_params)
    else:
        raise Exception('Error in select the model!')
    return model


def set_params_wrt_dataset(run_params, train_loader):
    # Configure input feature size
    if run_params.pre_fc is None or len(run_params.pre_fc) == 0:
        if len(run_params.dgm_layers[0]) > 0:
            run_params.dgm_layers[0][0] = train_loader.dataset.n_features
        run_params.conv_layers[0][0] = train_loader.dataset.n_features
    else:
        if run_params.dataset_name in ['PV', 'wind']:
            run_params.pre_fc[0] = train_loader.dataset.dataset.n_features
        elif run_params.dataset_name in ['METR-LA', 'solar', 'electricity']:
            run_params.pre_fc[0] = run_params.lags
        else:
            raise ValueError('Define dataset name correctly!')

    run_params.fc_layers[-1] = run_params.prediction_window
    run_params.node_features = run_params.lags
    return run_params


def make_dgm_network_parameters(emb_dim):
    cost = emb_dim / 32
    conv_layers = [[emb_dim, emb_dim], [emb_dim, int(emb_dim / 2)], [int(emb_dim / 2), int(emb_dim / 4)]]
    dgm_layers = [[emb_dim, int(emb_dim / 2), int(emb_dim / 8)], [int(36 * cost), int(emb_dim / 2), int(emb_dim / 8)],
                  []]
    fc_layers = [int(8 * cost), int(8 * cost), int(3 * cost)]
    pre_fc = [-1, emb_dim]
    return conv_layers, dgm_layers, fc_layers, pre_fc


def make_dgm_network_parameters_v2(emb_dim):
    pre_fc = [-1, emb_dim]

    dgm_layers = [[emb_dim, int(emb_dim / 2)], [emb_dim, int(emb_dim / 2)], []]
    conv_layers = [[emb_dim, int(emb_dim / 2)], [int(emb_dim / 2), int(emb_dim / 2)],
                   [int(emb_dim / 2), int(emb_dim / 4)]]

    fc_layers = [int(emb_dim / 4), -1]
    return conv_layers, dgm_layers, fc_layers, pre_fc


def initialize_parameters(cont, run_combination):
    model_item = run_combination[0]
    dataset_name_item = run_combination[1]
    emb_dim_item = run_combination[2]
    k_item = run_combination[3]
    batch_size_item = run_combination[4]
    lags_item = run_combination[5]
    prediction_window_item = run_combination[6]
    grid_params_name = ['Run',
                        'model',
                        'dataset_name',
                        'emb_dim',
                        'k',
                        'batch_size',
                        'lags',
                        'prediction_window',
                        'val_mse_mean',
                        'val_mse_std',
                        'val_rmse_mean',
                        'val_rmse_std',
                        'val_mae_mean',
                        'val_mae_std',
                        'val_mape_mean',
                        'val_mape_std',
                        'test_mse_mean',
                        'test_mse_std',
                        'test_rmse_mean',
                        'test_rmse_std',
                        'test_mae_mean',
                        'test_mae_std',
                        'test_mape_mean',
                        'test_mape_std']
    grid_params = [cont,
                   model_item,
                   dataset_name_item,
                   emb_dim_item,
                   k_item,
                   batch_size_item,
                   lags_item,
                   prediction_window_item,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.,
                   0.]
    grid_params_dict = dict(zip(grid_params_name, grid_params))
    output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])
    print(output_string)
    return grid_params_dict


def update_current_configuration(run_combination, seed, run_params, reproducible, save_ckpts, save_logs,
                                 early_stop_callback_flag):
    model_item = run_combination[0]
    dataset_name_item = run_combination[1]
    emb_dim_item = run_combination[2]
    k_item = run_combination[3]
    batch_size_item = run_combination[4]
    lags_item = run_combination[5]
    prediction_window_item = run_combination[6]

    run_params.seed = seed
    run_params.emb_dim = emb_dim_item
    # conv_layers, dgm_layers, fc_layers, pre_fc = make_dgm_network_parameters(emb_dim_item)
    conv_layers, dgm_layers, fc_layers, pre_fc = make_dgm_network_parameters_v2(emb_dim_item)

    run_params.conv_layers = conv_layers
    run_params.dgm_layers = dgm_layers
    run_params.fc_layers = fc_layers
    run_params.pre_fc = pre_fc
    run_params.model = model_item
    if 'dgm' in model_item:
        ffun_item = model_item.split('_')[-2]
        gfun_item = model_item.split('_')[-1]
        run_params.ffun = ffun_item
        run_params.gfun = gfun_item

    run_params.k = k_item
    run_params.batch_size = batch_size_item
    run_params.lags = lags_item
    run_params.prediction_window = prediction_window_item

    run_params.dataset_name = dataset_name_item
    run_params.dataset = dataset_name_item

    run_params.chkpt_dir = "../checkpoints/" + dataset_name_item + '/'
    run_params.reproducible = reproducible
    run_params.save_ckpts = save_ckpts
    run_params.early_stop_callback_flag = early_stop_callback_flag
    run_params.save_logs = save_logs
    return run_params


def get_datamodule_old(run_params):
    # TSL-datasets style
    if run_params.dataset_name in ['METR-LA', 'electricity', 'solar']:
        if run_params.dataset_name == 'METR-LA':
            dataset = MetrLA(root='../data')
        elif run_params.dataset_name == 'electricity':
            dataset = Elergone(root='../data')
        elif run_params.dataset_name == 'solar':
            solar_dataframe = pd.read_csv('../data/df_solar.csv')
            solar_dataframe = solar_dataframe.drop('Unnamed: 0', axis=1)
            dataset = SolarCustom(solar_dataframe=solar_dataframe)
        else:
            raise ValueError(f'Dataset {run_params.dataset_name} not recognized')

        run_params.num_nodes = dataset.shape[1]
        connectivity = dataset.get_connectivity(threshold=0.1,
                                                include_self=True,
                                                normalize_axis=1,
                                                layout="edge_index")
        df_dataset = dataset.dataframe()
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame
        df_dataset = pd.DataFrame(scaler.fit_transform(df_dataset), columns=df_dataset.columns)
        torch_dataset = SpatioTemporalDataset(target=df_dataset,
                                              connectivity=connectivity,  # edge_index
                                              horizon=run_params.prediction_window,
                                              window=run_params.lags,
                                              stride=1)
        # Normalize data using mean and std computed over time and node dimensions

        splitter = TemporalSplitter(val_len=0.2, test_len=0.1)
        data_module_instance = SpatioTemporalDataModule(
            dataset=torch_dataset,
            # scalers=scalers,
            splitter=splitter,
            batch_size=run_params.batch_size,
            workers=2
        )

    elif run_params.dataset_name in ['PV', 'wind']:
        if run_params.dataset_name == 'PV':
            run_params.dataset_path = '../data/Generated_time_series_output_31_with_weigth_multivariate_and_time.json'
        elif run_params.dataset_name == 'wind':
            run_params.dataset_path = '../data/wind_dataset.json'
        data_module_instance = MyDataModule(run_params)
        run_params.num_nodes = data_module_instance.num_station
    else:
        raise ValueError('Define dataset name correct!')

    return data_module_instance, run_params


def get_datamodule(run_params):
    # TSL-datasets style
    if run_params.dataset_name in ['METR-LA', 'electricity', 'solar']:
        if run_params.dataset_name == 'METR-LA':
            dataset = MetrLA(root='../data')
        elif run_params.dataset_name == 'electricity':
            dataset = Elergone(root='../data')
        elif run_params.dataset_name == 'solar':
            solar_dataframe = pd.read_csv('../data/df_solar.csv')
            solar_dataframe = solar_dataframe.drop('Unnamed: 0', axis=1)
            dataset = SolarCustom(solar_dataframe=solar_dataframe)
        else:
            raise ValueError(f'Dataset {run_params.dataset_name} not recognized')

        run_params.num_nodes = dataset.shape[1]
        connectivity = dataset.get_connectivity(threshold=0.1,
                                                include_self=True,
                                                normalize_axis=1,
                                                layout="edge_index")
        df_dataset = dataset.dataframe()
        # Initialize MinMaxScaler
        scaler = MinMaxScaler()

        # Apply the scaler to the DataFrame
        df_dataset = pd.DataFrame(scaler.fit_transform(df_dataset), columns=df_dataset.columns)
        torch_dataset = SpatioTemporalDataset(target=df_dataset,
                                              connectivity=connectivity,  # edge_index
                                              horizon=run_params.prediction_window,
                                              window=run_params.lags,
                                              stride=1)
        # Normalize data using mean and std computed over time and node dimensions

        splitter = TemporalSplitter(val_len=0.2, test_len=0.1)
        data_module_instance = SpatioTemporalDataModule(
            dataset=torch_dataset,
            # scalers=scalers,
            splitter=splitter,
            batch_size=run_params.batch_size,
            workers=2
        )

        data_module_instance.setup()

        train_loader = data_module_instance.train_dataloader()
        val_loader = data_module_instance.val_dataloader()
        test_loader = data_module_instance.test_dataloader()


    elif run_params.dataset_name in ['PV', 'wind']:
        if run_params.dataset_name == 'PV':
            run_params.dataset_path = '../../data/Generated_time_series_output_31_with_weigth_multivariate_and_time.json'
        elif run_params.dataset_name == 'wind':
            run_params.dataset_path = '../../data/wind_dataset.json'
        data_module_instance = MyDataModule(run_params)
        run_params.num_nodes = data_module_instance.num_station
        train_loader = data_module_instance.train_dataloader
        val_loader = data_module_instance.val_dataloader
        test_loader = data_module_instance.test_dataloader
    else:
        raise ValueError('Define dataset name correct!')

    # return data_module_instance, run_params
    return (train_loader,
            val_loader,
            test_loader,
            run_params)

def update_seed_metrics(model, res_test, val_results, test_results):
    best_val_mse, best_val_rmse, best_val_mae, best_val_mape = model.best_mse, model.best_rmse, model.best_mae, model.best_mape

    # Testing
    test_mse = res_test[0]['test_mse']
    test_rmse = res_test[0]['test_rmse']
    test_mae = res_test[0]['test_mae']
    test_mape = res_test[0]['test_mape']

    val_results.append([best_val_mse, best_val_rmse, best_val_mae, best_val_mape])
    test_results.append([test_mse, test_rmse, test_mae, test_mape])

    print(f'best_val_mse: {best_val_mse}')
    print(f'best_val_rmse: {best_val_rmse}')
    print(f'best_val_mae: {best_val_mae}')
    print(f'best_val_mape: {best_val_mape}')
    print(f'test_mse: {test_mse}')
    print(f'test_rmse {test_rmse}')
    print(f'test_mae: {test_mae}')
    print(f'test_mape: {test_mape}')
    return val_results, test_results


def update_run_metrics(val_results, test_results, grid_params_dict, run_params):
    val_results = torch.tensor(val_results)
    test_results = torch.tensor(test_results)
    val_mse_over_seeds = val_results[:, 0]
    val_rmse_over_seeds = val_results[:, 1]
    val_mae_over_seeds = val_results[:, 2]
    val_mape_over_seeds = val_results[:, 3]

    test_mse_over_seeds = test_results[:, 0]
    test_rmse_over_seeds = test_results[:, 1]
    test_mae_over_seeds = test_results[:, 2]
    test_mape_over_seeds = test_results[:, 3]

    grid_params_dict.update({
        'val_mse_mean': float(torch.mean(val_mse_over_seeds)),
        'val_mse_std': float(torch.std(val_mse_over_seeds)),
        'val_rmse_mean': float(torch.mean(val_rmse_over_seeds)),
        'val_rmse_std': float(torch.std(val_rmse_over_seeds)),
        'val_mae_mean': float(torch.mean(val_mae_over_seeds)),
        'val_mae_std': float(torch.std(val_mae_over_seeds)),
        'val_mape_mean': float(torch.mean(val_mape_over_seeds)),
        'val_mape_std': float(torch.std(val_mape_over_seeds)),
        'test_mse_mean': float(torch.mean(test_mse_over_seeds)),
        'test_mse_std': float(torch.std(test_mse_over_seeds)),
        'test_rmse_mean': float(torch.mean(test_rmse_over_seeds)),
        'test_rmse_std': float(torch.std(test_rmse_over_seeds)),
        'test_mae_mean': float(torch.mean(test_mae_over_seeds)),
        'test_mae_std': float(torch.std(test_mae_over_seeds)),
        'test_mape_mean': float(torch.mean(test_mape_over_seeds)),
        'test_mape_std': float(torch.std(test_mape_over_seeds))
    })
    output_string = ' '.join([f'{name}: {value}' for name, value in grid_params_dict.items()])

    if run_params.save_logs:
        with open(f'../logs/logs_{run_params.dataset_name}_mean_std_{run_params.lags}_{run_params.prediction_window}.txt', 'a') as file:
            print(output_string, file=file)


def train_and_test(model, train_l, val_l, test_l, run_params):
    if 'dgm' in run_params.model:
        train_and_test_dgm(model, train_l, val_l, test_l, run_params)
    else:
        train_and_test_baseline(model, train_l, val_l, test_l, run_params)