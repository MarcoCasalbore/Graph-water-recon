import json
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader import DataLoader as DataLoaderPyg


class MyDataModule(pl.LightningDataModule):
    def __init__(self, run_params):
        super().__init__()
        self.run_params = run_params
        self.train_data = None
        self.test_data = None

        dataset = Dataset_custom(self.run_params)
        self.num_station = dataset.number_of_station
        len_dataset = len(dataset)
        train_ratio = 0.7
        val_test_ratio = 0.5
        train_snapshots = int(train_ratio * len_dataset)
        val_test_snapshots = len_dataset - train_snapshots
        val_snapshots = int(val_test_ratio * val_test_snapshots)
        test_snapshots = len_dataset - train_snapshots - val_snapshots
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(dataset,
                                                             [train_snapshots, val_snapshots, test_snapshots])

        if self.train_data is None:
            raise Exception("Dataset %s not supported" % self.run_params.dataset)
        # if run_params.dataset_name in 'PV':
        self.train_dataloader = DataLoaderPyg(self.train_data, batch_size=run_params.batch_size, num_workers=4, shuffle=True)
        self.val_dataloader = DataLoaderPyg(self.val_data, batch_size=run_params.batch_size, num_workers=4)
        self.test_dataloader = DataLoaderPyg(self.test_data, batch_size=run_params.batch_size)
        # else:
        #     self.train_loader = DataLoader(self.train_data, batch_size=run_params.batch_size, num_workers=0)
        #     self.val_loader = DataLoader(self.val_data, batch_size=run_params.batch_size)
        #     self.test_loader = DataLoader(self.test_data, batch_size=run_params.batch_size)

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader

    def test_dataloader(self):
        return self.test_dataloader


class Dataset_custom(Dataset):

    def __init__(self, run_params):
        self.params = run_params
        self._read_json_data()
        self.lags = None
        self.n_features = 1 * run_params.lags
        self.features = None
        self.features_corrupted = None
        self.targets = None
        self.features_temperatures = None
        self.targets_temperatures = None
        self.features_winds = None
        self.targets_winds = None
        self.number_of_station = None
        self.encoded_data = []
        self.read_dataset(self.params.lags)

    def _read_json_data(self):
        with open(self.params.dataset_path) as f:
            self._dataset = json.load(f)

    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array(self._dataset["weights"]).T

    def _get_targets_and_features(self):
        # Power
        stacked_target = np.stack(self._dataset["block"])
        # stacked_target = stacked_target[self.params.INIT_LIMIT_TS:self.params.END_LIMIT_TS, :]
        # # #MOD ONLY_DAY_SERIES
        # if self.params.ONLY_DAY_VALUES:
        #     only_day_values_index = []
        #     for i in range(len(stacked_target)):
        #         elem = stacked_target[i]
        #         if float(0.0) in elem:
        #             continue
        #         else:
        #             only_day_values_index.append(i)
        # stacked_target = np.array(new_st)
        # # #ENDMOD
        scaler = MinMaxScaler()
        scaler.fit(stacked_target)
        standardized_target = scaler.transform(stacked_target)
        # Temperature
        # stacked_temp = np.stack(self._dataset["block_temp"])
        # # stacked_temp = stacked_temp[self.params.INIT_LIMIT_TS:self.params.END_LIMIT_TS, :]
        # scaler = MinMaxScaler()
        # scaler.fit(stacked_temp)
        # standardized_temp = scaler.transform(stacked_temp)
        # # Wind
        # stacked_wind = np.stack(self._dataset["block_wind"])
        # # stacked_wind = stacked_wind[self.params.INIT_LIMIT_TS:self.params.END_LIMIT_TS, :]
        # scaler = MinMaxScaler()
        # scaler.fit(stacked_wind)
        # standardized_wind = scaler.transform(stacked_wind)
        # # Month
        # stacked_month = np.stack(self._dataset["block_month"])
        # # stacked_month = stacked_month[self.params.INIT_LIMIT_TS:self.params.END_LIMIT_TS, :]
        # scaler = MinMaxScaler()
        # scaler.fit(stacked_month)
        # standardized_month = scaler.transform(stacked_month)
        # # Hour
        # stacked_hour = np.stack(self._dataset["block_hour"])
        # # stacked_hour = stacked_hour[self.params.INIT_LIMIT_TS:self.params.END_LIMIT_TS, :]
        # scaler = MinMaxScaler()
        # scaler.fit(stacked_hour)
        # standardized_hour = scaler.transform(stacked_hour)

        self.number_of_station = stacked_target.shape[1]

        # if self.params.ONLY_DAY_VALUES:
        #     standardized_target = np.take(standardized_target, only_day_values_index, 0)
        #     standardized_temp = np.take(standardized_temp, only_day_values_index, 0)
        #     standardized_wind = np.take(standardized_wind, only_day_values_index, 0)
        #     standardized_month = np.take(standardized_month, only_day_values_index, 0)
        #     standardized_hour = np.take(standardized_hour, only_day_values_index, 0)

        # self.features = [
        #     np.concatenate((standardized_target[i: i + self.lags, :].T,
        #                     standardized_temp[i: i + self.lags, :].T,
        #                     standardized_wind[i: i + self.lags, :].T,
        #                     standardized_month[i: i + self.lags, :].T,
        #                     standardized_hour[i: i + self.lags, :].T), axis=-1)
        #
        #     # list of (4, 3, 24)
        #     for i in range(0, standardized_target.shape[0] - self.lags - self.params.prediction_window, 4)
        # ]
        #
        # self.targets = [
        #     np.concatenate((standardized_target[i:i + self.params.prediction_window, :].T,
        #                     standardized_temp[i:i + self.params.prediction_window, :].T,
        #                     standardized_wind[i:i + self.params.prediction_window, :].T,
        #                     standardized_month[i:i + self.params.prediction_window, :].T,
        #                     standardized_hour[i:i + self.params.prediction_window, :].T), axis=-1)
        #
        #     for i in range(self.lags, standardized_target.shape[0] - self.params.prediction_window, 4)
        # ]

        self.features = [standardized_target[i: i + self.lags, :].T

                         for i in range(0, standardized_target.shape[0] - self.lags - self.params.prediction_window,
                                        self.params.time_series_step)
                         ]

        self.targets = [standardized_target[i:i + self.params.prediction_window, :].T
                        for i in range(self.lags, standardized_target.shape[0] - self.params.prediction_window,
                                       self.params.time_series_step)
                        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.encoded_data[idx]

    def read_dataset(self, lags):
        self.lags = lags
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features()
        for i in range(len(self.features)):
            self.encoded_data.append(Data(x=torch.FloatTensor(self.features[i]),
                                          edge_index=torch.LongTensor(self._edges),
                                          edge_attr=torch.FloatTensor(self._edge_weights),
                                          y=torch.FloatTensor(self.targets[i])))


class DataModule_GCN1D(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.num_station = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        dataset = Dataset_custom(self.params)
        self.num_station = dataset.number_of_station
        len_dataset = len(dataset)
        train_ratio = 0.7
        val_test_ratio = 0.5
        train_snapshots = int(train_ratio * len_dataset)
        val_test_snapshots = len_dataset - train_snapshots
        val_snapshots = int(val_test_ratio * val_test_snapshots)
        test_snapshots = len_dataset - train_snapshots - val_snapshots
        tr_db, val_db, te_db = torch.utils.data.random_split(dataset, [train_snapshots, val_snapshots, test_snapshots])
        self.train_loader = DataLoader(tr_db, batch_size=self.params.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_db, batch_size=self.params.BATCH_SIZE)
        self.test_loader = DataLoader(te_db, batch_size=self.params.BATCH_SIZE)

    # def setup(self, stage=None):

    def get_len_loader(self):
        len_train = 0
        len_val = 0
        len_test = 0
        for _ in self.train_loader: len_train += 1
        for _ in self.val_loader: len_val += 1
        for _ in self.test_loader: len_test += 1
        return len_train, len_val, len_test

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


