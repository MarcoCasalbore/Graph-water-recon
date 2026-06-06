import argparse
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch.nn import functional as F
from einops import rearrange
from sklearn.metrics import (
    mean_absolute_error as MAE,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import torch.optim as optim

from tsl.nn.models import DCRNNModel, GraphWaveNetModel, AGCRNModel

from src.config import Parameters
from src.model.gcn1d import GCN1DConv_big, GCN1DConv
from src.model.gconvrnn import GraphConvRNN_our
from src.model.dgm import MLP


class BaselineModelPV(nn.Module):
    def __init__(self, params: Parameters):
        super(BaselineModelPV, self).__init__()
        hidden_channels = params.emb_dim
        node_features = params.node_features

        self.params = params

        # Scelta del modello
        if params.model == "gcn":
            from torch_geometric.nn import GCNConv

            self.conv1 = GCNConv(node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        elif params.model == "gat":
            from torch_geometric.nn import GATConv

            self.conv1 = GATConv(node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, hidden_channels)

        elif params.model == "mlp":
            self.conv1 = nn.Linear(node_features, params.dgm_layers[0])
            self.conv2 = MLP(params.dgm_layers)
            self.conv3 = MLP(params.dgm_layers)

        elif params.model == "gcn1d":
            self.gcn1d_model = GCN1DConv(node_features,
                                         params.prediction_window,
                                         1,
                                         1,
                                         hid_dim=hidden_channels)

        elif params.model == "gcn1d-big":
            self.gcn1d_big_model = GCN1DConv_big(node_features,
                                                 params.prediction_window,
                                                 1,
                                                 1,
                                                 hid_dim=hidden_channels)

        elif params.model == "gConvLSTM":
            self.g_conv_lstm = GraphConvRNN_our(1,
                                                 params.prediction_window,
                                                 hidden_channels,
                                                 params,
                                                 cell_type="LSTM")

        elif params.model == "gConvGRU":
            self.g_conv_gru = GraphConvRNN_our(1,
                                                params.prediction_window,
                                                hidden_channels,
                                                params,
                                                cell_type="GRU")

        elif params.model == "DCRNN":
            self.dcrnn = DCRNNModel(input_size=1,
                                    output_size=1,
                                    horizon=params.prediction_window,
                                    hidden_size=hidden_channels,
                                    kernel_size=2,
                                    n_layers=2,
                                    dropout=0.0,
                                    activation="relu")

        elif params.model == "GraphWavenet":
            self.graph_wavenet = GraphWaveNetModel(input_size=1,
                                                    output_size=1,
                                                    horizon=params.prediction_window,
                                                    hidden_size=hidden_channels,
                                                    n_nodes=params.num_nodes,
                                                    dropout=0.3)

        elif params.model == "AGCRNModel":
            self.agcrn = AGCRNModel(input_size=1,
                                    output_size=1,
                                    horizon=params.prediction_window,
                                    hidden_size=hidden_channels,
                                    exog_size=0,
                                    n_nodes=params.num_nodes)

        # Linear finale usato da gcn/gat/mlp
        self.lin = Linear(hidden_channels, params.prediction_window)

        # best metrics (come prima, ma gestite nel loop)
        self.best_mse = 1000.0
        self.best_rmse = 1000.0
        self.best_mae = 1000.0
        self.best_mape = 1000.0

    def forward(self, x, edge_index, edge_weight=None):
        """
        x:
          - per 'gcn', 'gat', 'mlp', 'gcn1d', 'gcn1d-big', 'gConvLSTM', 'gConvGRU':
               [B*N, T*F] (se dataset spatio-temporale) o [num_nodes_tot, F]
          - per 'DCRNN', 'GraphWavenet', 'AGCRNModel':
               [B, T, N, F]
        edge_index: [2, E]
        """
        if self.params.model in ["gcn", "gat"]:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)

            x = F.dropout(x, p=self.params.dropout, training=self.training)
            x = self.lin(x)

        elif self.params.model == "mlp":
            # conv1 è Linear, conv2/3 sono MLP
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)

            x = F.dropout(x, p=self.params.dropout, training=self.training)
            x = self.lin(x)

        elif self.params.model == "gcn1d":
            x = self.gcn1d_model(x, edge_index)

        elif self.params.model == "gcn1d-big":
            x = self.gcn1d_big_model(x, edge_index)

        elif self.params.model == "gConvLSTM":
            x = self.g_conv_lstm(x, edge_index)

        elif self.params.model == "gConvGRU":
            x = self.g_conv_gru(x, edge_index)

        elif self.params.model == "DCRNN":
            x = self.dcrnn(x, edge_index)  # [B, T, N, F]
            x = rearrange(x, "b t n f -> (b n) (t f)")

        elif self.params.model == "GraphWavenet":
            x = self.graph_wavenet(x, edge_index)  # [B, T, N, F]
            x = rearrange(x, "b t n f -> (b n) (t f)")

        elif self.params.model == "AGCRNModel":
            x = self.agcrn(x)  # [B, T, N, F]
            x = rearrange(x, "b t n f -> (b n) (t f)")

        return x


def _prepare_inputs(batch,
                    hparams):
    """
    Replica la logica di reshape usata nel Lightning `training_step` + `forward`.

    Restituisce:
      x_model: input da dare al modello
      y:      target per la loss (sempre [B*N, T*F] quando dataset_name è spatio-temporale)
      edge_index, edge_weight
    """
    x = batch.x
    y = batch.y[:, : hparams.prediction_window]
    edge_index = batch.edge_index
    edge_weight = batch.edge_attr

    if hparams.dataset_name in ["METR-LA", "solar", "electricity"] and len(x.shape) == 4:
        if hparams.model in ["DCRNN", "GraphWavenet", "AGCRNModel"]:  # x: [B, T, N, F], y: [B, T, N, F]
            x_model = x  # questi modelli vogliono [B, T, N, F]
        else:
            x_model = rearrange(x, "b t n f -> (b n) (t f)")
        y = rearrange(y, "b t n f -> (b n) (t f)")
    else:
        # altri dataset, assumiamo x già [num_nodes_tot, feat]
        x_model = x

    return x_model, y, edge_index, edge_weight


def train_one_epoch(model,
                    dataloader,
                    optimizer,
                    device,
                    hparams):
    model.train()

    all_y_true = []
    all_y_pred = []

    running_mse = 0.0
    n_batches = 0

    for batch in dataloader:
        print(f"Running epoch {n_batches}/{len(dataloader)}")
        optimizer.zero_grad()
        batch = batch.to(device)
        x_model, y, edge_index, edge_weight = _prepare_inputs(batch, hparams)

        y, edge_index = y.to(device), edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        y_pred = model(x_model, edge_index, edge_weight)

        loss_forecasting = F.mse_loss(y_pred, y)
        loss_forecasting.backward()
        optimizer.step()

        running_mse += loss_forecasting.detach().item()
        n_batches += 1

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_pred.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    train_mae = MAE(all_y_pred, all_y_true)
    train_rmse = root_mean_squared_error(all_y_pred, all_y_true)
    train_mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    train_mse = running_mse / max(1, n_batches)

    stats = {
        "mse": train_mse,
        "rmse": train_rmse,
        "mae": train_mae,
        "mape": train_mape,
    }
    return stats


@torch.no_grad()
def evaluate(model,
             dataloader,
             device,
             hparams,
             split="val"):
    model.eval()

    all_y_true = []
    all_y_pred = []
    running_mse = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        x_model, y, edge_index, edge_weight = _prepare_inputs(batch, hparams)

        y, edge_index = y.to(device), edge_index.to(device)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)

        y_pred = model(x_model, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_pred, y)

        running_mse += loss_forecasting.detach().item()
        n_batches += 1

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_pred.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    mae = MAE(all_y_pred, all_y_true)
    rmse = root_mean_squared_error(all_y_pred, all_y_true)
    mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    mse = running_mse / max(1, n_batches)

    stats = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
    return stats


def train_and_test_baseline(model,
                            train_loader,
                            val_loader,
                            test_loader,
                            hparams):
    """
    Loop alto livello: training + validation (con ReduceLROnPlateau) + test finale.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode="min",
                                                     patience=5,
                                                     factor=0.15)

    n_epochs = hparams.max_epochs

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, device, hparams)
        print(f"Train - MSE: {train_stats['mse']:.4f} | RMSE: {train_stats['rmse']:.4f} "
            f"| MAE: {train_stats['mae']:.4f} | MAPE: {train_stats['mape']:.4f}")

        # Validazione
        if val_loader is not None and epoch % hparams.check_val_every_n_epoch == 0:
            val_stats = evaluate(model, val_loader, device, hparams, split="val")
            print(f"Val   - MSE: {val_stats['mse']:.4f} | RMSE: {val_stats['rmse']:.4f} "
                f"| MAE: {val_stats['mae']:.4f} | MAPE: {val_stats['mape']:.4f}")

            # aggiorno best nel modello
            if val_stats["mse"] < model.best_mse:
                model.best_mse = val_stats["mse"]
                model.best_rmse = val_stats["rmse"]
                model.best_mae = val_stats["mae"]
                model.best_mape = val_stats["mape"]

            # scheduler ReduceLROnPlateau
            scheduler.step(val_stats["mse"])

    # Test finale
    if test_loader is not None:
        test_stats = evaluate(model, test_loader, device, hparams, split="test")
        print(f"\nTEST - MSE: {test_stats['mse']:.4f} | RMSE: {test_stats['rmse']:.4f} "
            f"| MAE: {test_stats['mae']:.4f} | MAPE: {test_stats['mape']:.4f}")
        return test_stats

    return None

