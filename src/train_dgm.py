import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange
from sklearn.metrics import (
    mean_absolute_error as MAE,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import torch.optim as optim



def train_one_epoch(model, dataloader, optimizer, device, hparams):
    model.train()

    all_y_true = []
    all_y_pred = []

    running_mse = 0.0
    running_graph_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        # print(f"Running epoch {n_batches}/{len(dataloader)}")
        optimizer.zero_grad()

        # batch.x, batch.y, batch.edge_index, batch.edge_attr come in Lightning
        x = batch.x.to(device)
        y = batch.y[:, : hparams.prediction_window].to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

        if hparams.dataset_name in ["METR-LA", "solar", "electricity"] and len(
            x.shape
        ) == 4:
            x = rearrange(x, "b t n f -> (b n) (t f)")
            y = rearrange(y, "b t n f -> (b n) (t f)")

        y_predicted, logprobs = model(x, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_predicted, y)
        loss_forecasting.backward()

        # GRAPH LOSS (stessa logica del training_step Lightning)
        graph_loss_value = 0.0
        if logprobs is not None:
            corr_pred_ = F.mse_loss(y_predicted, y, reduction="none").detach()
            corr_pred = torch.reshape(corr_pred_, (-1, model.num_nodes, corr_pred_.shape[1]))
            # corr_pred = torch.sum(corr_pred, dim=0)
            # corr_pred = torch.sum(corr_pred, dim=1)  # provato ad usare mean invece di sum che point_w andava ad inf
            corr_pred = torch.mean(corr_pred, dim=0)
            corr_pred = torch.mean(corr_pred, dim=1)
            corr_pred = torch.unsqueeze(corr_pred, dim=0)

            if model.avg_accuracy is None:
                model.avg_accuracy = torch.ones_like(corr_pred) * F.mse_loss(y_predicted, y).detach()

            delta = model.avg_accuracy - corr_pred
            delta = torch.clamp(delta, min=-1, max=1)
            alpha = 0.1
            point_w = delta**2 * torch.exp(-alpha * delta)
            logprobs_ = logprobs.exp().mean([0, -1, -2])
            graph_loss = point_w.squeeze(0) * logprobs_
            graph_loss = graph_loss.mean()
            graph_loss.backward()

            # aggiorna moving average
            model.avg_accuracy = (model.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred)
            graph_loss_value = graph_loss.detach().item()

        optimizer.step()

        # metriche per epoch
        running_mse += loss_forecasting.detach().item()
        running_graph_loss += graph_loss_value
        n_batches += 1

        # accumulo per metriche globali
        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_predicted.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    train_mae = MAE(all_y_pred, all_y_true)
    train_rmse = root_mean_squared_error(all_y_pred, all_y_true)
    train_mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    train_mse = running_mse / max(1, n_batches)
    avg_graph_loss = running_graph_loss / max(1, n_batches)

    stats = {
        "mse": train_mse,
        "rmse": train_rmse,
        "mae": train_mae,
        "mape": train_mape,
        "graph_loss": avg_graph_loss,
    }
    return stats


@torch.no_grad()
def evaluate(model, dataloader, device, hparams, split="val"):
    model.eval()

    all_y_true = []
    all_y_pred = []
    running_mse = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch.x.to(device)
        y = batch.y[:, : hparams.prediction_window].to(device)
        edge_index = batch.edge_index.to(device)
        edge_weight = batch.edge_attr.to(device) if batch.edge_attr is not None else None

        if hparams.dataset_name in ["METR-LA", "solar", "electricity"] and len(
            x.shape
        ) == 4:
            x = rearrange(x, "b t n f -> (b n) (t f)")
            y = rearrange(y, "b t n f -> (b n) (t f)")

        y_predicted, _ = model(x, edge_index, edge_weight)
        loss_forecasting = F.mse_loss(y_predicted, y)

        running_mse += loss_forecasting.detach().item()
        n_batches += 1

        all_y_true.append(y.detach().cpu().numpy())
        all_y_pred.append(y_predicted.detach().cpu().numpy())

    all_y_true = np.concatenate(all_y_true, axis=0)
    all_y_pred = np.concatenate(all_y_pred, axis=0)

    mae = MAE(all_y_pred, all_y_true)
    rmse = root_mean_squared_error(all_y_pred, all_y_true)
    mape = mean_absolute_percentage_error(all_y_pred, all_y_true)
    mse = running_mse / max(1, n_batches)

    stats = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}
    return stats


def train_and_test_dgm(model, train_loader, val_loader, test_loader, hparams):
    """
    Funzione "alta livello" che fa train+val (con scheduler) e poi test.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode="min",
                                                     patience=2,
                                                     factor=0.8)

    n_epochs = hparams.max_epochs

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        train_stats = train_one_epoch(model, train_loader, optimizer, device, hparams)
        print(
            f"Train - MSE: {train_stats['mse']:.4f} | RMSE: {train_stats['rmse']:.4f} "
            f"| MAE: {train_stats['mae']:.4f} | MAPE: {train_stats['mape']:.4f} "
            f"| GraphLoss: {train_stats['graph_loss']:.4f}"
        )

        # fai la validazione ogni check_val_every_n_epoch epoche
        if epoch % hparams.check_val_every_n_epoch == 0 and val_loader is not None:
            val_stats = evaluate(model, val_loader, device, hparams, split="val")
            print(f"Val   - MSE: {val_stats['mse']:.4f} | RMSE: {val_stats['rmse']:.4f} "
                f"| MAE: {val_stats['mae']:.4f} | MAPE: {val_stats['mape']:.4f}")

            # aggiorna best
            if val_stats["mse"] < model.best_mse:
                model.best_mse = val_stats["mse"]
                model.best_rmse = val_stats["rmse"]
                model.best_mae = val_stats["mae"]
                model.best_mape = val_stats["mape"]

            # ReduceLROnPlateau step
            scheduler.step(val_stats["mse"])

    # --- TEST ---
    if test_loader is not None:
        test_stats = evaluate(model, test_loader, device, hparams, split="test")
        print(f"\nTEST - MSE: {test_stats['mse']:.4f} | RMSE: {test_stats['rmse']:.4f} "
            f"| MAE: {test_stats['mae']:.4f} | MAPE: {test_stats['mape']:.4f}")
        return test_stats

    return None


