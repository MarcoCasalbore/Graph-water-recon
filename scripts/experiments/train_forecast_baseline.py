"""Train deep forecasting baselines on synthetic water-like datasets.

These models do not use graph edges. They are meant as a forecasting reference
against the DGM pipeline, whose main scientific target is topology recovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def make_windows(series: np.ndarray, start: int, end: int, lags: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    first = max(start + lags, lags)
    for t in range(first, end):
        x_rows.append(series[t - lags : t])
        y_rows.append(series[t])
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.float32)


class GlobalGRU(nn.Module):
    def __init__(self, num_nodes: int, features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.features = features
        self.gru = nn.GRU(
            input_size=num_nodes * features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_nodes * features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, lags, nodes, features = x.shape
        out, _ = self.gru(x.reshape(batch, lags, nodes * features))
        y = self.head(out[:, -1])
        return y.reshape(batch, nodes, features)


class NodeMLP(nn.Module):
    def __init__(self, lags: int, features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lags * features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, lags, nodes, features = x.shape
        y = self.net(x.permute(0, 2, 1, 3).reshape(batch * nodes, lags * features))
        return y.reshape(batch, nodes, features)


def metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    denom = np.maximum(np.abs(true), 1e-6)
    mape = float(np.mean(np.abs(err) / denom))
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "pressure_mse": float(np.mean(err[:, :, 0] ** 2)),
        "demand_mse": float(np.mean(err[:, :, 1] ** 2)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
            trues.append(y.numpy())
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    out = {"normalized": metrics(pred, true)}
    if mean is not None and std is not None:
        out["raw"] = metrics(pred * std + mean, true * std + mean)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep non-graph forecasting baselines.")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", choices=["global_gru", "node_mlp"], default="global_gru")
    parser.add_argument("--lags", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_dir / "data.npz")
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    series_raw = np.stack([data["pressure"], data["demand"]], axis=-1).astype(np.float32)
    splits = meta["splits"]
    train_chunk = series_raw[splits["train"]["start"] : splits["train"]["end"]]
    mean = train_chunk.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (train_chunk.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    series = (series_raw - mean) / std

    x_train, y_train = make_windows(series, splits["train"]["start"], splits["train"]["end"], args.lags)
    x_val, y_val = make_windows(series, splits["val"]["start"], splits["val"]["end"], args.lags)
    x_test, y_test = make_windows(series, splits["test"]["start"], splits["test"]["end"], args.lags)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=args.batch_size)

    num_nodes = series.shape[1]
    features = series.shape[2]
    if args.model == "global_gru":
        model = GlobalGRU(num_nodes, features, args.hidden, args.layers, args.dropout)
    else:
        model = NodeMLP(args.lags, features, args.hidden)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val = evaluate(model, val_loader, device)
        val_mse = val["normalized"]["mse"]
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_mse": val_mse})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    result = {
        "model": args.model,
        "data_dir": str(args.data_dir),
        "lags": args.lags,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "layers": args.layers,
        "seed": args.seed,
        "validation": evaluate(model, val_loader, device, mean=mean, std=std),
        "test": evaluate(model, test_loader, device, mean=mean, std=std),
        "history": history,
    }
    (args.out_dir / "forecast_baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    torch.save(model.state_dict(), args.out_dir / f"{args.model}.pt")
    print(json.dumps(result["test"], indent=2))
    print(f"[OK] Saved baseline results to: {args.out_dir}")


if __name__ == "__main__":
    main()
