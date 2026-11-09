"""Train the controlled node-only baseline with the joint task-loss settings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.train_forecast_baseline import NodeMLP, evaluate, make_windows


def task_loss(pred, true, loss_type, normalization, pressure_weight, demand_weight):
    parts = []
    for feature, weight in ((0, pressure_weight), (1, demand_weight)):
        p, t = pred[..., feature], true[..., feature]
        if normalization == "target_std":
            scale = t.detach().std().clamp_min(1e-6)
            p, t = p / scale, t / scale
        elif normalization == "target_mad":
            center = t.detach().median()
            scale = (t.detach() - center).abs().median().clamp_min(1e-6)
            p, t = p / scale, t / scale
        if loss_type == "huber":
            value = F.huber_loss(p, t, delta=1.0)
        elif loss_type == "mae":
            value = torch.abs(p - t).mean()
        else:
            value = F.mse_loss(p, t)
        parts.append(weight * value)
    return sum(parts) / max(pressure_weight + demand_weight, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--lags", type=int, required=True)
    parser.add_argument("--prediction_window", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss_type", choices=["mse", "mae", "huber"], default="huber")
    parser.add_argument("--feature_normalization", choices=["none", "target_std", "target_mad"], default="target_std")
    parser.add_argument("--pressure_weight", type=float, default=1.0)
    parser.add_argument("--demand_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = np.load(args.data_dir / "data.npz")
    meta = json.loads((args.data_dir / "meta.json").read_text(encoding="utf-8"))
    raw = np.stack([data["pressure"], data["demand"]], axis=-1).astype(np.float32)
    splits = meta["splits"]
    train_chunk = raw[splits["train"]["start"]:splits["train"]["end"]]
    mean = train_chunk.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = (train_chunk.std(axis=(0, 1), keepdims=True) + 1e-8).astype(np.float32)
    series = (raw - mean) / std

    def windows(name):
        split = splits[name]
        return make_windows(series, split["start"], split["end"], args.lags, args.prediction_window)

    x_train, y_train = windows("train")
    x_val, y_val = windows("val")
    x_test, y_test = windows("test")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
                              batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val), torch.tensor(y_val)), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test), torch.tensor(y_test)), batch_size=args.batch_size)

    model = NodeMLP(args.lags, 2, args.hidden, args.prediction_window)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.8)
    best_state, best_epoch, best_val = None, None, float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = task_loss(model(x), y, args.loss_type, args.feature_normalization,
                             args.pressure_weight, args.demand_weight)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val = evaluate(model, val_loader, device)
        val_mse = val["normalized"]["mse"]
        scheduler.step(val_mse)
        history.append({"epoch": epoch, "train_task_loss": float(np.mean(losses)),
                        "val_mse": val_mse, "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_mse < best_val:
            best_val, best_epoch = val_mse, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.to(device)
    result = {
        "model": "node_mlp_controlled", "data_dir": str(args.data_dir),
        "lags": args.lags, "prediction_window": args.prediction_window,
        "epochs": args.epochs, "best_epoch": best_epoch, "seed": args.seed,
        "hidden": args.hidden, "parameter_count": parameter_count,
        "loss_type": args.loss_type, "feature_normalization": args.feature_normalization,
        "validation": evaluate(model, val_loader, device, mean=mean, std=std),
        "test": evaluate(model, test_loader, device, mean=mean, std=std),
    }
    (args.out_dir / "forecast_baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.out_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    torch.save(best_state, args.out_dir / "node_mlp_controlled.pt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
