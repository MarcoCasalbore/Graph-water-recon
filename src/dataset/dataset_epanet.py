import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


def _geometry_knn_edge_index(node_xy: np.ndarray, k: int = 3) -> np.ndarray:
    xy = np.asarray(node_xy, dtype=np.float64)
    if xy.ndim == 3:
        xy = xy[0]
    xy = xy[:, :2]
    n = xy.shape[0]
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    k = min(max(1, int(k)), n - 1)
    idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    edges = set()
    for i in range(n):
        for j in idx[i]:
            u, v = int(i), int(j)
            if u == v:
                continue
            edges.add((u, v))
            edges.add((v, u))
    return np.asarray(sorted(edges), dtype=np.int64).T


class EpanetWindowDataset(Dataset):
    """
    Builds windows from EPANET node time series.

    Input:
        x  -> [L, N, 2]
    Target:
        y  -> [H, N, 2]

    Features order:
        feature 0 = pressure
        feature 1 = demand
    """

    def __init__(
        self,
        data_npz,
        meta_json,
        split="train",
        lags=24,
        prediction_window=1,
        stride=1,
        normalize=True,
        stats=None,
    ):
        super().__init__()

        self.data_npz = str(data_npz)
        self.meta_json = str(meta_json)
        self.split = split
        self.lags = int(lags)
        self.prediction_window = int(prediction_window)
        self.stride = int(stride)
        self.normalize = bool(normalize)

        npz = np.load(self.data_npz)
        meta = json.loads(Path(self.meta_json).read_text(encoding="utf-8"))

        pressure = npz["pressure"].astype(np.float32)   # [T, N]
        demand = npz["demand"].astype(np.float32)       # [T, N]
        node_xy = npz["node_xy"].astype(np.float32)     # [N, 2]
        edge_index_gt = npz["edge_index"].astype(np.int64) # [2, E]
        edge_index = (
            npz["initial_edge_index"].astype(np.int64)
            if "initial_edge_index" in npz.files
            else _geometry_knn_edge_index(node_xy, k=3)
        )

        x_all = np.stack([pressure, demand], axis=-1)   # [T, N, 2]

        split_info = meta["splits"][split]
        start = int(split_info["start"])
        end = int(split_info["end"])

        self.x_all = x_all
        self.node_xy = node_xy
        self.edge_index = edge_index
        self.edge_index_gt = edge_index_gt
        self.num_nodes = x_all.shape[1]
        self.n_features = x_all.shape[2]

        # normalization statistics computed on TRAIN only
        if self.normalize:
            if stats is None:
                raise ValueError("stats must be provided when normalize=True")
            self.mean = stats["mean"].astype(np.float32)   # [1, 1, 2]
            self.std = stats["std"].astype(np.float32)     # [1, 1, 2]
        else:
            self.mean = None
            self.std = None

        # valid window start indices entirely inside this split
        self.indices = []
        last_start = end - self.lags - self.prediction_window
        for s in range(start, last_start + 1, self.stride):
            self.indices.append(s)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        l = self.lags
        h = self.prediction_window

        x = self.x_all[s : s + l]              # [L, N, 2]
        y = self.x_all[s + l : s + l + h]      # [H, N, 2]

        if self.normalize:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        data = Data(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(y).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_attr=None,
        )

        data.node_xy = torch.from_numpy(self.node_xy).float()  # [N, 2]
        return data


class EpanetDataModule:
    """
    Lightweight datamodule-like wrapper with train/val/test dataloaders.
    """

    def __init__(self, run_params):
        self.run_params = run_params

        data_npz = run_params.data_npz
        meta_json = run_params.meta_json

        npz = np.load(data_npz)
        pressure = npz["pressure"].astype(np.float32)
        demand = npz["demand"].astype(np.float32)
        x_all = np.stack([pressure, demand], axis=-1)   # [T, N, 2]

        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
        train_split = meta["splits"]["train"]
        tr_start = int(train_split["start"])
        tr_end = int(train_split["end"])

        train_chunk = x_all[tr_start:tr_end]            # [Ttr, N, 2]

        mean = train_chunk.mean(axis=(0, 1), keepdims=True)    # [1, 1, 2]
        std = train_chunk.std(axis=(0, 1), keepdims=True) + 1e-8

        stats = {"mean": mean, "std": std}

        self.train_dataset = EpanetWindowDataset(
            data_npz=data_npz,
            meta_json=meta_json,
            split="train",
            lags=run_params.lags,
            prediction_window=run_params.prediction_window,
            stride=getattr(run_params, "stride", 1),
            normalize=True,
            stats=stats,
        )

        self.val_dataset = EpanetWindowDataset(
            data_npz=data_npz,
            meta_json=meta_json,
            split="val",
            lags=run_params.lags,
            prediction_window=run_params.prediction_window,
            stride=getattr(run_params, "stride", 1),
            normalize=True,
            stats=stats,
        )

        self.test_dataset = EpanetWindowDataset(
            data_npz=data_npz,
            meta_json=meta_json,
            split="test",
            lags=run_params.lags,
            prediction_window=run_params.prediction_window,
            stride=getattr(run_params, "stride", 1),
            normalize=True,
            stats=stats,
        )

        self.num_nodes = self.train_dataset.num_nodes
        self.n_features = self.train_dataset.n_features

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.run_params.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.run_params.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.run_params.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
