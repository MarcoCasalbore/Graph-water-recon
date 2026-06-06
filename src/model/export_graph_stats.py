#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np
import torch

def edge_set_from_edge_index(edge_index, directed=True):
    # edge_index: [2, E]
    src = edge_index[0].astype(int)
    dst = edge_index[1].astype(int)
    if directed:
        return set(zip(src.tolist(), dst.tolist()))
    # undirected canonical
    return set((min(u,v), max(u,v)) for u,v in zip(src.tolist(), dst.tolist()))

def main():
    base = Path(".").resolve()

    learned_path = base / "dgm_learned_edges.pt"
    meta_path = base / "data" / "ctown_ts" / "meta.json"
    npz_path  = base / "data" / "ctown_ts" / "data.npz"

    edges = torch.load(learned_path, map_location="cpu").numpy()  # [2, E_global]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz  = np.load(npz_path)

    node_ids = meta["node_ids"]
    N = len(node_ids)

    node_xy = npz["node_xy"]          # [N,2] se salvato come statico; oppure [B,N,2] se per-batch
    if node_xy.ndim == 3:
        # se per-batch: usa il primo batch
        node_xy0 = node_xy[0]
    else:
        node_xy0 = node_xy

    src_g = edges[0].astype(int)
    dst_g = edges[1].astype(int)

    # Se edges sono "global" con offset batch: riduci modulo N
    src = src_g % N
    dst = dst_g % N

    # statistiche base
    E = len(src)
    deg_out = np.bincount(src, minlength=N)
    deg_in  = np.bincount(dst, minlength=N)

    # lunghezze archi (se coord finite)
    mask = np.isfinite(node_xy0[:,0]) & np.isfinite(node_xy0[:,1])
    ok = mask[src] & mask[dst]
    dxy = node_xy0[src[ok]] - node_xy0[dst[ok]]
    dist = np.sqrt((dxy**2).sum(1))

    density = E / (N*(N-1) + 1e-12)

    # bidirectional ratio
    S_dir = set(zip(src.tolist(), dst.tolist()))
    bi = sum((v,u) in S_dir for (u,v) in S_dir)
    bidir_ratio = bi / (E + 1e-12)

    print("\n=== LEARNED GRAPH STATS ===")
    print(f"N nodes: {N}")
    print(f"E edges: {E}")
    print(f"density (directed): {density:.6f}")
    print(f"mean out-degree: {deg_out.mean():.3f} (min {deg_out.min()}, max {deg_out.max()})")
    print(f"mean in-degree : {deg_in.mean():.3f} (min {deg_in.min()}, max {deg_in.max()})")
    if dist.size > 0:
        print(f"edge length mean: {dist.mean():.3f} | median: {np.median(dist):.3f} | max: {dist.max():.3f}")
    print(f"bidirectional ratio: {bidir_ratio:.3f}")

    # export edge list CSV
    out_csv = base / "learned_edges.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("src_id,dst_id,src_idx,dst_idx,dist\n")
        for u,v in zip(src, dst):
            d = ""
            if np.isfinite(node_xy0[u]).all() and np.isfinite(node_xy0[v]).all():
                d = float(np.linalg.norm(node_xy0[u]-node_xy0[v]))
            f.write(f"{node_ids[u]},{node_ids[v]},{u},{v},{d}\n")
    print(f"\n[OK] wrote: {out_csv}")

    # confronto con ground-truth se disponibile
    if "edge_index" in npz.files:
        gt = npz["edge_index"]
        S_gt = edge_set_from_edge_index(gt, directed=True)
        S_lr = edge_set_from_edge_index(np.vstack([src, dst]), directed=True)

        tp = len(S_lr & S_gt)
        fp = len(S_lr - S_gt)
        fn = len(S_gt - S_lr)

        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1   = 2*prec*rec / (prec + rec + 1e-12)

        print("\n=== EDGE RECOVERY (directed) ===")
        print(f"TP {tp} | FP {fp} | FN {fn}")
        print(f"Precision {prec:.4f} | Recall {rec:.4f} | F1 {f1:.4f}")
    else:
        print("\n[INFO] No edge_index in NPZ: skipping GT comparison.")

if __name__ == "__main__":
    main()