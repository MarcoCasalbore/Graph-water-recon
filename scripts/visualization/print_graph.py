"""Print and draw the initial graph and the learned DGM graph."""

from pathlib import Path

import torch
import networkx as nx
import matplotlib.pyplot as plt

from src.config import Parameters
from src.utils.utils import get_datamodule


def build_graph_from_edge_index(edge_index, edge_weight=None, name="GRAPH"):
    """Build a NetworkX graph from edge_index and print basic statistics."""
    edge_index = edge_index.detach().cpu()
    if edge_weight is not None:
        edge_weight = edge_weight.detach().cpu()
    else:
        edge_weight = torch.ones(edge_index.size(1))

    num_nodes = int(edge_index.max().item()) + 1

    print("\n====================")
    print(f"{name} INFO")
    print("====================")
    print(f"Number of nodes:  {num_nodes}")
    print(f"Number of edges:  {edge_index.shape[1]}")

    unique_nodes = torch.unique(edge_index)
    print(f"Unique nodes in edges: {unique_nodes.numel()}/{num_nodes}")

    degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)
    print(f"Avg degree: {degrees.float().mean().item():.2f}")
    print(f"Max degree: {degrees.max().item()}")
    print(f"Min degree: {degrees.min().item()}")

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))

    edges = edge_index.t().tolist()
    weights = edge_weight.tolist()
    for (u, v), w in zip(edges, weights):
        G.add_edge(u, v, weight=w)

    return G, num_nodes


def save_graph_png(G, pos, title, out_path):
    """Disegna e salva un grafo NetworkX in PNG."""
    print(f"\n[INFO] Drawing and saving graph to: {out_path}")
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos,
        node_size=40,
        node_color="skyblue",
        edge_color="gray",
        width=0.5,
        with_labels=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    print("\nLoading dataset and datamodule...")

    params = Parameters()
    train_l, val_l, test_l, run_params = get_datamodule(params)

    print("\nFetching one batch from train loader...\n")
    batch = next(iter(train_l))

    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        x_b = batch[0]
        y_b = batch[1]
        edge_index_b = batch[2]
        edge_weight_b = batch[3] if len(batch) > 3 else None
        print("[INFO] Batch unpacked as tuple:")
        print("  x_b.shape:", getattr(x_b, "shape", None))
        print("  edge_index_b.shape:", getattr(edge_index_b, "shape", None))
        print("  edge_weight_b.shape:", getattr(edge_weight_b, "shape", None))
    else:
        print("[INFO] Batch type:", type(batch))
        if hasattr(batch, "x") and hasattr(batch, "edge_index"):
            x_b = batch.x
            edge_index_b = batch.edge_index
            edge_weight_b = getattr(batch, "edge_weight", None)
            print("[INFO] Using attributes: x, edge_index, edge_weight")
            print("  x_b.shape:", x_b.shape)
            print("  edge_index_b.shape:", edge_index_b.shape)
            if edge_weight_b is not None:
                print("  edge_weight_b.shape:", edge_weight_b.shape)
            else:
                print("  edge_weight_b: None")
        else:
            raise RuntimeError("Unrecognized batch format; cannot extract initial graph.")

    G_init, num_nodes_init = build_graph_from_edge_index(
        edge_index_b, edge_weight_b, name="INITIAL GRAPH"
    )

    pos = nx.spring_layout(G_init, seed=42)
    save_graph_png(G_init, pos, "Initial Graph (before DGM)", "initial_graph.png")

    edges_path = Path("dgm_learned_edges.pt")
    if not edges_path.exists():
        print(
            f"\n[WARN] File {edges_path} non trovato.\n"
            "       Run training first so dgm_learned_edges.pt is generated."
        )
        return

    print(f"\n[INFO] Loading learned edges from: {edges_path}")
    learned_edges = torch.load(edges_path, map_location="cpu")

    if not isinstance(learned_edges, torch.Tensor):
        raise RuntimeError("dgm_learned_edges.pt does not contain an edge tensor.")

    if learned_edges.dim() != 2 or learned_edges.size(0) != 2:
        raise RuntimeError(f"Expected edge index shape [2, E], found {learned_edges.shape}")

    G_learned, num_nodes_learned = build_graph_from_edge_index(
        learned_edges, None, name="LEARNED GRAPH"
    )

    save_graph_png(G_learned, pos, "Learned Graph (after DGM)", "learned_graph.png")

    print("\nDone. Generated files:")
    print("  - initial_graph.png")
    print("  - learned_graph.png")
    print("  - dgm_learned_edges.pt")


if __name__ == "__main__":
    main()
