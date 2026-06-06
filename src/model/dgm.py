import torch
import torch.nn as nn
from torch.nn import ModuleList
from torch.nn import functional as F
from torch_geometric.nn import EdgeConv, GCNConv, GATConv

from src.config import Parameters
from src.model.gcn1d import GCN1DConv_big, GCN1DConv
from src.model.gconvrnn import GraphConvRNN_ourDGM


# Euclidean distance
def pairwise_euclidean_distances(x, dim=-1):
    dist = torch.cdist(x, x) ** 2
    return dist, x


# Poincarè disk distance r=1 (Hyperbolic)
def pairwise_poincare_distances(x, dim=-1):
    x_norm = (x ** 2).sum(dim, keepdim=True)
    x_norm = (x_norm.sqrt() - 1).relu() + 1
    x = x / (x_norm * (1 + 1e-2))
    x_norm = (x ** 2).sum(dim, keepdim=True)

    pq = torch.cdist(x, x) ** 2
    dist = torch.arccosh(1e-6 + 1 + 2 * pq / ((1 - x_norm) * (1 - x_norm.transpose(-1, -2)))) ** 2
    return dist, x


def sparse_eye(size):
    """
    Returns the identity matrix as a sparse matrix
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).float().expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


class DGM_d(nn.Module):
    def __init__(self, embed_f, k=5, distance="euclidean", num_nodes=1, sparse=True, hparams=None):
        super(DGM_d, self).__init__()

        self.candidate_mask = None
        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1.0 if distance == "hyperbolic" else 4.0).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k
        self.distance = distance
        self.num_of_nodes = 0
        self.num_nodes = num_nodes
        self.hparams = hparams
        self.stability_loss = None
        self.last_logits = None

        self.debug = False

    def forward(self, x, A, not_used=None, fixedges=None):
        # Get dimension and preprocess X
        if x.shape[0] == 1:
            x = x[0]

        if self.hparams is not None and self.hparams.ffun == "gconvLSTM":
            batch = int(x.shape[0] / self.hparams.num_nodes)
            x = x.unsqueeze(-1)
            x = torch.reshape(x, (batch, self.hparams.num_nodes, x.shape[1]))

        # NN embedding
        x = self.embed_f(x, A)

        if self.hparams is not None and self.hparams.ffun == "gconvLSTM":
            x = x.squeeze()

        if x.dim() == 2:
            x = x[None, ...]

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(
                    fixedges.shape[0],
                    fixedges.shape[-1] // self.k,
                    self.k,
                    dtype=torch.float,
                    device=x.device,
                ), None

            edges_hat, logprobs = self.sample_without_replacement(x)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(
                        fixedges.shape[0],
                        fixedges.shape[-1] // self.k,
                        self.k,
                        dtype=torch.float,
                        device=x.device,
                    ), None

                edges_hat, logprobs = self.sample_without_replacement(x)

        if self.debug:
            if self.distance == "euclidean":
                D, _x = pairwise_euclidean_distances(x)
            if self.distance == "hyperbolic":
                D, _x = pairwise_poincare_distances(x)

            self.D = (D * torch.exp(torch.clamp(self.temperature, -5, 5))).detach().cpu()
            self.edges_hat = edges_hat.detach().cpu()
            self.logprobs = logprobs.detach().cpu()
            self.x = x

        return x, edges_hat, logprobs, None

    def sample_without_replacement(self, x):
        """
        Input:
        x: [B, N, F]  (or [N, F])

        Output:
        edges:    [2, B*N*K]  batch-aware node indices (global indexing per batch)
        logprobs: [B, N, K]   negative squared distances scaled by temperature

        Notes:
        - Device agnostic
        - Supports optional hard candidate mask self.candidate_mask of shape [N, N]
        - Neighbor selection is performed row-wise on the masked cost matrix
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # [1, N, F]

        if self.num_nodes and x.shape[1] != self.num_nodes:
            if x.shape[1] % self.num_nodes != 0:
                raise ValueError(
                    f"DGM input has {x.shape[1]} nodes, which is not compatible "
                    f"with num_nodes={self.num_nodes}"
                )
            x = x.reshape(-1, self.num_nodes, x.shape[-1])

        b, n, d = x.shape
        dev = x.device

        if self.distance != "euclidean":
            raise NotImplementedError(
                "Only distance='euclidean' is currently supported in sample_without_replacement. "
                "If you need hyperbolic, we can reinstate the previous branch in a clean form."
            )

        # Pairwise squared distances per batch: [B, N, N]
        G_i = x.unsqueeze(2)  # [B, N, 1, F]
        X_j = x.unsqueeze(1)  # [B, 1, N, F]
        mD = ((G_i - X_j) ** 2).sum(-1)  # [B, N, N]
        if b > 1:
            self.stability_loss = mD.var(dim=0, unbiased=False).mean()
        else:
            self.stability_loss = torch.tensor(0.0, device=dev, dtype=x.dtype)

        # Scale and aggregate across batch
        temp_scale = torch.exp(torch.clamp(self.temperature, -5, 5))
        lq = mD * temp_scale  # [B, N, N]
        lq = torch.sum(lq, dim=0)  # [N, N]

        # Apply hard candidate mask if available
        if self.candidate_mask is not None:
            cand = self.candidate_mask.to(dev)

            if cand.dim() != 2 or cand.shape[0] != cand.shape[1]:
                raise ValueError(
                    f"candidate_mask must be square, got shape {tuple(cand.shape)}"
                )

            # Case 1: mask already matches current graph size
            if cand.shape == (n, n):
                pass

            # Case 2: mask is for a base graph and current n is a batch multiple
            else:
                base_n = cand.shape[0]

                if n % base_n != 0:
                    raise ValueError(
                        f"candidate_mask base size {base_n} is not compatible with current graph size {n}"
                    )

                batch_factor = n // base_n
                cand = torch.block_diag(*([cand] * batch_factor))

            # Disallow non-candidate edges
            lq = lq.masked_fill(~cand, float("inf"))

        # Disallow self-loops explicitly
        eye = torch.eye(n, device=dev, dtype=torch.bool)
        lq = lq.masked_fill(eye, float("inf"))
        self.last_logits = -lq

        # Row-wise nearest-neighbor selection from masked matrix
        valid_per_row = torch.isfinite(lq).sum(dim=1)
        if (valid_per_row < self.k).any():
            bad_rows = torch.where(valid_per_row < self.k)[0]
            raise ValueError(
                f"Not enough valid candidate neighbors for some rows. "
                f"Min valid count = {int(valid_per_row.min().item())}, "
                f"k = {self.k}, bad rows example = {bad_rows[:10].tolist()}"
            )

        dist, indices = torch.topk(lq, k=self.k, dim=1, largest=False)  # [N, K]
        indices = indices.clamp_(min=0, max=n - 1)  # [N, K]

        # Build logprobs per batch using actual x for each batch
        # x2: [B, N*K, F] = repeat each source node K times
        x2 = x.unsqueeze(2).expand(b, n, self.k, d).reshape(b, n * self.k, d)

        # x1: [B, N*K, F] = gather neighbor embeddings by indices
        idx_flat = indices.reshape(-1)  # [N*K]
        idx_flat = idx_flat.unsqueeze(0).repeat(b, 1)  # [B, N*K]
        idx_gather = idx_flat.unsqueeze(-1).expand(b, n * self.k, d)  # [B, N*K, F]
        x1 = torch.gather(x, 1, idx_gather)  # [B, N*K, F]

        logprobs = (
            -(x1 - x2).pow(2).sum(-1) * temp_scale
        ).reshape(b, n, self.k)  # [B, N, K]

        # Build batch-aware edges: global indexing per batch offset = batch_id * n
        # Convention: src = i, dst = selected neighbor(i)
        single_batch_src = (
            torch.arange(n, device=dev)
            .view(n, 1)
            .repeat(1, self.k)
            .reshape(-1)
        )  # [N*K]

        single_batch_dst = indices.reshape(-1).to(dev)  # [N*K]

        single_batch_src = single_batch_src.repeat(b)  # [B*N*K]
        single_batch_dst = single_batch_dst.repeat(b)  # [B*N*K]

        batch_offset = (
            (torch.arange(b, device=dev) * n)
            .unsqueeze(-1)
            .repeat(1, n * self.k)
            .reshape(-1)
        )  # [B*N*K]

        src = single_batch_src + batch_offset
        dst = single_batch_dst + batch_offset

        edges = torch.stack([src, dst], dim=0)  # [2, B*N*K]
        return edges, logprobs


class DGM_d_dense(nn.Module):
    def __init__(self, embed_f, k=5, distance=pairwise_euclidean_distances, sparse=True):
        super(DGM_d_dense, self).__init__()

        self.sparse = sparse
        self.temperature = nn.Parameter(torch.tensor(1.0 if distance == "hyperbolic" else 4.0).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.k = k

        self.debug = False
        if distance == "euclidean":
            self.distance = pairwise_euclidean_distances
        else:
            self.distance = pairwise_poincare_distances

    def forward(self, x, A, not_used=None, fixedges=None):
        x = self.embed_f(x, A)

        if self.training:
            if fixedges is not None:
                return x, fixedges, torch.zeros(
                    fixedges.shape[0],
                    fixedges.shape[-1] // self.k,
                    self.k,
                    dtype=torch.float,
                    device=x.device,
                ), None

            D, _x = self.distance(x)
            edges_hat, logprobs = self.sample_without_replacement(D)
        else:
            with torch.no_grad():
                if fixedges is not None:
                    return x, fixedges, torch.zeros(
                        fixedges.shape[0],
                        fixedges.shape[-1] // self.k,
                        self.k,
                        dtype=torch.float,
                        device=x.device,
                    ), None
                D, _x = self.distance(x)
                edges_hat, logprobs = self.sample_without_replacement(D)

        if self.debug:
            self.D = D
            self.edges_hat = edges_hat
            self.logprobs = logprobs
            self.x = x

        return x, edges_hat, logprobs, None

    def sample_without_replacement(self, logits):
        b, n, _ = logits.shape
        logits = logits * torch.exp(torch.clamp(self.temperature, -5, 5))

        q = torch.rand_like(logits) + 1e-8
        lq = logits - torch.log(-torch.log(q))
        logprobs, indices = torch.topk(-lq, self.k)

        rows = torch.arange(n).view(1, n, 1).to(logits.device).repeat(b, 1, self.k)
        edges = torch.stack((indices.view(b, -1), rows.view(b, -1)), -2)

        if self.sparse:
            return (edges + (torch.arange(b).to(logits.device) * n)[:, None, None]).transpose(0, 1).reshape(2, -1), logprobs
        return edges, logprobs


class DGM_c(nn.Module):
    input_dim = 4
    debug = False

    def __init__(self, embed_f, k=None, distance="euclidean"):
        super(DGM_c, self).__init__()
        self.temperature = nn.Parameter(torch.tensor(1).float())
        self.threshold = nn.Parameter(torch.tensor(0.5).float())
        self.embed_f = embed_f
        self.centroid = None
        self.scale = None
        self.distance = distance

        self.scale = nn.Parameter(torch.tensor(-1).float(), requires_grad=False)
        self.centroid = nn.Parameter(torch.zeros((1, 1, DGM_c.input_dim)).float(), requires_grad=False)

    def forward(self, x, A, not_used=None, fixedges=None, edges_weight=None):

        x = self.embed_f(x, A)

        # estimate normalization parameters
        if self.scale < 0:
            self.centroid.data = x.mean(-2, keepdim=True).detach()
            self.scale.data = (0.9 / (x - self.centroid).abs().max()).detach()

        if self.distance == "hyperbolic":
            D, _x = pairwise_poincare_distances((x - self.centroid) * self.scale)
        else:
            D, _x = pairwise_euclidean_distances((x - self.centroid) * self.scale)

        adj = torch.sigmoid(self.temperature * (self.threshold.abs() - D))

        if DGM_c.debug:
            self.A = adj.data.cpu()
            self._x = _x.data.cpu()

        if adj.dim() == 2:
            edge_index = adj.nonzero().t()
            edge_attr = adj[edge_index[0], edge_index[1]]
            return edge_index, edge_attr
        else:
            flatten_adj = adj.view(-1, adj.size(-1))
            edge_index = flatten_adj.nonzero().t()
            edges_weight = flatten_adj[edge_index[0], edge_index[1]]

            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
            edge_index[1] += offset[edge_index[0]]

        edge_index = torch.reshape(edge_index, (2, 1, edge_index.shape[1]))
        edge_index = torch.permute(edge_index, (1, 0, 2)).squeeze()
        return x, edge_index, None, edges_weight, None


class MLP(nn.Module):
    def __init__(self, layers_size, final_activation=False, dropout=0):
        super(MLP, self).__init__()
        layers = []
        for li in range(1, len(layers_size)):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(layers_size[li - 1], layers_size[li]))
            if li == len(layers_size) - 1 and not final_activation:
                continue
            layers.append(nn.LeakyReLU(0.1))
        self.MLP = nn.Sequential(*layers)

    def forward(self, x, e=None):
        x = self.MLP(x)
        return x


class Identity(nn.Module):
    def __init__(self, retparam=None):
        self.retparam = retparam
        super(Identity, self).__init__()

    def forward(self, *params):
        if self.retparam is not None:
            return params[self.retparam]
        return params


class DGMModel(nn.Module):
    """
    Versione 'plain PyTorch' del modello Lightning che avevi.
    Stessa logica di forward / costruttore, ma senza Lightning.
    """

    def __init__(self, hparams: Parameters):
        super(DGMModel, self).__init__()
        conv_layers = hparams.conv_layers
        dgm_layers = hparams.dgm_layers
        fc_layers = hparams.fc_layers
        self.num_nodes = hparams.num_nodes

        self.graph_f = ModuleList()
        self.node_g = ModuleList()
        for i, (dgm_l, conv_l) in enumerate(zip(dgm_layers, conv_layers)):
            # --- graph_f ---
            if len(dgm_l) > 0:
                if hparams.ffun == "gcn":
                    self.graph_f.append(DGM_d(GCNConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "gat":
                    self.graph_f.append(DGM_d(GATConv(dgm_l[0], dgm_l[-1]),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "mlp":
                    self.graph_f.append(DGM_d(MLP(dgm_l),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "knn":
                    self.graph_f.append(DGM_d(Identity(retparam=0),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "gcn1d":
                    self.graph_f.append(DGM_d(GCN1DConv(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "gcn1d-big":
                    self.graph_f.append(DGM_d(GCN1DConv_big(dgm_l[0], dgm_l[-1], 1, 1),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "gConvLSTM":
                    self.graph_f.append(DGM_d(GraphConvRNN_ourDGM(dgm_l[0], dgm_l[-1],
                                                                  hidden_channels=hparams.emb_dim,
                                                                  params=hparams,
                                                                  cell_type="LSTM"),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
                if hparams.ffun == "gConvGRU":
                    self.graph_f.append(DGM_d(GraphConvRNN_ourDGM(dgm_l[0], dgm_l[-1],
                                                                  hidden_channels=hparams.emb_dim,
                                                                  params=hparams,
                                                                  cell_type="GRU"),
                                              k=hparams.k,
                                              distance=hparams.distance,
                                              num_nodes=hparams.num_nodes))
            else:
                self.graph_f.append(Identity())

            # --- node_g ---
            if hparams.gfun == "edgeconv":
                conv_l = conv_l.copy()
                conv_l[0] = conv_l[0] * 2
                self.node_g.append(EdgeConv(MLP(conv_l), hparams.pooling))
            if hparams.gfun == "gcn":
                self.node_g.append(GCNConv(conv_l[0], conv_l[1]))
            if hparams.gfun == "gat":
                self.node_g.append(GATConv(conv_l[0], conv_l[1]))
            if hparams.gfun == "gcn1d":
                self.node_g.append(GCN1DConv(conv_l[0], conv_l[1], 1, 1))
            if hparams.gfun == "gcn1d-big":
                self.node_g.append(GCN1DConv_big(conv_l[0], conv_l[1], 1, 1))
            if hparams.gfun == "gConvLSTM":
                self.node_g.append(GraphConvRNN_ourDGM(conv_l[0], conv_l[-1],
                                                      hidden_channels=hparams.emb_dim,
                                                      params=hparams,
                                                      cell_type="LSTM"))
            if hparams.gfun == "gConvGRU":
                self.node_g.append(GraphConvRNN_ourDGM(conv_l[0], conv_l[-1],
                                                      hidden_channels=hparams.emb_dim,
                                                      params=hparams,
                                                      cell_type="GRU"))

        if hparams.pre_fc is not None and len(hparams.pre_fc) > 0:
            self.pre_fc = MLP(hparams.pre_fc, final_activation=True)
        else:
            self.pre_fc = None

        self.fc = nn.Linear(fc_layers[0], fc_layers[-1])

        self.avg_accuracy = None
        self.params = hparams
        self.best_mse = 1000
        self.best_mae = 1000
        self.best_mape = 1000
        self.best_rmse = 1000

    def forward(self, x, edges=None, edges_weight=None):
        if self.params.pre_fc is not None and len(self.params.pre_fc) > 0:
            x = self.pre_fc(x)

        detach_graph_input = bool(getattr(self.params, "graph_detach_input", True))

        graph_x = x.detach() if detach_graph_input else x
        lprobslist = []
        for f, g in zip(self.graph_f, self.node_g):
            graph_x, edges, lprobs, edges_weight = f(graph_x, edges, None, None)
            graph_x = graph_x.squeeze()

            x_ = torch.reshape(x, (int(x.shape[0] / self.params.num_nodes), self.params.num_nodes, -1))
            b, n, d = x_.shape

            self.edges = edges

            x_pre = g(torch.dropout(x.view(-1, d), self.params.dropout, train=self.training), edges)
            x = torch.relu(x_pre)

            x_for_graph = x.detach() if detach_graph_input else x
            graph_x = torch.cat([graph_x, x_for_graph], -1)

            if lprobs is not None:
                lprobslist.append(lprobs)

        x = F.dropout(x, p=self.params.dropout, training=self.training)

        if len(lprobslist) > 0:
            return self.fc(x), torch.stack(lprobslist, -1)
        else:
            return self.fc(x), None
