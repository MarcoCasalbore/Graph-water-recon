# -------------------------------------------------------------------------------------
# GCN1D model, from
# https://github.com/alessioverdone/Explainable-Spatio-Temporal-Graph-Neural-Network
# -------------------------------------------------------------------------------------

import torch
import math
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from torch import Tensor
from torch import nn
from torch.nn import Parameter, init
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor
from torch_geometric.nn.inits import zeros
from torch_geometric.utils import get_laplacian, remove_self_loops
from typing import Optional


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        padding=0,
        bias=True,
    ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.__padding = padding

    def forward(self, input):
        input = F.pad(input, (self.__padding * 2, 0))
        return super(CausalConv1d, self).forward(input)


class Conv_base(torch.nn.Module):
    def __init__(
        self,
        input_features_cnn,
        output_features_cnn,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
    ):
        super(Conv_base, self).__init__()

        self.in_channels = in_channels
        padding = int(
            (
                stride * (output_features_cnn - 1)
                - input_features_cnn
                + (dilation * (kernel_size - 1) + 1)
            )
            / 2
        )
        self.conv1d_1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            stride=stride,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(
            self.conv1d_1.weight, a=math.sqrt(5)
        )  # Kaiming uniform distribution

        if self.conv1d_1.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv1d_1.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.conv1d_1.bias, -bound, bound)

    def forward(self, x_conv):
        h = torch.reshape(x_conv, (x_conv.shape[0], self.in_channels, -1))
        h = self.conv1d_1(h)
        h = torch.reshape(h, (h.shape[0], -1))
        return h


class ChebConv(MessagePassing):
    def __init__(
        self,
        input_features_cnn,
        output_features_cnn,
        cnn1d_kernel_size,
        in_channels: int,
        out_channels: int,
        gconv_cheb_filter_size: int,
        normalization: Optional[str] = "sym",
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        assert gconv_cheb_filter_size > 0
        assert normalization in [None, "sym", "rw"], "Invalid normalization"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.lins = torch.nn.ModuleList(
            [
                Conv_base(
                    input_features_cnn,
                    output_features_cnn,
                    in_channels,
                    out_channels,
                    cnn1d_kernel_size,
                )
                for _ in range(gconv_cheb_filter_size)
            ]
        )

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def __norm__(
        self,
        edge_index,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization, dtype, num_nodes
        )

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float("inf"), 0)

        # It causes problems in GNNExplainer evaluation
        # edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
        #                                          fill_value=-1.,
        #                                          num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ):
        """"""
        if self.normalization != "sym" and lambda_max is None:
            raise ValueError(
                "You need to pass `lambda_max` to `forward() in`"
                "case the normalization is non-symmetric."
            )

        if lambda_max is None:
            lambda_max = torch.tensor(2.0, dtype=x.dtype, device=x.device)
        if not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=x.dtype, device=x.device)
        assert lambda_max is not None

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm, size=None)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            Tx_2 = 2.0 * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, K={len(self.lins)}, "
            f"normalization={self.normalization})"
        )


class GCN1D(nn.Module):
    """
    Model for doing forecasting on temporal series data
    Functions:
        __init__
        forward
    """

    def __init__(
        self,
        lags,
        input_size,
        exog_size,
        num_features,
        num_quantiles,
        horizon,
        gconv_filter_size,
        n_layers,
        night,
        normalization="sym",
        bias=None,
        **kwargs,
    ):
        super(GCN1D, self).__init__()

        # Define model setup
        input_dim = input_size + exog_size  # include exogenous features
        output_size = (
            num_features * num_quantiles
        )  # multi-feature, multi-quantile output
        self.num_features = num_features
        self.num_quantiles = num_quantiles
        self.horizon = horizon
        self.night = night
        model_config = get_model_config(
            num_layers=n_layers, lags=lags, input_dim=input_dim
        )

        # Build model components
        self.layer_list = nn.ModuleList()
        self.bias_list = nn.ParameterList()
        for i in range(n_layers):
            self.layer_list.append(
                ChebConv(
                    input_features_cnn=model_config[i][0],
                    output_features_cnn=model_config[i][1],
                    cnn1d_kernel_size=model_config[i][2],
                    in_channels=model_config[i][3],
                    out_channels=model_config[i][4],
                    gconv_cheb_filter_size=gconv_filter_size,
                    normalization=normalization,
                    bias=bias,
                )
            )
            self.bias_list.append(
                Parameter(torch.Tensor(1, model_config[i][1] * model_config[i][4]))
            )

        # Initiate bias
        for b in self.bias_list:
            zeros(b)

        # Final layers to adapt output to horizon
        horizon = output_size * horizon  # multi-step, multi-feature output
        self.linear = torch.nn.Linear(self.bias_list[-1].shape[1], horizon)
        self.linear_time = torch.nn.Linear(lags, horizon)

        # Rearrange layer to reshape output
        self.rearrange = Rearrange(
            "b n (h f q) -> b h n f q", h=self.horizon, f=num_features, q=num_quantiles
        )

    def forward(self, x, edge_index, edge_weight, u=None):
        # Initialize data
        if u is not None:
            x = torch.cat([x, u], dim=-1)

        b, t, n, f = x.shape
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (b * n, t * f))

        h = x
        # Main component
        for i in range(len(self.layer_list)):
            h = F.relu(
                self.layer_list[i](h, edge_index, edge_weight) + self.bias_list[i]
            )

        # Adapt embedding to horizon
        h = self.linear(h)

        # Night correction
        if self.night:
            x_hour = x[:, -self.lags :]
            h_time = F.sigmoid(self.linear_time(x_hour))
            h = h * h_time

        # Reshape to (batch, horizon, nodes, features, quantiles)
        # First disentangle b and n
        h = rearrange(h, "(b n) d -> b n d", b=b, n=n)
        # Then use the layer to reshape the output
        h = self.rearrange(h)

        return h


def get_model_config(num_layers, lags, input_dim):
    """
    Each value represent a model parameter at each layer:
    ["in_feat", "out_feat", "k_size", "in_ch", "out_ch"]

    e.g.
    [[-1,32,5,-1,24]  # layer 1
    [32,30,5,24,32] # layer 2
    [30,28,5,32,64]
    [28,26,3,64,64]
    [26,24,3,64,96]
    [24,22,3,96,96]
    [22,20,3,96,128]
    [20,18,3,128,192]
    [18,16,3,192,256]
    [16,14,3,256,320]
    [14,12,3,320,320]] # layer n
    """
    # Basic model parameters configuration, can be changed by controlling dimension consistency between each layer
    base_config = [
        [-1, 32, 5, -1, 24],
        [32, 30, 5, 24, 32],
        [30, 28, 5, 32, 64],
        [28, 26, 3, 64, 64],
        [26, 24, 3, 64, 96],
        [24, 22, 3, 96, 96],
        [22, 20, 3, 96, 128],
        [20, 18, 3, 128, 192],
        [18, 16, 3, 192, 256],
        [16, 14, 3, 256, 320],
        [14, 12, 3, 320, 320],
    ]

    base_config = base_config[:num_layers]
    base_config[0][0] = lags
    base_config[0][3] = input_dim
    return base_config
