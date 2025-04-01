"""
This is the original model form the GRANT software.
For better usage experience it was refactored to avoid excessive usage of setattr and exec.
The initial code is still preserved for compatibility reasons and reference and educational purposes.
"""

from typing import List
import torch
import torch_geometric
from torch_geometric.nn import (
    global_add_pool as gsp,
    global_mean_pool as gap,
    BatchNorm,
    ResGatedGraphConv,
)
from torch.nn import functional as F, Linear, Dropout, Sigmoid, ReLU
import torch.nn as nn


class ResGatedGraphConvRefactored(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernels: List[int],
        nb_conv_layers: int = 2,
        nb_hidden_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.nb_conv_layers = nb_conv_layers
        self.nb_hidden_layers = nb_hidden_layers

        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        for i in range(self.nb_conv_layers):
            in_dim = in_channels if i == 0 else kernels[i - 1]
            self.conv_layers.append(ResGatedGraphConv(in_dim, kernels[i]))
            self.cl_norms.append(BatchNorm(kernels[i]))

        self.drop = Dropout(p=dropout)

        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        in_dim = sum(kernels) * 2  # Corrected concatenation dimension
        for i in range(self.nb_hidden_layers):
            out_dim = in_dim // 2  # Reduce by half each layer
            self.hidden_layers.append(Linear(in_dim, out_dim))
            self.hl_norms.append(BatchNorm(out_dim))
            in_dim = out_dim  # Update for next iteration

        self.classifier = Linear(in_dim, 2)

        self.activation_cl = Sigmoid()
        self.activation_hl = Sigmoid()

    def forward(self, graphs: torch_geometric.data.Dataset) -> float:
        batch = graphs.batch
        x = graphs.x
        edge_index = graphs.edge_index

        conv_outputs = []

        for i in range(self.nb_conv_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.activation_cl(x)
            x = self.cl_norms[i](x)
            x = self.drop(x)
            conv_outputs.append(x)

        xs_gap = [gap(x, batch=batch) for x in conv_outputs]
        xs_gsp = [gsp(x, batch=batch) for x in conv_outputs]

        out = torch.cat(xs_gap + xs_gsp, dim=1)

        for i in range(self.nb_hidden_layers):
            out = self.hidden_layers[i](out)
            out = self.activation_hl(out)
            out = self.hl_norms[i](out)

        out = self.classifier(out)
        return out


# original GRANT code:

# class MyEdgeConv(torch.nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         kernels: List[int],
#         nb_conv_layers: int = 2,
#         nb_hidden_layers: int = 2,
#         dropout: float = 0.2,
#     ) -> None:

#         super().__init__()

#         self.nb_conv_layers = nb_conv_layers
#         self.nb_hidden_layers = nb_hidden_layers

#         # Define the convolutional and normalization layers
#         for i in range(self.nb_conv_layers):
#             if i == 0:
#                 setattr(
#                     self,
#                     "conv_layer1",
#                     ResGatedGraphConv(in_channels, kernels[0]),
#                 )
#             else:
#                 setattr(
#                     self,
#                     f"conv_layer{i + 1}",
#                     ResGatedGraphConv(kernels[i - 1], kernels[i]),
#                 )
#             setattr(self, f"cl_norm{i + 1}", BatchNorm(kernels[i]))

#         self.drop = Dropout(p=dropout)

#         # Define the hidden and normalization layers
#         for i in range(self.nb_hidden_layers):
#             setattr(
#                 self,
#                 f"hidden_layer_bipool{i + 1}",
#                 Linear(int(sum(kernels) / 2 ** (i - 1)), int(sum(kernels) / 2**i)),
#             )
#             setattr(self, f"hl_norm{i + 1}", BatchNorm(int(sum(kernels) / 2**i)))

#         self.classifier = Linear(
#             int(sum(kernels) / 2 ** (self.nb_hidden_layers - 1)), 2
#         )

#         # Define the activation and output functions
#         self.activation_cl = Sigmoid()  # ReLU()
#         self.activation_hl = Sigmoid()  # ReLU()
#         # self.output = LogSoftmax(dim=1)

#     def forward(self, graphs: torch_geometric.data.Dataset) -> float:
#         batch = graphs.batch
#         x0 = graphs.x
#         edge_index = graphs.edge_index

#         for i in range(1, self.nb_conv_layers + 1):
#             exec(f"x{i} = self.conv_layer{i}(x{i - 1},edge_index)")
#             exec(f"x{i} = self.activation_cl(x{i})")
#             exec(f"x{i} = self.cl_norm{i}(x{i})")
#             exec(f"x{i} = self.drop(x{i})")
#         xs_gap = []
#         xs_gsp = []
#         for i in range(1, self.nb_conv_layers + 1):
#             exec(f"xs_gap.append(gap(x{i}, batch=batch))")
#             exec(f"xs_gsp.append(gsp(x{i}, batch=batch))")

#         out = torch.cat(
#             tuple(xs_gap[i] for i in range(self.nb_conv_layers))
#             + tuple(xs_gsp[i] for i in range(self.nb_conv_layers)),
#             1,
#         )

#         for i in range(1, self.nb_hidden_layers + 1):
#             if i == 1:
#                 exec(f"out_loop = self.hidden_layer_bipool{i}(out)")
#             else:
#                 exec(f"out_loop = self.hidden_layer_bipool{i}(out_loop)")
#             exec("out_loop = self.activation_hl(out_loop)")
#             if i < self.nb_hidden_layers:
#                 exec(f"out_loop = self.hl_norm{i}(out_loop)")
#             else:
#                 exec(f"out_loop = self.hl_norm{i}(out_loop)")

#         exec("out_loop = self.classifier(out_loop)")
#         # print(eval("torch.sigmoid(out_loop)"))
#         # return eval("torch.sigmoid(out_loop)")
#         return eval("out_loop")
