from typing import List

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import ResGatedGraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.norm import BatchNorm, LayerNorm, GraphNorm

from torch.nn import Dropout, Linear, Sigmoid, LogSoftmax, ReLU



# The v_1 would be the "BaseResGateConv" only using Linear at the end (not batchnorm and dropout)
# nor global max et global add pooling
class ResGatedConv_v3(torch.nn.Module):
    r""""
    Args :
        in_channels : Number of features per node in the graph
        conv_kernels : The number of filters for each layer starting from the 2nd. The len defines 
            the number of layers.
        linear_out_features : The number of neurones at the end of each Linear layer. 
            The len defines the number of layers. 
            The last element of the list gives the number of classes the model will predict
    """
    def __init__(
        self,
        in_channels: int,
        conv_in_channels: List[int],
        linear_out_features: List[int],
        dropout: float = 0.2,
        use_nhits: bool = False,
        use_event_total_charge: bool = False
    ) -> None:

        super().__init__() # If more heritage than torch.nn.Module is added, modify this according to the changes

        self.use_nhits = use_nhits
        self.use_event_total_charge = use_event_total_charge
        self.conv_layers = torch.nn.ModuleList()
        self.cl_norms = torch.nn.ModuleList()

        # Define the convolutional and normalization layers
        for i in range(len(conv_in_channels)):
            in_channels = in_channels if i == 0 else conv_in_channels[i - 1] # Erwan : Sure to be a good idea ? Maybe enforce some check from the user (just use in_conv_feat)
            out_channels = conv_in_channels[i]
            self.conv_layers.append(ResGatedGraphConv(in_channels, out_channels))
            self.cl_norms.append(GraphNorm(out_channels))

        self.hidden_layers = torch.nn.ModuleList()
        self.hl_norms = torch.nn.ModuleList()

        # Define the hidden and normalization layers
        # first_linear_in_features = 2 * conv_in_channels[-1] # Times 2 because of the torch.cat on gap and gsp
        first_linear_in_features = conv_in_channels[-1]
        first_linear_in_features += int(self.use_nhits)
        first_linear_in_features += int(self.use_event_total_charge)

        for j in range(len(linear_out_features) - 1):
            in_features  = first_linear_in_features if j == 0 else linear_out_features[j - 1]
            out_features = linear_out_features[j]
            self.hidden_layers.append(Linear(in_features, out_features))
            self.hl_norms.append(LayerNorm(out_features))

        # Last layer doesn't have batch_norm
        self.last_layer = Linear(linear_out_features[-2], linear_out_features[-1])

        self.drop = Dropout(p=dropout)
        # self.activation_cl = ReLU()
        self.activation_cl = torch.nn.GELU()
        self.activation_hl = ReLU()
        # self.activation_hl = torch.nn.SiLU()
        
        # DO NOT add a sigmoid layer for the output (it is handled by the loss)
        # if it's not a sigmoid then define the output activation here
        # And then use BCELoss as a loss function (and not BCEwithlogitsloss)
        self.output_activation = None # ReLU()..


    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
 
        # Apply convolutional layers
        for conv_layer, norm_layer in zip(self.conv_layers, self.cl_norms):
            x = conv_layer(x, edge_index)
            x = self.activation_cl(x)
            x = norm_layer(x)
            x = self.drop(x)

        # Global pooling 
        # Terminologie : gap, très moyen vu que global_add_pool existe aussi..
        # xs_gap = global_mean_pool(x, batch)
        # xs_gsp = global_max_pool(x, batch)
        # out = torch.cat((xs_gap, xs_gsp), dim=1) # out.shape = (batch_size, 2 * conv_in_channels[-1])
        
        out = global_mean_pool(x, batch)
        

        # Add the n_hits info to the input of the mlp
        if self.use_nhits:
            assert hasattr(data, 'n_hits'), f"Use n_hits set to true but data has not n_hits attributs. Data : {data}"
            nhits = data.n_hits.to(out.dtype).to(out.device).unsqueeze(1)
            out = torch.cat([out, nhits], dim=1) 

        if self.use_event_total_charge:
            assert hasattr(data, 'event_total_charge'), f"Use event_total_charge set to true but data has not event_total_charge attributs. Data : {data}"
            event_total_charge = data.event_total_charge.to(out.dtype).to(out.device).unsqueeze(1)
            out = torch.cat([out, event_total_charge], dim=1)  

        # Apply hidden layers
        for hidden_layer, norm_layer in zip(self.hidden_layers, self.hl_norms):
            out = hidden_layer(out)
            out = self.activation_hl(out)
            out = norm_layer(out)
            out = self.drop(out)
    
        out =  self.last_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out
