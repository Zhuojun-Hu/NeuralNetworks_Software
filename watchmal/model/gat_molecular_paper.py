import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax

class NodeEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, max_position=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )
        # Optional positional embedding (e.g. towall bin index or hit time rank)
        self.pos_embed = nn.Embedding(max_position, hidden_channels)

    def forward(self, x, pos_idx):
        x = self.mlp(x)
        x += self.pos_embed(pos_idx)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, hidden_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        assert hidden_channels % num_heads == 0

        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)

        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        Q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(-1, self.num_heads, self.head_dim)

        attn_scores = (Q[row] * K[col]).sum(dim=-1) / self.head_dim**0.5
        attn_weights = softmax(attn_scores, index=row)

        out = attn_weights.unsqueeze(-1) * V[col]
        out = out.view(-1, self.num_heads * self.head_dim)

        agg = torch.zeros_like(x).scatter_add_(0, row.unsqueeze(-1).expand_as(out), out)

        return self.mlp(torch.cat([x, agg], dim=-1)) + x

class GraphAttentionNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=4, num_heads=4, max_position=10):
        super().__init__()
        self.encoder = NodeEncoder(in_channels, hidden_channels, max_position=max_position)
        self.layers = nn.ModuleList([
            AttentionLayer(hidden_channels, num_heads) for _ in range(num_layers)
        ])
        self.pool = global_mean_pool
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if hasattr(data, "pos_idx"):
            pos_idx = data.pos_idx  # e.g. binned towall or rank/time
        else:
            pos_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.encoder(x, pos_idx)
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.pool(x, batch)
        return self.classifier(x)
