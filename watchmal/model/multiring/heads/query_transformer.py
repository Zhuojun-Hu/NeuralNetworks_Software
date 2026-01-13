# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- PE ----
class SinPE(nn.Module):
    def __init__(self, bands: int, base: float = 2*math.pi, scale: float = 2.0, d_out: int = 256):
        super().__init__()
        self.omegas = nn.Parameter(torch.tensor([base*(scale**j) for j in range(bands)], dtype=torch.float32), requires_grad=False)
        self.proj = nn.Linear(6*bands, d_out)

    def forward(self, zyx: torch.Tensor, grid: Tuple[int,int,int]) -> torch.Tensor:
        Z,Y,X = [float(g) for g in grid]
        x = zyx[:,2] / max(X-1,1.0)
        y = zyx[:,1] / max(Y-1,1.0)
        z = zyx[:,0] / max(Z-1,1.0)
        xyz = torch.stack([x,y,z], dim=1)  # [U,3]
        phase = xyz.unsqueeze(-1)*self.omegas  # [U,3,B]
        pe = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1)  # [U,6B]
        return self.proj(pe)

# ---- decoder ----
class QueryDecoder(nn.Module):
    def __init__(self, d: int, nhead: int, layers: int, n_queries: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d))
        dec_layer = nn.TransformerDecoderLayer(d_model=d, nhead=nhead, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=layers)

    def forward(self, mem: torch.Tensor, mem_pe: torch.Tensor) -> torch.Tensor:
        src = mem + mem_pe             # [U,D]
        tgt = self.query.unsqueeze(0)  # [1,N,D]
        out = self.dec(tgt, src.unsqueeze(0)).squeeze(0)  # [N,D]
        return out

# ---- head ----
class QueryPerVoxelSoftmaxHead(nn.Module):
    """
    Outputs per-event:
      Z: [V_b, N+1], Pi: [V_b, N+1], H: [N+1,D]
    """
    def __init__(
        self,
        d_model: int = 256,
        c_vox: int = 64,
        c_mask: int = 96,
        n_queries: int = 2,
        nhead: int = 8,
        layers: int = 3,
        pe_bands: int = 8,
        voxel_attention: bool = False,
    ):
        super().__init__()
        self.pe = SinPE(pe_bands, d_out=d_model)

        # background is now treated like the other queries:
        # total queries = (n_queries + 1) with query 0 = background
        self.dec = QueryDecoder(d=d_model, nhead=nhead, layers=layers, n_queries=n_queries + 1)

        self.vox_to_mask = nn.Linear(c_vox, c_mask)
        self.q_to_mask = nn.Linear(d_model, c_mask)

        self.n_queries = n_queries + 1

        self.voxel_attention = voxel_attention
        self.vox_to_dec = nn.Linear(c_vox, d_model)

    def forward(self, enc: Dict[str, Any], batch: Dict[str, Any]) -> Dict[str, Any]:
        grid = batch["meta"][0]["grid_size"]

        Z_list: List[torch.Tensor] = []
        Pi_list: List[torch.Tensor] = []
        H_list: List[torch.Tensor] = []

        voxel_idx_list = enc.get("voxel_idx_list", None)

        for idx, (mem_f, mem_c, vox_f) in enumerate(
            zip(enc["mem_feat_list"], enc["mem_coord_list"], enc["voxel_feat_list"])
        ):

            if vox_f.numel() == 0:
                Z_list.append(vox_f.new_zeros((0, self.n_queries)))
                Pi_list.append(vox_f.new_zeros((0, self.n_queries)))
                H_list.append(
                    vox_f.new_zeros((self.n_queries, self.q_to_mask.in_features))
                )
                continue

            if self.voxel_attention:
                if voxel_idx_list is None:
                    raise KeyError("Expected enc['voxel_idx_list'] when voxel_attention=True.")
                vox_idx = voxel_idx_list[idx]                 # [V,3] (z,y,x)
                mem_for_dec = self.vox_to_dec(vox_f)          # [V,D]
                pe_for_dec = self.pe(vox_idx.float(), grid)   # [V,D]
            else:
                if mem_f.numel() == 0:
                    Z_list.append(vox_f.new_zeros((0, self.n_queries)))
                    Pi_list.append(vox_f.new_zeros((0, self.n_queries)))
                    H_list.append(
                        vox_f.new_zeros((self.n_queries, self.q_to_mask.in_features))
                    )
                    continue
                mem_for_dec = mem_f                           # [U,D]
                pe_for_dec = self.pe(mem_c.float(), grid)     # [U,D]

            H = self.dec(mem_for_dec, pe_for_dec)             # [N+1,D]

            E = self.vox_to_mask(vox_f)                       # [V,Cm]
            P = self.q_to_mask(H)                             # [N+1,Cm]
            Z = E @ P.t() / math.sqrt(E.shape[1])             # [V,N+1]
            Pi = F.softmax(Z, dim=1)

            Z_list.append(Z)
            Pi_list.append(Pi)
            H_list.append(H)

        return {"Z_list": Z_list, "Pi_list": Pi_list, "H_list": H_list}
