from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn

import spconv.pytorch as spconv
from spconv.pytorch import (
    SparseConvTensor,
    SubMConv3d,
    SparseConv3d,
    SparseInverseConv3d,
)


class BNAct1d(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(x))


class SubMBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, key: str):
        super().__init__()
        self.c1 = SubMConv3d(c_in, c_out, 3, bias=False, indice_key=key)
        self.n1 = BNAct1d(c_out)
        self.c2 = SubMConv3d(c_out, c_out, 3, bias=False, indice_key=key)
        self.n2 = BNAct1d(c_out)
    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        y = self.c1(x)
        y = y.replace_feature(self.n1(y.features))
        y = self.c2(y)
        y = y.replace_feature(self.n2(y.features))
        return y


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int, key: str):
        super().__init__()
        self.conv = SparseConv3d(c_in, c_out, 2, stride=2, bias=False, indice_key=key)
        self.n = BNAct1d(c_out)
    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        y = self.conv(x)
        return y.replace_feature(self.n(y.features))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, key_down: str, key_up: str):
        super().__init__()
        self.deconv = SparseInverseConv3d(c_in, c_out, 2, indice_key=key_down, bias=False)
        self.n = BNAct1d(c_out)
        self.fuse = SubMConv3d(c_out + c_skip, c_out, 1, bias=False, indice_key=key_up)
    def forward(self, x: SparseConvTensor, skip: SparseConvTensor) -> SparseConvTensor:
        y = self.deconv(x)
        y = y.replace_feature(self.n(y.features))
        y = y.replace_feature(torch.cat([y.features, skip.features], 1))
        return self.fuse(y)


class SparseUNet3D(nn.Module):
    """
    Coords: [N,4]=[b,z,y,x] or [N,5]=[b,t,z,y,x]. For 4D, time is folded into a composite batch.
    Returns:
      - mem_feat_list:  List[[U_b, d_mem]]
      - mem_coord_list: List[[U_b, 3]]
      - voxel_feat_list: List[[V_b, c_out]]
      - voxel_idx_list:  List[[V_b, 3]]
    """
    def __init__(
        self,
        c_in: int = 5,
        c_stem: int = 64,
        channels: Tuple[int, ...] = (96, 160, 256),
        mem_level: int = 2,
        d_mem: int = 256,
    ):
        super().__init__()
        self.channels = tuple(channels)
        self.mem_level = int(mem_level)

        self.stem = SubMBlock(c_in, c_stem, key="subm1")

        self.downs, self.blocks = nn.ModuleList(), nn.ModuleList()
        c_prev = c_stem
        for i, c in enumerate(self.channels):
            self.downs.append(Down(c_prev, c, key=f"down{i+1}"))
            self.blocks.append(SubMBlock(c, c, key=f"subm_down{i+1}"))
            c_prev = c

        self.ups, self.upblocks, self.up_skip_levels = nn.ModuleList(), nn.ModuleList(), []
        c_in_up = self.channels[-1]
        L = len(self.channels)
        for d in range(L - 1, -1, -1):
            c_out = (self.channels[d - 1] if d > 0 else c_stem)
            c_skip = (self.channels[d - 1] if d > 0 else c_stem)
            self.ups.append(Up(c_in_up, c_skip, c_out, key_down=f"down{d+1}", key_up=f"up{d+1}"))
            self.upblocks.append(SubMBlock(c_out, c_out, key=f"subm_up{d+1}"))
            self.up_skip_levels.append(d)
            c_in_up = c_out

        self.out_proj = SubMConv3d(c_in_up, c_stem, 1, bias=False, indice_key="subm1")
        self.mem_proj = nn.Linear(self.channels[self.mem_level - 1], d_mem)

    @staticmethod
    def _fold_time(coords: torch.Tensor, meta: List[Dict[str, Any]]) -> Tuple[torch.Tensor, List[List[int]]]:
        if coords.size(1) == 4:
            B = int(coords[:, 0].max().item()) + 1 if coords.numel() else 0
            return coords, [[b] for b in range(B)]
        b, t = coords[:, 0].long(), coords[:, 1].long()
        B = int(b.max().item()) + 1 if coords.numel() else 0
        t_bins = []
        for bi in range(B):
            tb = meta[bi].get("t_bins")
            if tb is None:
                tb = int((t[b == bi].max().item() if (b == bi).any() else -1) + 1)
                tb = max(tb, 1)
            t_bins.append(int(tb))
        offs = [0]
        for i in range(1, B):
            offs.append(offs[-1] + t_bins[i - 1])
        offs = torch.tensor(offs, device=coords.device, dtype=torch.long)
        comp = offs[b] + t
        comps_per_event = [list(range(int(offs[bi]), int(offs[bi] + t_bins[bi]))) for bi in range(B)]
        zyx = coords[:, 2:].int()
        return torch.stack([comp.int(), zyx[:, 0], zyx[:, 1], zyx[:, 2]], 1), comps_per_event

    @staticmethod
    def _build_sparse(feats: torch.Tensor, coords4: torch.Tensor, grid: Tuple[int, int, int]) -> SparseConvTensor:
        return spconv.SparseConvTensor(
            features=feats,
            indices=coords4.int(),
            spatial_shape=torch.Size(list(grid)),
            batch_size=int(coords4[:, 0].max().item()) + 1,
        )

    @staticmethod
    def _regroup(x: torch.Tensor, comp_idx: torch.Tensor, groups: List[List[int]]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        if x.numel() == 0:
            return out
        for g in groups:
            if len(g) == 1:
                out.append(x[comp_idx == g[0]])
            else:
                m = torch.zeros_like(comp_idx, dtype=torch.bool)
                for c in g:
                    m |= (comp_idx == c)
                out.append(x[m])
        return out

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        coords_in: torch.Tensor = batch["coords"]
        feats: torch.Tensor = batch["feats"]
        meta: List[Dict[str, Any]] = batch["meta"]
        grid = meta[0]["grid_size"]

        coords4, groups = self._fold_time(coords_in, meta)
        x = self._build_sparse(feats, coords4, grid)

        x0 = self.stem(x)
        levels = [x0]
        cur = x0
        for d, blk in zip(self.downs, self.blocks):
            cur = blk(d(cur))
            levels.append(cur)

        mem = levels[self.mem_level]
        mem_feat_all = self.mem_proj(mem.features)
        mem_idx_all = mem.indices

        cur = levels[-1]
        for k, d in enumerate(self.up_skip_levels):
            cur = self.upblocks[k](self.ups[k](cur, levels[d]))

        full = self.out_proj(cur)
        full_idx_all = full.indices
        full_feat_all = full.features

        comp_vox = full_idx_all[:, 0]
        comp_mem = mem_idx_all[:, 0]

        return {
            "full": full,
            "full_feat": full_feat_all,
            "full_idx": full_idx_all[:, 1:],
            "mem_feat_list": self._regroup(mem_feat_all, comp_mem, groups),
            "mem_coord_list": self._regroup(mem_idx_all[:, 1:], comp_mem, groups),
            "voxel_feat_list": self._regroup(full_feat_all, comp_vox, groups),
            "voxel_idx_list": self._regroup(full_idx_all[:, 1:], comp_vox, groups),
        }
