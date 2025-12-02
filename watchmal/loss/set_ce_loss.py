# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F


def dice_coeff(prob: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * target_bin).sum()
    den = prob.sum() + target_bin.sum() + eps
    return num / den


def smallK_match(cost: torch.Tensor) -> List[int]:
    N, K = cost.shape
    if K == 0 or N == 0:
        return []
    assign: List[int] = []
    for kk in range(K):
        col = cost[:, kk]
        qidx = int(torch.argmin(col).item())
        assign.append(qidx)
    return assign


def _stable_probs(pi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    pi = pi.clamp(0.0, 1.0)
    row_sum = pi.sum(dim=1, keepdim=True).clamp_min(eps)
    return pi / row_sum


def nll_with_label_smoothing(logp: torch.Tensor,
                             target_idx: torch.Tensor,
                             eps: float = 0.05,
                             reduction: str = "sum") -> torch.Tensor:
    V, C = logp.shape
    with torch.no_grad():
        y = F.one_hot(target_idx.long(), num_classes=C).to(logp.dtype)
        y = (1.0 - eps) * y + eps / C
    loss_vec = -(y * logp).sum(dim=1)
    if reduction == "sum":
        return loss_vec.sum()
    elif reduction == "mean":
        return loss_vec.mean()
    return loss_vec


def loss_set_ce_dice(lambda_dice: float = 2.0,
                     label_smoothing: float = 0.05,
                     eps: float = 1e-6,
                     poisson: bool = True):

    def _loss(batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "Pi_list" not in out:
            raise KeyError("Expected out['Pi_list'] = list of [V_b, C] probabilities per batch element.")

        Pi_list: List[torch.Tensor] = out["Pi_list"]
        device = Pi_list[0].device if len(Pi_list) > 0 else "cpu"

        coords_all = batch.get("coords", None)
        feats_all = batch.get("feats", None)
        if poisson:
            if not isinstance(coords_all, torch.Tensor) or not isinstance(feats_all, torch.Tensor):
                raise KeyError("Poisson weighting requires batch['coords'] and batch['feats'].")

            coords_all = coords_all.to(device)
            feats_all = feats_all.to(device)

        ce_sum = torch.tensor(0.0, device=device)
        dice_sum = torch.tensor(0.0, device=device)
        total_weight = 0.0
        total_vox = 0
        B = len(Pi_list)

        for b in range(B):
            Pi_b = Pi_list[b]
            if Pi_b.numel() == 0:
                continue

            Pi_b = _stable_probs(Pi_b, eps=eps)
            logPi_b = Pi_b.clamp_min(eps).log()

            meta_b = batch["meta"][b]
            if "voxel_parent_frac" not in meta_b:
                raise KeyError("Expected batch['meta'][b]['voxel_parent_frac'].")

            y_b = meta_b["voxel_parent_frac"].to(Pi_b.device).float()
            if y_b.dim() == 1:
                y_b = y_b.unsqueeze(1)

            V_b, C = Pi_b.shape
            if y_b.shape[0] != V_b:
                raise ValueError(f"Shape mismatch: Pi_b has {V_b} voxels, voxel_parent_frac has {y_b.shape[0]}.")

            C_t = y_b.shape[1]
            if C_t < C:
                pad = torch.zeros((V_b, C - C_t), device=Pi_b.device, dtype=y_b.dtype)
                y_b = torch.cat([y_b, pad], dim=1)
            elif C_t > C:
                y_b = y_b[:, :C]

            y_b = _stable_probs(y_b, eps=eps)
            ce_vec = -(y_b * logPi_b).sum(dim=1)

            if poisson:
                mask_b = (coords_all[:, 0] == b)
                if int(mask_b.sum().item()) != V_b:
                    raise ValueError(f"Poisson mode: coords mask for batch {b} has {int(mask_b.sum().item())} voxels, "
                                     f"but Pi_list[{b}] has {V_b}.")
                w_v = feats_all[mask_b, 0].float().clamp_min(eps)
                ce_sum = ce_sum + (w_v * ce_vec).sum()
                total_weight += float(w_v.sum().item())
            else:
                ce_sum = ce_sum + ce_vec.sum()
                total_vox += V_b

            if lambda_dice > 0.0 and C > 1:
                dsum = 0.0
                for c in range(1, C):
                    pc = Pi_b[:, c]
                    tc = y_b[:, c]
                    dsum += (1.0 - dice_coeff(pc, tc, eps=eps))
                dice_sum = dice_sum + dsum

        if poisson:
            denom = max(total_weight, 1.0)
            ce = ce_sum / float(denom)
        else:
            vox_ = max(total_vox, 1)
            ce = ce_sum / float(vox_)

        if lambda_dice > 0.0 and B > 0:
            dice = dice_sum / float(B)
        else:
            dice = torch.tensor(0.0, device=device)

        total = ce + lambda_dice * dice
        return {"loss": total, "loss_ce": ce, "loss_dice": dice}

    return _loss
