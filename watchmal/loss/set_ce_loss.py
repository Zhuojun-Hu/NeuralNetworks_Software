# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def dice_coeff(prob: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Sørensen–Dice coefficient for one binary mask vs. one probability field.
    prob        : [V] in [0,1]
    target_bin  : [V] in {0,1}
    """
    num = 2.0 * (prob * target_bin).sum()
    den = prob.sum() + target_bin.sum() + eps
    return num / den


def smallK_match(cost: torch.Tensor) -> List[int]:
    """
    Greedy column-wise assignment:
      - cost: [N, K] where N = #queries, K = #present GT sets
      - for each GT column kk, pick the query row qi with minimal cost
      - queries can be reused (no one-to-one constraint)
    Returns a python list of length K with values in [0..N-1].
    """
    N, K = cost.shape
    if K == 0 or N == 0:
        return []
    assign: List[int] = []
    for kk in range(K):
        col = cost[:, kk]
        qidx = int(torch.argmin(col).item())  # safe even if N==1
        assign.append(qidx)
    return assign


def _stable_probs(pi: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Sanitize and row-normalize probabilities:
      - replace NaN/Inf
      - clamp to [0,1]
      - renormalize across classes so rows sum to 1
    """
    pi = torch.nan_to_num(pi, nan=0.0, posinf=1.0, neginf=0.0)
    pi = pi.clamp(0.0, 1.0)
    row_sum = pi.sum(dim=1, keepdim=True).clamp_min(eps)
    return pi / row_sum


def nll_with_label_smoothing(logp: torch.Tensor,
                             target_idx: torch.Tensor,
                             eps: float = 0.05,
                             reduction: str = "sum") -> torch.Tensor:
    """
    Version-agnostic NLL with label smoothing implemented on log-probabilities.

    logp       : [V, C] log-probabilities (e.g. Pi.log())
    target_idx : [V] integer targets in [0 .. C-1]
    eps        : smoothing factor; eps=0 -> standard NLL
    reduction  : 'sum' | 'mean' | 'none'
    """
    V, C = logp.shape
    with torch.no_grad():
        y = F.one_hot(target_idx.long(), num_classes=C).to(logp.dtype)  # [V, C]
        y = (1.0 - eps) * y + eps / C
    loss_vec = -(y * logp).sum(dim=1)  # [V]
    if reduction == "sum":
        return loss_vec.sum()
    elif reduction == "mean":
        return loss_vec.mean()
    return loss_vec


# ---------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------

def loss_set_ce_dice(lambda_dice: float = 2.0,
                     label_smoothing: float = 0.05,
                     eps: float = 1e-6):
    """
    Returns a callable loss(batch, out) that combines:
      (A) Set-aware CE on per-voxel (background + queries) after matching GT sets to queries.
      (B) Penalty on 1 - Dice for the matched (query, GT) pairs, averaged per-batch and scaled by lambda_dice.

    Expected model output (per batch element b):
      out["Pi_list"][b] is a [V_b, N+1] tensor of probabilities:
        - column 0 : background
        - columns 1..N : per-query probabilities

    Expected labels (per batch element b):
      batch["meta"][b]["voxel_parent_id"] is [V_b] with values in {0,1,..,R}
        - 0 : background
        - k>0 : belongs to GT set k (e.g., a ring index)
      We only *use* labels that appear in y_b and are <= N (number of model queries).

    Matching:
      - Build cost[qi, kk] = (1 - Dice(p_qi, g_k)) * lambda_dice + BCE(p_qi, g_k)
      - For each GT set kk, pick argmin over qi (queries can be reused).

    Cross-entropy:
      - After matching, form per-voxel targets t in {0..N}:
          t[v] = 0 if y_b[v]==0 (background),
          t[v] = 1+assigned_query_index_of_GTy  if y_b[v] == GTy>0.
      - Compute NLL on log-probs with label smoothing (manual).

    Returns:
      dict(loss=..., loss_ce=..., loss_dice=...)
    """

    def _loss(batch: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "Pi_list" not in out:
            raise KeyError("Expected out['Pi_list'] = list of [V_b, N+1] probabilities per batch element.")

        Pi_list: List[torch.Tensor] = out["Pi_list"]

        ce_sum = torch.tensor(0.0, device=Pi_list[0].device if len(Pi_list) else "cpu")
        dice_sum = torch.tensor(0.0, device=ce_sum.device)
        vox = 0
        B = len(Pi_list)

        for b in range(B):
            Pi_b = Pi_list[b]  # [V_b, N+1]
            if Pi_b.numel() == 0:
                continue

            Pi_b = _stable_probs(Pi_b, eps=eps)             # sanitize & renormalize
            logPi_b = Pi_b.clamp_min(eps).log()             # [V_b, N+1], stable logs

            # Labels
            y_b = batch["meta"][b]["voxel_parent_id"].to(Pi_b.device).long()  # [V_b] in {0,1,...}
            V_b = int(Pi_b.size(0))
            vox += V_b

            # Present GT labels (exclude background=0), capped by available queries N
            N = Pi_b.size(1) - 1  # number of query channels
            if N < 0:
                raise ValueError("Pi_b must have at least one column for background.")

            unique_pos = torch.unique(y_b)
            present_labels: List[int] = [int(k.item()) for k in unique_pos if (k.item() > 0 and k.item() <= N)]

            # If no positive GT present, compute CE vs pure background targets and skip Dice
            if len(present_labels) == 0 or N == 0:
                t_bg = torch.zeros_like(y_b)  # all background
                ce_sum = ce_sum + nll_with_label_smoothing(logPi_b, t_bg, eps=label_smoothing, reduction="sum")
                # Optional: add a constant dice penalty if you want to discourage false positives
                # dice_sum = dice_sum + float(len(present_labels))
                continue

            # Build binary masks g_k for present GT sets
            g_bins: List[torch.Tensor] = [(y_b == k).float() for k in present_labels]
            K = len(g_bins)  # number of present sets

            # Compute matching cost: lambda_dice*(1 - Dice) + BCE
            cost = torch.zeros((N, K), device=Pi_b.device)
            for qi in range(N):
                pq = Pi_b[:, qi + 1].clamp(eps, 1.0 - eps)  # query probability field for this qi
                for kk, gk in enumerate(g_bins):
                    bce = F.binary_cross_entropy(pq, gk, reduction="mean")
                    dsc = dice_coeff(pq, gk, eps=eps)
                    cost[qi, kk] = lambda_dice * (1.0 - dsc) + bce

            assign = smallK_match(cost)  # length K

            # Build per-voxel target t in {0..N} using the matched queries
            t = torch.zeros_like(y_b)
            for kk, klabel in enumerate(present_labels):
                qidx = assign[kk]  # in [0..N-1]
                t = torch.where(y_b == klabel, torch.full_like(t, qidx + 1), t)

            # Cross-entropy on (background + matched queries) with smoothing
            ce_sum = ce_sum + nll_with_label_smoothing(logPi_b, t, eps=label_smoothing, reduction="sum")

            # Dice penalty for matched pairs (average over present sets per batch element)
            dsum = 0.0
            for kk, gk in enumerate(g_bins):
                qidx = assign[kk]
                pq = Pi_b[:, qidx + 1].clamp(eps, 1.0 - eps)
                dsum += (1.0 - dice_coeff(pq, gk, eps=eps))
            dice_sum = dice_sum + dsum

        vox_ = max(vox, 1)
        ce = ce_sum / vox_                   # per-voxel CE
        dice = dice_sum / max(B, 1)          # per-sample average Dice penalty
        total = ce + lambda_dice * dice

        return {"loss": total, "loss_ce": ce, "loss_dice": dice}

    return _loss
