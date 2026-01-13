# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def _centers_from_coords(
    coords: torch.Tensor,
    origin: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    """
    coords: [N, 4] = [batch, z, y, x]
    return centers [N, 3] in world coordinates.
    """
    c = coords.detach().cpu().numpy().astype(np.int32)
    zyx = c[:, 1:4]
    xyz = np.stack([zyx[:, 2], zyx[:, 1], zyx[:, 0]], axis=1).astype(np.float32)
    centers = origin.reshape(1, 3) + (xyz + 0.5) * float(voxel_size)
    return centers


def _build_cube_mesh_uniform(
    centers: np.ndarray,
    size: float,
    color: str,
    name: Optional[str],
    opacity: float,
) -> go.Mesh3d:
    """
    Single color for all cubes (background).
    """
    if centers.size == 0:
        kwargs = dict(
            x=[], y=[], z=[],
            i=[], j=[], k=[],
            color=color,
            opacity=opacity,
            flatshading=True,
        )
        if name is not None:
            kwargs["name"] = name
        return go.Mesh3d(**kwargs)

    h = float(size) * 0.5
    offs = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [h, -h,  h], [h,  h,  h], [-h,  h,  h],
        ],
        dtype=float,
    )
    verts = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    n = centers.shape[0]

    tri = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=int,
    )
    base = (np.arange(n, dtype=int) * 8)[:, None, None]
    tri_all = (base + tri[None, :, :]).reshape(-1, 3)
    i, j, k = tri_all[:, 0], tri_all[:, 1], tri_all[:, 2]

    kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i,
        j=j,
        k=k,
        color=color,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1),
    )
    if name is not None:
        kwargs["name"] = name
    return go.Mesh3d(**kwargs)


def _build_cube_mesh_scalar(
    centers: np.ndarray,
    size: float,
    values: np.ndarray,
    vmin: float,
    vmax: float,
    colorscale,
    name: Optional[str],
    opacity: float,
    showscale: bool = False,
    colorbar_title: Optional[str] = None,
    customdata_voxel: Optional[np.ndarray] = None,
    hovertemplate: Optional[str] = None,
) -> go.Mesh3d:
    """
    Cubes colored by a scalar value in [vmin, vmax].
    Also supports per-voxel customdata for hover.
    """
    if centers.size == 0:
        kwargs = dict(
            x=[], y=[], z=[],
            i=[], j=[], k=[],
            intensity=[],
            colorscale=colorscale,
            opacity=opacity,
            flatshading=True,
            showscale=showscale,
        )
        if showscale and colorbar_title is not None:
            kwargs["colorbar"] = dict(
                title=dict(text=colorbar_title, font=dict(size=14)),
                thickness=40,
                len=0.95,
                x=1.06,
                xpad=60,
                ypad=40,
                tickfont=dict(size=12),
            )
        if customdata_voxel is not None:
            kwargs["customdata"] = []
        if hovertemplate is not None:
            kwargs["hovertemplate"] = hovertemplate
        if name is not None:
            kwargs["name"] = name
        return go.Mesh3d(**kwargs)

    h = float(size) * 0.5
    offs = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [h, -h,  h], [h,  h,  h], [-h,  h,  h],
        ],
        dtype=float,
    )
    verts = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    n = centers.shape[0]

    tri = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=int,
    )
    base = (np.arange(n, dtype=int) * 8)[:, None, None]
    tri_all = (base + tri[None, :, :]).reshape(-1, 3)
    i, j, k = tri_all[:, 0], tri_all[:, 1], tri_all[:, 2]

    vals = np.asarray(values, dtype=float)
    span = max(vmax - vmin, 1e-9)
    vals = (vals - vmin) / span
    vals = np.clip(vals, 0.0, 1.0)

    intensity = np.repeat(vals, 8)

    colorbar = None
    if showscale and colorbar_title is not None:
        colorbar = dict(
            title=dict(text=colorbar_title, font=dict(size=14)),
            thickness=40,
            len=0.95,
            x=1.06,
            xpad=60,
            ypad=40,
            tickfont=dict(size=12),
        )

    kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i,
        j=j,
        k=k,
        intensity=intensity,
        colorscale=colorscale,
        opacity=opacity,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1),
        showscale=showscale,
    )

    if colorbar is not None:
        kwargs["colorbar"] = colorbar

    if customdata_voxel is not None:
        cdv = np.asarray(customdata_voxel)
        if cdv.shape[0] != n:
            raise ValueError("customdata_voxel must have same length as centers")
        # repeat per vertex
        kwargs["customdata"] = np.repeat(cdv, 8, axis=0)

    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate

    if name is not None:
        kwargs["name"] = name

    return go.Mesh3d(**kwargs)


def _build_cube_mesh_vertexcolor(
    centers: np.ndarray,
    size: float,
    rgba_per_voxel: np.ndarray,   # [N,4] floats in [0,1]
    name: Optional[str],
    customdata_voxel: Optional[np.ndarray] = None,
    hovertemplate: Optional[str] = None,
    showscale: bool = False,
) -> go.Mesh3d:
    """
    Cubes colored by per-voxel RGBA (implemented via Mesh3d.vertexcolor).
    """
    if centers.size == 0:
        kwargs = dict(
            x=[], y=[], z=[],
            i=[], j=[], k=[],
            vertexcolor=[],
            flatshading=True,
            showscale=showscale,
        )
        if customdata_voxel is not None:
            kwargs["customdata"] = []
        if hovertemplate is not None:
            kwargs["hovertemplate"] = hovertemplate
        if name is not None:
            kwargs["name"] = name
        return go.Mesh3d(**kwargs)

    h = float(size) * 0.5
    offs = np.array(
        [
            [-h, -h, -h], [h, -h, -h], [h,  h, -h], [-h,  h, -h],
            [-h, -h,  h], [h, -h,  h], [h,  h,  h], [-h,  h,  h],
        ],
        dtype=float,
    )
    verts = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    n = centers.shape[0]

    tri = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [3, 2, 6], [3, 6, 7],
            [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=int,
    )
    base = (np.arange(n, dtype=int) * 8)[:, None, None]
    tri_all = (base + tri[None, :, :]).reshape(-1, 3)
    i, j, k = tri_all[:, 0], tri_all[:, 1], tri_all[:, 2]

    rgba = np.asarray(rgba_per_voxel, dtype=float)
    if rgba.shape != (n, 4):
        raise ValueError("rgba_per_voxel must have shape [N,4] matching centers")

    rgba = np.clip(rgba, 0.0, 1.0)
    r = (rgba[:, 0] * 255.0).astype(np.int32)
    g = (rgba[:, 1] * 255.0).astype(np.int32)
    b = (rgba[:, 2] * 255.0).astype(np.int32)
    a = rgba[:, 3].astype(np.float32)

    col_vox = [f"rgba({rr},{gg},{bb},{aa:.4f})" for rr, gg, bb, aa in zip(r, g, b, a)]
    vertexcolor = np.repeat(np.array(col_vox, dtype=object), 8)

    kwargs = dict(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=i,
        j=j,
        k=k,
        vertexcolor=vertexcolor,
        flatshading=True,
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1),
        showscale=showscale,
    )

    if customdata_voxel is not None:
        cdv = np.asarray(customdata_voxel)
        if cdv.shape[0] != n:
            raise ValueError("customdata_voxel must have same length as centers")
        kwargs["customdata"] = np.repeat(cdv, 8, axis=0)

    if hovertemplate is not None:
        kwargs["hovertemplate"] = hovertemplate

    if name is not None:
        kwargs["name"] = name

    return go.Mesh3d(**kwargs)


def _annotation(
    meta: Dict[str, Any],
    label: str,
    extra: Optional[str],
    x: float,
) -> dict:
    evt = meta.get("event_index", None)
    txt = f"<b>{label}</b>"
    if evt is not None:
        txt += f"<br>Event {evt}"
    if extra:
        txt += "<br>" + extra.replace("\n", "<br>")

    return dict(
        x=x,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=txt,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.65)",
        bordercolor="white",
        borderwidth=1,
        borderpad=8,
        font=dict(color="white", size=12),
    )


def _scene_layout(axis_limit: float) -> dict:
    rng = [-float(axis_limit), float(axis_limit)]
    ax = dict(range=rng, showbackground=False, gridcolor="rgba(255,255,255,0.18)")
    return dict(
        xaxis=ax,
        yaxis=ax,
        zaxis=ax,
        aspectmode="cube",
        bgcolor="black",
    )

def _dice_coeff(prob: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * target_bin).sum()
    den = prob.sum() + target_bin.sum() + eps
    return num / den


def _map_queries_to_parents(
    Pi: torch.Tensor,
    y_b: torch.Tensor,
    lambda_dice: float = 2.0,
) -> Dict[int, int]:
    """
    Map query channels to parent labels {1,2,3,4} by minimizing BCE + Dice cost.
    Pi: [V, N+1], channel 0 is background.
    y_b: [V] integer slots {0,1,2,3,4,...}.
    """
    V = int(Pi.size(0))
    if V == 0:
        return {}
    N = int(Pi.size(1)) - 1
    if N <= 0:
        return {}

    present: List[int] = []
    g_bins: List[torch.Tensor] = []
    for k in (1, 2, 3, 4):
        gk = (y_b == k).float()
        if gk.sum() > 0:
            present.append(k)
            g_bins.append(gk)
    K = len(present)
    if K == 0:
        return {}

    cost = Pi.new_zeros((N, K))
    for qi in range(N):
        pq = Pi[:, qi + 1].clamp(1e-6, 1 - 1e-6)
        for kk, gk in enumerate(g_bins):
            bce = -(gk * torch.log(pq) + (1 - gk) * torch.log(1 - pq)).mean()
            dice = _dice_coeff(pq, gk)
            cost[qi, kk] = lambda_dice * (1.0 - dice) + bce

    assign: Dict[int, int] = {}
    used_q = set()
    used_k = set()

    pairs = []
    for qi in range(N):
        for kk in range(K):
            pairs.append((float(cost[qi, kk].item()), qi, kk))
    pairs.sort(key=lambda x: x[0])

    for _, qi, kk in pairs:
        if qi in used_q or kk in used_k:
            continue
        assign[int(qi)] = int(present[kk])
        used_q.add(qi)
        used_k.add(kk)
        if len(used_k) == K:
            break

    return assign


def _fractions_from_meta_frac(
    voxel_frac: torch.Tensor,
    eps: float = 1e-6,
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    From meta["voxel_parent_frac"] (shape [V,G]) extract:
      p1_norm, p2_norm, ratio = P2/(P1+P2), fg_mask
    Background voxels (no P1+P2) are excluded by fg_mask.
    """
    if voxel_frac.numel() == 0 or voxel_frac.size(1) < 3:
        V = int(voxel_frac.size(0))
        zero = np.zeros((V,), dtype=np.float32)
        fg = np.zeros((V,), dtype=bool)
        return zero, zero, zero, fg

    f = voxel_frac.detach().cpu().float()
    w1 = f[:, 1].clamp(min=0.0)  # P1 charge
    w2 = f[:, 2].clamp(min=0.0)  # P2 charge
    s = w1 + w2

    fg_mask = (s > eps)
    s_safe = torch.where(fg_mask, s, torch.ones_like(s))

    p1 = torch.where(fg_mask, w1 / s_safe, torch.zeros_like(w1))
    p2 = torch.where(fg_mask, w2 / s_safe, torch.zeros_like(w2))
    ratio = p2.clone()  # already == P2/(P1+P2) where fg, 0 elsewhere

    return (
        p1.cpu().numpy().astype(np.float32),
        p2.cpu().numpy().astype(np.float32),
        ratio.cpu().numpy().astype(np.float32),
        fg_mask.cpu().numpy().astype(bool),
    )


def _fractions4_from_meta_frac(
    voxel_frac: torch.Tensor,
    eps: float = 1e-6,
) -> (np.ndarray, np.ndarray):
    """
    From meta["voxel_parent_frac"] (shape [V,G]) extract:
      frac4 = [V,4] (P1..P4 fractions normalized by P1+P2+P3+P4), fg_mask
    Background voxels (no P1+P2+P3+P4) are excluded by fg_mask.
    """
    V = int(voxel_frac.size(0)) if voxel_frac is not None else 0
    if voxel_frac is None or voxel_frac.numel() == 0 or voxel_frac.size(1) < 2:
        frac = np.zeros((V, 4), dtype=np.float32)
        fg = np.zeros((V,), dtype=bool)
        return frac, fg

    f = voxel_frac.detach().cpu().float()
    w = []
    for k in (1, 2, 3, 4):
        if f.size(1) > k:
            w.append(f[:, k].clamp(min=0.0))
        else:
            w.append(torch.zeros((V,), dtype=f.dtype))

    W = torch.stack(w, dim=1)  # [V,4]
    s = W.sum(dim=1)
    fg_mask = (s > eps)
    s_safe = torch.where(fg_mask, s, torch.ones_like(s))
    F = torch.where(fg_mask[:, None], W / s_safe[:, None], torch.zeros_like(W))

    return F.cpu().numpy().astype(np.float32), fg_mask.cpu().numpy().astype(bool)


# =========================
#  Main HTML diagnostic
# =========================
@torch.no_grad()
def save_val_true_pred_split_html(
    model: torch.nn.Module,
    val_subset,
    device: torch.device,
    out_html_path: Path,
    n_events: int = 30,
    start_at: int = 0,
    max_voxels: int = 3000,
    colors: Optional[Dict[str, str]] = None,
    poisson_fluctuate_pred: bool = True,
    poisson_seed: int = 0,
) -> None:
    """
    Left: GT P1..P4 fractions per voxel.
    Right: Predicted P1..P4 fractions per voxel.

    Color scheme (minimal extension to 4 parents):
      - background                          -> light gray
      - P1..P3 encoded as RGB intensities   -> (R,G,B) = (P1,P2,P3), alpha fixed
      - P4 overlay                          -> red cubes with alpha proportional to P4 fraction

    If poisson_fluctuate_pred=True:
      - for each voxel and each predicted parent k:
            Nk ~ Poisson( (pred_frac_k) * (true_total_charge_in_voxel) )
        then pred fractions are replaced by:
            pred_frac_k := Nk / sum_j Nj   (if sum_j Nj > 0 else zeros)
      - this is applied ONLY on the display-side predicted fractions
        (model output is unchanged).
    """
    colors = colors or dict(bg="#d9d9d9", p1="#d62728", p2="#ffdd00")
    was_training = model.training
    model.eval()

    rng = np.random.RandomState(int(poisson_seed))

    ds = getattr(val_subset, "dataset", val_subset)
    indices = getattr(val_subset, "indices", list(range(len(val_subset))))
    start = max(0, int(start_at))
    stop = min(len(indices), start + max(1, int(n_events)))
    chosen = indices[start:stop]

    frames: List[go.Frame] = []
    base_left: Optional[List[go.Mesh3d]] = None
    base_right: Optional[List[go.Mesh3d]] = None

    base_alpha_rgb = 0.98
    base_alpha_p4 = 0.25

    for ev_i, ds_idx in enumerate(chosen):
        sample = ds[ds_idx]
        coords: torch.Tensor = sample["coords"]
        feats: torch.Tensor = sample["feats"]
        meta: Dict[str, Any] = sample["meta"]

        origin = np.asarray(
            meta.get("origin", np.array([-4000.0, -4000.0, -4000.0], dtype=np.float32)),
            dtype=np.float32,
        )
        vsize = float(meta.get("voxel_size", 1.0))
        centers = _centers_from_coords(coords, origin, vsize)
        V = centers.shape[0]

        gt_slots = meta.get("voxel_parent_id", None)
        if isinstance(gt_slots, torch.Tensor):
            gt_slots_np = gt_slots.detach().cpu().numpy().astype(np.int64)
        else:
            gt_slots_np = np.zeros((V,), dtype=np.int64)

        voxel_frac = meta.get("voxel_parent_frac", None)
        if isinstance(voxel_frac, torch.Tensor):
            frac4_gt, fg_mask_gt = _fractions4_from_meta_frac(voxel_frac)
        else:
            frac4_gt = np.zeros((V, 4), dtype=np.float32)
            for kk in (1, 2, 3, 4):
                frac4_gt[gt_slots_np == kk, kk - 1] = 1.0
            fg_mask_gt = (frac4_gt.sum(axis=1) > 0)

        # ---------------------------
        # Predict (P1..P4)
        # ---------------------------
        batch = {
            "coords": coords.clone(),
            "feats": feats.clone(),
            "meta": [meta],
        }
        if batch["coords"].numel():
            batch["coords"][:, 0] = 0
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = model(batch)

        frac4_pr = np.zeros((V, 4), dtype=np.float32)

        if isinstance(out, dict) and ("Pi_list" in out) and len(out["Pi_list"]) > 0:
            Pi: torch.Tensor = out["Pi_list"][0]  # [V, N+1]
            Pi = torch.nan_to_num(Pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            Pi = Pi / Pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

            y_b = torch.as_tensor(gt_slots_np, device=Pi.device, dtype=torch.long)
            q2parent = _map_queries_to_parents(Pi, y_b, lambda_dice=2.0)
            Nq = Pi.size(1) - 1

            w = []
            for lab in (1, 2, 3, 4):
                idxs = [qi + 1 for qi, lb in q2parent.items() if lb == lab]
                if Nq > 0 and len(idxs) > 0:
                    w.append(Pi[:, idxs].sum(dim=1))
                else:
                    w.append(torch.zeros((V,), device=Pi.device))
            W = torch.stack(w, dim=1)  # [V,4]
            s = W.sum(dim=1).clamp_min(1e-6)
            frac4_pr = (W / s[:, None]).detach().cpu().numpy().astype(np.float32)
        else:
            # Fallback: classification output (expects at least 5 classes: bg + 4 parents)
            if isinstance(out, dict) and "probs_list" in out and len(out["probs_list"]) > 0:
                p = out["probs_list"][0]
            elif isinstance(out, dict) and "logits_list" in out and len(out["logits_list"]) > 0:
                z = out["logits_list"][0]
                p = torch.softmax(z, dim=1) if z.numel() else z
            else:
                p = coords.new_zeros((coords.size(0), 5))

            if p.dim() == 2 and p.size(1) >= 5 and p.size(0) == V:
                preds = torch.argmax(p, dim=1).detach().cpu().numpy().astype(np.int64)
                for kk in (1, 2, 3, 4):
                    frac4_pr[preds == kk, kk - 1] = 1.0

        # ---------------------------------------------------------
        # Poisson fluctuation on *predicted* parent contributions:
        # Nk ~ Poisson(pred_frac_k * true_total_charge_in_voxel)
        # ---------------------------------------------------------
        if poisson_fluctuate_pred and V > 0:
            # "true_total_charge_in_voxel" must be provided in meta as voxel_charge_pe
            q_raw = meta.get("voxel_charge_pe", None)
            if isinstance(q_raw, torch.Tensor):
                q_raw_np = q_raw.detach().cpu().numpy().astype(np.float32)
            elif q_raw is not None:
                q_raw_np = np.asarray(q_raw, dtype=np.float32)
            else:
                q_raw_np = None

            if q_raw_np is not None and q_raw_np.shape[0] == V:
                q_raw_np = np.clip(q_raw_np, 0.0, None)
                frac = np.clip(frac4_pr, 0.0, 1.0).astype(np.float32)
                denom = np.clip(frac.sum(axis=1, keepdims=True), 1e-6, None)
                frac = frac / denom  # ensure sums to 1 over the 4 parents

                lam = frac * q_raw_np.reshape(-1, 1)  # [V,4]
                Nk = rng.poisson(lam=lam).astype(np.float32)  # integers as float
                S = Nk.sum(axis=1, keepdims=True)
                frac4_pr = np.where(S > 0.0, Nk / S, 0.0).astype(np.float32)

        # Background vs foreground based on GT fractions
        fg_mask = fg_mask_gt
        bg_mask = ~fg_mask

        centers_bg = centers[bg_mask]
        centers_fg = centers[fg_mask]

        frac4_fg_gt = frac4_gt[fg_mask]
        frac4_fg_pr = frac4_pr[fg_mask]

        # If too many voxels, subsample
        if centers_fg.shape[0] > int(max_voxels):
            keep = np.arange(centers_fg.shape[0])[: int(max_voxels)]
            centers_fg = centers_fg[keep]
            frac4_fg_gt = frac4_fg_gt[keep]
            frac4_fg_pr = frac4_fg_pr[keep]

        # Summary stats for annotation
        if centers_fg.shape[0] > 0:
            only = []
            for k in range(4):
                other = np.delete(frac4_fg_gt, k, axis=1).sum(axis=1)
                only_k = (frac4_fg_gt[:, k] > 0.5) & (other < 0.5)
                only.append(float(only_k.mean()))
        else:
            only = [0.0, 0.0, 0.0, 0.0]

        extra_left = (
            "GT voxel colors:\n"
            f"P1-only voxels: {only[0]*100:.0f}%\n"
            f"P2-only voxels: {only[1]*100:.0f}%\n"
            f"P3-only voxels: {only[2]*100:.0f}%\n"
            f"P4-only voxels: {only[3]*100:.0f}%"
        )
        extra_right = "Pred voxel colors:\n"
        if poisson_fluctuate_pred:
            extra_right += "Poisson on pred contribs × true charge\n"

        # Hovertext
        hover_gt = (
            "GT P1=%{customdata[0]:.2f}<br>"
            "GT P2=%{customdata[1]:.2f}<br>"
            "GT P3=%{customdata[2]:.2f}<br>"
            "GT P4=%{customdata[3]:.2f}<extra></extra>"
        )
        hover_pr = (
            "Pred P1=%{customdata[0]:.2f}<br>"
            "Pred P2=%{customdata[1]:.2f}<br>"
            "Pred P3=%{customdata[2]:.2f}<br>"
            "Pred P4=%{customdata[3]:.2f}<extra></extra>"
        )

        custom_gt = frac4_fg_gt.astype(np.float32)
        custom_pr = frac4_fg_pr.astype(np.float32)

        rgb_gt = np.clip(frac4_fg_gt[:, 0:3], 0.0, 1.0)
        rgb_pr = np.clip(frac4_fg_pr[:, 0:3], 0.0, 1.0)

        rgba_rgb_gt = np.concatenate(
            [rgb_gt, np.full((rgb_gt.shape[0], 1), base_alpha_rgb, dtype=np.float32)],
            axis=1,
        )
        rgba_rgb_pr = np.concatenate(
            [rgb_pr, np.full((rgb_pr.shape[0], 1), base_alpha_rgb, dtype=np.float32)],
            axis=1,
        )

        a4_gt = (base_alpha_p4 * np.clip(frac4_fg_gt[:, 3], 0.0, 1.0)).astype(np.float32)
        a4_pr = (base_alpha_p4 * np.clip(frac4_fg_pr[:, 3], 0.0, 1.0)).astype(np.float32)

        rgba_p4_gt = np.stack(
            [np.ones_like(a4_gt), np.zeros_like(a4_gt), np.zeros_like(a4_gt), a4_gt],
            axis=1,
        )
        rgba_p4_pr = np.stack(
            [np.ones_like(a4_pr), np.zeros_like(a4_pr), np.zeros_like(a4_pr), a4_pr],
            axis=1,
        )

        left_traces: List[go.Mesh3d] = []
        right_traces: List[go.Mesh3d] = []

        left_bg = _build_cube_mesh_uniform(
            centers_bg, vsize, colors["bg"], "GT: Background", opacity=0.25
        )
        right_bg = _build_cube_mesh_uniform(
            centers_bg, vsize, colors["bg"], "Pred: Background", opacity=0.25
        )
        right_bg.update(scene="scene2")
        left_traces.append(left_bg)
        right_traces.append(right_bg)

        left_rgb = _build_cube_mesh_vertexcolor(
            centers_fg,
            vsize,
            rgba_rgb_gt,
            name="GT: RGB (P1,P2,P3)",
            customdata_voxel=custom_gt,
            hovertemplate=hover_gt,
        )
        right_rgb = _build_cube_mesh_vertexcolor(
            centers_fg,
            vsize,
            rgba_rgb_pr,
            name="Pred: RGB (P1,P2,P3)",
            customdata_voxel=custom_pr,
            hovertemplate=hover_pr,
        )
        right_rgb.update(scene="scene2")
        left_traces.append(left_rgb)
        right_traces.append(right_rgb)

        left_p4 = _build_cube_mesh_vertexcolor(
            centers_fg,
            vsize,
            rgba_p4_gt,
            name="GT: P4 overlay",
            customdata_voxel=custom_gt,
            hovertemplate=hover_gt,
        )
        right_p4 = _build_cube_mesh_vertexcolor(
            centers_fg,
            vsize,
            rgba_p4_pr,
            name="Pred: P4 overlay",
            customdata_voxel=custom_pr,
            hovertemplate=hover_pr,
        )
        right_p4.update(scene="scene2")
        left_traces.append(left_p4)
        right_traces.append(right_p4)

        if base_left is None:
            base_left = [
                _build_cube_mesh_uniform(
                    np.zeros((0, 3)), vsize, colors["bg"], "GT: Background", 0.25
                ),
                _build_cube_mesh_vertexcolor(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0, 4), dtype=np.float32),
                    name="GT: RGB (P1,P2,P3)",
                    customdata_voxel=np.zeros((0, 4), dtype=np.float32),
                    hovertemplate=hover_gt,
                ),
                _build_cube_mesh_vertexcolor(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0, 4), dtype=np.float32),
                    name="GT: P4 overlay",
                    customdata_voxel=np.zeros((0, 4), dtype=np.float32),
                    hovertemplate=hover_gt,
                ),
            ]
            base_right = [
                _build_cube_mesh_uniform(
                    np.zeros((0, 3)), vsize, colors["bg"], "Pred: Background", 0.25
                ),
                _build_cube_mesh_vertexcolor(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0, 4), dtype=np.float32),
                    name="Pred: RGB (P1,P2,P3)",
                    customdata_voxel=np.zeros((0, 4), dtype=np.float32),
                    hovertemplate=hover_pr,
                ),
                _build_cube_mesh_vertexcolor(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0, 4), dtype=np.float32),
                    name="Pred: P4 overlay",
                    customdata_voxel=np.zeros((0, 4), dtype=np.float32),
                    hovertemplate=hover_pr,
                ),
            ]
            for tr in base_right:
                tr.update(scene="scene2")

        ann_left = _annotation(meta, "Ground truth", extra_left, x=0.01)
        ann_right = _annotation(meta, "Prediction", extra_right, x=0.51)

        frames.append(
            go.Frame(
                name=f"evt{int(meta.get('event_index', ev_i))}",
                data=left_traces + right_traces,
                traces=[0, 1, 2, 3, 4, 5],
                layout=go.Layout(annotations=[ann_left, ann_right]),
            )
        )

    out_html_path = Path(out_html_path)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)

    if base_left is None:
        with open(out_html_path, "w", encoding="utf-8") as f:
            f.write("<html><body><h3>No validation events to display.</h3></body></html>")
        if was_training:
            model.train()
        return

    try:
        axis_limit = float(ds[chosen[0]]["meta"].get("axis_limit", 4000.0))
    except Exception:
        axis_limit = 4000.0

    layout = go.Layout(
        title="Validation — True (left) vs Predicted (right)",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        margin=dict(l=10, r=220, t=60, b=10),
        scene=_scene_layout(axis_limit),
        scene2=_scene_layout(axis_limit),
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.9)",
            bordercolor="white",
            font=dict(size=12),
            namelength=-1,
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.08,
                x=1.0,
                xanchor="right",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Event: "),
                pad=dict(t=30),
                steps=[
                    dict(
                        label=fr.name,
                        method="animate",
                        args=[
                            [fr.name],
                            dict(
                                mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0),
                            ),
                        ],
                    )
                    for fr in frames
                ],
            )
        ],
    )

    fig = go.Figure(data=(base_left + base_right), frames=frames, layout=layout)
    fig.write_html(str(out_html_path))

    if was_training:
        model.train()


def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return np.nan
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_parent_energies_pdgs(meta: Dict[str, Any], max_parents: int = 4) -> (np.ndarray, np.ndarray):
    """
    Extract per-parent energy [MeV] and PDG ID from meta["parent_info"].
    """
    energies = np.full((max_parents,), np.nan, dtype=float)
    pdgs = np.zeros((max_parents,), dtype=int)

    info = meta.get("parent_info", [])
    if not isinstance(info, list):
        return energies, pdgs

    for i in range(min(len(info), max_parents)):
        pi = info[i] if isinstance(info[i], dict) else {}
        # common variants people use
        e = pi.get("energy_mev", pi.get("E_mev", pi.get("energy", np.nan)))
        energies[i] = float(e) if e is not None else np.nan

        pdg = pi.get("pdg_id", pi.get("pdg", pi.get("pdgid", 0)))
        pdgs[i] = _safe_int(pdg, default=0)

    return energies, pdgs


def _nanmean_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(~np.isfinite(x)):
        return np.nan
    return float(np.nanmean(x))


def _nanmin_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(~np.isfinite(x)):
        return np.nan
    return float(np.nanmin(x))


def _nanmax_safe(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(~np.isfinite(x)):
        return np.nan
    return float(np.nanmax(x))


@torch.no_grad()
def collect_val_event_stats(
    model,
    criterion,            
    val_subset,
    device: torch.device,
    max_events: Optional[int] = None,
    start_at: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Collect per-event diagnostics:

    - angle between parents [deg]
    - event-level mean Dice score (0..1) over present rings
    - min/max parent energy (MeV)
    """
    model_was_train = model.training
    model.eval()

    ds = getattr(val_subset, "dataset", val_subset)
    indices = getattr(val_subset, "indices", list(range(len(val_subset))))
    start = max(0, int(start_at))
    stop = len(indices) if max_events is None else min(len(indices), start + int(max_events))

    angle_list: List[float] = []
    metric_list: List[float] = []
    emin_list: List[float] = []
    emax_list: List[float] = []

    true_rings: List[int] = []
    pred_rings: List[int] = []

    eavg_list: List[float] = []
    ediff_list: List[float] = []
    pdg_maxE_list: List[int] = []
    pred_rings_gt1_list: List[int] = []

    for pos in range(start, stop):
        i = indices[pos]
        sample = ds[i]
        coords = sample["coords"].clone()
        feats = sample["feats"].clone()
        meta = dict(sample["meta"])

        V = coords.size(0)
        if V == 0:
            continue

        # ----- GT slots & fractions -----
        gt_slots = meta.get("voxel_parent_id", None)
        if isinstance(gt_slots, torch.Tensor):
            gt_slots_np = gt_slots.detach().cpu().numpy().astype(np.int64)
        else:
            gt_slots_np = np.zeros((V,), dtype=np.int64)

        voxel_frac = meta.get("voxel_parent_frac", None)
        if isinstance(voxel_frac, torch.Tensor):
            frac4_gt, fg_mask_gt = _fractions4_from_meta_frac(voxel_frac)
        else:
            frac4_gt = np.zeros((V, 4), dtype=np.float32)
            for kk in (1, 2, 3, 4):
                frac4_gt[gt_slots_np == kk, kk - 1] = 1.0
            fg_mask_gt = (frac4_gt.sum(axis=1) > 0)

        frac4_pr = np.zeros((V, 4), dtype=np.float32)

        if coords.numel():
            coords[:, 0] = 0
        batch = {"coords": coords.to(device), "feats": feats.to(device), "meta": [meta]}
        out = model(batch)

        if isinstance(out, dict) and ("Pi_list" in out) and len(out["Pi_list"]) > 0:
            Pi: torch.Tensor = out["Pi_list"][0]  # [V, N+1]
            Pi = torch.nan_to_num(Pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            Pi = Pi / Pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

            y_b = torch.as_tensor(gt_slots_np, device=Pi.device, dtype=torch.long)
            q2parent = _map_queries_to_parents(Pi, y_b, lambda_dice=2.0)
            Nq = Pi.size(1) - 1

            w = []
            for lab in (1, 2, 3, 4):
                idxs = [qi + 1 for qi, lb in q2parent.items() if lb == lab]
                if Nq > 0 and len(idxs) > 0:
                    w.append(Pi[:, idxs].sum(dim=1))
                else:
                    w.append(torch.zeros((V,), device=Pi.device))
            W = torch.stack(w, dim=1)  # [V,4]
            s = W.sum(dim=1).clamp_min(1e-6)
            frac4_pr = (W / s[:, None]).detach().cpu().numpy().astype(np.float32)
        else:
            if isinstance(out, dict) and "probs_list" in out and len(out["probs_list"]) > 0:
                p = out["probs_list"][0]
            elif isinstance(out, dict) and "logits_list" in out and len(out["logits_list"]) > 0:
                z = out["logits_list"][0]
                p = torch.softmax(z, dim=1) if z.numel() else z
            else:
                p = coords.new_zeros((coords.size(0), 5))

            if p.dim() == 2 and p.size(1) >= 5 and p.size(0) == V:
                preds = torch.argmax(p, dim=1).detach().cpu().numpy().astype(np.int64)
                for kk in (1, 2, 3, 4):
                    frac4_pr[preds == kk, kk - 1] = 1.0

        # ----- Ring-count (per event)  -----
        fg_mask = fg_mask_gt
        if not np.any(fg_mask):
            tcount = 0
            pcount = 0
        else:
            gt_fg = frac4_gt[fg_mask]   # [Vfg,4]
            pr_fg = frac4_pr[fg_mask]   # [Vfg,4]

            gt_has = [(gt_fg[:, k] > 0.5).any() for k in range(4)]
            pr_has = [(pr_fg[:, k] > 0.5).any() for k in range(4)]

            tcount = int(np.sum(gt_has))
            pcount = int(np.sum(pr_has))

        true_rings.append(tcount)
        pred_rings.append(pcount)

        info = meta.get("parent_info", [])
        if not isinstance(info, list) or len(info) < 2:
            continue

        d1 = np.asarray(info[0].get("dir", (0.0, 0.0, 0.0)), dtype=float)
        d2 = np.asarray(info[1].get("dir", (0.0, 0.0, 0.0)), dtype=float)
        ang = _angle_between_deg(d1, d2)
        if not np.isfinite(ang):
            continue

        e1 = float(info[0].get("energy_mev", 0.0))
        e2 = float(info[1].get("energy_mev", 0.0))
        e_min, e_max = min(e1, e2), max(e1, e2)

        if np.any(fg_mask):
            gt_fg = frac4_gt[fg_mask]
            pr_fg = frac4_pr[fg_mask]

            dice_vals: List[float] = []
            for k in range(4):
                gt_bin = torch.from_numpy((gt_fg[:, k] > 0.5).astype(np.float32))
                pr_bin = torch.from_numpy((pr_fg[:, k] > 0.5).astype(np.float32))
                if gt_bin.sum() > 0:
                    dice_vals.append(float(_dice_coeff(pr_bin, gt_bin).item()))

            if len(dice_vals) == 0:
                continue
            mean_dice = float(np.mean(dice_vals))
        else:
            continue

        energies4, pdgs4 = _extract_parent_energies_pdgs(meta, max_parents=4)
        eavg = _nanmean_safe(energies4)
        emin_meta = _nanmin_safe(energies4)
        emax_meta = _nanmax_safe(energies4)
        ediff = (emax_meta - emin_meta) if (np.isfinite(emax_meta) and np.isfinite(emin_meta)) else np.nan

        if np.isfinite(emax_meta):
            imax = int(np.nanargmax(energies4))
            pdg_maxE = int(pdgs4[imax])
        else:
            pdg_maxE = 0

        angle_list.append(ang)
        metric_list.append(mean_dice)
        emin_list.append(e_min)
        emax_list.append(e_max)

        eavg_list.append(eavg)
        ediff_list.append(ediff)
        pdg_maxE_list.append(pdg_maxE)
        pred_rings_gt1_list.append(int(pcount > 1))

    if model_was_train:
        model.train()

    angle_arr = np.asarray(angle_list, dtype=float)
    metric_arr = np.asarray(metric_list, dtype=float)
    emin_arr = np.asarray(emin_list, dtype=float)
    emax_arr = np.asarray(emax_list, dtype=float)
    true_rings_arr = np.asarray(true_rings, dtype=int)
    pred_rings_arr = np.asarray(pred_rings, dtype=int)

    eavg_arr = np.asarray(eavg_list, dtype=float)
    ediff_arr = np.asarray(ediff_list, dtype=float)
    pdg_maxE_arr = np.asarray(pdg_maxE_list, dtype=int)
    pred_rings_gt1_arr = np.asarray(pred_rings_gt1_list, dtype=int)

    return {
        "loss": metric_arr,
        "dice_mean": metric_arr,
        "angle_deg": angle_arr,
        "emin": emin_arr,
        "emax": emax_arr,
        "n_true_rings": true_rings_arr,
        "n_pred_rings": pred_rings_arr,

        "eavg": eavg_arr,
        "ediff": ediff_arr,
        "pdg_maxE": pdg_maxE_arr,
        "pred_rings_gt1": pred_rings_gt1_arr,
    }


def save_val_angle_loss_png(
    out_path: Path,
    angle_deg: np.ndarray,
    losses: np.ndarray,
    nbins: int = 18,
) -> None:
    """
    Plot mean Dice score (0..1) vs opening angle.
    The 'losses' argument is interpreted as a metric in [0,1] (e.g. mean Dice).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if angle_deg.size == 0 or losses.size == 0:
        plt.figure()
        plt.title("Dice vs opening angle (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    bins = np.linspace(0.0, 180.0, nbins + 1, dtype=float)
    idx = np.digitize(angle_deg, bins) - 1
    valid = (idx >= 0) & (idx < nbins) & np.isfinite(losses)
    idx = idx[valid]
    w = losses[valid]

    sum_w = np.bincount(idx, weights=w, minlength=nbins).astype(float)
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    mean = sum_w / np.maximum(cnt, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure()
    plt.bar(centers, mean, width=(bins[1] - bins[0]) * 0.9, align="center")
    plt.xlabel("Angle between parents [deg]")
    plt.ylabel("Mean Dice score")
    plt.ylim(0.0, 1.0)
    plt.title("Segmentation quality vs opening angle")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_val_energy2d_loss_png(
    out_path: Path,
    emin: np.ndarray,
    emax: np.ndarray,
    losses: np.ndarray,
    nbins_x: int = 20,
    nbins_y: int = 20,
) -> None:
    """
    2D heatmap of metric (currently mean Dice) vs (E_min, E_max).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if emin.size == 0 or emax.size == 0:
        plt.figure()
        plt.title("Emax vs Emin (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    ex = np.asarray(emax, dtype=float)
    en = np.asarray(emin, dtype=float)
    lw = np.asarray(losses, dtype=float)

    m = np.isfinite(ex) & np.isfinite(en) & np.isfinite(lw) & (ex >= en)
    ex = ex[m]
    en = en[m]
    lw = lw[m]
    if ex.size == 0:
        plt.figure()
        plt.title("Emax vs Emin (no valid points)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    x_low, x_high = np.percentile(ex, [1.0, 99.0])
    y_low, y_high = np.percentile(en, [1.0, 99.0])
    xbins = np.linspace(x_low, x_high, nbins_x + 1)
    ybins = np.linspace(y_low, y_high, nbins_y + 1)
    H_sum, _, _ = np.histogram2d(en, ex, bins=[ybins, xbins], weights=lw)
    H_cnt, _, _ = np.histogram2d(en, ex, bins=[ybins, xbins])
    H_mean = H_sum / np.maximum(H_cnt, 1.0)

    plt.figure()
    X, Y = np.meshgrid(xbins, ybins)
    mesh = plt.pcolormesh(X, Y, H_mean, shading="auto")
    plt.xlabel("E_max [MeV]")
    plt.ylabel("E_min [MeV]")
    plt.title("Mean Dice heatmap: E_max vs E_min")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Mean Dice score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_val_ring_count_confusion_png(
    out_path: Path,
    n_true_rings: np.ndarray,
    n_pred_rings: np.ndarray,
    normalize: bool = True,
) -> None:
    """
    Plot an N×N confusion matrix for the number of rings (per event).

    n_true_rings, n_pred_rings: arrays of integers (0,1,2,...).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_true_rings = np.asarray(n_true_rings, dtype=int)
    n_pred_rings = np.asarray(n_pred_rings, dtype=int)

    if n_true_rings.size == 0:
        plt.figure()
        plt.title("Ring-count confusion (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    max_class = int(max(n_true_rings.max(), n_pred_rings.max()))
    K = max_class + 1

    cm = np.zeros((K, K), dtype=float)
    for t, p in zip(n_true_rings, n_pred_rings):
        if 0 <= t < K and 0 <= p < K:
            cm[t, p] += 1.0

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = cm / np.maximum(row_sums, 1.0)
    else:
        cm_display = cm

    fig, ax = plt.subplots()
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    plt.title("Ring-count confusion matrix")
    plt.xlabel("Predicted # rings")
    plt.ylabel("True # rings")
    fig.colorbar(im, ax=ax, label="Fraction" if normalize else "Count")

    tick_marks = np.arange(K)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([str(i) for i in range(K)])
    ax.set_yticklabels([str(i) for i in range(K)])

    # Write numbers in cells
    fmt = ".3f" if normalize else "d"
    thresh = cm_display.max() / 2.0 if cm_display.size > 0 else 0.0
    for i in range(K):
        for j in range(K):
            val = cm_display[i, j]
            if normalize and cm.sum() == 0:
                text_str = "0.00"
            else:
                text_str = format(val, fmt)
            ax.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
#  NEW diagnostics plots
# =========================

def save_val_avg_energy_loss_png(
    out_path: Path,
    eavg: np.ndarray,
    losses: np.ndarray,
    nbins: int = 20,
) -> None:
    """
    Plot mean Dice score vs average parent energy (MeV).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eavg = np.asarray(eavg, dtype=float)
    losses = np.asarray(losses, dtype=float)

    m = np.isfinite(eavg) & np.isfinite(losses)
    eavg = eavg[m]
    losses = losses[m]
    if eavg.size == 0:
        plt.figure()
        plt.title("Dice vs average parent energy (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    lo, hi = np.percentile(eavg, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(eavg)), float(np.max(eavg) + 1e-6)

    bins = np.linspace(lo, hi, nbins + 1, dtype=float)
    idx = np.digitize(eavg, bins) - 1
    valid = (idx >= 0) & (idx < nbins)
    idx = idx[valid]
    w = losses[valid]

    sum_w = np.bincount(idx, weights=w, minlength=nbins).astype(float)
    cnt = np.bincount(idx, minlength=nbins).astype(float)
    mean = sum_w / np.maximum(cnt, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure()
    plt.bar(centers, mean, width=(bins[1] - bins[0]) * 0.9, align="center")
    plt.xlabel("Average parent energy [MeV]")
    plt.ylabel("Mean Dice score")
    plt.ylim(0.0, 1.0)
    plt.title("Segmentation quality vs average parent energy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_val_pdg_loss_png(
    out_path: Path,
    pdg: np.ndarray,
    losses: np.ndarray,
    top_k: int = 12,
    min_count: int = 5,
) -> None:
    """
    Plot mean Dice per PDG ID (PDG of max-energy parent).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdg = np.asarray(pdg, dtype=int)
    losses = np.asarray(losses, dtype=float)

    m = np.isfinite(losses)
    pdg = pdg[m]
    losses = losses[m]
    if losses.size == 0:
        plt.figure()
        plt.title("Dice vs PDG (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    unique = np.unique(pdg)

    filtered_unique: List[int] = []
    filtered_counts: List[int] = []
    filtered_means: List[float] = []

    for u in unique:
        mask = (pdg == u)
        c = int(mask.sum())
        if c < int(min_count):
            continue
        filtered_unique.append(int(u))
        filtered_counts.append(c)
        filtered_means.append(float(np.mean(losses[mask])))

    if len(filtered_unique) == 0:
        plt.figure()
        plt.title("Dice vs PDG (no groups with enough stats)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    filtered_counts_np = np.asarray(filtered_counts, dtype=int)
    order = np.argsort(-filtered_counts_np)[: int(top_k)]

    sel_pdgs = [filtered_unique[i] for i in order]
    sel_means = [filtered_means[i] for i in order]
    sel_counts = [filtered_counts[i] for i in order]

    x = np.arange(len(sel_pdgs))

    plt.figure(figsize=(max(6, len(sel_pdgs) * 0.8), 4))
    plt.bar(x, sel_means)
    plt.xticks(x, [f"{p}\n(n={c})" for p, c in zip(sel_pdgs, sel_counts)], rotation=0)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Mean Dice score")
    plt.xlabel("PDG of max-energy parent")
    plt.title("Segmentation quality vs PDG (max-energy parent)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_val_energy2d_diff_avg_loss_png(
    out_path: Path,
    ediff: np.ndarray,
    eavg: np.ndarray,
    losses: np.ndarray,
    pred_rings_gt1: np.ndarray,
    nbins_x: int = 20,
    nbins_y: int = 20,
) -> None:
    """
    2D heatmap of metric (mean Dice) vs (E_max - E_min, E_avg),
    restricted to events where predicted #rings > 1.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ediff = np.asarray(ediff, dtype=float)
    eavg = np.asarray(eavg, dtype=float)
    losses = np.asarray(losses, dtype=float)
    pred_rings_gt1 = np.asarray(pred_rings_gt1, dtype=int)

    m = (pred_rings_gt1 > 0) & np.isfinite(ediff) & np.isfinite(eavg) & np.isfinite(losses) & (ediff >= 0.0)
    ediff = ediff[m]
    eavg = eavg[m]
    losses = losses[m]

    if ediff.size == 0:
        plt.figure()
        plt.title("Dice heatmap vs (Emax-Emin, Eavg) for n_pred>1 (no data)")
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    x_low, x_high = np.percentile(ediff, [1.0, 99.0])
    y_low, y_high = np.percentile(eavg, [1.0, 99.0])

    if not np.isfinite(x_low) or not np.isfinite(x_high) or x_high <= x_low:
        x_low, x_high = float(np.min(ediff)), float(np.max(ediff) + 1e-6)
    if not np.isfinite(y_low) or not np.isfinite(y_high) or y_high <= y_low:
        y_low, y_high = float(np.min(eavg)), float(np.max(eavg) + 1e-6)

    xbins = np.linspace(x_low, x_high, nbins_x + 1)
    ybins = np.linspace(y_low, y_high, nbins_y + 1)

    H_sum, _, _ = np.histogram2d(eavg, ediff, bins=[ybins, xbins], weights=losses)
    H_cnt, _, _ = np.histogram2d(eavg, ediff, bins=[ybins, xbins])
    H_mean = H_sum / np.maximum(H_cnt, 1.0)

    plt.figure()
    X, Y = np.meshgrid(xbins, ybins)
    mesh = plt.pcolormesh(X, Y, H_mean, shading="auto")
    plt.xlabel("E_max - E_min [MeV]")
    plt.ylabel("Average parent energy [MeV]")
    plt.title("Mean Dice heatmap vs energy spread and average (pred rings > 1)")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Mean Dice score")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
