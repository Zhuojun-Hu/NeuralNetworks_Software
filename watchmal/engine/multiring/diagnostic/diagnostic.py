# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# =========================
#  Geometry helpers
# =========================

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


# =========================
#  Parent mapping + fractions
# =========================

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
    Map query channels to parent labels {1,2} by minimizing BCE + Dice cost.
    Pi: [V, N+1], channel 0 is background.
    y_b: [V] integer slots {0,1,2}.
    """
    V = int(Pi.size(0))
    if V == 0:
        return {}
    N = int(Pi.size(1)) - 1
    if N <= 0:
        return {}

    present: List[int] = []
    g_bins: List[torch.Tensor] = []
    for k in (1, 2):
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
    for kk, klabel in enumerate(present):
        qidx = int(torch.argmin(cost[:, kk]).item())
        assign[qidx] = int(klabel)
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
) -> None:
    """
    Left: GT P1/P2 fractions per voxel.
    Right: Predicted P1/P2 fractions per voxel.
      - background  (no P1+P2)      -> light gray
      - only P1     (P2 ~ 0)        -> red
      - only P2     (P1 ~ 0)        -> yellow
      - both P1,P2  -> gradient red (P1) -> yellow (P2)
    """
    colors = colors or dict(bg="#d9d9d9", p1="#d62728", p2="#ffdd00")
    was_training = model.training
    model.eval()

    ds = getattr(val_subset, "dataset", val_subset)
    indices = getattr(val_subset, "indices", list(range(len(val_subset))))
    start = max(0, int(start_at))
    stop = min(len(indices), start + max(1, int(n_events)))
    chosen = indices[start:stop]

    frames: List[go.Frame] = []
    base_left: Optional[List[go.Mesh3d]] = None
    base_right: Optional[List[go.Mesh3d]] = None

    # red -> yellow for P2 fraction (0=P1, 1=P2)
    colorscale = [
        [0.0, colors["p1"]],
        [1.0, colors["p2"]],
    ]

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
        axis_limit = float(meta.get("axis_limit", 4000.0))
        centers = _centers_from_coords(coords, origin, vsize)
        V = centers.shape[0]

        # Ground-truth slots (0 = bg, 1=P1, 2=P2)
        gt_slots = meta.get("voxel_parent_id", None)
        if isinstance(gt_slots, torch.Tensor):
            gt_slots_np = gt_slots.detach().cpu().numpy().astype(np.int64)
        else:
            gt_slots_np = np.zeros((V,), dtype=np.int64)

        # Ground-truth fractions
        voxel_frac = meta.get("voxel_parent_frac", None)
        if isinstance(voxel_frac, torch.Tensor):
            p1_gt, p2_gt, ratio_gt, fg_mask_gt = _fractions_from_meta_frac(voxel_frac)
        else:
            # Fallback: discrete labels
            p1_gt = np.zeros((V,), dtype=np.float32)
            p2_gt = np.zeros((V,), dtype=np.float32)
            p1_gt[gt_slots_np == 1] = 1.0
            p2_gt[gt_slots_np == 2] = 1.0
            s = p1_gt + p2_gt
            fg_mask_gt = s > 0
            ratio_gt = np.zeros((V,), dtype=np.float32)
            ratio_gt[s > 0] = p2_gt[s > 0] / s[s > 0]

        # Predicted fractions: try Pi_list (queries) first
        batch = {
            "coords": coords.clone(),
            "feats": feats.clone(),
            "meta": [meta],
        }
        if batch["coords"].numel():
            batch["coords"][:, 0] = 0  # batch index = 0
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = model(batch)

        if isinstance(out, dict) and ("Pi_list" in out) and len(out["Pi_list"]) > 0:
            Pi: torch.Tensor = out["Pi_list"][0]  # [V, N+1]
            Pi = torch.nan_to_num(Pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            Pi = Pi / Pi.sum(dim=1, keepdim=True).clamp_min(1e-6)

            y_b = torch.as_tensor(gt_slots_np, device=Pi.device, dtype=torch.long)
            q2parent = _map_queries_to_parents(Pi, y_b, lambda_dice=2.0)
            Nq = Pi.size(1) - 1

            if Nq > 0 and q2parent:
                p1_idx = [qi + 1 for qi, lab in q2parent.items() if lab == 1]
                p2_idx = [qi + 1 for qi, lab in q2parent.items() if lab == 2]

                if len(p1_idx) == 0 and len(p2_idx) == 0:
                    p1_pr = torch.full((V,), 0.5, device=Pi.device)
                    p2_pr = torch.full((V,), 0.5, device=Pi.device)
                else:
                    w1 = Pi[:, p1_idx].sum(dim=1) if len(p1_idx) > 0 else torch.zeros((V,), device=Pi.device)
                    w2 = Pi[:, p2_idx].sum(dim=1) if len(p2_idx) > 0 else torch.zeros((V,), device=Pi.device)
                    s_pr = (w1 + w2).clamp_min(1e-6)
                    p1_pr = w1 / s_pr
                    p2_pr = w2 / s_pr
            else:
                p1_pr = torch.full((V,), 0.5, device=Pi.device)
                p2_pr = torch.full((V,), 0.5, device=Pi.device)

            p1_pr_np = p1_pr.detach().cpu().numpy().astype(np.float32)
            p2_pr_np = p2_pr.detach().cpu().numpy().astype(np.float32)
            ratio_pr_np = p2_pr_np.copy()
        else:
            # Fallback: use classification (0=bg,1=P1,2=P2)
            if isinstance(out, dict) and "probs_list" in out and len(out["probs_list"]) > 0:
                p = out["probs_list"][0]  # [V,3]
            elif isinstance(out, dict) and "logits_list" in out and len(out["logits_list"]) > 0:
                z = out["logits_list"][0]
                p = torch.softmax(z, dim=1) if z.numel() else z
            else:
                p = coords.new_zeros((coords.size(0), 3))

            preds = torch.argmax(p, dim=1)
            preds_np = preds.detach().cpu().numpy().astype(np.int64)
            p1_pr_np = np.zeros((V,), dtype=np.float32)
            p2_pr_np = np.zeros((V,), dtype=np.float32)
            p1_pr_np[preds_np == 1] = 1.0
            p2_pr_np[preds_np == 2] = 1.0
            s_pr = p1_pr_np + p2_pr_np
            ratio_pr_np = np.zeros((V,), dtype=np.float32)
            ratio_pr_np[s_pr > 0] = p2_pr_np[s_pr > 0] / s_pr[s_pr > 0]

        # Background vs foreground based on GT fractions
        fg_mask = fg_mask_gt
        bg_mask = ~fg_mask

        centers_bg = centers[bg_mask]
        centers_fg = centers[fg_mask]

        p1_fg_gt = p1_gt[fg_mask]
        p2_fg_gt = p2_gt[fg_mask]
        ratio_fg_gt = ratio_gt[fg_mask]

        p1_fg_pr = p1_pr_np[fg_mask]
        p2_fg_pr = p2_pr_np[fg_mask]
        ratio_fg_pr = ratio_pr_np[fg_mask]

        # If too many voxels, subsample
        if centers_fg.shape[0] > int(max_voxels):
            keep = np.arange(centers_fg.shape[0])[: int(max_voxels)]
            centers_fg = centers_fg[keep]
            p1_fg_gt = p1_fg_gt[keep]
            p2_fg_gt = p2_fg_gt[keep]
            ratio_fg_gt = ratio_fg_gt[keep]
            p1_fg_pr = p1_fg_pr[keep]
            p2_fg_pr = p2_fg_pr[keep]
            ratio_fg_pr = ratio_fg_pr[keep]

        # Summary stats for annotation
        if ratio_fg_gt.size > 0:
            only_p1 = (p2_fg_gt < 1e-3).sum()
            only_p2 = (p1_fg_gt < 1e-3).sum()
            n = float(ratio_fg_gt.size)
            frac_only_p1 = only_p1 / n
            frac_only_p2 = only_p2 / n
        else:
            frac_only_p1 = 0.0
            frac_only_p2 = 0.0

        extra_left = (
            "GT voxel colors:\n"
            "background = light gray\n"
            "P1 only     = red\n"
            "P2 only     = yellow\n"
            "mix         = red→yellow\n"
            f"P1-only voxels: {frac_only_p1*100:.0f}%\n"
            f"P2-only voxels: {frac_only_p2*100:.0f}%"
        )
        extra_right = (
            "Pred voxel colors:\n"
            "same scheme as GT\n"
            "based on predicted P1/P2"
        )

        # Hovertext
        hover_gt = (
            "GT P1=%{customdata[0]:.2f}<br>"
            "GT P2=%{customdata[1]:.2f}<br>"
            "GT P2/(P1+P2)=%{customdata[2]:.2f}<extra></extra>"
        )
        hover_pr = (
            "Pred P1=%{customdata[0]:.2f}<br>"
            "Pred P2=%{customdata[1]:.2f}<br>"
            "Pred P2/(P1+P2)=%{customdata[2]:.2f}<extra></extra>"
        )

        custom_gt = np.stack([p1_fg_gt, p2_fg_gt, ratio_fg_gt], axis=1)
        custom_pr = np.stack([p1_fg_pr, p2_fg_pr, ratio_fg_pr], axis=1)

        # Traces for this frame
        left_traces: List[go.Mesh3d] = []
        right_traces: List[go.Mesh3d] = []

        # Background (same both sides)
        left_bg = _build_cube_mesh_uniform(
            centers_bg, vsize, colors["bg"], "GT: Background", opacity=0.25
        )
        right_bg = _build_cube_mesh_uniform(
            centers_bg, vsize, colors["bg"], "Pred: Background", opacity=0.25
        )
        right_bg.update(scene="scene2")
        left_traces.append(left_bg)
        right_traces.append(right_bg)

        # Foreground voxels colored by P2 fraction
        left_fg = _build_cube_mesh_scalar(
            centers_fg,
            vsize,
            ratio_fg_gt,
            vmin=0.0,
            vmax=1.0,
            colorscale=colorscale,
            name="GT: P2/(P1+P2)",
            opacity=0.98,
            showscale=True,
            colorbar_title="P2/(P1+P2)",
            customdata_voxel=custom_gt,
            hovertemplate=hover_gt,
        )
        right_fg = _build_cube_mesh_scalar(
            centers_fg,
            vsize,
            ratio_fg_pr,
            vmin=0.0,
            vmax=1.0,
            colorscale=colorscale,
            name="Pred: P2/(P1+P2)",
            opacity=0.98,
            showscale=False,
            colorbar_title=None,
            customdata_voxel=custom_pr,
            hovertemplate=hover_pr,
        )
        right_fg.update(scene="scene2")

        left_traces.append(left_fg)
        right_traces.append(right_fg)

        if base_left is None:
            # Base traces with zero vertices (for initial figure)
            base_left = [
                _build_cube_mesh_uniform(
                    np.zeros((0, 3)), vsize, colors["bg"], "GT: Background", 0.25
                ),
                _build_cube_mesh_scalar(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0,), dtype=float),
                    vmin=0.0,
                    vmax=1.0,
                    colorscale=colorscale,
                    name="GT: P2/(P1+P2)",
                    opacity=0.98,
                    showscale=True,
                    colorbar_title="P2/(P1+P2)",
                    customdata_voxel=np.zeros((0, 3), dtype=float),
                    hovertemplate=hover_gt,
                ),
            ]
            base_right = [
                _build_cube_mesh_uniform(
                    np.zeros((0, 3)), vsize, colors["bg"], "Pred: Background", 0.25
                ),
                _build_cube_mesh_scalar(
                    np.zeros((0, 3)),
                    vsize,
                    np.zeros((0,), dtype=float),
                    vmin=0.0,
                    vmax=1.0,
                    colorscale=colorscale,
                    name="Pred: P2/(P1+P2)",
                    opacity=0.98,
                    showscale=False,
                    colorbar_title=None,
                    customdata_voxel=np.zeros((0, 3), dtype=float),
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
                traces=[0, 1, 2, 3],
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


# =========================
#  Angle / ring-count diagnostics
# =========================

def _angle_between_deg(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return np.nan
    c = float(np.clip((u @ v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


@torch.no_grad()
def collect_val_event_stats(
    model,
    criterion,            # kept for API, not used now
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
    - predicted vs true number of rings (0,1,2)
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
            p1_gt, p2_gt, ratio_gt, fg_mask_gt = _fractions_from_meta_frac(voxel_frac)
        else:
            p1_gt = np.zeros((V,), dtype=np.float32)
            p2_gt = np.zeros((V,), dtype=np.float32)
            p1_gt[gt_slots_np == 1] = 1.0
            p2_gt[gt_slots_np == 2] = 1.0
            s = p1_gt + p2_gt
            fg_mask_gt = s > 0
            ratio_gt = np.zeros((V,), dtype=np.float32)
            ratio_gt[s > 0] = p2_gt[s > 0] / s[s > 0]

        # ----- Predictions (P1/P2 fractions) -----
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

            if Nq > 0 and q2parent:
                p1_idx = [qi + 1 for qi, lab in q2parent.items() if lab == 1]
                p2_idx = [qi + 1 for qi, lab in q2parent.items() if lab == 2]

                if len(p1_idx) == 0 and len(p2_idx) == 0:
                    p1_pr = torch.full((V,), 0.5, device=Pi.device)
                    p2_pr = torch.full((V,), 0.5, device=Pi.device)
                else:
                    w1 = Pi[:, p1_idx].sum(dim=1) if len(p1_idx) > 0 else torch.zeros((V,), device=Pi.device)
                    w2 = Pi[:, p2_idx].sum(dim=1) if len(p2_idx) > 0 else torch.zeros((V,), device=Pi.device)
                    s_pr = (w1 + w2).clamp_min(1e-6)
                    p1_pr = w1 / s_pr
                    p2_pr = w2 / s_pr
            else:
                p1_pr = torch.full((V,), 0.5, device=Pi.device)
                p2_pr = torch.full((V,), 0.5, device=Pi.device)

            p1_pr_np = p1_pr.detach().cpu().numpy().astype(np.float32)
            p2_pr_np = p2_pr.detach().cpu().numpy().astype(np.float32)
        else:
            # classification fallback
            if isinstance(out, dict) and "probs_list" in out and len(out["probs_list"]) > 0:
                p = out["probs_list"][0]  # [V,3]
            elif isinstance(out, dict) and "logits_list" in out and len(out["logits_list"]) > 0:
                z = out["logits_list"][0]
                p = torch.softmax(z, dim=1) if z.numel() else z
            else:
                p = coords.new_zeros((coords.size(0), 3))

            preds = torch.argmax(p, dim=1)
            preds_np = preds.detach().cpu().numpy().astype(np.int64)
            p1_pr_np = np.zeros((V,), dtype=np.float32)
            p2_pr_np = np.zeros((V,), dtype=np.float32)
            p1_pr_np[preds_np == 1] = 1.0
            p2_pr_np[preds_np == 2] = 1.0

        # ----- Ring-count (per event)  -----
        fg_mask = fg_mask_gt
        if not np.any(fg_mask):
            # no foreground voxels → treat as 0 rings
            true_rings.append(0)
            pred_rings.append(0)
        else:
            p1_fg_gt = p1_gt[fg_mask]
            p2_fg_gt = p2_gt[fg_mask]
            p1_fg_pr = p1_pr_np[fg_mask]
            p2_fg_pr = p2_pr_np[fg_mask]

            gt_bin_p1 = p1_fg_gt > 0.5
            gt_bin_p2 = p2_fg_gt > 0.5
            pr_bin_p1 = p1_fg_pr > 0.5
            pr_bin_p2 = p2_fg_pr > 0.5

            true_has_p1 = bool(gt_bin_p1.any())
            true_has_p2 = bool(gt_bin_p2.any())
            pred_has_p1 = bool(pr_bin_p1.any())
            pred_has_p2 = bool(pr_bin_p2.any())

            true_rings.append(int(true_has_p1) + int(true_has_p2))
            pred_rings.append(int(pred_has_p1) + int(pred_has_p2))

        # ----- Angle + per-event Dice metric (only if we have 2 parents) -----
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

        # Dice on foreground voxels
        if np.any(fg_mask):
            p1_fg_gt = p1_gt[fg_mask]
            p2_fg_gt = p2_gt[fg_mask]
            p1_fg_pr = p1_pr_np[fg_mask]
            p2_fg_pr = p2_pr_np[fg_mask]

            gt_bin_p1 = torch.from_numpy((p1_fg_gt > 0.5).astype(np.float32))
            gt_bin_p2 = torch.from_numpy((p2_fg_gt > 0.5).astype(np.float32))
            pr_bin_p1 = torch.from_numpy((p1_fg_pr > 0.5).astype(np.float32))
            pr_bin_p2 = torch.from_numpy((p2_fg_pr > 0.5).astype(np.float32))

            dice_vals: List[float] = []
            if gt_bin_p1.sum() > 0:
                dice_vals.append(float(_dice_coeff(pr_bin_p1, gt_bin_p1).item()))
            if gt_bin_p2.sum() > 0:
                dice_vals.append(float(_dice_coeff(pr_bin_p2, gt_bin_p2).item()))

            if len(dice_vals) == 0:
                # if somehow neither ring has GT voxels, skip this event for angle metric
                continue
            mean_dice = float(np.mean(dice_vals))
        else:
            continue

        angle_list.append(ang)
        metric_list.append(mean_dice)
        emin_list.append(e_min)
        emax_list.append(e_max)

    if model_was_train:
        model.train()

    angle_arr = np.asarray(angle_list, dtype=float)
    metric_arr = np.asarray(metric_list, dtype=float)
    emin_arr = np.asarray(emin_list, dtype=float)
    emax_arr = np.asarray(emax_list, dtype=float)
    true_rings_arr = np.asarray(true_rings, dtype=int)
    pred_rings_arr = np.asarray(pred_rings, dtype=int)

    return {
        # For backward compatibility "loss" is now actually a metric in [0,1]
        "loss": metric_arr,
        "dice_mean": metric_arr,
        "angle_deg": angle_arr,
        "emin": emin_arr,
        "emax": emax_arr,
        "n_true_rings": true_rings_arr,
        "n_pred_rings": pred_rings_arr,
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
    fmt = ".2f" if normalize else "d"
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
