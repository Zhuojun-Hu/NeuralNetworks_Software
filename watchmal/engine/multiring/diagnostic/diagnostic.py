# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def _centers_from_coords(coords: torch.Tensor, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    c = coords.detach().cpu().numpy().astype(np.int32)
    zyx = c[:, 1:4]
    xyz = np.stack([zyx[:, 2], zyx[:, 1], zyx[:, 0]], axis=1).astype(np.float32)
    centers = origin.reshape(1, 3) + (xyz + 0.5) * float(voxel_size)
    return centers


def _mesh_cubes(centers: np.ndarray, size: float, color: str, name: Optional[str], opacity: float) -> go.Mesh3d:
    if centers.size == 0:
        kwargs = dict(x=[], y=[], z=[], i=[], j=[], k=[], color=color, opacity=opacity, flatshading=True)
        if name is not None:
            kwargs["name"] = name
        return go.Mesh3d(**kwargs)

    h = float(size) * 0.5
    offs = np.array(
        [[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
         [-h, -h,  h], [h, -h,  h], [h, h,  h], [-h, h,  h]],
        dtype=float,
    )
    verts = (centers[:, None, :] + offs[None, :, :]).reshape(-1, 3)
    n = centers.shape[0]
    tri = np.array(
        [[0,1,2],[0,2,3], [4,5,6],[4,6,7],
         [0,1,5],[0,5,4], [3,2,6],[3,6,7],
         [0,4,7],[0,7,3], [1,2,6],[1,6,5]],
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


def _annotation(meta: Dict[str, Any], label: str) -> dict:
    evt = meta.get("event_index", None)
    txt = f"<b>{label}</b>" + (f"<br>Event {evt}" if evt is not None else "")
    return dict(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        text=txt,
        showarrow=False,
        align="left",
        bgcolor="rgba(0,0,0,0.5)",
        bordercolor="white",
        borderwidth=1,
        font=dict(color="white", size=12),
    )


def _scene_layout(axis_limit: float) -> dict:
    rng = [-float(axis_limit), float(axis_limit)]
    ax = dict(range=rng, showbackground=False, gridcolor="rgba(255,255,255,0.18)")
    return dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="black")


def _dice_coeff(prob: torch.Tensor, target_bin: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num = 2.0 * (prob * target_bin).sum()
    den = prob.sum() + target_bin.sum() + eps
    return num / den


def _map_queries_to_parents(Pi: torch.Tensor, y_b: torch.Tensor, lambda_dice: float = 2.0) -> Dict[int, int]:
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


@torch.no_grad()
def save_val_true_pred_split_html(
    model: torch.nn.Module,
    val_subset,
    device: torch.device,
    out_html_path: Path,
    n_events: int = 10,
    start_at: int = 0,
    max_voxels: int = 3000,
    colors: Optional[Dict[str, str]] = None,
) -> None:
    colors = colors or dict(bg="#bdbdbd", p1="#ff7f0e", p2="#1f77b4")
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

    for ev_i, ds_idx in enumerate(chosen):
        sample = ds[ds_idx]
        coords = sample["coords"]
        meta = sample["meta"]

        origin = np.asarray(meta.get("origin", np.array([-4000.0, -4000.0, -4000.0], dtype=np.float32)), dtype=np.float32)
        vsize = float(meta.get("voxel_size", 1.0))
        axis_limit = float(meta.get("axis_limit", 4000.0))
        centers = _centers_from_coords(coords, origin, vsize)

        gt = meta.get("voxel_parent_id", None)
        if isinstance(gt, torch.Tensor):
            gt_slots = gt.detach().cpu().numpy().astype(np.int64)
        else:
            gt_slots = np.zeros((centers.shape[0],), dtype=np.int64)

        batch = {
            "coords": coords.clone(),
            "feats": sample["feats"].clone(),
            "meta": [meta],
        }
        if batch["coords"].numel():
            batch["coords"][:, 0] = 0
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        out = model(batch)

        if isinstance(out, dict) and ("Pi_list" in out) and len(out["Pi_list"]) > 0:
            Pi = out["Pi_list"][0]
            Pi = torch.nan_to_num(Pi, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            Pi = Pi / Pi.sum(dim=1, keepdim=True).clamp_min(1e-6)
            y_b = torch.as_tensor(gt_slots, device=Pi.device, dtype=torch.long)
            q2parent = _map_queries_to_parents(Pi, y_b, lambda_dice=2.0)
            arg = torch.argmax(Pi, dim=1)
            labels = torch.zeros_like(arg)
            fg = arg > 0
            if fg.any():
                qidx = (arg[fg] - 1).tolist()
                mapped = [q2parent.get(int(qi), 0) for qi in qidx]
                labels[fg] = torch.as_tensor(mapped, device=labels.device, dtype=labels.dtype)
            pred_labels = labels.detach().cpu().numpy().astype(np.int64)
        else:
            if "probs_list" in out and len(out["probs_list"]) > 0:
                p = out["probs_list"][0]
            elif "logits_list" in out and len(out["logits_list"]) > 0:
                z = out["logits_list"][0]
                p = torch.softmax(z, dim=1) if z.numel() else z
            else:
                p = coords.new_zeros((coords.size(0), 3))
            pred_labels = torch.argmax(p, dim=1).detach().cpu().numpy().astype(np.int64)

        if centers.shape[0] > int(max_voxels):
            keep = np.arange(centers.shape[0])[: int(max_voxels)]
            centers_vis = centers[keep]
            gt_slots_vis = gt_slots[keep]
            pred_labels_vis = pred_labels[keep]
        else:
            centers_vis = centers
            gt_slots_vis = gt_slots
            pred_labels_vis = pred_labels

        m_bg_gt = gt_slots_vis == 0
        m_p1_gt = gt_slots_vis == 1
        m_p2_gt = gt_slots_vis == 2
        m_bg_pr = pred_labels_vis == 0
        m_p1_pr = pred_labels_vis == 1
        m_p2_pr = pred_labels_vis == 2

        left_traces = [
            _mesh_cubes(centers_vis[m_bg_gt], vsize, colors["bg"], None, 0.35),
            _mesh_cubes(centers_vis[m_p1_gt], vsize, colors["p1"], "GT: Parent 1", 0.95),
            _mesh_cubes(centers_vis[m_p2_gt], vsize, colors["p2"], "GT: Parent 2", 0.95),
        ]
        right_traces = [
            _mesh_cubes(centers_vis[m_bg_pr], vsize, colors["bg"], None, 0.35),
            _mesh_cubes(centers_vis[m_p1_pr], vsize, colors["p1"], "Pred: Parent 1", 0.95),
            _mesh_cubes(centers_vis[m_p2_pr], vsize, colors["p2"], "Pred: Parent 2", 0.95),
        ]
        for tr in right_traces:
            tr.update(scene="scene2")

        if base_left is None:
            base_left = [
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["bg"], "GT: Background", 0.35),
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["p1"], "GT: Parent 1", 0.95),
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["p2"], "GT: Parent 2", 0.95),
            ]
            base_right = [
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["bg"], "Pred: Background", 0.35),
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["p1"], "Pred: Parent 1", 0.95),
                _mesh_cubes(np.zeros((0, 3)), vsize, colors["p2"], "Pred: Parent 2", 0.95),
            ]
            for tr in base_right:
                tr.update(scene="scene2")

        ann_left = _annotation(meta, "Ground truth")
        ann_right = _annotation(meta, "Prediction")
        ann_right = {**ann_right}
        ann_right["x"] = 0.51

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
        margin=dict(l=10, r=10, t=40, b=10),
        scene=_scene_layout(axis_limit),
        scene2=_scene_layout(axis_limit),
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
                        args=[[None], dict(frame=dict(duration=0, redraw=True), fromcurrent=True, transition=dict(duration=0))],
                    ),
                    dict(label="Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False), transition=dict(duration=0))]),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Event: "),
                pad=dict(t=30),
                steps=[
                    dict(label=fr.name, method="animate", args=[[fr.name], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
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


@torch.no_grad()
def collect_val_event_stats(
    model,
    criterion,
    val_subset,
    device: torch.device,
    max_events: Optional[int] = None,
    start_at: int = 0,
) -> Dict[str, np.ndarray]:
    model_was_train = model.training
    model.eval()

    ds = getattr(val_subset, "dataset", val_subset)
    indices = getattr(val_subset, "indices", list(range(len(val_subset))))
    start = max(0, int(start_at))
    stop = len(indices) if max_events is None else min(len(indices), start + int(max_events))

    losses: List[float] = []
    angles: List[float] = []
    emin: List[float] = []
    emax: List[float] = []

    for pos in range(start, stop):
        i = indices[pos]
        sample = ds[i]
        coords = sample["coords"].clone()
        feats = sample["feats"].clone()
        meta = dict(sample["meta"])
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

        if coords.numel():
            coords[:, 0] = 0
        batch1 = {"coords": coords.to(device), "feats": feats.to(device), "meta": [meta]}
        out1 = model(batch1)
        loss_d = criterion(batch=batch1, out=out1)
        loss_v = float(loss_d["loss"].detach().item())

        losses.append(loss_v)
        angles.append(ang)
        emin.append(e_min)
        emax.append(e_max)

    if model_was_train:
        model.train()

    return {
        "loss": np.asarray(losses, dtype=float),
        "angle_deg": np.asarray(angles, dtype=float),
        "emin": np.asarray(emin, dtype=float),
        "emax": np.asarray(emax, dtype=float),
    }


def save_val_angle_loss_png(out_path: Path, angle_deg: np.ndarray, losses: np.ndarray, nbins: int = 18) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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
    plt.ylabel("Mean validation loss")
    plt.title("Val loss vs opening angle")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_val_energy2d_loss_png(
    out_path: Path, emin: np.ndarray, emax: np.ndarray, losses: np.ndarray, nbins_x: int = 20, nbins_y: int = 20
) -> None:
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
    plt.title("Val loss heatmap: E_max vs E_min")
    cbar = plt.colorbar(mesh)
    cbar.set_label("Mean validation loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
