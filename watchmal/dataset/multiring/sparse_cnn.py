from __future__ import annotations
import os, glob, math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
os.environ["SPCONV_ALGO"] = "native"
from spconv.pytorch.utils import PointToVoxel as Point2Voxel  # type: ignore


@dataclass
class VoxelGridConfig:
    axis_limit: float = 4000.0
    grid_size: int = 60
    time_bin_ns: float = 10.0

    @property
    def voxel_size(self) -> float:
        return (2.0 * self.axis_limit) / float(self.grid_size)

    @property
    def origin(self) -> np.ndarray:
        return np.array([-self.axis_limit, -self.axis_limit, -self.axis_limit],
                        dtype=np.float32)


def _rotate_xy_inplace(xyz: np.ndarray, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    xyz[:, 0] = c * x - s * y
    xyz[:, 1] = s * x + c * y


def _safe_f32(a):
    return np.asarray(a, dtype=np.float32)


def _safe_i32(a):
    return np.asarray(a, dtype=np.int32)


def _nan_to_num_(x: np.ndarray):
    np.nan_to_num(x, copy=False)


class _BaseSparseDataset(Dataset):
    def __init__(
        self,
        base_dir: str,
        min_charge: float = 2.5,
        grid: Optional[VoxelGridConfig] = None,
        num_batches: Optional[int] = None,
        seed: Optional[int] = None,
        feat_norm: str = "none",
        feat_norm_indices: Optional[List[int]] = None,
        stats_max_events: Optional[int] = None,
        rotate_p: float = 10.0,
        max_parents: int = 3,
        cache_in_ram: bool = False,
        use_digit_time: bool = False,
        true_charge_only_if_true_time_ge_digi_time: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.min_charge = float(min_charge)
        self.grid = grid or VoxelGridConfig()
        self.rotate_p = float(rotate_p)
        self.max_parents = int(max_parents)
        if self.max_parents < 0:
            raise ValueError("max_parents must be >= 0")
        self.rng = np.random.RandomState(seed if seed is not None else 0)

        self.cache_in_ram = bool(cache_in_ram)
        self._cache: Dict[int, Dict[str, Any]] = {}

        self.use_digit_time = bool(use_digit_time)
        self.true_charge_only_if_true_time_ge_digi_time = bool(
            true_charge_only_if_true_time_ge_digi_time
        )

        files = sorted(
            glob.glob(
                os.path.join(self.base_dir, "**", "wcsim_output_multihit_with_digit_time.h5"),
                recursive=True,
            )
        )
        if num_batches is not None and len(files) > 0:
            idx = np.arange(len(files))
            self.rng.shuffle(idx)
            files = [files[i] for i in idx[: max(1, int(num_batches))]]

        self.files: List[str] = []
        self._handlers: List[h5py.File] = []
        self._index: List[Tuple[int, int]] = []
        self._events_per_file: Dict[str, int] = {}

        for path in files:
            try:
                f = h5py.File(path, "r")
                n_evt = int(f["/events/event_id"].shape[0])
            except Exception:
                continue
            self.files.append(path)
            self._handlers.append(f)
            self._events_per_file[path] = n_evt
            for e in range(n_evt):
                self._index.append((len(self.files) - 1, e))

        if len(self._index) == 0:
            raise RuntimeError(f"No events found under {base_dir}, found {len(files)} files.")

        self.feat_norm = feat_norm
        self.feat_norm_idx = list(feat_norm_indices or [])
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._fmin: Optional[np.ndarray] = None
        self._fmax: Optional[np.ndarray] = None
        if self.feat_norm != "none" and len(self.feat_norm_idx) > 0:
            self._init_feature_stats(stats_max_events)

        print(f"Loaded {_BaseSparseDataset.__name__} with {len(self)} events "
              f"from {len(self.files)} files.")

    def __len__(self) -> int:
        return len(self._index)

    def close(self):
        for f in self._handlers:
            try:
                f.close()
            except Exception:
                pass
        self._handlers = []

    def __del__(self):
        self.close()

    def clear_cache(self):
        self._cache.clear()

    def events_per_h5(self) -> Dict[str, int]:
        return dict(self._events_per_file)

    def _read_event(self, f: h5py.File, eidx: int):
        E = f["/events"]
        hit_ptr = E["hit_index"]
        part_ptr = E["particle_index"]
        hits = E["hits"]
        parts = E["particles"]
        h0, h1 = int(hit_ptr[eidx]), int(hit_ptr[eidx + 1])
        p0, p1 = int(part_ptr[eidx]), int(part_ptr[eidx + 1])
        return hits[h0:h1], parts[p0:p1]

    def _read_digi_event(self, f: h5py.File, eidx: int):
        """
        Optional: read digitized hits if present in the HDF5:
          /events/digi_hit_index and /events/digi_hits
        Returns empty array if missing.
        """
        try:
            E = f["/events"]
            if ("digi_hit_index" not in E) or ("digi_hits" not in E):
                return np.zeros((0,), dtype=E["hits"].dtype)
            ptr = E["digi_hit_index"]
            digis = E["digi_hits"]
            d0, d1 = int(ptr[eidx]), int(ptr[eidx + 1])
            return digis[d0:d1]
        except Exception:
            return np.zeros((0,), dtype=np.dtype([]))

    @staticmethod
    def _compute_primary_groups(parts_evt) -> Tuple[List[int], Dict[int, int]]:
        if len(parts_evt) == 0:
            return [], {}
        tid = parts_evt["track_id"].astype(int)
        pid = parts_evt["parent_id"].astype(int)
        pdg = parts_evt["pdg"].astype(int)
        ene = (
            parts_evt["energy_mev"].astype(float)
            if "energy_mev" in parts_evt.dtype.names
            else np.zeros((len(parts_evt),), dtype=float)
        )

        have_dir = {"dir_x", "dir_y", "dir_z"}.issubset(parts_evt.dtype.names)
        if have_dir:
            dxyz = np.stack(
                [parts_evt["dir_x"], parts_evt["dir_y"], parts_evt["dir_z"]], axis=1
            ).astype(float)
        else:
            dxyz = np.zeros((len(parts_evt), 3), dtype=float)

        info = {
            int(t): (int(p), int(g), float(e), dxyz[i])
            for i, (t, p, g, e) in enumerate(zip(tid, pid, pdg, ene))
        }

        mask = (pid == 0) & (pdg != 0) & (ene > 0)
        prim_ids = tid[mask].tolist()

        if have_dir and len(prim_ids) > 1:
            dirs = np.array([info[t][3] for t in prim_ids], dtype=float)
            n = np.linalg.norm(dirs, axis=1)
            n[n == 0] = 1.0
            dirs = dirs / n[:, None]
            used = np.zeros(len(dirs), dtype=bool)
            keep = []
            cos_tol = np.cos(np.radians(0.2))
            for i in range(len(dirs)):
                if used[i]:
                    continue
                keep.append(i)
                dot = dirs @ dirs[i]
                used |= dot >= cos_tol
            prim_ids = [prim_ids[i] for i in keep]

        cache: Dict[int, int] = {}

        def ancestor(t: int) -> int:
            if t in cache:
                return cache[t]
            seen = set()
            cur = t
            while True:
                rec = info.get(cur)
                if rec is None or cur in seen:
                    cache[t] = -1
                    return -1
                p, g, e, _ = rec
                if p == 0 and g != 0 and e > 0:
                    cache[t] = cur
                    return cur
                seen.add(cur)
                cur = p

        return prim_ids, {int(tt): ancestor(int(tt)) for tt in tid}

    def _assign_primary_hits(
        self,
        hits,
        parts,
        q: np.ndarray,
        topk: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[int], List[int], List[Dict[str, Any]]]:
        if topk is None:
            topk = max(0, int(getattr(self, "max_parents", 2)))
        else:
            topk = max(0, int(topk))

        prim_ids, tid2prim = self._compute_primary_groups(parts)

        if isinstance(hits.dtype.names, tuple) and ("primary_track_id" in hits.dtype.names):
            hit_primary_tid = hits["primary_track_id"].astype(int)
        else:
            hit_tid = (
                hits["track_id"].astype(int)
                if (isinstance(hits.dtype.names, tuple) and ("track_id" in hits.dtype.names))
                else np.full((hits.shape[0],), -1, dtype=int)
            )
            hit_primary_tid = np.array([tid2prim.get(int(t), -1) for t in hit_tid],
                                       dtype=int)

        if hit_primary_tid.size and topk > 0:
            u, inv = np.unique(hit_primary_tid, return_inverse=True)
            sums = np.zeros(u.shape[0], dtype=np.float64)
            for k in range(u.shape[0]):
                sums[k] = float(q[inv == k].sum())
            mask_valid = u != -1
            u_valid = u[mask_valid]
            sums_valid = sums[mask_valid]
            order = np.argsort(-sums_valid)
            top = u_valid[order][:topk]
        else:
            top = np.array([], dtype=int)

        primary_slots: Dict[int, int] = {int(tid): (i + 1) for i, tid in enumerate(list(top))}

        try:
            pdg_arr = parts["pdg"].astype(int)
            track_arr = parts["track_id"].astype(int)
            energy_arr = (
                parts["energy_mev"].astype(float)
                if "energy_mev" in parts.dtype.names
                else np.zeros((len(parts),), dtype=float)
            )
            dir_x = (
                parts["dir_x"].astype(float)
                if "dir_x" in parts.dtype.names
                else np.zeros((len(parts),), dtype=float)
            )
            dir_y = (
                parts["dir_y"].astype(float)
                if "dir_y" in parts.dtype.names
                else np.zeros((len(parts),), dtype=float)
            )
            dir_z = (
                parts["dir_z"].astype(float)
                if "dir_z" in parts.dtype.names
                else np.zeros((len(parts),), dtype=float)
            )
            pdg_map = {int(t): int(p) for t, p in zip(track_arr, pdg_arr)}
            energy_map = {int(t): float(e) for t, e in zip(track_arr, energy_arr)}
            dir_map = {
                int(t): (float(dx), float(dy), float(dz))
                for t, dx, dy, dz in zip(track_arr, dir_x, dir_y, dir_z)
            }
        except Exception:
            pdg_map = {}
            energy_map = {}
            dir_map = {}

        top_track_ids: List[int] = [int(t) for t in top]
        top_pdgs = [int(pdg_map.get(int(t), 0)) for t in top_track_ids]
        if len(top_pdgs) < topk:
            top_pdgs = top_pdgs + [0] * (topk - len(top_pdgs))

        top_info: List[Dict[str, Any]] = []
        for tid in top_track_ids:
            pdg_val = int(pdg_map.get(tid, 0))
            ene = float(energy_map.get(tid, 0.0))
            dx, dy, dz = dir_map.get(tid, (0.0, 0.0, 0.0))
            mag = (dx * dx + dy * dy + dz * dz) ** 0.5
            if mag > 0:
                dx /= mag
                dy /= mag
                dz /= mag
            top_info.append(
                {"track_id": tid, "pdg": pdg_val, "energy_mev": ene,
                 "dir": (dx, dy, dz)}
            )

        pid_hits = np.array(
            [primary_slots.get(int(tp), 0) for tp in hit_primary_tid], dtype=np.int64
        )
        return pid_hits, top_track_ids, top_pdgs[:topk], top_info

    def _voxel_centers(self, indices_zyx: np.ndarray) -> np.ndarray:
        g = self.grid
        v = g.voxel_size
        z = indices_zyx[:, 0].astype(np.float32)
        y = indices_zyx[:, 1].astype(np.float32)
        x = indices_zyx[:, 2].astype(np.float32)
        xyz = np.stack([x, y, z], axis=1)
        centers = g.origin + (xyz + 0.5) * v
        return centers.astype(np.float32)

    def _voxelize_spconv_3d(
        self,
        xyz: np.ndarray,
        q: np.ndarray,
        t: Optional[np.ndarray] = None,
        parent_ids: Optional[np.ndarray] = None,
        max_points_per_voxel: int = 64,
        max_voxels: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if xyz.shape[0] == 0:
            feats = np.zeros((0, 1 + (1 if t is not None else 0)), dtype=np.float32)
            coords = np.zeros((0, 3), dtype=np.int32)
            return feats, {
                "coords_zyx": coords,
                "voxel_parent_id": np.zeros((0,), dtype=np.int64),
                "voxel_parent_conflicts": np.array(0, dtype=np.int64),
                "voxel_parent_frac": np.zeros((0, 0), dtype=np.float32),
                "voxel_parent_num_groups": np.array(0, dtype=np.int64),
            }

        feat_list = [xyz.astype(np.float32), q.reshape(-1, 1).astype(np.float32)]
        if t is not None:
            feat_list.append(t.reshape(-1, 1).astype(np.float32))

        n_groups = 0
        if parent_ids is not None and parent_ids.size > 0:
            parent_ids = parent_ids.astype(np.int64)
            n_groups = int(parent_ids.max()) + 1
            feat_list.append(parent_ids.reshape(-1, 1).astype(np.float32))

        points = np.concatenate(feat_list, axis=1)
        v = float(self.grid.voxel_size)
        L = float(self.grid.axis_limit)
        vxl = Point2Voxel(
            vsize_xyz=[v, v, v],
            coors_range_xyz=[-L, -L, -L, L, L, L],
            num_point_features=points.shape[1],
            max_num_points_per_voxel=int(max_points_per_voxel),
            max_num_voxels=int(
                max_voxels
                if max_voxels is not None
                else max(1, min(points.shape[0], self.grid.grid_size**3))
            ),
        )

        pt = torch.from_numpy(points.astype(np.float32))
        voxels, coors, num_points = vxl(pt)
        M, T, F = voxels.shape

        if M == 0:
            feats = np.zeros((0, 1 + (1 if t is not None else 0)), dtype=np.float32)
            coords = np.zeros((0, 3), dtype=np.int32)
            return feats, {
                "coords_zyx": coords,
                "voxel_parent_id": np.zeros((0,), dtype=np.int64),
                "voxel_parent_conflicts": np.array(0, dtype=np.int64),
                "voxel_parent_frac": np.zeros((0, n_groups), dtype=np.float32),
                "voxel_parent_num_groups": np.array(n_groups, dtype=np.int64),
            }

        mask = (torch.arange(T, device=voxels.device)[None, :] < num_points[:, None])

        idx_q = 3
        idx_t = 4 if (t is not None) else None
        idx_pid = 4 + (1 if idx_t is not None else 0) if (parent_ids is not None) else None

        q_vals = voxels[..., idx_q]
        q_sums = (q_vals * mask.to(q_vals.dtype)).sum(dim=1)
        feat_out = [q_sums.unsqueeze(1)]

        if idx_t is not None:
            t_vals = voxels[..., idx_t]
            t_sums = (t_vals * mask.to(t_vals.dtype)).sum(dim=1)
            counts = num_points.to(t_vals.dtype).clamp_min_(1)
            t_mean = (t_sums / counts).unsqueeze(1)
            feat_out.append(t_mean)

        feats = torch.cat(feat_out, dim=1).cpu().numpy().astype(np.float32)
        coords_zyx = coors.int().cpu().numpy().astype(np.int32)

        voxel_pid = np.zeros((M,), dtype=np.int64)
        voxel_conflicts = 0
        frac = np.zeros((M, n_groups), dtype=np.float32)

        if idx_pid is not None and n_groups > 0:
            pid_vals = voxels[..., idx_pid]
            q_np = q_vals.cpu().numpy()
            pid_np = pid_vals.cpu().numpy().astype(np.int64)
            mask_np = mask.cpu().numpy()

            for vi in range(M):
                mvi = mask_np[vi]
                p = pid_np[vi][mvi]
                qv = q_np[vi][mvi]
                if p.size == 0:
                    voxel_pid[vi] = 0
                    continue
                tot = float(qv.sum())
                if tot > 0.0:
                    bc = np.bincount(p, weights=qv, minlength=n_groups)
                    frac[vi, :] = (bc / tot).astype(np.float32)
                idx_max_hit = int(np.argmax(qv))
                voxel_pid[vi] = int(p[idx_max_hit])
                if np.unique(p).size > 1:
                    voxel_conflicts += 1

        extra: Dict[str, np.ndarray] = {
            "coords_zyx": coords_zyx,
            "voxel_parent_id": voxel_pid,
            "voxel_parent_conflicts": np.array(voxel_conflicts, dtype=np.int64),
            "voxel_parent_frac": frac,
            "voxel_parent_num_groups": np.array(n_groups, dtype=np.int64),
        }
        return feats, extra

    def _apply_norm(self, feats: np.ndarray):
        if self._mean is None:
            return feats
        cols = self.feat_norm_idx
        x = feats[:, cols]
        if self.feat_norm == "standard":
            x = (x - self._mean) / np.clip(self._std, 1e-6, None)
        elif self.feat_norm == "minmax":
            x = (x - self._fmin) / np.clip(self._fmax - self._fmin, 1e-6, None)
        feats[:, cols] = x
        return feats

    def _accum_stats(self, feats: np.ndarray, s: Dict[str, np.ndarray]):
        cols = self.feat_norm_idx
        if feats.shape[0] == 0:
            return
        x = feats[:, cols].astype(np.float64)
        s["n"] += x.shape[0]
        s["sum"] += x.sum(0)
        s["sum2"] += (x * x).sum(0)
        s["min"] = np.minimum(s["min"], x.min(0))
        s["max"] = np.maximum(s["max"], x.max(0))

    @staticmethod
    def _tube_to_min_digi_time(digi_hits) -> Dict[int, float]:
        """Map tube_id -> minimum digi time (ns) within an event."""
        d: Dict[int, float] = {}
        if digi_hits is None or len(digi_hits) == 0:
            return d
        if not (isinstance(digi_hits.dtype.names, tuple) and ("tube_id" in digi_hits.dtype.names) and ("time_ns" in digi_hits.dtype.names)):
            return d
        for row in digi_hits:
            try:
                tube = int(row["tube_id"])
                tt = float(row["time_ns"])
            except Exception:
                continue
            prev = d.get(tube)
            if prev is None or tt < prev:
                d[tube] = tt
        return d

    def _init_feature_stats(self, stats_max_events: Optional[int]):
        N = len(self)
        take = N if (stats_max_events is None) else min(N, int(stats_max_events))
        idxs = np.arange(N)
        self.rng.shuffle(idxs)
        idxs = idxs[:take]
        k = len(self.feat_norm_idx)
        s = {
            "n": 0.0,
            "sum": np.zeros((k,), dtype=np.float64),
            "sum2": np.zeros((k,), dtype=np.float64),
            "min": np.full((k,), np.inf, dtype=np.float64),
            "max": np.full((k,), -np.inf, dtype=np.float64),
        }
        for i in idxs:
            fidx, eidx = self._index[int(i)]
            f = self._handlers[fidx]
            hits, parts = self._read_event(f, eidx)
            if len(hits) == 0:
                continue
            hits_f = hits
            if len(hits_f) == 0:
                continue

            xyz = np.stack([hits_f["x"], hits_f["y"], hits_f["z"]],
                           axis=1).astype(np.float32)
            t_true = hits_f["time_ns"].astype(np.float32)
            q = hits_f["charge_pe"].astype(np.float32)

            if self.use_digit_time or self.true_charge_only_if_true_time_ge_digi_time:
                digi_hits = self._read_digi_event(f, eidx)
                dmap = self._tube_to_min_digi_time(digi_hits)
                if dmap and ("tube_id" in hits_f.dtype.names):
                    tubes = hits_f["tube_id"].astype(np.int32)
                    t_digi = np.array([dmap.get(int(tu), np.nan) for tu in tubes], dtype=np.float32)
                else:
                    t_digi = np.full_like(t_true, np.nan, dtype=np.float32)

                if self.true_charge_only_if_true_time_ge_digi_time:
                    have = np.isfinite(t_digi)
                    mask = (~have) | (t_true >= t_digi)
                    q = q * mask.astype(np.float32)

                if self.use_digit_time:
                    t = t_digi.copy()
                    miss = ~np.isfinite(t)
                    if np.any(miss):
                        t[miss] = t_true[miss]
                else:
                    t = t_true
            else:
                t = t_true

            if xyz.shape[0] == 0:
                continue

            pid_hits, _, _, _ = self._assign_primary_hits(
                hits_f, parts, q, topk=self.max_parents,
            )
            feats3, extra3 = self._voxelize_spconv_3d(xyz, q, t, parent_ids=pid_hits)

            if self.min_charge > 0.0 and feats3.shape[0] > 0:
                mv = feats3[:, 0] >= self.min_charge
                feats3 = feats3[mv]
                extra3["coords_zyx"] = extra3["coords_zyx"][mv]

            if feats3.shape[0] == 0:
                continue

            centers = self._voxel_centers(extra3["coords_zyx"])
            charge = feats3[:, 0:1]
            t_mean = feats3[:, 1:2]
            feats = np.concatenate([charge, centers, t_mean],
                                   axis=1).astype(np.float32)
            self._accum_stats(feats, s)

        n = max(1.0, float(s["n"]))
        mean = s["sum"] / n
        var = s["sum2"] / n - mean * mean
        std = np.sqrt(np.clip(var, 0.0, None))
        self._mean = mean.astype(np.float32)
        self._std = std.astype(np.float32)
        self._fmin = s["min"].astype(np.float32)
        self._fmax = s["max"].astype(np.float32)


class HyperKSparseCNN3D(_BaseSparseDataset):
    def __getitem__(self, i: int) -> Dict[str, Any]:
        if self.cache_in_ram and i in self._cache:
            return self._cache[i]

        fidx, eidx = self._index[int(i)]
        f = self._handlers[fidx]
        path = self.files[fidx]
        hits, parts = self._read_event(f, eidx)

        if len(hits) > 0:
            hits_f = hits

            if len(hits_f) > 0:
                xyz = np.stack([hits_f["x"], hits_f["y"], hits_f["z"]],
                               axis=1).astype(np.float32)
                t_true = hits_f["time_ns"].astype(np.float32)
                q = hits_f["charge_pe"].astype(np.float32)

                if self.use_digit_time or self.true_charge_only_if_true_time_ge_digi_time:
                    digi_hits = self._read_digi_event(f, eidx)
                    dmap = self._tube_to_min_digi_time(digi_hits)
                    if dmap and ("tube_id" in hits_f.dtype.names):
                        tubes = hits_f["tube_id"].astype(np.int32)
                        t_digi = np.array([dmap.get(int(tu), np.nan) for tu in tubes], dtype=np.float32)
                    else:
                        t_digi = np.full_like(t_true, np.nan, dtype=np.float32)

                    if self.true_charge_only_if_true_time_ge_digi_time:
                        have = np.isfinite(t_digi)
                        mask = (~have) | (t_true >= t_digi)
                        q = q * mask.astype(np.float32)

                    if self.use_digit_time:
                        t = t_digi.copy()
                        miss = ~np.isfinite(t)
                        if np.any(miss):
                            t[miss] = t_true[miss]
                    else:
                        t = t_true
                else:
                    t = t_true
            else:
                xyz = np.zeros((0, 3), dtype=np.float32)
                t = np.zeros((0,), dtype=np.float32)
                q = np.zeros((0,), dtype=np.float32)

            if xyz.shape[0] > 0 and self.rotate_p > 0.0 and self.rng.rand() < self.rotate_p:
                angle = float(self.rng.uniform(0.0, 2.0 * math.pi))
                _rotate_xy_inplace(xyz, angle)

            if xyz.shape[0] == 0:
                coords = torch.zeros((0, 4), dtype=torch.int32)
                feats = torch.zeros((0, 5), dtype=torch.float32)
                voxel_parent = torch.zeros((0,), dtype=torch.long)
                voxel_parent_frac = torch.zeros((0, self.max_parents + 1),
                                                dtype=torch.float32)
                voxel_charge_pe = torch.zeros((0,), dtype=torch.float32)
                top_pdgs, top_tids, top_info = [0] * self.max_parents, [], []
                extra3: Dict[str, Any] = {}
            else:
                pid_hits, top_tids, top_pdgs, top_info = self._assign_primary_hits(
                    hits_f, parts, q, topk=self.max_parents
                )
                feats3, extra3 = self._voxelize_spconv_3d(xyz, q, t,
                                                          parent_ids=pid_hits)

                coords_zyx = extra3["coords_zyx"]
                voxel_parent_np = extra3["voxel_parent_id"]
                frac_np = extra3["voxel_parent_frac"]

                if self.min_charge > 0.0 and feats3.shape[0] > 0:
                    mv = feats3[:, 0] >= self.min_charge
                    feats3 = feats3[mv]
                    coords_zyx = coords_zyx[mv]
                    voxel_parent_np = voxel_parent_np[mv]
                    frac_np = frac_np[mv]

                if feats3.shape[0] == 0:
                    coords = torch.zeros((0, 4), dtype=torch.int32)
                    feats = torch.zeros((0, 5), dtype=torch.float32)
                    voxel_parent = torch.zeros((0,), dtype=torch.long)
                    voxel_parent_frac = torch.zeros((0, self.max_parents + 1),
                                                    dtype=torch.float32)
                    voxel_charge_pe = torch.zeros((0,), dtype=torch.float32)
                else:
                    centers = self._voxel_centers(coords_zyx)
                    charge = feats3[:, 0:1]
                    t_mean = feats3[:, 1:2]
                    feats_np = np.concatenate(
                        [charge, centers, t_mean], axis=1
                    ).astype(np.float32)
                    _nan_to_num_(feats_np)

                    raw_charge_pe_np = feats_np[:, 0].copy()

                    feats_np = self._apply_norm(feats_np)

                    b = np.zeros((coords_zyx.shape[0], 1), dtype=np.int32)
                    coords_np = np.concatenate([b, coords_zyx], axis=1)
                    coords = torch.from_numpy(_safe_i32(coords_np))
                    feats = torch.from_numpy(_safe_f32(feats_np))

                    voxel_charge_pe = torch.from_numpy(_safe_f32(raw_charge_pe_np)).view(-1)

                    voxel_parent = torch.from_numpy(
                        voxel_parent_np.astype(np.int64)
                    ).to(torch.long)

                    G_full = self.max_parents + 1
                    frac_np = frac_np.astype(np.float32)
                    if frac_np.shape[1] < G_full:
                        pad = np.zeros(
                            (frac_np.shape[0], G_full - frac_np.shape[1]),
                            dtype=np.float32,
                        )
                        frac_np = np.concatenate([frac_np, pad], axis=1)
                    elif frac_np.shape[1] > G_full:
                        frac_np = frac_np[:, :G_full]
                    voxel_parent_frac = torch.from_numpy(frac_np)
        else:
            coords = torch.zeros((0, 4), dtype=torch.int32)
            feats = torch.zeros((0, 5), dtype=torch.float32)
            voxel_parent = torch.zeros((0,), dtype=torch.long)
            voxel_parent_frac = torch.zeros((0, self.max_parents + 1),
                                            dtype=torch.float32)
            voxel_charge_pe = torch.zeros((0,), dtype=torch.float32)
            top_pdgs, top_tids, top_info = [0] * self.max_parents, [], []
            extra3 = {}

        meta = {
            "event_index": eidx,
            "file_path": path,
            "events_in_file": int(self._events_per_file[path]),
            "events_are_separate": True,
            "grid_size": (self.grid.grid_size,
                          self.grid.grid_size,
                          self.grid.grid_size),
            "axis_limit": self.grid.axis_limit,
            "voxel_size": self.grid.voxel_size,
            "origin": self.grid.origin.copy(),
            "voxel_parent_id": voxel_parent,
            "voxel_parent_conflicts": int(
                extra3.get("voxel_parent_conflicts", 0)
            ) if len(hits) > 0 else 0,
            "voxel_parent_frac": voxel_parent_frac,
            "voxel_charge_pe": voxel_charge_pe,
            "parent_pdgs": top_pdgs,
            "parent_track_ids": top_tids,
            "parent_info": top_info,
            "max_parents": self.max_parents,
        }
        sample = {"coords": coords, "feats": feats, "meta": meta}
        if self.cache_in_ram:
            self._cache[i] = sample
        return sample
