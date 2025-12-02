# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

import torch
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

from watchmal.engine.reconstruction import ReconstructionEngine
from watchmal.dataset.multiring.sparse_cnn import (
    VoxelGridConfig, HyperKSparseCNN3D
)
from watchmal.utils.logging_utils import setup_logging
from watchmal.engine.multiring.diagnostic.diagnostic import (
    save_val_true_pred_split_html,
    collect_val_event_stats,
    save_val_angle_loss_png,          # now plots Dice vs angle
    save_val_energy2d_loss_png,       # now plots Dice vs energies
    save_val_ring_count_confusion_png # NEW: confusion matrix for ring counts
)

log = setup_logging(__name__)


def _collate_sparse(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for sparse CNN
    """
    coords_list, feats_list, meta_list = [], [], []

    for b, s in enumerate(samples):
        c = s["coords"].clone()
        c[:, 0] = b
        coords_list.append(c)
        feats_list.append(s["feats"])

        m = dict(s["meta"])
        meta_list.append(m)

    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "meta": meta_list,
    }


@dataclass
class _LoaderCfg:
    batch_size: int = 1
    num_workers: int = 0
    split: float = 0.9
    # Peut être une string ("subset_random") ou un dict Hydra
    sampler_config: Optional[Any] = None


def _build_sampler(sampler_cfg: Optional[Any], indices: List[int]):
    """
    Build a torch Sampler from a sampler_config value.

    Supports:
      - "subset_random"
      - dict with name/_name_/_target_ == "subset_random"
    """
    if sampler_cfg is None:
        return None

    if isinstance(sampler_cfg, str):
        name = sampler_cfg.lower()
    elif isinstance(sampler_cfg, dict):
        name = (
            sampler_cfg.get("name")
            or sampler_cfg.get("_name_")
            or sampler_cfg.get("_target_")
            or ""
        )
        name = str(name).lower()
    else:
        return None

    if name == "subset_random":
        return SubsetRandomSampler(indices)

    # Extend here if you add more sampler types
    return None


class MultiRingSegEngine(ReconstructionEngine):
    """
    Engine for multiring voxel segmentation with sparse CNNs.
    """

    def __init__(self, truth_key: str = "voxel_parent_frac", **kwargs):
        super().__init__(truth_key=truth_key, **kwargs)

        self.history = {
            "train": {"loss": [], "loss_ce": [], "loss_dice": []},
            "val":   {"loss": [], "loss_ce": [], "loss_dice": []},
        }

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._move_to_device(v) for v in obj]
            return type(obj)(t)
        return obj

    def _get_run_img_dir(self) -> Path:
        """
        Return <hydra_run_dir>/img, creating it if needed.
        """
        base = Path(os.getcwd())
        img_dir = base / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def _append_epoch_history(self, split: str, metrics_mean: Dict[str, float]) -> None:
        for k in ("loss", "loss_ce", "loss_dice"):
            if k in metrics_mean:
                self.history[split][k].append(float(metrics_mean[k]))

    def _plot_curves(self) -> None:
        """
        Save three plots (Loss, CE, DICE) comparing train vs val over epochs
        into <hydra_run_dir>/img/.
        """
        if self.rank != 0:
            return

        img_dir = self._get_run_img_dir()

        def _save2(name: str, ytr: List[float], yva: List[float], ylabel: str):
            n = min(len(ytr), len(yva))
            if n == 0:
                return
            x = list(range(1, n + 1))
            plt.figure()
            plt.plot(x, ytr[:n], label="train")
            plt.plot(x, yva[:n], label="val")
            plt.xlabel("epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(img_dir / f"{name}.png", dpi=150)
            plt.close()

        _save2(
            "train_val_loss",
            self.history["train"]["loss"],
            self.history["val"]["loss"],
            ylabel="Loss (mean per epoch)",
        )
        _save2(
            "train_val_ce",
            self.history["train"]["loss_ce"],
            self.history["val"]["loss_ce"],
            ylabel="Cross-Entropy (mean per epoch)",
        )
        _save2(
            "train_val_dice",
            self.history["train"]["loss_dice"],
            self.history["val"]["loss_dice"],
            ylabel="Dice penalty (mean per epoch)",
        )

    def configure_data_loaders(self, data_config, loaders_config):
        ds_cfg = data_config.dataset
        params = OmegaConf.to_container(ds_cfg.params, resolve=True) or {}

        if "grid" in params and isinstance(params["grid"], dict):
            params["grid"] = VoxelGridConfig(**params["grid"])

        # Only 3D variant is used here
        ds_cls = HyperKSparseCNN3D
        full_ds = ds_cls(**params)

        train_cfg = _LoaderCfg(**(loaders_config.get("train", {}) or {}))
        val_cfg   = _LoaderCfg(**(loaders_config.get("validation", {}) or {}))

        n = len(full_ds)
        k = int(n * train_cfg.split)
        idx = torch.randperm(n).tolist()
        if k < n:
            train_idx, val_idx = idx[:k], idx[k:]
        else:
            train_idx, val_idx = idx, idx[:0]

        d_train = Subset(full_ds, train_idx)
        d_val   = Subset(full_ds, val_idx)
        self.val_subset = d_val

        # Build samplers from sampler_config (no DDP-specific sampler here)
        train_sampler = _build_sampler(train_cfg.sampler_config, train_idx)
        val_sampler   = _build_sampler(val_cfg.sampler_config, val_idx)

        self.data_loaders["train"] = DataLoader(
            d_train,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
            collate_fn=_collate_sparse,
        )
        self.data_loaders["validation"] = DataLoader(
            d_val,
            batch_size=val_cfg.batch_size or train_cfg.batch_size,
            num_workers=val_cfg.num_workers,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            collate_fn=_collate_sparse,
        )

    def forward(self, forward_type: str):
        """
        Called by sub_train / sub_validate.
        """
        batch = self.data
        out = self.model(batch)

        loss_dict = self.criterion(batch=batch, out=out)
        total = loss_dict["loss"]

        self.loss = total

        outputs: Dict[str, torch.Tensor] = {}
        return outputs, loss_dict

    def _extract_target_if_any_(self, batch: Dict[str, Any]) -> None:
        """
        Build a flat target tensor from batch['meta'][b][truth_key] if available.

        - If truth_key == "voxel_parent_id"  -> cast to long (class indices).
        - If truth_key == "voxel_parent_frac" -> keep as float (soft labels / fractions).
        - Otherwise: infer from dtype (ints -> long, floats stay float).
        """
        meta = batch.get("meta", None)
        self.target = None

        def _prepare_tensor(x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)
            # explicit special cases
            if self.truth_key == "voxel_parent_id":
                return x.long()
            if self.truth_key == "voxel_parent_frac":
                return x.float()
            return x.long()

        if isinstance(meta, list):
            t_list = []
            for m in meta:
                if self.truth_key in m:
                    val = m[self.truth_key]
                    if not isinstance(val, torch.Tensor):
                        val = torch.as_tensor(val)
                    t_list.append(_prepare_tensor(val))
            if t_list:
                self.target = torch.cat(t_list, dim=0)

        elif isinstance(meta, dict) and self.truth_key in meta:
            val = meta[self.truth_key]
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val)
            self.target = _prepare_tensor(val)

    def sub_train(self, loader, val_interval):
        self.model.train()
        metrics_epoch_sum: Dict[str, float] = {"loss": 0.0}
        n_steps = 0

        for step, train_batch in enumerate(loader):
            n_steps += 1
            batch = self._move_to_device(train_batch)
            self.data = batch
            self._extract_target_if_any_(batch)

            outputs, metrics = self.forward(forward_type="train")
            self.loss = metrics["loss"]
            self.backward()

            if self.rank == 0:
                metrics = {k: float(v.item()) for k, v in metrics.items()}
                outputs = {k: float(v.item()) for k, v in outputs.items()}

                for k, v in metrics.items():
                    metrics_epoch_sum[k] = metrics_epoch_sum.get(k, 0.0) + v

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {f"train_batch_{k}": v for k, v in outputs.items()}
                        | {f"train_batch_{k}": v for k, v in metrics.items()}
                    )

            if (step % val_interval == 0) and (self.rank == 0):
                log.info(
                    f"GPU : {self.device} | Steps : {step + 1}/{len(loader)} "
                    f"| Iteration : {self.iteration + step} | Batch Size : {loader.batch_size}"
                )
                log.info(
                    "Batch metrics "
                    + ", ".join(f"{k}: {metrics.get(k, float('nan')):.5g}" for k in sorted(metrics_epoch_sum))
                )

        self.iteration += n_steps

        # Convert sums to means
        metrics_epoch_mean = {}
        if self.rank == 0 and n_steps > 0:
            for k, v in metrics_epoch_sum.items():
                metrics_epoch_mean[k] = v / n_steps

            # store in history for plotting
            self._append_epoch_history("train", metrics_epoch_mean)

        return metrics_epoch_mean

    def sub_validate(self, loader, forward_type: str = "val"):
        """
        Validation loop adapted for sparse multiring batches.
        """
        metrics_epoch_sum: Dict[str, float] = {"loss": 0.0}
        n_steps = 0

        self.model.eval()
        with torch.no_grad():
            for step, val_batch in enumerate(loader):
                n_steps += 1
                batch = self._move_to_device(val_batch)
                self.data = batch
                self._extract_target_if_any_(batch)

                outputs, metrics = self.forward(forward_type=forward_type)

                if self.is_distributed:
                    metrics = self.get_reduced(metrics)
                    outputs = self.get_reduced(outputs)

                if self.rank == 0:
                    metrics = {k: float(v.item()) for k, v in metrics.items()}
                    outputs = {k: float(v.item()) for k, v in outputs.items()}

                    for k, v in metrics.items():
                        metrics_epoch_sum[k] = metrics_epoch_sum.get(k, 0.0) + v

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {f"val_batch_{k}": v for k, v in outputs.items()}
                            | {f"val_batch_{k}": v for k, v in metrics.items()}
                        )

        metrics_epoch_mean = {}
        if self.rank == 0 and n_steps > 0:
            for k, v in metrics_epoch_sum.items():
                metrics_epoch_mean[k] = v / n_steps

            self._append_epoch_history("val", metrics_epoch_mean)
            self._plot_curves()

            # --- Diagnostics: HTML + plots ---
            try:
                img_dir = self._get_run_img_dir()
                out_html = img_dir / "val_true_vs_pred_split.html"
                save_val_true_pred_split_html(
                    model=self.model,
                    val_subset=self.val_subset,
                    device=self.device,
                    out_html_path=out_html,
                    n_events=30,
                    start_at=0,
                    max_voxels=8000,
                )
                log.info(f"Wrote event display: {out_html}")
            except Exception as e:
                log.warning(f"Diagnostics display failed: {e}")

            try:
                stats = collect_val_event_stats(
                    model=self.model,
                    criterion=self.criterion,
                    val_subset=self.val_subset,
                    device=self.device,
                    max_events=None,   # all validation events
                    start_at=0,
                )

                img_dir = self._get_run_img_dir()
                # Note: 'loss' in stats is now actually a Dice-based metric in [0,1]
                angle_png  = img_dir / "val_dice_vs_angle.png"
                energy_png = img_dir / "val_dice_vs_energy2d.png"
                cm_png     = img_dir / "val_ring_count_confusion.png"

                save_val_angle_loss_png(
                    out_path=angle_png,
                    angle_deg=stats["angle_deg"],
                    losses=stats["loss"],
                    nbins=18,
                )
                log.info(f"Wrote Dice-vs-angle plot: {angle_png}")

                save_val_energy2d_loss_png(
                    out_path=energy_png,
                    emin=stats["emin"],
                    emax=stats["emax"],
                    losses=stats["loss"],
                    nbins_x=20,
                    nbins_y=20,
                )
                log.info(f"Wrote Dice-vs-energy2D plot: {energy_png}")

                save_val_ring_count_confusion_png(
                    out_path=cm_png,
                    n_true_rings=stats["n_true_rings"],
                    n_pred_rings=stats["n_pred_rings"],
                    normalize=True,
                )
                log.info(f"Wrote ring-count confusion matrix: {cm_png}")

            except Exception as e:
                log.warning(f"Angle/Energy/Confusion diagnostics failed: {e}")

        return metrics_epoch_mean

    def to_disk_data_reformat(self, **kw):
        """
        Kept for compatibility
        """
        return kw

    def make_plots(self, *a, **k):
        """
        Kept for compatibility
        """
        try:
            self._plot_curves()
        except Exception as e:
            if self.rank == 0:
                log.warning(f"make_plots failed: {e}")
