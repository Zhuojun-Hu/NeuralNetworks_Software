# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import importlib

os.environ["SPCONV_ALGO"] = "native"

import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from torch.utils.data import DataLoader, Subset

from watchmal.dataset.multiring.sparse_cnn import VoxelGridConfig
from watchmal.utils.logging_utils import setup_logging

from watchmal.engine.multiring.diagnostic.diagnostic import (
    save_val_true_pred_split_html,
    collect_val_event_stats,
    save_val_angle_loss_png,
    save_val_energy2d_loss_png,
    save_val_ring_count_confusion_png,
    save_val_avg_energy_loss_png,
    save_val_pdg_loss_png,
    save_val_energy2d_diff_avg_loss_png,
)

log = setup_logging(__name__)


def _collate_sparse(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    coords_list, feats_list, meta_list = [], [], []
    for b, s in enumerate(samples):
        c = s["coords"].clone()
        c[:, 0] = b
        coords_list.append(c)
        feats_list.append(s["feats"])
        meta_list.append(dict(s["meta"]))
    return {
        "coords": torch.cat(coords_list, dim=0),
        "feats": torch.cat(feats_list, dim=0),
        "meta": meta_list,
    }


def _import_from_path(path: str):
    if not isinstance(path, str) or "." not in path:
        raise ValueError(f"Invalid import path: {path}")
    mod_name, obj_name = path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)


@dataclass
class _TestLoaderCfg:
    batch_size: int = 1
    num_workers: int = 0
    indices_path: Optional[str] = None
    max_events: Optional[int] = None


class MultiRingSegTestEngine:
    """
    Test-time engine that automatically loads:
      - training hydra config:  <train_output_dir>/.hydra/config.yaml
      - checkpoint:             <train_output_dir>/MultiRingSegEngine_MultiRingModel_BEST.pth

    Output is written directly into the CURRENT Hydra run directory (os.getcwd()),
    which you already set via hydra.run.dir in your SLURM script.
    """

    CKPT_NAME = "MultiRingSegEngine_MultiRingModel_BEST.pth"

    def __init__(
        self,
        truth_key: str,
        rank: int,
        device: str | torch.device,
        dump_path: str,
        train_output_dir: str,
        wandb_run=None,
        model=None,    # passed by main.py, ignored
        dataset=None,  # unused for cnn; kept for signature compatibility
    ):
        self.truth_key = truth_key
        self.rank = int(rank)
        self.device = torch.device(device)
        self.dump_path = str(dump_path)
        self.wandb_run = wandb_run

        self.train_output_dir = str(train_output_dir)
        self.train_output_dir_p = Path(self.train_output_dir)

        self.run_cfg: Optional[DictConfig] = None
        self.model: Optional[torch.nn.Module] = None

        self.dataset = None
        self.test_subset = None
        self.data_loaders: Dict[str, DataLoader] = {}


    def _get_run_img_dir(self) -> Path:
        """
        All images/html go under:
          <hydra.run.dir>/img
        """
        img_dir = Path(os.getcwd()) / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    # -------------------------
    # Auto-loading train artifacts
    # -------------------------
    def _load_run_cfg(self) -> DictConfig:
        cfg_path = self.train_output_dir_p / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Run config not found: {cfg_path}")
        return OmegaConf.load(str(cfg_path))

    def _resolve_checkpoint_path(self) -> Path:
        ckpt = self.train_output_dir_p / self.CKPT_NAME
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    def _build_model_from_run_cfg(self) -> torch.nn.Module:
        assert self.run_cfg is not None
        if "model" not in self.run_cfg:
            raise KeyError("Saved run config has no 'model' section")
        m = instantiate(self.run_cfg.model)
        m = m.to(self.device)
        m.eval()
        return m

    def restore_state(self, weight_file: str) -> None:
        assert self.model is not None
        ckpt = torch.load(str(weight_file), map_location=self.device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=True)

    def configure_data_loaders(self, data_config, loaders_config) -> None:
        """
        main_multiring.py calls:
          engine.configure_data_loaders(hydra_config.data, task_config.pop("data_loaders"))
        """
        if self.rank != 0:
            return


        self.run_cfg = self._load_run_cfg()
        self.model = self._build_model_from_run_cfg()
        eff_cfg = OmegaConf.merge(self.run_cfg, {"data": data_config})
        ds_cfg = eff_cfg.data.dataset

        params = OmegaConf.to_container(ds_cfg.params, resolve=True) or {}
        if "grid" in params and isinstance(params["grid"], dict):
            params["grid"] = VoxelGridConfig(**params["grid"])

        variant = getattr(ds_cfg, "variant", None)
        if variant != "3d":
            raise ValueError(f"Unsupported dataset variant: {variant} (expected '3d')")

        target_path = getattr(ds_cfg, "target_3d", None)
        if target_path is None:
            raise ValueError("data.dataset.target_3d is missing")

        ds_cls = _import_from_path(str(target_path))
        full_ds = ds_cls(**params)
        self.dataset = full_ds

        loaders_cfg = OmegaConf.to_container(loaders_config, resolve=True) or {}
        test_cfg = _TestLoaderCfg(**(loaders_cfg.get("test", {}) or {}))

        indices = list(range(len(full_ds)))
        print(f"Full TEST dataset has {len(full_ds)} events")
        if test_cfg.indices_path:
            p = Path(test_cfg.indices_path)
            if not p.exists():
                raise FileNotFoundError(f"indices_path not found: {p}")
            if p.suffix == ".npy":
                indices = np.load(str(p)).astype(int).tolist()
            else:
                indices = [int(x) for x in p.read_text().strip().split()]

        if test_cfg.max_events is not None:
            indices = indices[: int(test_cfg.max_events)]

        self.test_subset = Subset(full_ds, indices)

        self.data_loaders["test"] = DataLoader(
            self.test_subset,
            batch_size=int(test_cfg.batch_size),
            num_workers=int(test_cfg.num_workers),
            shuffle=False,
            pin_memory=True,
            persistent_workers=(int(test_cfg.num_workers) > 0),
            collate_fn=_collate_sparse,
        )

        log.info(f"[TEST] Train cfg : {self.train_output_dir_p / '.hydra' / 'config.yaml'}")
        log.info(f"[TEST] Train ckpt: {self._resolve_checkpoint_path()}")
        log.info(f"[TEST] Test out : {Path(os.getcwd()).resolve()}")
        try:
            log.info(f"[TEST] Test base_dir: {ds_cfg.params.base_dir}")
        except Exception:
            pass

    @torch.no_grad()
    def test(
        self,
        n_events_html: int = 30,
        start_at: int = 0,
        max_voxels: int = 8000,
        poisson_fluctuate_pred: bool = True,
        poisson_seed: int = 0,
        stats_max_events: Optional[int] = None,
    ) -> Dict[str, float]:
        if self.rank != 0:
            return {}

        if self.model is None or self.test_subset is None:
            raise RuntimeError("Not initialized: configure_data_loaders did not run?")

        ckpt_path = self._resolve_checkpoint_path()
        self.restore_state(str(ckpt_path))
        self.model.eval()

        if stats_max_events is None:
            try:
                stats_max_events = self.run_cfg.data.dataset.params.get("stats_max_events", None)
            except Exception:
                stats_max_events = None

        img_dir = self._get_run_img_dir()

        out_html = img_dir / "test_true_vs_pred_split.html"
        save_val_true_pred_split_html(
            model=self.model,
            val_subset=self.test_subset,
            device=self.device,
            out_html_path=out_html,
            n_events=int(n_events_html),
            start_at=int(start_at),
            max_voxels=int(max_voxels),
            poisson_fluctuate_pred=bool(poisson_fluctuate_pred),
            poisson_seed=int(poisson_seed),
        )
        log.info(f"Wrote HTML: {out_html}")

        stats = collect_val_event_stats(
            model=self.model,
            criterion=None,
            val_subset=self.test_subset,
            device=self.device,
            max_events=stats_max_events,
            start_at=int(start_at),
        )

        angle_png = img_dir / "test_dice_vs_angle.png"
        energy_png = img_dir / "test_dice_vs_energy2d.png"
        cm_png = img_dir / "test_ring_count_confusion.png"
        avgE_png = img_dir / "test_dice_vs_avg_energy.png"
        pdg_png  = img_dir / "test_dice_vs_pdg.png"
        e2_png   = img_dir / "test_dice_vs_ediff_eavg_predgt1.png"

        save_val_angle_loss_png(angle_png, stats["angle_deg"], stats["loss"], nbins=18)
        save_val_energy2d_loss_png(energy_png, stats["emin"], stats["emax"], stats["loss"], nbins_x=20, nbins_y=20)
        save_val_ring_count_confusion_png(cm_png, stats["n_true_rings"], stats["n_pred_rings"], normalize=True)

        save_val_avg_energy_loss_png(
            out_path=avgE_png,
            eavg=stats["eavg"],
            losses=stats["loss"],
            nbins=20,
        )

        save_val_pdg_loss_png(
            out_path=pdg_png,
            pdg=stats["pdg_maxE"],
            losses=stats["loss"],
            top_k=12,
            min_count=5,
        )

        save_val_energy2d_diff_avg_loss_png(
            out_path=e2_png,
            ediff=stats["ediff"],
            eavg=stats["eavg"],
            losses=stats["loss"],
            pred_rings_gt1=stats["pred_rings_gt1"],
            nbins_x=20,
            nbins_y=20,
        )

        log.info(f"Wrote PNG: {angle_png}")
        log.info(f"Wrote PNG: {energy_png}")
        log.info(f"Wrote PNG: {cm_png}")

        summary: Dict[str, float] = {}
        if stats["dice_mean"].size > 0:
            summary["dice_mean/mean"] = float(np.mean(stats["dice_mean"]))
            summary["dice_mean/median"] = float(np.median(stats["dice_mean"]))

        if self.wandb_run is not None and summary:
            self.wandb_run.log({f"test/{k}": v for k, v in summary.items()})

        return summary
