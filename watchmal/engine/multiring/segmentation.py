# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
import math
import random 
os.environ["SPCONV_ALGO"] = "native"

import numpy as np
from datetime import datetime
import wandb
from hydra.utils import instantiate
import torch
import torch.nn.utils as nn_utils
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from watchmal.dataset.multiring.sparse_cnn import (
    VoxelGridConfig, HyperKSparseCNN3D
)
from watchmal.utils.logging_utils import setup_logging
from watchmal.utils.early_stopping import EarlyStopping
from watchmal.engine.multiring.diagnostic.diagnostic import (
    save_val_true_pred_split_html,
    collect_val_event_stats,
    save_val_angle_loss_png,
    save_val_energy2d_loss_png,
    save_val_ring_count_confusion_png,
)

log = setup_logging(__name__)


def _collate_sparse(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    sampler_config: Optional[Any] = None


def _build_sampler(sampler_cfg: Optional[Any], indices: List[int]):
    if sampler_cfg is None:
        return None
    if isinstance(sampler_cfg, str):
        name = sampler_cfg.lower()
        if "subset_random" in name or "subsetrandomsampler" in name:
            return SubsetRandomSampler(indices)
        return None
    if isinstance(sampler_cfg, dict):
        raw = (
            sampler_cfg.get("name")
            or sampler_cfg.get("_name_")
            or sampler_cfg.get("_target_")
            or ""
        )
        name = str(raw).lower()
        if "subsetrandomsampler" in name or "subset_random" in name:
            return SubsetRandomSampler(indices)
    return None


class MultiRingSegEngine:
    def __init__(
        self,
        truth_key,
        model,
        rank,
        device,
        dump_path,
        wandb_run=None,
        dataset=None,
    ):
        self.dump_path = dump_path
        self.wandb_run = wandb_run

        self.rank = int(rank)
        self.device = torch.device(device)
        self.model = model

        self.epoch = 0
        self.step = 0
        self.iteration = 0
        self.best_validation_loss = 1.0e10
        self.best_training_loss = 1.0e10

        self.dataset = dataset if dataset is not None else None
        self.split_path = ""
        self.truth_key = truth_key

        self.data_loaders = {}

        if isinstance(self.model, DistributedDataParallel):
            self.is_distributed = True
            self.module = self.model.module
            self.n_gpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.module = self.model
            self.n_gpus = 1

        self.data = None
        self.target = None
        self.loss = None
        self.outputs_epoch_history = []

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.early_stopping = None

        self.history = {
            "train": {"loss": [], "loss_ce": [], "loss_dice": []},
            "val": {"loss": [], "loss_ce": [], "loss_dice": []},
        }
        self.val_subset = None

        self.max_stable_loss = 10.0
        self.last_good_ckpt = None
        self.lastgood_every_steps = 200 
        self.recover_count = 0
        self.recover_seed_base = 12345
        self.recover_lr_factor = 1
    def configure_loss(self, loss_config):
        self.criterion = instantiate(loss_config)

    def configure_optimizers(self, optimizer_config):
        self.optimizer = instantiate(optimizer_config, params=self.module.parameters())

    def configure_scheduler(self, scheduler_config):
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)

    def configure_early_stopping(self, early_stopping_config):
        self.early_stopping = instantiate(early_stopping_config)

    def set_dataset(self, dataset, dataset_config):
        if self.dataset is not None:
            print(f'Error : Dataset is already set in the engine of the process : {self.rank}')
            raise ValueError
        self.dataset = dataset
        self.split_path = dataset_config.split_path
        self.target_names = list(dataset_config.target_names)
        for trf in dataset.transforms.transforms:
            if trf.__class__.__name__ == 'Normalize':
                ft_norm, target_norm = trf.feat_norm, trf.target_norm
                break
        self.feat_norm, self.target_norm = ft_norm, target_norm

    def _move_to_device(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._move_to_device(v) for v in obj]
            return type(obj)(t)
        return obj

    def _get_run_img_dir(self) -> Path:
        base = Path(os.getcwd())
        img_dir = base / "img"
        img_dir.mkdir(parents=True, exist_ok=True)
        return img_dir

    def _append_epoch_history(self, split, metrics_mean):
        for k in ("loss", "loss_ce", "loss_dice"):
            if k in metrics_mean:
                self.history[split][k].append(float(metrics_mean[k]))

    def _plot_curves(self):
        if self.rank != 0:
            return
        img_dir = self._get_run_img_dir()

        def _save2(name, ytr, yva, ylabel):
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

        _save2("train_val_loss", self.history["train"]["loss"], self.history["val"]["loss"], ylabel="Loss (mean per epoch)")
        _save2("train_val_ce", self.history["train"]["loss_ce"], self.history["val"]["loss_ce"], ylabel="Cross-Entropy (mean per epoch)")
        _save2("train_val_dice", self.history["train"]["loss_dice"], self.history["val"]["loss_dice"], ylabel="Dice penalty (mean per epoch)")

    def configure_data_loaders(self, data_config, loaders_config):
        ds_cfg = data_config.dataset
        params = OmegaConf.to_container(ds_cfg.params, resolve=True) or {}
        if "grid" in params and isinstance(params["grid"], dict):
            params["grid"] = VoxelGridConfig(**params["grid"])
        ds_cls = HyperKSparseCNN3D
        full_ds = ds_cls(**params)

        loaders_cfg = OmegaConf.to_container(loaders_config, resolve=True) or {}
        train_cfg = _LoaderCfg(**(loaders_cfg.get("train", {}) or {}))
        val_cfg = _LoaderCfg(**(loaders_cfg.get("validation", {}) or {}))

        n = len(full_ds)
        k = int(n * train_cfg.split)
        idx = torch.randperm(n).tolist()
        if k < n:
            train_idx, val_idx = idx[:k], idx[k:]
        else:
            train_idx, val_idx = idx, idx[:0]

        d_train = Subset(full_ds, train_idx)
        d_val = Subset(full_ds, val_idx)
        self.val_subset = d_val

        if self.is_distributed:
            world_size = self.n_gpus
            rank = self.rank
            train_sampler = DistributedSampler(
                d_train,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            val_sampler = DistributedSampler(
                d_val,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            train_shuffle = False
        else:
            train_sampler = _build_sampler(train_cfg.sampler_config, train_idx)
            val_sampler = _build_sampler(val_cfg.sampler_config, val_idx)
            train_shuffle = train_sampler is None

        self.data_loaders["train"] = DataLoader(
            d_train,
            batch_size=train_cfg.batch_size,
            num_workers=train_cfg.num_workers,
            shuffle=train_shuffle,
            sampler=train_sampler,
            pin_memory=True,
            persistent_workers=(train_cfg.num_workers > 0),
            collate_fn=_collate_sparse,
        )
        self.data_loaders["validation"] = DataLoader(
            d_val,
            batch_size=val_cfg.batch_size or train_cfg.batch_size,
            num_workers=val_cfg.num_workers,
            shuffle=False,
            sampler=val_sampler,
            pin_memory=True,
            persistent_workers=(val_cfg.num_workers > 0),
            collate_fn=_collate_sparse,
        )

    def get_reduced(self, outputs, op=torch.distributed.ReduceOp.SUM):
        if not self.is_distributed:
            return outputs
        new_outputs = {}
        for name, tensor in outputs.items():
            if not torch.is_tensor(tensor):
                continue
            torch.distributed.reduce(tensor, 0, op=op)
            if self.rank == 0:
                new_outputs[name] = tensor / self.n_gpus
        return new_outputs


    def _extract_target_if_any_(self, batch: Dict[str, Any]) -> None:
        meta = batch.get("meta", None)
        self.target = None

        def _prepare_tensor(x: torch.Tensor) -> torch.Tensor:
            x = x.to(self.device)
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

    def forward(self, forward_type):
        batch = self.data
        out = self.model(batch)
        loss_dict = self.criterion(batch=batch, out=out)
        total = loss_dict["loss"]
        self.loss = total
        outputs = {}
        return outputs, loss_dict

    def sub_train(self, loader, val_interval):
        self.model.train()
        metrics_epoch_history = {'loss': 0.0}
        for step, train_data in enumerate(loader):
            batch = self._move_to_device(train_data)
            self.data = batch
            self._extract_target_if_any_(batch)

            if self.rank == 0 and self.last_good_ckpt is None:
                self.last_good_ckpt = self.save_state(suffix="_LASTGOOD")
            if self.rank == 0 and (self.iteration + step) % self.lastgood_every_steps == 0:
                self.last_good_ckpt = self.save_state(suffix="_LASTGOOD")

            try:
                outputs, metrics = self.forward(forward_type='train')
                self.loss = metrics['loss']
                self.backward()

            except RuntimeError as e:
                msg = str(e)
                if ("Non-finite logits detected" in msg) and (self.last_good_ckpt is not None):
                    if self.rank == 0:
                        log.error(f"Non-finite logits at epoch={self.epoch} step={step} iter={self.iteration + step}. Rolling back.")
                        if self.wandb_run is not None:
                            self.wandb_run.log({
                                "nan/epoch": self.epoch,
                                "nan/step": step,
                                "nan/iter": self.iteration + step,
                                "nan/error": msg[:300],
                            })

                    self.restore_state(self.last_good_ckpt)
                    self.recover_count += 1
                    new_seed = self.recover_seed_base + self.recover_count
                    random.seed(new_seed)
                    np.random.seed(new_seed)
                    torch.manual_seed(new_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(new_seed)

                    try:
                        for pg in self.optimizer.param_groups:
                            pg["lr"] *= self.recover_lr_factor
                        if self.rank == 0 and self.wandb_run is not None:
                            self.wandb_run.log({"nan/recover_seed": new_seed, "nan/new_lr": self.optimizer.param_groups[0]["lr"]})
                    except Exception:
                        pass

                    continue

                raise

            metrics = {k: v.item() for k, v in metrics.items()}
            outputs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in outputs.items()}

            if self.rank == 0:
                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {'train_batch_' + k: v for k, v in outputs.items()} |
                        {'train_batch_' + k: v for k, v in metrics.items()}
                    )
                for k in metrics.keys():
                    if k not in metrics_epoch_history:
                        metrics_epoch_history[k] = 0.0
                    metrics_epoch_history[k] += metrics[k]

            if (step % val_interval == 0) and (self.rank == 0):
                log.info(
                    f"GPU : {self.device} | Steps : {step + 1}/{len(loader)} "
                    f"| Iteration : {self.iteration + step} | Batch Size : {loader.batch_size}"
                )
                log.info(
                    "Batch metrics " +
                    ", ".join(f"{k}: {v:.5g}" for k, v in metrics.items())
                )

        self.iteration += (step + 1)

        if self.rank == 0:
            for k in metrics_epoch_history.keys():
                metrics_epoch_history[k] /= (step + 1)
            self._append_epoch_history("train", metrics_epoch_history)

        return metrics_epoch_history

    def sub_validate(self, loader, forward_type='val', **kwargs):
        metrics_epoch_history = {'loss': 0.0}
        wandb_prefix = ""

        self.model.eval()
        with torch.no_grad():
            for step, val_batch in enumerate(loader):
                batch = self._move_to_device(val_batch)
                self.data = batch
                self._extract_target_if_any_(batch)

                outputs, metrics = self.forward(forward_type=forward_type)

                if self.is_distributed:
                    metrics = self.get_reduced(metrics)
                    outputs = self.get_reduced(outputs)

                if self.rank == 0:
                    metrics = {k: v.item() for k, v in metrics.items()}
                    outputs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in outputs.items()}

                    for k in metrics.keys():
                        if k not in metrics_epoch_history:
                            metrics_epoch_history[k] = 0.0
                        metrics_epoch_history[k] += metrics[k]

                    if self.wandb_run is not None:
                        self.wandb_run.log(
                            {wandb_prefix + 'val_batch_' + k: v for k, v in outputs.items()} |
                            {wandb_prefix + 'val_batch_' + k: v for k, v in metrics.items()}
                        )

        if self.rank == 0:
            for k in metrics_epoch_history.keys():
                metrics_epoch_history[k] /= (step + 1)

            self._append_epoch_history("val", metrics_epoch_history)
            self._plot_curves()

            eval_model = self.module if self.is_distributed else self.model

            try:
                img_dir = self._get_run_img_dir()
                out_html = img_dir / "val_true_vs_pred_split.html"
                save_val_true_pred_split_html(
                    model=eval_model,
                    val_subset=self.val_subset,
                    device=self.device,
                    out_html_path=out_html,
                    n_events=30,
                    start_at=0,
                    max_voxels=8000,
                    poisson_fluctuate_pred=True,
                    poisson_seed=0,
                )
                log.info(f"Wrote event display: {out_html}")
            except Exception as e:
                log.warning(f"Diagnostics display failed: {e}")

            try:
                stats = collect_val_event_stats(
                    model=eval_model,
                    criterion=self.criterion,
                    val_subset=self.val_subset,
                    device=self.device,
                    max_events=None,
                    start_at=0,
                )

                img_dir = self._get_run_img_dir()
                angle_png = img_dir / "val_dice_vs_angle.png"
                energy_png = img_dir / "val_dice_vs_energy2d.png"
                cm_png = img_dir / "val_ring_count_confusion.png"

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

        return metrics_epoch_history

    def backward(self):
        loss = self.loss
        loss_val = float(loss.detach().cpu().item())

        if (not math.isfinite(loss_val)) or (loss_val > self.max_stable_loss):
            if self.rank == 0:
                log.warning(f"Unstable loss detected: {loss_val}. Skipping optimizer step.")
            self.optimizer.zero_grad(set_to_none=True)
            return

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.rank == 0:
            total_norm = 0.0
            for p in self.module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if self.wandb_run is not None:
                self.wandb_run.log({'grad_total_norm': total_norm})

        nn_utils.clip_grad_norm_(self.module.parameters(), max_norm=1.0)
        self.optimizer.step()

    def train(self, epochs=0, val_interval=20, checkpointing=False, save_interval=None):
        start_run_time = datetime.now()
        log.info(f"Engine : {self.rank} | Dataloaders : {self.data_loaders}")
        if self.rank == 0:
            print("\n")
            log.info(f"\033[1;96m********** 🚀 Starting training for {epochs} epochs 🚀 **********\033[0m")

        self.iteration = 1
        self.step = 0

        self.best_training_loss = np.inf
        self.best_validation_loss = np.inf

        train_loader = self.data_loaders["train"]
        val_loader = self.data_loaders["validation"]

        cache_refresh_epochs =  20

        if self.wandb_run is not None and self.rank == 0:
            self.wandb_run.watch(self.module, log='all', log_freq=val_interval * 2, log_graph=True)
            self.wandb_run.log({'max_datapoints_seen': epochs * len(train_loader) * train_loader.batch_size})

        for epoch in range(epochs):
            self.epoch = epoch

            if cache_refresh_epochs and (epoch % cache_refresh_epochs == 0):
                ds = train_loader.dataset
                root_ds = getattr(ds, "dataset", ds)
                if hasattr(root_ds, "clear_cache"):
                    if self.rank == 0:
                        log.info(f"[Epoch {epoch + 1}] Clearing train dataset cache")
                    root_ds.clear_cache()

            epoch_start_time = datetime.now()
            if self.rank == 0:
                log.info(f"\n\nTraining epoch {self.epoch + 1}/{epochs} starting at {epoch_start_time}")
                if self.optimizer is not None:
                    current_lr0 = self.optimizer.param_groups[0]['lr']
                    log.info(f"[Epoch {self.epoch + 1}] Current LR: {current_lr0:.6g}")
                    if self.wandb_run is not None:
                        self.wandb_run.log({'learning_rate': current_lr0, 'epoch': self.epoch})

            if self.is_distributed:
                sampler = getattr(train_loader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)

            metrics_epoch_history = self.sub_train(train_loader, val_interval)

            if (self.scheduler is not None) and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = self.scheduler.get_last_lr()
                self.scheduler.step()
                if self.scheduler.get_last_lr() != current_lr:
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")

            epoch_end_time = datetime.now()

            if self.rank == 0:
                log.info(f"(Train) Epoch : {epoch + 1} completed in {(epoch_end_time - epoch_start_time)} | Iteration : {self.iteration} ")
                log.info(f"Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info("Metrics over the (train) epoch " +
                         ", ".join(f"{k}: {v:.5g}" for k, v in metrics_epoch_history.items()))

                if self.wandb_run is not None:
                    self.wandb_run.log({'epoch': epoch})
                    self.wandb_run.log(
                        {'train_epoch_' + k: v for k, v in metrics_epoch_history.items()}
                    )

                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.wandb_run.log({'learning_rate': self.optimizer.param_groups[0]['lr']})
                        else:
                            self.wandb_run.log({'learning_rate': self.scheduler.get_last_lr()[0]})

                    if metrics_epoch_history['loss'] < self.best_training_loss:
                        self.best_training_loss = metrics_epoch_history['loss']
                        self.wandb_run.log({'best_train_epoch_loss': self.best_training_loss})

            epoch_start_time = datetime.now()
            if self.rank == 0:
                log.info("")
                log.info(f" -- Validation epoch starting at {epoch_start_time}")

            metrics_epoch_history = self.sub_validate(val_loader, forward_type='val')

            if self.early_stopping is not None:
                stop_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)
                if self.rank == 0:
                    self.early_stopping(metrics_epoch_history['loss'])
                    stop_flag.fill_(int(self.early_stopping.should_stop))
                if self.is_distributed:
                    torch.distributed.broadcast(stop_flag, src=0)
            else:
                stop_flag = torch.tensor(0, dtype=torch.uint8, device=self.device)

            if (self.scheduler is not None) and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(metrics_epoch_history['loss'])
                if self.optimizer.param_groups[0]['lr'] != current_lr:
                    log.info("Applied scheduler")
                    log.info(f"New learning rate is {self.scheduler.get_last_lr()}")

            epoch_end_time = datetime.now()

            if self.rank == 0:
                log.info(f" -- Validation epoch completed in {epoch_end_time - epoch_start_time} | Iteration : {self.iteration}")
                log.info(f" -- Total time since the beginning of the run : {epoch_end_time - start_run_time}")
                log.info(" -- Metrics over the (val) epoch " +
                         ", ".join(f"{k}: {v:.5g}" for k, v in metrics_epoch_history.items()))

                if self.wandb_run is not None:
                    self.wandb_run.log(
                        {'val_epoch_' + k: v for k, v in metrics_epoch_history.items()}
                    )

                if checkpointing:
                    self.save_state()

                if metrics_epoch_history["loss"] < self.best_validation_loss:
                    log.info(" ... Best validation loss so far!")
                    self.best_validation_loss = metrics_epoch_history["loss"]
                    self.save_state(suffix="_BEST")

            if self.early_stopping is not None:
                if stop_flag.item():
                    if self.rank == 0:
                        log.info("Early stopping triggered.")
                        if self.wandb_run is not None:
                            self.wandb_run.log({'early_stopped': True})
                    if self.is_distributed:
                        torch.distributed.barrier()
                    break

    def to_disk_data_reformat(self, **kw):
        return kw

    def make_plots(self, *a, **k):
        try:
            self._plot_curves()
        except Exception as e:
            if self.rank == 0:
                log.warning(f"make_plots failed: {e}")

    def save_state(self, suffix="", name=None):
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        filename = f"{self.dump_path}{name}{suffix}.pth"

        model_dict = self.module.state_dict()
        optimizer_dict = self.optimizer.state_dict() if self.optimizer is not None else {}
        scheduler_dict = self.scheduler.state_dict() if self.scheduler is not None else {}  # <-- ADDED

        torch.save({
            'global_step': self.iteration,
            'epoch': self.epoch,                 # <-- ADDED
            'optimizer': optimizer_dict,
            'scheduler': scheduler_dict,         # <-- ADDED
            'state_dict': model_dict
        }, filename)

        if self.wandb_run is not None and self.rank == 0 and suffix != "_LASTGOOD":
            artifact = wandb.Artifact(name=f"model-and-opti-checkpoints-{self.wandb_run.id}", type="model-and-opti")
            artifact.add_file(filename)
            artifact.metadata['checkpoints_dir'] = filename
            aliases = ['ite_' + str(self.iteration)]
            if suffix:
                aliases.append(suffix)
            if suffix == "_BEST":
                artifact.description = f"Validation loss : {self.best_validation_loss:.4g}"
                self.wandb_run.log({'best_val_epoch_loss': self.best_validation_loss})
            self.wandb_run.log_artifact(artifact, aliases=aliases)
            log.info("Save state on wandb")

        log.info(f"Saved state as: {filename}")
        return filename

    def restore_state(self, weight_file):
        with open(weight_file, 'rb') as f:
            log.info("\n\n")
            log.info(f"Restoring state from {weight_file}\n")

            if self.is_distributed:
                torch.distributed.barrier()

            checkpoint = torch.load(f, map_location=self.device)

            self.module.load_state_dict(checkpoint['state_dict'])

            if self.optimizer is not None and 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if self.scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler']:  # <-- ADDED
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                except Exception as e:
                    log.warning(f"Could not restore scheduler state: {e}")

            self.iteration = checkpoint.get('global_step', 0)
            self.epoch = checkpoint.get('epoch', self.epoch)  # <-- ADDED

    def restore_best_state(self, name=None, complete_path=False):
        if name is None:
            name = f"{self.__class__.__name__}_{self.module.__class__.__name__}"
        if complete_path:
            full_path = name
        else:
            full_path = f"{self.dump_path}{name}_BEST.pth"
        self.restore_state(full_path)
