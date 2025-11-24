# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from importlib import import_module
from hydra.utils import instantiate

def build_from_path(path: str, kwargs: Dict[str, Any]):
    mod_name, cls_name = path.rsplit(":", 1)
    mod = import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls(**(kwargs or {}))

def build_encoder(cfg: Dict[str, Any]):
    return build_from_path(cfg["target"], cfg.get("params", {}))

def build_head(cfg: Dict[str, Any]):
    return build_from_path(cfg["target"], cfg.get("params", {}))


def build_segmentation_model(encoder, head, wrapper):
    enc = instantiate(encoder)
    hd  = instantiate(head)
    model = instantiate(wrapper, encoder=enc, head=hd)
    return model

