# -*- coding: utf-8 -*-
"""
source/data/normalization.py

Normalization utilities for nrs_imitation.

Supported modes
---------------
minmax_01:
    x_norm = (x - x_min) / (x_max - x_min)          -> [0, 1]

minmax_m11:
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1  -> [-1, 1]

zscore:
    x_norm = (x - mean) / std

Notes
-----
- HDF5 raw data is never modified.
- The normalization mode is stored inside dataset_stats.pkl so that inference
  can denormalize policy output correctly.
- Backward compatibility:
    old mode names such as "minmax" and missing mode keys are treated as
    "minmax_01".
"""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-6


def canonical_norm_mode(mode: Any, default: str = "minmax_01") -> str:
    if mode is None:
        mode = default

    m = str(mode).strip().lower()

    aliases_01 = {
        "minmax",
        "minmax_01",
        "01",
        "0_1",
        "[0,1]",
        "0to1",
        "zero_one",
    }
    aliases_m11 = {
        "minmax_m11",
        "m11",
        "-1_1",
        "[-1,1]",
        "minus1_1",
        "minus_one_one",
        "neg1_pos1",
    }
    aliases_z = {
        "zscore",
        "standard",
        "standardize",
        "meanstd",
        "mean_std",
    }

    if m in aliases_01:
        return "minmax_01"
    if m in aliases_m11:
        return "minmax_m11"
    if m in aliases_z:
        return "zscore"

    raise ValueError(f"Unknown normalization mode: {mode}")


def sanitize_minmax(vmin, vmax, eps: float = EPS):
    vmin = np.asarray(vmin, dtype=np.float32)
    vmax = np.asarray(vmax, dtype=np.float32)
    rng = np.maximum(vmax - vmin, eps)
    return vmin.astype(np.float32), (vmin + rng).astype(np.float32)


def sanitize_std(std, eps: float = EPS):
    std = np.asarray(std, dtype=np.float32)
    return np.maximum(std, eps).astype(np.float32)


def normalize_minmax_01(x, vmin, vmax, eps: float = EPS):
    vmin, vmax = sanitize_minmax(vmin, vmax, eps=eps)
    return ((np.asarray(x, dtype=np.float32) - vmin) / (vmax - vmin + eps)).astype(np.float32)


def denormalize_minmax_01(x, vmin, vmax, eps: float = EPS):
    vmin, vmax = sanitize_minmax(vmin, vmax, eps=eps)
    return (np.asarray(x, dtype=np.float32) * (vmax - vmin) + vmin).astype(np.float32)


def normalize_minmax_m11(x, vmin, vmax, eps: float = EPS):
    x01 = normalize_minmax_01(x, vmin, vmax, eps=eps)
    return (2.0 * x01 - 1.0).astype(np.float32)


def denormalize_minmax_m11(x, vmin, vmax, eps: float = EPS):
    x01 = 0.5 * (np.asarray(x, dtype=np.float32) + 1.0)
    return denormalize_minmax_01(x01, vmin, vmax, eps=eps)


def normalize_zscore(x, mean, std, eps: float = EPS):
    mean = np.asarray(mean, dtype=np.float32)
    std = sanitize_std(std, eps=eps)
    return ((np.asarray(x, dtype=np.float32) - mean) / std).astype(np.float32)


def denormalize_zscore(x, mean, std, eps: float = EPS):
    mean = np.asarray(mean, dtype=np.float32)
    std = sanitize_std(std, eps=eps)
    return (np.asarray(x, dtype=np.float32) * std + mean).astype(np.float32)


def normalize(x, a, b, mode: str = "minmax_01", eps: float = EPS):
    mode = canonical_norm_mode(mode)
    if mode == "minmax_01":
        return normalize_minmax_01(x, a, b, eps=eps)
    if mode == "minmax_m11":
        return normalize_minmax_m11(x, a, b, eps=eps)
    if mode == "zscore":
        return normalize_zscore(x, a, b, eps=eps)
    raise ValueError(mode)


def denormalize(x, a, b, mode: str = "minmax_01", eps: float = EPS):
    mode = canonical_norm_mode(mode)
    if mode == "minmax_01":
        return denormalize_minmax_01(x, a, b, eps=eps)
    if mode == "minmax_m11":
        return denormalize_minmax_m11(x, a, b, eps=eps)
    if mode == "zscore":
        return denormalize_zscore(x, a, b, eps=eps)
    raise ValueError(mode)


def norm_range_for_mode(mode: str):
    mode = canonical_norm_mode(mode)
    if mode == "minmax_01":
        return [0.0, 1.0]
    if mode == "minmax_m11":
        return [-1.0, 1.0]
    if mode == "zscore":
        return None
    raise ValueError(mode)