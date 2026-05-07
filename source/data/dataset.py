# -*- coding: utf-8 -*-
"""
source/data/dataset.py

ACT-style HDF5 dataset for nrs_imitation.

Expected common HDF5 layout
---------------------------
observations/position        (T, 6)
observations/force           (T, 3)
observations/images/cam0     (T, H, W, 3) or (T, 3, H, W)

action/position              (T, 6)
action/force                 (T, 3)

The dataset returns:
    image         : torch.FloatTensor (K, 3, H, W), image in [0,1]
    qpos          : torch.FloatTensor (9,)
    action        : torch.FloatTensor (seq_len, 9)
    is_pad        : torch.BoolTensor  (seq_len,)
    force_history : torch.FloatTensor (L, 3), optional

Normalization is controlled by stats:
    stats["qpos_norm_mode"]
    stats["action_norm_mode"]

Raw HDF5 files are not rewritten.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .normalization import normalize, canonical_norm_mode


def natural_key(path):
    name = Path(path).name
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def _read_dataset(f: h5py.File, candidates: Sequence[str]):
    for key in candidates:
        k = key[1:] if key.startswith("/") else key
        if k in f and isinstance(f[k], h5py.Dataset):
            return f[k]
    raise KeyError(f"None of dataset keys exists: {candidates}")


def _find_image_dataset(f: h5py.File, camera_name: str):
    candidates = [
        f"observations/images/{camera_name}",
        f"/observations/images/{camera_name}",
        f"images/{camera_name}",
        f"/images/{camera_name}",
        camera_name,
        f"/{camera_name}",
    ]
    for key in candidates:
        k = key[1:] if key.startswith("/") else key
        if k in f and isinstance(f[k], h5py.Dataset):
            return f[k]

    # fallback: first image-like dataset
    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            lname = name.lower()
            if "image" in lname or "cam" in lname:
                found.append(name)
    f.visititems(visitor)
    if found:
        return f[found[0]]
    raise KeyError(f"Could not find image dataset for camera={camera_name}")


def _decode_frame(arr: np.ndarray) -> np.ndarray:
    """Return uint8 RGB image with shape (H,W,3)."""
    arr = np.asarray(arr)

    if arr.ndim == 1 and arr.dtype == np.uint8:
        # compressed image bytes
        try:
            import cv2
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise RuntimeError(f"Failed to decode compressed image: {e}")

    if arr.ndim == 3:
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] != 3:
            raise RuntimeError(f"Unsupported image shape: {arr.shape}")
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=-1)

    raise RuntimeError(f"Unsupported frame shape: {arr.shape}, dtype={arr.dtype}")


def _to_chw_float01(rgb: np.ndarray) -> np.ndarray:
    rgb = _decode_frame(rgb)
    chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return chw


def _concat_pos_force(pos, force) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    force = np.asarray(force, dtype=np.float32)

    if pos.ndim == 1:
        pos = pos.reshape(1, -1)
    if force.ndim == 1:
        force = force.reshape(1, -1)

    if pos.shape[-1] >= 6:
        pos = pos[..., :6]
    if force.shape[-1] >= 3:
        force = force[..., :3]

    return np.concatenate([pos, force], axis=-1).astype(np.float32)


def read_qpos_action_arrays(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read full qpos/action arrays from one episode file.

    Returns:
        qpos   : (T,9)
        action : (T,9)
    """
    with h5py.File(h5_path, "r") as f:
        # observations
        if "observations/qpos" in f:
            qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
            if qpos.shape[-1] != 9:
                # fallback to explicit fields
                obs_pos = _read_dataset(f, ["observations/position", "observations/pose", "qpos/position"])
                obs_force = _read_dataset(f, ["observations/force", "force", "observations/ft"])
                qpos = _concat_pos_force(obs_pos[:], obs_force[:])
        else:
            obs_pos = _read_dataset(f, ["observations/position", "observations/pose", "position"])
            obs_force = _read_dataset(f, ["observations/force", "force", "observations/ft"])
            qpos = _concat_pos_force(obs_pos[:], obs_force[:])

        # actions
        if "action" in f and isinstance(f["action"], h5py.Dataset):
            action = np.asarray(f["action"], dtype=np.float32)
            if action.shape[-1] != 9:
                act_pos = _read_dataset(f, ["action/position", "action/pose", "target/position"])
                act_force = _read_dataset(f, ["action/force", "target/force"])
                action = _concat_pos_force(act_pos[:], act_force[:])
        elif "actions" in f and isinstance(f["actions"], h5py.Dataset):
            action = np.asarray(f["actions"], dtype=np.float32)
            if action.shape[-1] != 9:
                act_pos = _read_dataset(f, ["action/position", "action/pose", "target/position"])
                act_force = _read_dataset(f, ["action/force", "target/force"])
                action = _concat_pos_force(act_pos[:], act_force[:])
        else:
            act_pos = _read_dataset(f, ["action/position", "action/pose", "target/position"])
            act_force = _read_dataset(f, ["action/force", "target/force"])
            action = _concat_pos_force(act_pos[:], act_force[:])

    qpos = np.asarray(qpos, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)

    if qpos.ndim != 2 or qpos.shape[-1] != 9:
        raise RuntimeError(f"qpos must be (T,9), got {qpos.shape} in {h5_path}")
    if action.ndim != 2 or action.shape[-1] != 9:
        raise RuntimeError(f"action must be (T,9), got {action.shape} in {h5_path}")

    T = min(qpos.shape[0], action.shape[0])
    return qpos[:T], action[:T]


class EpisodicDataset(Dataset):
    def __init__(
        self,
        episode_files: Sequence[str],
        camera_names: Sequence[str],
        stats: Dict,
        seq_len: int,
        samples_per_episode: int = 50,
        seed: int = 0,
        return_force_history: bool = False,
        use_force_history: bool = False,
        force_history_len: int = 10,
    ):
        super().__init__()
        self.episode_files = [str(Path(p).expanduser()) for p in episode_files]
        self.camera_names = list(camera_names)
        self.stats = dict(stats)
        self.seq_len = int(seq_len)
        self.samples_per_episode = int(samples_per_episode)
        self.return_force_history = bool(return_force_history or use_force_history)
        self.use_force_history = bool(use_force_history or return_force_history)
        self.force_history_len = int(force_history_len)

        self.qpos_norm_mode = canonical_norm_mode(self.stats.get("qpos_norm_mode", "minmax_01"))
        self.action_norm_mode = canonical_norm_mode(self.stats.get("action_norm_mode", "minmax_01"))

        self.index_map: List[Tuple[int, int]] = []
        rng = np.random.default_rng(seed)

        for ep_idx, p in enumerate(self.episode_files):
            qpos, action = read_qpos_action_arrays(p)
            T = min(qpos.shape[0], action.shape[0])
            if T <= 0:
                continue

            if self.samples_per_episode > 0:
                starts = rng.integers(low=0, high=T, size=self.samples_per_episode).tolist()
            else:
                starts = list(range(T))

            for s in starts:
                self.index_map.append((ep_idx, int(s)))

        if len(self.index_map) == 0:
            raise RuntimeError("No dataset samples were created. Check episode files.")

    def __len__(self):
        return len(self.index_map)

    def _read_action_chunk(self, action_raw: np.ndarray, start: int):
        T = action_raw.shape[0]
        end = start + self.seq_len

        chunk = np.zeros((self.seq_len, 9), dtype=np.float32)
        is_pad = np.zeros((self.seq_len,), dtype=bool)

        valid_end = min(end, T)
        valid_len = max(0, valid_end - start)

        if valid_len > 0:
            chunk[:valid_len] = action_raw[start:valid_end]

        if valid_len < self.seq_len:
            is_pad[valid_len:] = True
            if T > 0:
                fill = action_raw[T - 1]
                chunk[valid_len:] = fill

        return chunk, is_pad

    def _read_force_history(self, qpos_raw: np.ndarray, start: int) -> np.ndarray:
        L = max(1, self.force_history_len)
        T = qpos_raw.shape[0]
        force = qpos_raw[:, 6:9].astype(np.float32)

        end = min(max(start, 0), T - 1)
        begin = end - L + 1

        if begin < 0:
            pad_count = -begin
            pad_value = force[0:1]
            hist = force[0:end + 1]
            hist = np.concatenate([np.repeat(pad_value, pad_count, axis=0), hist], axis=0)
        else:
            hist = force[begin:end + 1]

        if hist.shape[0] < L:
            pad_count = L - hist.shape[0]
            pad_value = hist[0:1] if hist.shape[0] > 0 else force[0:1]
            hist = np.concatenate([np.repeat(pad_value, pad_count, axis=0), hist], axis=0)

        hist = hist[-L:].astype(np.float32)

        qmin = np.asarray(self.stats["qpos_min"], dtype=np.float32)[6:9]
        qmax = np.asarray(self.stats["qpos_max"], dtype=np.float32)[6:9]
        hist_n = normalize(hist, qmin, qmax, mode=self.qpos_norm_mode)
        return hist_n.astype(np.float32)

    def __getitem__(self, index: int):
        ep_idx, start = self.index_map[index]
        h5_path = self.episode_files[ep_idx]

        with h5py.File(h5_path, "r") as f:
            qpos_raw, action_raw = read_qpos_action_arrays(h5_path)
            T = min(qpos_raw.shape[0], action_raw.shape[0])
            start = int(np.clip(start, 0, max(T - 1, 0)))

            # Image stack, current frame only.
            imgs = []
            for cam in self.camera_names:
                dset = _find_image_dataset(f, cam)
                img_idx = min(start, int(dset.shape[0]) - 1)
                imgs.append(_to_chw_float01(dset[img_idx]))
            image = np.stack(imgs, axis=0).astype(np.float32)  # (K,3,H,W)

        qpos = qpos_raw[start].astype(np.float32)
        action, is_pad = self._read_action_chunk(action_raw, start)

        qpos_n = normalize(
            qpos,
            self.stats["qpos_min"],
            self.stats["qpos_max"],
            mode=self.qpos_norm_mode,
        ).astype(np.float32)

        action_n = normalize(
            action,
            self.stats["action_min"],
            self.stats["action_max"],
            mode=self.action_norm_mode,
        ).astype(np.float32)

        image_t = torch.from_numpy(image)
        qpos_t = torch.from_numpy(qpos_n)
        action_t = torch.from_numpy(action_n)
        is_pad_t = torch.from_numpy(is_pad)

        if self.return_force_history:
            force_hist = self._read_force_history(qpos_raw, start)
            force_hist_t = torch.from_numpy(force_hist)
            return image_t, qpos_t, action_t, is_pad_t, force_hist_t

        return image_t, qpos_t, action_t, is_pad_t