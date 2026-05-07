#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_flow.py

RGB-conditioned Flow Matching training script for nrs_imitation.

This is the first-stage Flow RGB baseline:
    - Use the existing ACT-compatible HDF5 dataset.
    - Use cam0 RGB + qpos + optional force_history as observation.
    - Predict the same 9D action chunk as ACT:
        [x, y, z, wx, wy, wz, fx, fy, fz]
    - No virtual pose.
    - No impedance/compliance output.
    - Low-level admittance controller remains unchanged.

Default dataset:
    /home/eunseop/nrs_imitation/datasets/ACT/<latest_timestamp>/episodes_ft

Default checkpoint:
    /home/eunseop/nrs_imitation/checkpoints/flow/ur10e_swing/<timestamp>

Run:
    cd ~/nrs_imitation
    python3 scripts/flow/train_flow.py

Eval-load check:
    python3 scripts/flow/train_flow.py --eval
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")

for p in [_PROJECT_ROOT, _SOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import numpy as np
import h5py
from tqdm import tqdm

from data.loader import load_data
from common.utils import set_seed
from common.fs import find_latest_timestamped_subdir
from models.flow_core import build_flow_rgb_policy_and_optimizer


# =============================================================================
# Dataset helpers
# =============================================================================

TASK_CONFIGS = {}
try:
    from custom.custom_constants import TASK_CONFIGS as _TC
    TASK_CONFIGS = _TC
except Exception:
    TASK_CONFIGS = {}


def _is_probably_timestamp_dir(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            continue
    return False


def _episode_files(dataset_dir: str) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return []
    return sorted(d.glob("episode_*.hdf5"))


def _count_episodes(dataset_dir: str) -> int:
    return len(_episode_files(dataset_dir))


def find_latest_episode_dir(
    root_dir: str = "/home/eunseop/nrs_imitation/datasets/ACT",
    subdir_name: str = "episodes_ft",
) -> str:
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        ep_dir = run_dir / subdir_name
        if not ep_dir.is_dir():
            continue
        n = _count_episodes(str(ep_dir))
        if n <= 0:
            continue
        timestamp_bonus = 1 if _is_probably_timestamp_dir(run_dir.name) else 0
        candidates.append((timestamp_bonus, run_dir.name, ep_dir.stat().st_mtime, ep_dir, n))

    if not candidates:
        for ep_dir in root.rglob(subdir_name):
            if not ep_dir.is_dir():
                continue
            n = _count_episodes(str(ep_dir))
            if n <= 0:
                continue
            parent_name = ep_dir.parent.name
            timestamp_bonus = 1 if _is_probably_timestamp_dir(parent_name) else 0
            candidates.append((timestamp_bonus, parent_name, ep_dir.stat().st_mtime, ep_dir, n))

    if not candidates:
        raise FileNotFoundError(
            f"No usable dataset directory found for subdir={subdir_name} under {root}"
        )

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def resolve_dataset_dir(dataset_dir: Optional[str], cam_preprocess: str, task_name: str) -> str:
    if dataset_dir is not None and str(dataset_dir).strip():
        resolved = os.path.expanduser(dataset_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        return resolved

    subdir_name = "episodes_ft_camproc" if cam_preprocess == "stabilize_crop" else "episodes_ft"

    if task_name in TASK_CONFIGS and "dataset_dir" in TASK_CONFIGS[task_name]:
        resolved = os.path.expanduser(str(TASK_CONFIGS[task_name]["dataset_dir"]))
        if os.path.isdir(resolved) and _count_episodes(resolved) > 0:
            return resolved

    latest = find_latest_episode_dir(
        root_dir="/home/eunseop/nrs_imitation/datasets/ACT",
        subdir_name=subdir_name,
    )
    print(f"[AUTO] dataset_dir not provided -> using latest {subdir_name}: {latest}")
    return latest


def parse_camera_names(camera_names_arg) -> List[str]:
    if camera_names_arg is None:
        return ["cam0"]
    if isinstance(camera_names_arg, str):
        raw = [camera_names_arg]
    else:
        raw = list(camera_names_arg)

    out = []
    for item in raw:
        for part in str(item).split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out if out else ["cam0"]


# =============================================================================
# Demo-start pose statistics
# =============================================================================

def _read_first_dataset_row(f: h5py.File, candidate_keys: List[str]) -> Optional[np.ndarray]:
    """
    Read the first row of the first existing dataset among candidate_keys.

    Supported final episode layouts:
      - /observations/position, /observations/force
      - /position, /ft  (fallback for merged-like files)
    """
    for key in candidate_keys:
        if key not in f:
            continue
        arr = f[key]
        if arr.shape[0] <= 0:
            continue
        row = np.asarray(arr[0], dtype=np.float32).reshape(-1)
        return row.copy()
    return None


def collect_demo_start_pose_stats(
    dataset_dir: str,
    num_episodes: int = 0,
    max_store_episodes: int = 10000,
) -> Dict[str, object]:
    """
    Collect initial pose statistics from final ACT-format episode_*.hdf5 files.

    This stores the average initial pose of all training episodes into dataset_stats.pkl,
    so the inference node can later move the robot from its current pose to the
    demonstration initial pose before starting policy inference.

    Saved keys:
      demo_start_pose_mean : (6,)  [x, y, z, wx, wy, wz]
      demo_start_pose_std  : (6,)
      demo_start_pose_min  : (6,)
      demo_start_pose_max  : (6,)
      demo_start_pose_all  : (N,6)

      demo_start_qpos_mean : (9,)  [x, y, z, wx, wy, wz, fx, fy, fz]
      demo_start_qpos_std  : (9,)
      demo_start_qpos_min  : (9,)
      demo_start_qpos_max  : (9,)
      demo_start_qpos_all  : (N,9)

      demo_start_num_episodes
      demo_start_source_dataset_dir
      demo_start_episode_files
    """
    files = _episode_files(dataset_dir)
    if num_episodes is not None and int(num_episodes) > 0:
        files = files[: int(num_episodes)]

    poses: List[np.ndarray] = []
    qposes: List[np.ndarray] = []
    used_files: List[str] = []
    skipped: List[Tuple[str, str]] = []

    for path in files:
        try:
            with h5py.File(str(path), "r") as f:
                p0 = _read_first_dataset_row(
                    f,
                    [
                        "observations/position",
                        "/observations/position",
                        "position",
                        "/position",
                    ],
                )
                if p0 is None or p0.size < 6:
                    skipped.append((str(path), "missing first position row"))
                    continue

                pose0 = p0[:6].astype(np.float32)

                f0 = _read_first_dataset_row(
                    f,
                    [
                        "observations/force",
                        "/observations/force",
                        "ft",
                        "/ft",
                        "force",
                        "/force",
                    ],
                )
                if f0 is None or f0.size < 3:
                    force0 = np.zeros(3, dtype=np.float32)
                else:
                    force0 = f0[:3].astype(np.float32)

                qpos0 = np.concatenate([pose0, force0], axis=0).astype(np.float32)

                poses.append(pose0)
                qposes.append(qpos0)
                used_files.append(str(path))

        except Exception as e:
            skipped.append((str(path), repr(e)))

    if not poses:
        print("[WARN] demo-start stats: no valid episode initial poses were found.")
        if skipped:
            print("[WARN] demo-start stats: first skipped examples:")
            for path, reason in skipped[:5]:
                print(f"       - {path}: {reason}")
        return {}

    pose_all = np.stack(poses, axis=0).astype(np.float32)
    qpos_all = np.stack(qposes, axis=0).astype(np.float32)

    if max_store_episodes is not None and int(max_store_episodes) > 0:
        pose_store = pose_all[: int(max_store_episodes)].copy()
        qpos_store = qpos_all[: int(max_store_episodes)].copy()
        file_store = used_files[: int(max_store_episodes)]
    else:
        pose_store = pose_all.copy()
        qpos_store = qpos_all.copy()
        file_store = used_files

    out = {
        "demo_start_pose_mean": pose_all.mean(axis=0).astype(np.float32),
        "demo_start_pose_std": pose_all.std(axis=0).astype(np.float32),
        "demo_start_pose_min": pose_all.min(axis=0).astype(np.float32),
        "demo_start_pose_max": pose_all.max(axis=0).astype(np.float32),
        "demo_start_pose_all": pose_store,

        "demo_start_qpos_mean": qpos_all.mean(axis=0).astype(np.float32),
        "demo_start_qpos_std": qpos_all.std(axis=0).astype(np.float32),
        "demo_start_qpos_min": qpos_all.min(axis=0).astype(np.float32),
        "demo_start_qpos_max": qpos_all.max(axis=0).astype(np.float32),
        "demo_start_qpos_all": qpos_store,

        "demo_start_num_episodes": int(pose_all.shape[0]),
        "demo_start_source_dataset_dir": str(Path(dataset_dir).expanduser()),
        "demo_start_episode_files": file_store,
    }

    print("[DEMO_START] collected initial pose statistics")
    print(f"[DEMO_START] used episodes = {pose_all.shape[0]} / candidate files = {len(files)}")
    if skipped:
        print(f"[DEMO_START] skipped episodes = {len(skipped)}")
    print(
        "[DEMO_START] pose_mean [x y z wx wy wz] = "
        + np.array2string(out["demo_start_pose_mean"], precision=4, separator=", ")
    )
    print(
        "[DEMO_START] pose_std  [x y z wx wy wz] = "
        + np.array2string(out["demo_start_pose_std"], precision=4, separator=", ")
    )
    print(
        "[DEMO_START] qpos_mean [x y z wx wy wz fx fy fz] = "
        + np.array2string(out["demo_start_qpos_mean"], precision=4, separator=", ")
    )

    return out


# =============================================================================
# Training helpers
# =============================================================================

def _unpack_batch(batch, device: torch.device):
    """
    Supports existing loader outputs:
      1) image, qpos, action, is_pad
      2) image, qpos, action, is_pad, force_history
    """
    if len(batch) == 4:
        image, qpos, action, is_pad = batch
        force_history = None
    elif len(batch) == 5:
        image, qpos, action, is_pad, force_history = batch
    else:
        raise RuntimeError(f"Unexpected batch length: {len(batch)}")

    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)

    if force_history is not None:
        force_history = force_history.to(device, non_blocking=True)

    return image, qpos, action, is_pad, force_history


def _scalar_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for k, v in loss_dict.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu().item())
        else:
            out[k] = float(v)
    return out


def _mean_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {k: sum(d[k] for d in items) / len(items) for k in keys}


@torch.no_grad()
def validate(policy, val_loader, device, amp: bool = False) -> Dict[str, float]:
    policy.eval()
    vals = []

    for batch in val_loader:
        image, qpos, action, is_pad, force_history = _unpack_batch(batch, device)
        if amp and device.type == "cuda":
            with torch.cuda.amp.autocast():
                out = policy(qpos, image, action, is_pad, force_history=force_history)
        else:
            out = policy(qpos, image, action, is_pad, force_history=force_history)
        vals.append(_scalar_dict(out))

    return _mean_dict(vals)


def save_checkpoint(
    path: str,
    epoch: int,
    policy,
    optimizer,
    train_summary,
    val_summary,
    config: dict,
):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "config": config,
    }
    torch.save(payload, path)


def train_flow(train_loader, val_loader, config: dict):
    device = config["device"]
    ckpt_dir = config["ckpt_dir"]
    num_epochs = int(config["num_epochs"])
    seed = int(config.get("seed", 0))
    save_every = int(config.get("save_every", 100))
    debug_batches = int(config.get("debug_batches", 3))
    amp = bool(config.get("amp", False))

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)

    policy, optimizer = build_flow_rgb_policy_and_optimizer(config["policy_config"])
    policy = policy.to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[FlowRGBPolicy] trainable params = {n_params / 1e6:.2f}M")

    best_val = float("inf")
    best_epoch = -1
    best_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    last_path = os.path.join(ckpt_dir, "policy_last.ckpt")

    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        print(f"Epoch {epoch}")

        val_summary = validate(policy, val_loader, device, amp=amp)
        val_loss = val_summary.get("loss", float("nan"))
        if val_summary:
            val_msg = " | ".join(f"{k}:{v:.6f}" for k, v in sorted(val_summary.items()))
            print(f"Val: {val_msg}")

        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            save_checkpoint(
                path=best_path,
                epoch=epoch,
                policy=policy,
                optimizer=optimizer,
                train_summary=None,
                val_summary=val_summary,
                config=config,
            )

        policy.train()
        train_items = []

        for batch_idx, batch in enumerate(train_loader):
            image, qpos, action, is_pad, force_history = _unpack_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)

            if amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    out = policy(qpos, image, action, is_pad, force_history=force_history)
                    loss = out["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = policy(qpos, image, action, is_pad, force_history=force_history)
                loss = out["loss"]
                loss.backward()
                optimizer.step()

            scalars = _scalar_dict(out)
            train_items.append(scalars)

            if batch_idx < debug_batches:
                print(f"[DEBUG] Epoch {epoch}, batch {batch_idx}, train loss = {scalars['loss']:.6f}")

        train_summary = _mean_dict(train_items)
        train_loss = train_summary.get("loss", float("nan"))

        if save_every > 0 and epoch % save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"),
                epoch=epoch,
                policy=policy,
                optimizer=optimizer,
                train_summary=train_summary,
                val_summary=val_summary,
                config=config,
            )

        save_checkpoint(
            path=last_path,
            epoch=epoch,
            policy=policy,
            optimizer=optimizer,
            train_summary=train_summary,
            val_summary=val_summary,
            config=config,
        )

        pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

    print("[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_epoch}")
    print(f"[INFO] Best val loss  = {best_val:.6f}")
    print(f"[INFO] Best ckpt path = {best_path}")
    print(f"[INFO] Last ckpt path = {last_path}")

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_ckpt_path": best_path,
        "last_ckpt_path": last_path,
    }


# =============================================================================
# Main
# =============================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    task_name = args.task_name
    dataset_dir = resolve_dataset_dir(
        dataset_dir=args.dataset_dir,
        cam_preprocess=args.cam_preprocess,
        task_name=task_name,
    )

    num_episodes = int(args.num_episodes)
    if num_episodes <= 0:
        num_episodes = _count_episodes(dataset_dir)

    camera_names = parse_camera_names(args.camera_names)

    if args.train_seq_len is None:
        args.train_seq_len = int(args.chunk_size)
    if args.val_seq_len is None:
        args.val_seq_len = int(args.chunk_size)

    print(f"[INFO] task_name          = {task_name}")
    print(f"[INFO] dataset_dir        = {dataset_dir}")
    print(f"[INFO] cam_preprocess     = {args.cam_preprocess}")
    print(f"[INFO] norm_mode          = {args.norm_mode}")
    print(f"[INFO] num_episodes       = {num_episodes}")
    print(f"[INFO] camera_names       = {camera_names}")
    print(f"[INFO] chunk_size         = {args.chunk_size}")
    print(f"[INFO] train_seq_len      = {args.train_seq_len}")
    print(f"[INFO] val_seq_len        = {args.val_seq_len}")
    print(f"[INFO] samples/ep         = {args.samples_per_episode}")
    print(f"[INFO] save_every         = {args.save_every}")
    print(f"[INFO] batch_size         = {args.batch_size}")
    print(f"[INFO] AMP                = {args.amp}")
    print(f"[INFO] use_force_history  = {args.use_force_history}")
    print(f"[INFO] force_history_len  = {args.force_history_len}")
    print(f"[INFO] flow_infer_steps   = {args.flow_infer_steps}")
    if args.norm_mode == "minmax_m11":
        print("[INFO] qpos/action norm  = min-max per-dim -> [-1,1]")
    else:
        print("[INFO] qpos/action norm  = min-max per-dim -> [0,1]")
    print("[INFO] image norm        = raw RGB -> [0,1], then ImageNet normalization inside policy")

    policy_config = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "beta1": args.beta1,
        "beta2": args.beta2,

        "num_queries": args.chunk_size,
        "state_dim": args.state_dim,
        "action_dim": args.action_dim,
        "force_dim": args.force_dim,

        "camera_names": camera_names,
        "pretrained_backbone": not args.no_pretrained,

        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
        "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
        "force_encoder_num_layers": args.force_encoder_num_layers,
        "force_encoder_dropout": args.force_encoder_dropout,

        "flow_obs_hidden_dim": args.flow_obs_hidden_dim,
        "flow_image_feature_dim": args.flow_image_feature_dim,
        "flow_global_cond_dim": args.flow_global_cond_dim,

        "flow_time_embed_dim": args.flow_time_embed_dim,
        "flow_down_dims": args.flow_down_dims,
        "flow_kernel_size": args.flow_kernel_size,
        "flow_n_groups": args.flow_n_groups,
        "flow_cond_predict_scale": args.flow_cond_predict_scale,

        "flow_infer_steps": args.flow_infer_steps,
        "flow_train_eps": args.flow_train_eps,
        "flow_loss_type": args.flow_loss_type,
        "norm_mode": args.norm_mode,
    }

    if args.eval:
        ckpt_dir = args.ckpt_dir
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest = find_latest_timestamped_subdir(ckpt_dir)
            if latest is None:
                raise FileNotFoundError(f"No policy_best.ckpt in {ckpt_dir}, and no timestamp subdir found.")
            ckpt_dir = latest
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"policy_best.ckpt not found: {best_ckpt}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found: {stats_path}")

        policy, _ = build_flow_rgb_policy_and_optimizer(policy_config)
        policy = policy.to(device)

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        ckpt = torch.load(best_ckpt, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        policy.eval()

        print(f"[INFO] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("[WARN] first missing keys:", list(missing)[:10])
        if len(unexpected) > 0:
            print("[WARN] first unexpected keys:", list(unexpected)[:10])

        with open(stats_path, "rb") as f:
            _ = pickle.load(f)
        print(f"[INFO] Loaded dataset stats from {stats_path}")
        print("\n✅ FLOW-RGB model ready for inference wrapper.\n")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = os.path.join(args.ckpt_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    train_loader, val_loader, stats, meta = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        seq_len_train=args.train_seq_len,
        seq_len_val=args.val_seq_len,
        seed=args.seed,
        samples_per_episode=args.samples_per_episode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        return_force_history=args.use_force_history,
        use_force_history=args.use_force_history,
        force_history_len=args.force_history_len,
        qpos_norm_mode=args.norm_mode,
        action_norm_mode=args.norm_mode,
    )
    print(f"[INFO] data meta: {meta}")

    # ------------------------------------------------------------
    # Demo-start pose statistics for inference-time auto alignment
    # ------------------------------------------------------------
    demo_start_stats = collect_demo_start_pose_stats(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
    )
    if demo_start_stats:
        stats.update(demo_start_stats)
    else:
        print("[WARN] dataset_stats.pkl will be saved without demo_start_pose_mean.")

    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {stats_path}")

    config = {
        "device": device,
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "ckpt_dir": ckpt_dir,
        "save_every": args.save_every,
        "amp": args.amp,
        "debug_batches": args.debug_batches,
        "policy_class": "FLOW",
        "policy_config": policy_config,
    }

    train_flow(train_loader, val_loader, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--task_name", type=str, default="ur10e_swing")
    parser.add_argument("--ckpt_dir", type=str, default="/home/eunseop/nrs_imitation/checkpoints/flow/ur10e_swing")

    parser.add_argument("--dataset_dir", type=str, default=None)
    parser.add_argument("--cam_preprocess", type=str, default="stabilize_crop", choices=["off", "stabilize_crop"])
    parser.add_argument("--norm_mode", type=str, default="minmax_m11", choices=["minmax_01", "minmax_m11"])
    parser.add_argument("--num_episodes", type=int, default=0)
    parser.add_argument("--camera_names", nargs="+", default=["cam0"])

    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--train_seq_len", type=int, default=None)
    parser.add_argument("--val_seq_len", type=int, default=None)
    parser.add_argument("--samples_per_episode", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--state_dim", type=int, default=9)
    parser.add_argument("--action_dim", type=int, default=9)
    parser.add_argument("--force_dim", type=int, default=3)

    parser.add_argument("--use_force_history", dest="use_force_history", action="store_true", default=True)
    parser.add_argument("--no_force_history", dest="use_force_history", action="store_false")
    parser.add_argument("--force_history_len", type=int, default=10)
    parser.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    parser.add_argument("--force_encoder_num_layers", type=int, default=1)
    parser.add_argument("--force_encoder_dropout", type=float, default=0.0)

    parser.add_argument("--no_pretrained", action="store_true", default=False)

    parser.add_argument("--flow_obs_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_image_feature_dim", type=int, default=512)
    parser.add_argument("--flow_global_cond_dim", type=int, default=256)
    parser.add_argument("--flow_time_embed_dim", type=int, default=256)
    parser.add_argument("--flow_down_dims", type=str, default="256,512,1024")
    parser.add_argument("--flow_kernel_size", type=int, default=5)
    parser.add_argument("--flow_n_groups", type=int, default=8)
    parser.add_argument("--flow_cond_predict_scale", action="store_true", default=False)
    parser.add_argument("--flow_train_eps", type=float, default=1e-4)
    parser.add_argument("--flow_loss_type", type=str, default="mse", choices=["mse", "l1"])
    parser.add_argument("--flow_infer_steps", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--debug_batches", type=int, default=3)

    args = parser.parse_args()
    main(args)