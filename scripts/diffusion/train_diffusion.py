#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "source")

for p in [_PROJECT_ROOT, _SOURCE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from training.engine import train_bc, make_policy
from common.fs import find_latest_timestamped_subdir
from data.loader import load_data


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
            pass
    return False


def _episode_files(dataset_dir: str) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    return sorted(d.glob("episode_*.hdf5")) if d.is_dir() else []


def _count_episodes(dataset_dir: str) -> int:
    return len(_episode_files(dataset_dir))


def find_latest_episode_dir(root_dir: str = "/home/eunseop/nrs_imitation/datasets/ACT", subdir_name: str = "episodes_ft") -> str:
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        ep_dir = run_dir / subdir_name
        if ep_dir.is_dir() and _count_episodes(str(ep_dir)) > 0:
            stat = ep_dir.stat()
            timestamp_bonus = 1 if _is_probably_timestamp_dir(run_dir.name) else 0
            candidates.append((timestamp_bonus, run_dir.name, stat.st_mtime, ep_dir))

    if not candidates:
        for ep_dir in root.rglob(subdir_name):
            if ep_dir.is_dir() and _count_episodes(str(ep_dir)) > 0:
                parent_name = ep_dir.parent.name
                stat = ep_dir.stat()
                timestamp_bonus = 1 if _is_probably_timestamp_dir(parent_name) else 0
                candidates.append((timestamp_bonus, parent_name, stat.st_mtime, ep_dir))

    if not candidates:
        raise FileNotFoundError(f"No usable dataset directory found for {subdir_name}")

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def resolve_dataset_dir(dataset_dir: Optional[str], cam_preprocess: str, task_name: str) -> str:
    if dataset_dir is not None and str(dataset_dir).strip():
        resolved = os.path.expanduser(dataset_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        return resolved

    subdir = "episodes_ft_camproc" if cam_preprocess == "stabilize_crop" else "episodes_ft"

    if task_name in TASK_CONFIGS and "dataset_dir" in TASK_CONFIGS[task_name]:
        resolved = os.path.expanduser(str(TASK_CONFIGS[task_name]["dataset_dir"]))
        if os.path.isdir(resolved) and _count_episodes(resolved) > 0:
            return resolved

    latest = find_latest_episode_dir("/home/eunseop/nrs_imitation/datasets/ACT", subdir_name=subdir)
    print(f"[AUTO] dataset_dir not provided -> using latest {subdir}: {latest}")
    return latest


def parse_camera_names(camera_names_arg) -> List[str]:
    raw = [camera_names_arg] if isinstance(camera_names_arg, str) else list(camera_names_arg)
    out = []
    for item in raw:
        for part in str(item).split(","):
            s = part.strip()
            if s:
                out.append(s)
    return out if out else ["cam0"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    task_name = args.task_name
    policy_class = "DIFFUSION"
    dataset_dir = resolve_dataset_dir(args.dataset_dir, args.cam_preprocess, task_name)
    num_episodes = int(args.num_episodes) if int(args.num_episodes) > 0 else _count_episodes(dataset_dir)
    camera_names = parse_camera_names(args.camera_names)

    print(f"[INFO] task_name            = {task_name}")
    print(f"[INFO] dataset_dir          = {dataset_dir}")
    print(f"[INFO] cam_preprocess       = {args.cam_preprocess}")
    print(f"[INFO] num_episodes         = {num_episodes}")
    print(f"[INFO] camera_names         = {camera_names}")
    print(f"[INFO] chunk_size           = {args.chunk_size}")
    print(f"[INFO] train_seq_len        = {args.train_seq_len}")
    print(f"[INFO] val_seq_len          = {args.val_seq_len}")
    print(f"[INFO] samples/ep           = {args.samples_per_episode}")
    print(f"[INFO] save_every           = {args.save_every}")
    print(f"[INFO] use_force_history    = {args.use_force_history}")
    print(f"[INFO] force_history_len    = {args.force_history_len}")
    print(f"[INFO] diffusion_train_steps= {args.diffusion_train_steps}")
    print(f"[INFO] diffusion_infer_steps= {args.diffusion_infer_steps}")

    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "camera_names": camera_names,
        "state_dim": 9,
        "action_dim": 9,
        "image_resize_hw": args.image_resize_hw,
        "image_pool_hw": args.image_pool_hw,
        "pretrained_backbone": (not args.no_pretrained),

        "position_dim": args.position_dim,
        "force_dim": args.force_dim,
        "position_encoder_hidden_dim": args.position_encoder_hidden_dim,
        "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
        "force_encoder_num_layers": args.force_encoder_num_layers,
        "force_encoder_dropout": args.force_encoder_dropout,
        "observation_encoder_activation": args.observation_encoder_activation,
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,

        "diffusion_train_steps": args.diffusion_train_steps,
        "diffusion_infer_steps": args.diffusion_infer_steps,
        "diffusion_beta_start": args.diffusion_beta_start,
        "diffusion_beta_end": args.diffusion_beta_end,
        "diffusion_loss_type": args.diffusion_loss_type,
        "diffusion_time_embed_dim": args.diffusion_time_embed_dim,
        "diffusion_down_dims": args.diffusion_down_dims,
        "diffusion_kernel_size": args.diffusion_kernel_size,
        "diffusion_n_groups": args.diffusion_n_groups,
        "diffusion_cond_predict_scale": args.diffusion_cond_predict_scale,
        "diffusion_obs_hidden_dim": args.diffusion_obs_hidden_dim,
        "diffusion_image_feature_dim": args.diffusion_image_feature_dim,
        "diffusion_global_cond_dim": args.diffusion_global_cond_dim,
        "weight_decay": args.weight_decay,
        "optimizer_betas": (args.beta1, args.beta2),
    }

    if args.eval:
        ckpt_dir = args.ckpt_dir
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest_sub = find_latest_timestamped_subdir(ckpt_dir)
            if latest_sub is None:
                raise FileNotFoundError(f"[EVAL] No policy_best.ckpt in {ckpt_dir}")
            ckpt_dir = latest_sub
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"[EVAL] policy_best.ckpt not found in {ckpt_dir}")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[EVAL] dataset_stats.pkl not found in {ckpt_dir}")

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        policy = make_policy(policy_class, policy_config).to(device)
        ckpt = torch.load(best_ckpt, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        policy.load_state_dict(state_dict, strict=False)
        policy.eval()

        with open(stats_path, "rb") as f:
            _ = pickle.load(f)
        print(f"[INFO] Loaded dataset stats from {stats_path}")
        print("\n✅ Diffusion model ready for inference!\n")
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
    )
    print(f"[INFO] data meta: {meta}")

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    config = {
        "num_epochs": args.num_epochs,
        "ckpt_dir": ckpt_dir,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "seed": args.seed,
        "device": device,
        "amp": args.amp,
        "debug_norm": args.debug_norm,
        "debug_norm_batches": 1,
        "save_every": args.save_every,
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
    }

    best = train_bc(train_loader, val_loader, config)
    print(f"[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best['best_epoch']}")
    print(f"[INFO] Best val loss  = {best['best_val_loss']:.6f}")
    print(f"[INFO] Best ckpt path = {best['best_ckpt_path']}")
    print(f"[INFO] Last ckpt path = {best['last_ckpt_path']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--ckpt_dir", type=str, default="/home/eunseop/nrs_imitation/checkpoints/diffusion/ur10e_swing")
    p.add_argument("--task_name", type=str, default="ur10e_swing")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--num_episodes", type=int, default=0)
    p.add_argument("--camera_names", nargs="+", default=["cam0"])
    p.add_argument("--cam_preprocess", type=str, default="off", choices=["off", "stabilize_crop"])
    p.add_argument("--chunk_size", type=int, default=200)
    p.add_argument("--train_seq_len", type=int, default=None)
    p.add_argument("--val_seq_len", type=int, default=None)
    p.add_argument("--samples_per_episode", type=int, default=50)
    p.add_argument("--save_every", type=int, default=100)

    p.add_argument("--no_pretrained", action="store_true", default=False)
    p.add_argument("--image_resize_hw", type=int, default=256)
    p.add_argument("--image_pool_hw", type=int, default=4)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--debug_norm", action="store_true")

    p.add_argument("--use_force_history", dest="use_force_history", action="store_true", default=True)
    p.add_argument("--no_force_history", dest="use_force_history", action="store_false")
    p.add_argument("--force_history_len", type=int, default=10)
    p.add_argument("--position_dim", type=int, default=6)
    p.add_argument("--force_dim", type=int, default=3)
    p.add_argument("--position_encoder_hidden_dim", type=int, default=128)
    p.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    p.add_argument("--force_encoder_num_layers", type=int, default=1)
    p.add_argument("--force_encoder_dropout", type=float, default=0.0)
    p.add_argument("--observation_encoder_activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])

    p.add_argument("--diffusion_train_steps", type=int, default=100)
    p.add_argument("--diffusion_infer_steps", type=int, default=10)
    p.add_argument("--diffusion_beta_start", type=float, default=1e-4)
    p.add_argument("--diffusion_beta_end", type=float, default=2e-2)
    p.add_argument("--diffusion_loss_type", type=str, default="mse", choices=["mse", "l1"])
    p.add_argument("--diffusion_time_embed_dim", type=int, default=256)
    p.add_argument("--diffusion_down_dims", type=str, default="256,512,1024")
    p.add_argument("--diffusion_kernel_size", type=int, default=5)
    p.add_argument("--diffusion_n_groups", type=int, default=8)
    p.add_argument("--diffusion_cond_predict_scale", action="store_true", default=False)
    p.add_argument("--diffusion_obs_hidden_dim", type=int, default=256)
    p.add_argument("--diffusion_image_feature_dim", type=int, default=512)
    p.add_argument("--diffusion_global_cond_dim", type=int, default=256)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--beta1", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.999)

    args = p.parse_args()
    if args.train_seq_len is None:
        args.train_seq_len = int(args.chunk_size)
    if args.val_seq_len is None:
        args.val_seq_len = int(args.chunk_size)
    if getattr(args, "debug_norm", False):
        sys.argv = [a for a in sys.argv if a != "--debug_norm"]
    main(args)
