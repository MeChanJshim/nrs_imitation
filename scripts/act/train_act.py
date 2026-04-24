#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ACT training & evaluation script (position+force+image -> position+force)

Normalization:
- qpos/action (9D): per-dimension MIN-MAX -> [0,1]
  x y z wx wy wz fx fy fz  각각 min/max로 정규화
- image: uint8 -> float in [0,1]  (policy.py 내부에서 ImageNet normalize는 그대로 유지)

IMPORTANT:
- policy.py 유지
- 데이터 로딩(episode_*.hdf5 검색) 유지
- 멀티카메라 입력 형태 (B,K,3,H,W) 유지
- debug_norm: 학습 시작 시 "정규화 이후" 9D + RGB(3ch) 각각 mean/std 출력

Force history:
- --use_force_history 를 켜면 dataset/loader/engine 경로를 통해
  force_history (B,T,3)를 policy/model까지 전달
- .hdf5 원본은 수정하지 않고 dataset.py 에서 on-the-fly 생성
"""

import os
import sys
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# -------------------------------------------------------------------------
# source/ 경로 추가
# scripts/act/train_act.py 에서 source/*.py 를 import 가능하게 함
# -------------------------------------------------------------------------
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


# -----------------------------
# Robust TASK_CONFIGS import (optional)
# -----------------------------
TASK_CONFIGS = {}
try:
    from custom.custom_constants import TASK_CONFIGS as _TC  # optional
    TASK_CONFIGS = _TC
except Exception:
    TASK_CONFIGS = {}


# ============================================================
# Dataset auto-resolve helpers
# ============================================================
def _is_probably_timestamp_dir(name: str) -> bool:
    """Return True for run directory names such as 20260421_1509 or 202604211509."""
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def _episode_files(dataset_dir: str) -> List[Path]:
    d = Path(dataset_dir).expanduser()
    if not d.is_dir():
        return []
    return sorted(d.glob("episode_*.hdf5"))


def _count_episodes(dataset_dir: str) -> int:
    return len(_episode_files(dataset_dir))


def find_latest_episodes_ft_dir(root_dir: str = "/home/eunseop/nrs_act/datasets/ACT") -> str:
    """
    Find the latest dataset directory that contains episode_*.hdf5.

    Preferred new layout:
      /home/eunseop/nrs_act/datasets/ACT/<RUN_ID>/episodes_ft/episode_*.hdf5

    Fallbacks:
      - any directory named episodes_ft under root
      - root itself, if it directly contains episode_*.hdf5
    """
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    candidates = []

    # Preferred: root/<RUN_ID>/episodes_ft
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        ep_dir = run_dir / "episodes_ft"
        if ep_dir.is_dir() and _count_episodes(str(ep_dir)) > 0:
            stat = ep_dir.stat()
            timestamp_bonus = 1 if _is_probably_timestamp_dir(run_dir.name) else 0
            candidates.append((timestamp_bonus, run_dir.name, stat.st_mtime, ep_dir))

    # Fallback: recursive search for episodes_ft
    if not candidates:
        for ep_dir in root.rglob("episodes_ft"):
            if ep_dir.is_dir() and _count_episodes(str(ep_dir)) > 0:
                parent_name = ep_dir.parent.name
                stat = ep_dir.stat()
                timestamp_bonus = 1 if _is_probably_timestamp_dir(parent_name) else 0
                candidates.append((timestamp_bonus, parent_name, stat.st_mtime, ep_dir))

    # Fallback: root directly contains episode_*.hdf5
    if not candidates and _count_episodes(str(root)) > 0:
        stat = root.stat()
        candidates.append((0, root.name, stat.st_mtime, root))

    if not candidates:
        raise FileNotFoundError(
            "No usable ACT dataset directory found. Expected episode_*.hdf5 under:\n"
            f"  {root}/<RUN_ID>/episodes_ft/"
        )

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return str(candidates[0][3])


def resolve_dataset_dir(dataset_dir: Optional[str], task_name: str) -> str:
    """
    Resolution order:
      1. --dataset_dir, if provided
      2. TASK_CONFIGS[task_name]['dataset_dir'], if available
      3. latest /home/eunseop/nrs_act/datasets/ACT/<RUN_ID>/episodes_ft
    """
    if dataset_dir is not None and str(dataset_dir).strip() != "":
        resolved = os.path.expanduser(dataset_dir)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"dataset_dir does not exist: {resolved}")
        return resolved

    if task_name in TASK_CONFIGS and "dataset_dir" in TASK_CONFIGS[task_name]:
        resolved = os.path.expanduser(str(TASK_CONFIGS[task_name]["dataset_dir"]))
        if os.path.isdir(resolved) and _count_episodes(resolved) > 0:
            return resolved
        print(f"[WARN] TASK_CONFIGS dataset_dir is invalid or empty, fallback to latest: {resolved}")

    latest = find_latest_episodes_ft_dir("/home/eunseop/nrs_act/datasets/ACT")
    print(f"[AUTO] dataset_dir not provided -> using latest episodes_ft: {latest}")
    return latest


def parse_camera_names(camera_names_arg) -> List[str]:
    """
    Accept either:
      --camera_names cam0
      --camera_names cam_top cam_ee
      --camera_names cam_top,cam_ee
    """
    if camera_names_arg is None:
        return ["cam0"]
    if isinstance(camera_names_arg, str):
        raw = [camera_names_arg]
    else:
        raw = list(camera_names_arg)

    out = []
    for item in raw:
        for part in str(item).split(','):
            s = part.strip()
            if s:
                out.append(s)
    return out if out else ["cam0"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device = {device}")

    is_eval = args.eval
    task_name = args.task_name
    ckpt_root_dir = args.ckpt_dir
    policy_class = args.policy_class
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    seed = args.seed
    lr = args.lr

    dataset_dir = resolve_dataset_dir(args.dataset_dir, task_name)
    # Use explicit --num_episodes only when the user passes a positive value.
    # Otherwise, use the actual number of episode_*.hdf5 files in the resolved dataset_dir.
    # This avoids stale TASK_CONFIGS values such as num_episodes=51 when the latest dataset has fewer episodes.
    num_episodes = int(args.num_episodes)
    if num_episodes <= 0:
        num_episodes = _count_episodes(dataset_dir)

    # Single-camera default for the new VR demo pipeline.
    # Keep B,K,3,H,W multi-camera tensor format with K=1.
    camera_names = parse_camera_names(args.camera_names)

    print(f"[INFO] task_name         = {task_name}")
    print(f"[INFO] dataset_dir       = {dataset_dir}")
    print(f"[INFO] num_episodes      = {num_episodes if num_episodes > 0 else 'ALL'}")
    print(f"[INFO] camera_names      = {camera_names}")
    print(f"[INFO] chunk_size        = {args.chunk_size}")
    print(f"[INFO] train_seq_len     = {args.train_seq_len}")
    print(f"[INFO] val_seq_len       = {args.val_seq_len}")
    print(f"[INFO] samples/ep        = {args.samples_per_episode}")
    print(f"[INFO] AMP               = {args.amp}")
    print(f"[INFO] use_force_history = {args.use_force_history}")
    print(f"[INFO] force_history_len = {args.force_history_len}")
    print(f"[INFO] NORM              = min-max per-dim -> [0,1] (qpos/action), image -> [0,1]")

    # policy config
    lr_backbone = args.lr_backbone
    backbone = args.backbone

    if policy_class == "ACT":
        policy_config = {
            "lr": lr,
            "num_queries": args.chunk_size,
            "kl_weight": args.kl_weight,
            "hidden_dim": args.hidden_dim,
            "dim_feedforward": args.dim_feedforward,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "nheads": args.nheads,
            "camera_names": camera_names,
            "state_dim": 9,
            "action_dim": 9,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "temporal_agg": args.temporal_agg,
            "pretrained_backbone": (not args.no_pretrained),

            # split observation encoder / force GRU config
            "position_dim": args.position_dim,
            "force_dim": args.force_dim,
            "position_encoder_hidden_dim": args.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
            "force_encoder_num_layers": args.force_encoder_num_layers,
            "force_encoder_dropout": args.force_encoder_dropout,
            "observation_encoder_activation": args.observation_encoder_activation,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": lr,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
            "state_dim": 9,
            "action_dim": 9,
            "image_resize_hw": args.image_resize_hw,
            "image_pool_hw": args.image_pool_hw,
            "temporal_agg": args.temporal_agg,
            "pretrained_backbone": (not args.no_pretrained),

            # split observation encoder / force GRU config
            "position_dim": args.position_dim,
            "force_dim": args.force_dim,
            "position_encoder_hidden_dim": args.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": args.force_encoder_hidden_dim,
            "force_encoder_num_layers": args.force_encoder_num_layers,
            "force_encoder_dropout": args.force_encoder_dropout,
            "observation_encoder_activation": args.observation_encoder_activation,
            "cnnmlp_observation_embed_dim": args.cnnmlp_observation_embed_dim,
        }
    else:
        raise NotImplementedError(policy_class)

    # ============================================================
    # [EVAL MODE]
    # ============================================================
    if is_eval:
        ckpt_dir = ckpt_root_dir
        best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(best_ckpt):
            latest_sub = find_latest_timestamped_subdir(ckpt_root_dir)
            if latest_sub is None:
                raise FileNotFoundError(
                    f"[EVAL] No policy_best.ckpt in {ckpt_root_dir} "
                    f"and no timestamped subdirectories were found."
                )
            ckpt_dir = latest_sub
            best_ckpt = os.path.join(ckpt_dir, "policy_best.ckpt")

        if not os.path.exists(best_ckpt):
            raise FileNotFoundError(f"[EVAL] policy_best.ckpt not found in {ckpt_dir}")

        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"[EVAL] dataset_stats.pkl not found in {ckpt_dir}")

        print(f"[EVAL] Using checkpoint dir: {ckpt_dir}")
        print(f"[INFO] Loading checkpoint from {best_ckpt}")

        policy = make_policy(policy_class, policy_config).to(device)

        ckpt = torch.load(best_ckpt, map_location=device)

        # backward compatibility:
        # - some old ckpts may save raw state_dict
        # - newer ckpts may save dict with model_state_dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        policy.load_state_dict(state_dict, strict=False)
        policy.eval()

        with open(stats_path, "rb") as f:
            _ = pickle.load(f)
        print(f"[INFO] Loaded dataset stats from {stats_path}")

        print("\n✅ Model ready for inference!")
        if args.use_force_history:
            print("   - current inference config expects optional force_history input")
            print(f"   - configured force_history_len = {args.force_history_len}")
        print("   (주의: action은 [0,1] 정규화 스케일이므로, 실제 제어 시 stats로 denormalize 필요)\n")
        return

    # ============================================================
    # [TRAIN MODE]
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    ckpt_dir = os.path.join(ckpt_root_dir, timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[TRAIN] Checkpoints will be saved under: {ckpt_dir}")

    train_loader, val_loader, stats, meta = load_data(
        dataset_dir=dataset_dir,
        num_episodes=num_episodes,
        camera_names=camera_names,
        batch_size_train=batch_size,
        batch_size_val=batch_size,
        seq_len_train=args.train_seq_len,
        seq_len_val=args.val_seq_len,
        seed=seed,
        samples_per_episode=args.samples_per_episode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,

        # force history
        return_force_history=args.use_force_history,
        use_force_history=args.use_force_history,
        force_history_len=args.force_history_len,
    )
    print(f"[INFO] data meta: {meta}")

    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)
    print(f"[INFO] saved dataset stats -> {ckpt_dir}/dataset_stats.pkl")

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "seed": seed,
        "device": device,
        "amp": args.amp,
        "debug_norm": args.debug_norm,
        "debug_norm_batches": 1,
        "temporal_agg": args.temporal_agg,

        # logging / reproducibility
        "use_force_history": args.use_force_history,
        "force_history_len": args.force_history_len,
    }

    best_ckpt_info = train_bc(train_loader, val_loader, config)

    best_epoch = best_ckpt_info["best_epoch"]
    best_val_loss = best_ckpt_info["best_val_loss"]
    best_ckpt_path = best_ckpt_info["best_ckpt_path"]
    last_ckpt_path = best_ckpt_info["last_ckpt_path"]

    print(f"[INFO] Training finished.")
    print(f"[INFO] Best epoch     = {best_epoch}")
    print(f"[INFO] Best val loss  = {best_val_loss:.6f}")
    print(f"[INFO] Best ckpt path = {best_ckpt_path}")
    print(f"[INFO] Last ckpt path = {last_ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--eval", action="store_true", help="run inference instead of training")
    p.add_argument("--ckpt_dir", type=str, default="/home/eunseop/nrs_act/checkpoints/ur10e_swing", help="root checkpoint directory")
    p.add_argument("--policy_class", type=str, default="ACT", choices=["ACT", "CNNMLP"])
    p.add_argument("--task_name", type=str, default="ur10e_swing")

    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)

    # dataset override
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--num_episodes", type=int, default=0)
    p.add_argument(
        "--camera_names",
        nargs="+",
        default=["cam0"],
        help="camera dataset names under observations/images. Default: cam0",
    )

    # IMPORTANT: sequence length
    p.add_argument("--chunk_size", type=int, default=200)
    p.add_argument("--train_seq_len", type=int, default=None)
    p.add_argument("--val_seq_len", type=int, default=None)

    # IMPORTANT: restore epoch steps
    p.add_argument("--samples_per_episode", type=int, default=50)

    # ACT params
    p.add_argument("--kl_weight", type=float, default=10)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)

    # transformer depth
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=7)

    # backbone
    p.add_argument("--backbone", type=str, default="resnet18")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--no_pretrained", action="store_true", default=False)

    # image perf knobs
    p.add_argument("--image_resize_hw", type=int, default=256)
    p.add_argument("--image_pool_hw", type=int, default=4)

    # dataloader knobs
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--persistent_workers", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2)

    # amp
    p.add_argument("--amp", action="store_true", default=False)

    # compatibility with old custom_imitate_episodes.py alias
    # This flag is saved in config/policy_config but does not otherwise alter training here.
    p.add_argument("--temporal_agg", action="store_true", default=True)
    p.add_argument("--no_temporal_agg", dest="temporal_agg", action="store_false")

    # debug normalization (print-only)
    p.add_argument(
        "--debug_norm",
        action="store_true",
        help="print post-normalization per-dim mean/std (9D + RGB), then continue"
    )

    # ------------------------------------------------------------
    # force history dataset / GRU observation encoder
    # ------------------------------------------------------------
    p.set_defaults(use_force_history=True)
    p.add_argument(
        "--use_force_history",
        dest="use_force_history",
        action="store_true",
        help="enable force history input (default: enabled)",
    )
    p.add_argument(
        "--no_force_history",
        dest="use_force_history",
        action="store_false",
        help="disable force history input",
    )
    p.add_argument(
        "--force_history_len",
        type=int,
        default=10,
        help="force history window length L for GRU encoder",
    )

    # split observation encoder params
    p.add_argument("--position_dim", type=int, default=6)
    p.add_argument("--force_dim", type=int, default=3)
    p.add_argument("--position_encoder_hidden_dim", type=int, default=128)
    p.add_argument("--force_encoder_hidden_dim", type=int, default=64)
    p.add_argument("--force_encoder_num_layers", type=int, default=1)
    p.add_argument("--force_encoder_dropout", type=float, default=0.0)
    p.add_argument("--observation_encoder_activation", type=str, default="gelu", choices=["relu", "gelu", "silu"])

    # CNNMLP-specific observation embed dim
    p.add_argument("--cnnmlp_observation_embed_dim", type=int, default=256)

    args = p.parse_args()

    # ACT encoder expects action sequence length to match num_queries/chunk_size.
    # If these are left unspecified, couple them to chunk_size.
    if args.train_seq_len is None:
        args.train_seq_len = int(args.chunk_size)
    if args.val_seq_len is None:
        args.val_seq_len = int(args.chunk_size)

    # CRITICAL FIX:
    # Some DETR/ACT modules parse sys.argv again internally.
    # Remove our custom flag to prevent "unrecognized arguments" crash.
    remove_flags = []
    if getattr(args, "debug_norm", False):
        remove_flags.append("--debug_norm")
    if getattr(args, "temporal_agg", False):
        remove_flags.append("--temporal_agg")
    if not getattr(args, "temporal_agg", True):
        remove_flags.append("--no_temporal_agg")
    if getattr(args, "use_force_history", False):
        remove_flags.append("--use_force_history")
    if not getattr(args, "use_force_history", True):
        remove_flags.append("--no_force_history")

    if remove_flags:
        sys.argv = [a for a in sys.argv if a not in remove_flags]

    main(args)
