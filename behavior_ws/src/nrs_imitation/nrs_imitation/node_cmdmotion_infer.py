#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node_cmdmotion_infer.py
(Updated for split observation encoder + force-history GRU model)

Current training-side architecture:
  - position encoder for [x y z wx wy wz]
  - force encoder (GRU) for force history [fx fy fz]_{t-L+1:t}
  - fusion encoder
  - image encoder + ACT

  
This node keeps the ROS topic interface mostly unchanged, but updates inference-side
preprocessing to match the new single-camera training structure:

  qpos current  : [x y z wx wy wz fx fy fz]
  force_history : recent L-step force history (online buffer), normalized using
                  the same qpos force statistics as training dataset.py

Stages / safety logic are kept the same as the previous version. Only the image input is changed from two cameras to one camera (cam0).

Usage:

ros2 run nrs_imitation node_act_cmdmotion_infer --ros-args \
  -p act_root:=/home/eunseop/nrs_act \
  -p ckpt_dir:=/home/eunseop/nrs_act/checkpoints/ur10e_swing/20260317_0043 \
  -p image_topic:=/realsense/vr/color/image_raw \
  -p use_force_history:=true \
  -p force_history_len:=10

"""


import os
import sys
import time
import math
import pickle
import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional, Deque, List
from enum import Enum

import numpy as np
import torch

try:
    import cv2
except Exception:
    cv2 = None

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image


# ============================================================
# Helpers (QoS / time / math)
# ============================================================

def _monotonic() -> float:
    return time.monotonic()


def _reliability_from_str(s: str) -> ReliabilityPolicy:
    s = (s or "").strip().lower()
    if s in ["reliable", "rel"]:
        return ReliabilityPolicy.RELIABLE
    if s in ["best_effort", "besteffort", "best"]:
        return ReliabilityPolicy.BEST_EFFORT
    return ReliabilityPolicy.BEST_EFFORT


def _qos(depth: int, reliability: ReliabilityPolicy) -> QoSProfile:
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
    )


def _exp_decay_weight(age_steps: int, tau_steps: float) -> float:
    if tau_steps <= 1e-9:
        return 1.0
    age_steps = max(0, int(age_steps))
    return float(math.exp(-float(age_steps) / float(tau_steps)))


def _beta_from_tau(dt: float, tau: float) -> float:
    if tau <= 1e-9:
        return 1.0
    return float(1.0 - math.exp(-float(dt) / float(tau)))


# ============================================================
# Helpers (Image decode)
# ============================================================

def _img_to_rgb_numpy(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image -> np.uint8 (H,W,3) RGB
    Supports: rgb8, bgr8, rgba8, bgra8
    """
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return buf.reshape((h, w, 3))
    if enc == "bgr8":
        img = buf.reshape((h, w, 3))
        return img[..., ::-1].copy()
    if enc == "rgba8":
        return buf.reshape((h, w, 4))[..., :3]
    if enc == "bgra8":
        img = buf.reshape((h, w, 4))[..., :3]
        return img[..., ::-1].copy()

    try:
        return buf.reshape((h, w, 3))
    except Exception as e:
        raise RuntimeError(f"Unsupported image encoding={msg.encoding}, size=({h},{w}), err={e}")


def _to_tensor_image_stack(
    cam0_rgb: np.ndarray,
    device: torch.device,
    resize_hw: int = 0,
) -> torch.Tensor:
    """
    (H,W,3) -> (1,1,3,H,W) float in [0,1]
    cam order fixed: [cam0]
    """
    if cam0_rgb is None:
        raise RuntimeError("cam0 image is None")

    if resize_hw and resize_hw > 0:
        try:
            import cv2
            cam0_rgb = cv2.resize(cam0_rgb, (resize_hw, resize_hw), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            raise RuntimeError(f"cv2 resize failed (resize_hw={resize_hw}): {e}")

    cam0 = np.transpose(cam0_rgb, (2, 0, 1))
    img = np.stack([cam0], axis=0).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,1,3,H,W)
    return img_t




# ============================================================
# Helpers (online camera stabilization / jitter diagnostics)
# ============================================================
def _estimate_pair_transform(prev_gray: np.ndarray, curr_gray: np.ndarray):
    if cv2 is None:
        return 0.0, 0.0, 0.0

    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=3,
    )
    if prev_pts is None or len(prev_pts) < 8:
        return 0.0, 0.0, 0.0

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    if curr_pts is None or status is None:
        return 0.0, 0.0, 0.0

    good_prev = prev_pts[status.flatten() == 1]
    good_curr = curr_pts[status.flatten() == 1]
    if len(good_prev) < 8 or len(good_curr) < 8:
        return 0.0, 0.0, 0.0

    m, _ = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
    if m is None:
        return 0.0, 0.0, 0.0

    dx = float(m[0, 2])
    dy = float(m[1, 2])
    da = float(np.arctan2(m[1, 0], m[0, 0]))
    return dx, dy, da


def _warp_rgb_affine(rgb: np.ndarray, dx: float, dy: float, da: float, border_mode: str = "reflect") -> np.ndarray:
    if cv2 is None:
        return rgb.copy()
    H, W = int(rgb.shape[0]), int(rgb.shape[1])
    c = float(np.cos(da))
    s = float(np.sin(da))
    m = np.array([[c, -s, dx], [s, c, dy]], dtype=np.float32)
    b = str(border_mode).strip().lower()
    if b == "constant":
        border_flag = cv2.BORDER_CONSTANT
    elif b == "replicate":
        border_flag = cv2.BORDER_REPLICATE
    else:
        border_flag = cv2.BORDER_REFLECT
    return cv2.warpAffine(
        rgb,
        m,
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=border_flag,
    )

# ============================================================
# Helpers (Stats)
# ============================================================

@dataclass
class StatsPack:
    qpos_mode: str   # "minmax_01", "minmax_m11", or "zscore"
    act_mode: str    # "minmax_01", "minmax_m11", or "zscore"
    qpos_a: np.ndarray   # min or mean
    qpos_b: np.ndarray   # max or std
    act_a: np.ndarray    # min or mean
    act_b: np.ndarray    # max or std


def _canonical_norm_mode(mode: str) -> str:
    if mode is None:
        return "minmax_01"
    m = str(mode).strip().lower()
    if m in ["minmax", "minmax_01", "01", "0_1", "[0,1]", "zero_one"]:
        return "minmax_01"
    if m in ["minmax_m11", "m11", "-1_1", "[-1,1]", "minus1_1", "neg1_pos1"]:
        return "minmax_m11"
    if m in ["zscore", "standard", "meanstd", "mean_std"]:
        return "zscore"
    return m


def _sanitize_std(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return np.maximum(x, eps)


def _sanitize_range_minmax(vmin: np.ndarray, vmax: np.ndarray, eps: float = 1e-6):
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)
    if vmin.size != 9 or vmax.size != 9:
        raise ValueError(f"min/max size must be 9. got {vmin.size}, {vmax.size}")
    rng = np.maximum(vmax - vmin, eps)
    vmax_fix = vmin + rng
    return vmin.astype(np.float32), vmax_fix.astype(np.float32)


def _load_dataset_stats(ckpt_dir: str) -> Optional[StatsPack]:
    """
    Priority:
      1) qpos_min/qpos_max/action_min/action_max with explicit norm mode
      2) qpos_mean/qpos_std/action_mean/action_std legacy zscore

    Backward compatibility:
      - old dataset_stats.pkl without qpos_norm_mode/action_norm_mode is treated as [0,1].
      - old mode name "minmax" is treated as "minmax_01".
    """
    p = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.exists(p):
        return None

    with open(p, "rb") as f:
        st = pickle.load(f)

    if all(k in st for k in ["qpos_min", "qpos_max", "action_min", "action_max"]):
        qmin = np.asarray(st["qpos_min"], dtype=np.float32).reshape(9)
        qmax = np.asarray(st["qpos_max"], dtype=np.float32).reshape(9)
        amin = np.asarray(st["action_min"], dtype=np.float32).reshape(9)
        amax = np.asarray(st["action_max"], dtype=np.float32).reshape(9)

        qmin, qmax = _sanitize_range_minmax(qmin, qmax)
        amin, amax = _sanitize_range_minmax(amin, amax)

        qmode = _canonical_norm_mode(
            st.get("qpos_norm_mode", st.get("qpos_mode", "minmax_01"))
        )
        amode = _canonical_norm_mode(
            st.get("action_norm_mode", st.get("act_mode", "minmax_01"))
        )

        # Legacy stats may store "minmax"; force it to [0,1].
        if qmode == "minmax":
            qmode = "minmax_01"
        if amode == "minmax":
            amode = "minmax_01"

        return StatsPack(
            qpos_mode=qmode,
            act_mode=amode,
            qpos_a=qmin,
            qpos_b=qmax,
            act_a=amin,
            act_b=amax,
        )

    if all(k in st for k in ["qpos_mean", "qpos_std", "action_mean", "action_std"]):
        qm = np.asarray(st["qpos_mean"], dtype=np.float32).reshape(9)
        qs = _sanitize_std(np.asarray(st["qpos_std"], dtype=np.float32).reshape(9))
        am = np.asarray(st["action_mean"], dtype=np.float32).reshape(9)
        astd = _sanitize_std(np.asarray(st["action_std"], dtype=np.float32).reshape(9))

        return StatsPack(
            qpos_mode="zscore",
            act_mode="zscore",
            qpos_a=qm,
            qpos_b=qs,
            act_a=am,
            act_b=astd,
        )

    return None


def _normalize_qpos(q: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    qa = torch.tensor(stats.qpos_a, dtype=torch.float32, device=q.device).view(1, 9)
    qb = torch.tensor(stats.qpos_b, dtype=torch.float32, device=q.device).view(1, 9)

    if stats.qpos_mode in ["minmax_01", "minmax_m11"]:
        den = torch.clamp(qb - qa, min=1e-6)
        q01 = (q - qa) / den
        if stats.qpos_mode == "minmax_m11":
            return torch.clamp(2.0 * q01 - 1.0, -1.0, 1.0)
        return torch.clamp(q01, 0.0, 1.0)

    return (q - qa) / torch.clamp(qb, min=1e-6)


def _normalize_force_history(force_hist: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    """
    force_hist: (1,L,3)
    Must use the same qpos force statistics as training dataset.py.
    force dims in qpos/action = indices [6:9].
    """
    if force_hist.dim() != 3 or force_hist.shape[-1] != 3:
        raise RuntimeError(f"force_hist must be (B,L,3), got {tuple(force_hist.shape)}")

    if stats.qpos_mode in ["minmax_01", "minmax_m11"]:
        fmin = torch.tensor(stats.qpos_a[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
        fmax = torch.tensor(stats.qpos_b[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
        den = torch.clamp(fmax - fmin, min=1e-6)
        f01 = (force_hist - fmin) / den
        if stats.qpos_mode == "minmax_m11":
            return torch.clamp(2.0 * f01 - 1.0, -1.0, 1.0)
        return torch.clamp(f01, 0.0, 1.0)

    fmean = torch.tensor(stats.qpos_a[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
    fstd = torch.tensor(stats.qpos_b[6:9], dtype=torch.float32, device=force_hist.device).view(1, 1, 3)
    return (force_hist - fmean) / torch.clamp(fstd, min=1e-6)


def _denorm_action_seq(seq: torch.Tensor, stats: StatsPack) -> torch.Tensor:
    if seq.dim() == 2:
        aa = torch.tensor(stats.act_a, dtype=torch.float32, device=seq.device).view(1, 9)
        ab = torch.tensor(stats.act_b, dtype=torch.float32, device=seq.device).view(1, 9)
    elif seq.dim() == 3:
        aa = torch.tensor(stats.act_a, dtype=torch.float32, device=seq.device).view(1, 1, 9)
        ab = torch.tensor(stats.act_b, dtype=torch.float32, device=seq.device).view(1, 1, 9)
    else:
        raise RuntimeError(f"unexpected seq dim: {seq.shape}")

    if stats.act_mode in ["minmax_01", "minmax_m11"]:
        den = torch.clamp(ab - aa, min=1e-6)
        if stats.act_mode == "minmax_m11":
            seq01 = 0.5 * (seq + 1.0)
            return seq01 * den + aa
        return seq * den + aa

    return seq * torch.clamp(ab, min=1e-6) + aa

# ============================================================
# Helpers (Policy output shape)
# ============================================================

def _fix_a_hat_shape(a_hat: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Standardize output to (T,9)
    Handles:
      - (1,T,9)
      - (T,1,9)
      - (T,9)
    """
    if a_hat.dim() == 2:
        return a_hat
    if a_hat.dim() != 3:
        raise RuntimeError(f"Unexpected a_hat dim: {a_hat.shape}")

    B0, B1, B2 = a_hat.shape
    if B2 != 9:
        raise RuntimeError(f"Unexpected last dim (need 9): {a_hat.shape}")

    if B0 == 1 and B1 == chunk_size:
        return a_hat[0]
    if B0 == chunk_size and B1 == 1:
        return a_hat[:, 0, :]
    if B1 == chunk_size:
        return a_hat[0]
    raise RuntimeError(f"Cannot interpret a_hat shape={a_hat.shape} with chunk_size={chunk_size}")


def _fix_policy_output_seq(seq: torch.Tensor, chunk_size: int, policy_class: str) -> torch.Tensor:
    """
    Standardize ACT / DIFFUSION policy output to (T,9).
    ACT:
      usually (1,T,9) or (T,9)
    DIFFUSION:
      usually (B,T,9) or (T,9)
    """
    if seq.dim() == 2:
        if seq.shape[-1] != 9:
            raise RuntimeError(f"Unexpected 2D seq shape: {tuple(seq.shape)}")
        return seq

    if seq.dim() != 3:
        raise RuntimeError(f"Unexpected policy output dim: {tuple(seq.shape)}")

    if seq.shape[-1] != 9:
        raise RuntimeError(f"Unexpected last dim in policy output: {tuple(seq.shape)}")

    policy_class = str(policy_class).upper()
    if policy_class == "DIFFUSION":
        if seq.shape[0] == 1:
            return seq[0]
        if seq.shape[1] == chunk_size:
            return seq[0]
        raise RuntimeError(f"Cannot interpret diffusion output shape={tuple(seq.shape)} with chunk_size={chunk_size}")

    return _fix_a_hat_shape(seq, chunk_size)


# ============================================================
# Plan buffer entry
# ============================================================

@dataclass
class Plan:
    t0: float
    seq_den: np.ndarray  # (T,9) denorm


# ============================================================
# Stage machine
# ============================================================

class Stage(Enum):
    APPROACH = 0
    PRELOAD = 1
    TRACK = 2
    RELEASE = 3
    RECOVER = 4  # deprecated / unused


# ============================================================
# State dict compatibility loader
# ============================================================

def _strip_prefix_from_state_dict(sd: dict, prefixes: List[str]) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _try_load_state_dict_compat(target: torch.nn.Module, state_dict: dict):
    """
    Try several key transforms and pick the best (min missing+unexpected).
    """
    candidates = []
    candidates.append(("orig", state_dict))
    candidates.append(("strip_model.", _strip_prefix_from_state_dict(state_dict, ["model."])))
    candidates.append(("strip_module.", _strip_prefix_from_state_dict(state_dict, ["module."])))
    candidates.append(("strip_policy.", _strip_prefix_from_state_dict(state_dict, ["policy."])))
    candidates.append(("strip_model+module", _strip_prefix_from_state_dict(state_dict, ["module.", "model."])))
    candidates.append(("strip_policy+module", _strip_prefix_from_state_dict(state_dict, ["module.", "policy."])))

    best_missing = None
    best_unexpected = None
    best_score = None

    for _, sd in candidates:
        try:
            missing, unexpected = target.load_state_dict(sd, strict=False)
            score = len(missing) + len(unexpected)
            if (best_score is None) or (score < best_score):
                best_score = score
                best_missing = missing
                best_unexpected = unexpected
        except Exception:
            continue

    if best_missing is None:
        missing, unexpected = target.load_state_dict(state_dict, strict=False)
        return missing, unexpected

    return best_missing, best_unexpected


# ============================================================
# ROS2 Node
# ============================================================

class NodeCmdMotionInfer(Node):
    def __init__(self):
        super().__init__("node_cmdmotion_infer")

        # -----------------------------
        # Parameters (paths / IO)
        # -----------------------------
        self.declare_parameter("ckpt_dir", "")
        self.declare_parameter("act_root", "")   # e.g. /home/eunseop/nrs_act
        self.declare_parameter("policy_class", "ACT")  # ACT | DIFFUSION | FLOW
        self.declare_parameter("chunk_size", 200)

        self.declare_parameter("pose_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("image_topic", "/realsense/vr/color/image_raw")
        self.declare_parameter("cmd_topic", "/ur10skku/cmdMotion")

        self.declare_parameter("image_qos", "best_effort")

        # Camera preprocessing for online inference.
        # Default ON: real-time stabilization of incoming RGB before policy observation.
        self.declare_parameter("camera_preprocess_mode", "stabilize")  # off | stabilize
        self.declare_parameter("camera_stabilize_alpha", 0.92)          # cumulative trajectory EMA
        self.declare_parameter("camera_stabilize_border_mode", "reflect")
        self.declare_parameter("camera_jitter_report_enable", True)
        self.declare_parameter("camera_jitter_log_every_n", 100)

        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("infer_hz", 7.5)

        # -----------------------------
        # New observation encoder / force history
        # -----------------------------
        self.declare_parameter("use_force_history", True)
        self.declare_parameter("force_history_len", 10)

        self.declare_parameter("position_dim", 6)
        self.declare_parameter("force_dim", 3)
        self.declare_parameter("position_encoder_hidden_dim", 128)
        self.declare_parameter("force_encoder_hidden_dim", 64)
        self.declare_parameter("force_encoder_num_layers", 1)
        self.declare_parameter("force_encoder_dropout", 0.0)
        self.declare_parameter("observation_encoder_activation", "gelu")

        # -----------------------------
        # Baseline safety (QP-safe)
        # -----------------------------
        self.declare_parameter("tau_sec", 0.35)
        self.declare_parameter("startup_ramp_sec", 1.5)
        self.declare_parameter("step_cap_pos_mm", 0.30)
        self.declare_parameter("step_cap_ang_rad", 0.0004)
        self.declare_parameter("step_cap_fz", 0.20)

        self.declare_parameter("use_temporal_agg", True)
        self.declare_parameter("temporal_agg_mode", "exp")
        self.declare_parameter("temporal_agg_tau_steps", 20.0)
        self.declare_parameter("pred_step_offset", 1)
        self.declare_parameter("max_plans", 6)

        # contact gating
        self.declare_parameter("contact_on_thr", 2.0)
        self.declare_parameter("contact_off_thr", 1.0)
        self.declare_parameter("clear_plans_on_contact_change", False)

        # touch detection
        self.declare_parameter("touch_fz_thr", 0.5)
        self.declare_parameter("touch_ok_count", 3)
        self.declare_parameter("touch_min_after_start_sec", 1.0)
        self.declare_parameter("touch_baseline_tau_sec", 0.5)
        self.declare_parameter("touch_use_delta", True)

        # preload
        self.declare_parameter("preload_target_source", "stats_mean")  # stats_mean | fixed
        self.declare_parameter("preload_fixed_N", 10.0)
        self.declare_parameter("preload_target_scale", 1.0)
        self.declare_parameter("preload_min_N", 10.0)
        self.declare_parameter("preload_timeout_sec", 5.0)
        self.declare_parameter("preload_ok_count", 10)
        self.declare_parameter("preload_kp_mm_per_N", 0.02)
        self.declare_parameter("preload_dz_max_mm", 0.08)
        self.declare_parameter("preload_tol_N", 0.2)
        self.declare_parameter("press_force_cmd_mode", "target")  # keep|zero|target
        self.declare_parameter("press_hold_xy", True)
        self.declare_parameter("press_hold_rpy", True)

        # optional release assist
        self.declare_parameter("release_assist_enable", False)
        self.declare_parameter("release_ramp_sec", 1.0)

        # I/O shaping
        self.declare_parameter("force_indices", [0, 1, 2])
        self.declare_parameter("first_cmd_fz", 0.0)
        self.declare_parameter("action_type", "absolute")  # absolute | delta
        self.declare_parameter("normalize_qpos", True)
        self.declare_parameter("denorm_action", True)
        self.declare_parameter("resize_hw", 0)
        self.declare_parameter("debug_every_n", 30)

        # force safety
        self.declare_parameter("fz_hard_limit", 25.0)

        # policy config
        self.declare_parameter("kl_weight", 10.0)
        self.declare_parameter("hidden_dim", 512)
        self.declare_parameter("dim_feedforward", 3200)
        self.declare_parameter("lr_backbone", 1e-5)
        self.declare_parameter("backbone", "resnet18")
        self.declare_parameter("enc_layers", 4)
        self.declare_parameter("dec_layers", 7)
        self.declare_parameter("nheads", 8)
        self.declare_parameter("image_resize_hw", 256)
        self.declare_parameter("image_pool_hw", 4)
        self.declare_parameter("pretrained_backbone", True)

        # diffusion policy config
        self.declare_parameter("diffusion_train_steps", 100)
        self.declare_parameter("diffusion_infer_steps", 10)
        self.declare_parameter("diffusion_beta_start", 1e-4)
        self.declare_parameter("diffusion_beta_end", 2e-2)
        self.declare_parameter("diffusion_loss_type", "mse")

        # FLOW policy config
        self.declare_parameter("flow_infer_steps", 10)
        self.declare_parameter("flow_train_eps", 1e-4)
        self.declare_parameter("flow_loss_type", "mse")
        self.declare_parameter("flow_obs_hidden_dim", 256)
        self.declare_parameter("flow_image_feature_dim", 512)
        self.declare_parameter("flow_global_cond_dim", 256)
        self.declare_parameter("flow_time_embed_dim", 256)
        self.declare_parameter("flow_down_dims", "256,512,1024")
        self.declare_parameter("flow_kernel_size", 5)
        self.declare_parameter("flow_n_groups", 8)
        self.declare_parameter("flow_cond_predict_scale", False)

        # stall + recover
        self.declare_parameter("stall_sec", 1.2)
        self.declare_parameter("stall_min_after_start_sec", 1.0)
        self.declare_parameter("stall_lpf_tau_sec", 0.40)
        self.declare_parameter("stall_window_net_pos_eps_mm", 0.25)
        self.declare_parameter("stall_window_net_ang_eps_rad", 0.0006)

        self.declare_parameter("fz_kick_N", 1.5)
        self.declare_parameter("fz_kick_dur_sec", 0.35)
        self.declare_parameter("fz_kick_cooldown_sec", 0.8)

        self.declare_parameter("recover_enable", True)
        self.declare_parameter("recover_cooldown_sec", 2.0)
        self.declare_parameter("recover_timeout_sec", 6.0)
        self.declare_parameter("recover_pos_tol_mm", 0.35)
        self.declare_parameter("recover_ang_tol_rad", 0.0008)
        self.declare_parameter("recover_ok_count", 10)

        # dither + improved recover
        self.declare_parameter("dither_enable", True)
        self.declare_parameter("dither_only_track", True)
        self.declare_parameter("dither_min_after_start_sec", 2.0)
        self.declare_parameter("dither_win_sec", 1.0)
        self.declare_parameter("dither_sec", 1.0)
        self.declare_parameter("dither_net_pos_thr_mm", 0.8)
        self.declare_parameter("dither_net_ang_thr_rad", 0.0015)
        self.declare_parameter("dither_path_ratio_thr", 6.0)
        self.declare_parameter("dither_rms_pos_thr_mm", 0.10)
        self.declare_parameter("dither_rms_ang_thr_rad", 0.00025)
        self.declare_parameter("dither_decay", 0.5)

        self.declare_parameter("kick_max_before_recover", 2)
        self.declare_parameter("kick_reset_sec", 6.0)

        self.declare_parameter("recover_check_lpf_tau_sec", 0.25)

        self.declare_parameter("recover_use_overrides", True)
        self.declare_parameter("recover_tau_sec", 0.25)
        self.declare_parameter("recover_startup_ramp_sec", 0.6)
        self.declare_parameter("recover_step_cap_pos_mm", 1.0)
        self.declare_parameter("recover_step_cap_ang_rad", 0.0012)
        self.declare_parameter("recover_step_cap_fz", 0.30)

        self.declare_parameter("recover_timeout_min_margin_sec", 1.0)
        self.declare_parameter("recover_timeout_scale", 1.4)

        # -----------------------------
        # Read params
        # -----------------------------
        self.ckpt_dir = str(self.get_parameter("ckpt_dir").value)
        self.act_root = str(self.get_parameter("act_root").value)
        self.policy_class = str(self.get_parameter("policy_class").value).strip().upper()
        self.chunk_size = int(self.get_parameter("chunk_size").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.image_qos_str = str(self.get_parameter("image_qos").value)
        self.camera_preprocess_mode = str(self.get_parameter("camera_preprocess_mode").value).strip().lower()
        self.camera_stabilize_alpha = float(self.get_parameter("camera_stabilize_alpha").value)
        self.camera_stabilize_border_mode = str(self.get_parameter("camera_stabilize_border_mode").value).strip().lower()
        self.camera_jitter_report_enable = bool(self.get_parameter("camera_jitter_report_enable").value)
        self.camera_jitter_log_every_n = max(1, int(self.get_parameter("camera_jitter_log_every_n").value))
        if self.camera_preprocess_mode not in ("off", "none", "raw", "stabilize"):
            raise RuntimeError(f"camera_preprocess_mode must be off or stabilize, got: {self.camera_preprocess_mode}")

        self.control_hz = float(self.get_parameter("control_hz").value)
        self.infer_hz = float(self.get_parameter("infer_hz").value)

        self.use_force_history = bool(self.get_parameter("use_force_history").value)
        self.force_history_len = int(self.get_parameter("force_history_len").value)

        self.position_dim = int(self.get_parameter("position_dim").value)
        self.force_dim = int(self.get_parameter("force_dim").value)
        self.position_encoder_hidden_dim = int(self.get_parameter("position_encoder_hidden_dim").value)
        self.force_encoder_hidden_dim = int(self.get_parameter("force_encoder_hidden_dim").value)
        self.force_encoder_num_layers = int(self.get_parameter("force_encoder_num_layers").value)
        self.force_encoder_dropout = float(self.get_parameter("force_encoder_dropout").value)
        self.observation_encoder_activation = str(self.get_parameter("observation_encoder_activation").value)

        self.tau_sec = float(self.get_parameter("tau_sec").value)
        self.startup_ramp_sec = float(self.get_parameter("startup_ramp_sec").value)
        self.step_cap_pos_mm = float(self.get_parameter("step_cap_pos_mm").value)
        self.step_cap_ang_rad = float(self.get_parameter("step_cap_ang_rad").value)
        self.step_cap_fz = float(self.get_parameter("step_cap_fz").value)

        self.use_temporal_agg = bool(self.get_parameter("use_temporal_agg").value)
        self.temporal_agg_mode = str(self.get_parameter("temporal_agg_mode").value).strip().lower()
        self.temporal_agg_tau_steps = float(self.get_parameter("temporal_agg_tau_steps").value)
        self.pred_step_offset = int(self.get_parameter("pred_step_offset").value)
        self.max_plans = int(self.get_parameter("max_plans").value)

        self.contact_on_thr = float(self.get_parameter("contact_on_thr").value)
        self.contact_off_thr = float(self.get_parameter("contact_off_thr").value)
        self.clear_plans_on_contact_change = bool(self.get_parameter("clear_plans_on_contact_change").value)

        self.touch_fz_thr = float(self.get_parameter("touch_fz_thr").value)
        self.touch_ok_count = int(self.get_parameter("touch_ok_count").value)
        self.touch_min_after_start_sec = float(self.get_parameter("touch_min_after_start_sec").value)
        self.touch_baseline_tau_sec = float(self.get_parameter("touch_baseline_tau_sec").value)
        self.touch_use_delta = bool(self.get_parameter("touch_use_delta").value)

        self.preload_target_source = str(self.get_parameter("preload_target_source").value).strip().lower()
        self.preload_fixed_N = float(self.get_parameter("preload_fixed_N").value)
        self.preload_target_scale = float(self.get_parameter("preload_target_scale").value)
        self.preload_min_N = float(self.get_parameter("preload_min_N").value)
        self.preload_timeout_sec = float(self.get_parameter("preload_timeout_sec").value)
        self.preload_ok_count = int(self.get_parameter("preload_ok_count").value)
        self.preload_kp_mm_per_N = float(self.get_parameter("preload_kp_mm_per_N").value)
        self.preload_dz_max_mm = float(self.get_parameter("preload_dz_max_mm").value)
        self.preload_tol_N = float(self.get_parameter("preload_tol_N").value)
        self.press_force_cmd_mode = str(self.get_parameter("press_force_cmd_mode").value).strip().lower()
        self.press_hold_xy = bool(self.get_parameter("press_hold_xy").value)
        self.press_hold_rpy = bool(self.get_parameter("press_hold_rpy").value)

        self.release_assist_enable = bool(self.get_parameter("release_assist_enable").value)
        self.release_ramp_sec = float(self.get_parameter("release_ramp_sec").value)

        self.force_indices = tuple(int(x) for x in self.get_parameter("force_indices").value)
        self.first_cmd_fz = float(self.get_parameter("first_cmd_fz").value)
        self.action_type = str(self.get_parameter("action_type").value).strip().lower()

        self.normalize_qpos_enabled = bool(self.get_parameter("normalize_qpos").value)
        self.denorm_action_enabled = bool(self.get_parameter("denorm_action").value)

        self.resize_hw = int(self.get_parameter("resize_hw").value)
        self.debug_every_n = max(1, int(self.get_parameter("debug_every_n").value))

        self.fz_hard_limit = float(self.get_parameter("fz_hard_limit").value)

        self.stall_sec = float(self.get_parameter("stall_sec").value)
        self.stall_min_after_start_sec = float(self.get_parameter("stall_min_after_start_sec").value)
        self.stall_lpf_tau_sec = float(self.get_parameter("stall_lpf_tau_sec").value)
        self.stall_window_net_pos_eps_mm = float(self.get_parameter("stall_window_net_pos_eps_mm").value)
        self.stall_window_net_ang_eps_rad = float(self.get_parameter("stall_window_net_ang_eps_rad").value)

        self.fz_kick_N = float(self.get_parameter("fz_kick_N").value)
        self.fz_kick_dur_sec = float(self.get_parameter("fz_kick_dur_sec").value)
        self.fz_kick_cooldown_sec = float(self.get_parameter("fz_kick_cooldown_sec").value)

        self.recover_enable = False  # RECOVER logic removed
        self.recover_cooldown_sec = float(self.get_parameter("recover_cooldown_sec").value)
        self.recover_timeout_sec = float(self.get_parameter("recover_timeout_sec").value)
        self.recover_pos_tol_mm = float(self.get_parameter("recover_pos_tol_mm").value)
        self.recover_ang_tol_rad = float(self.get_parameter("recover_ang_tol_rad").value)
        self.recover_ok_count = int(self.get_parameter("recover_ok_count").value)

        self.dither_enable = bool(self.get_parameter("dither_enable").value)
        self.dither_only_track = bool(self.get_parameter("dither_only_track").value)
        self.dither_min_after_start_sec = float(self.get_parameter("dither_min_after_start_sec").value)
        self.dither_win_sec = float(self.get_parameter("dither_win_sec").value)
        self.dither_sec = float(self.get_parameter("dither_sec").value)
        self.dither_net_pos_thr_mm = float(self.get_parameter("dither_net_pos_thr_mm").value)
        self.dither_net_ang_thr_rad = float(self.get_parameter("dither_net_ang_thr_rad").value)
        self.dither_path_ratio_thr = float(self.get_parameter("dither_path_ratio_thr").value)
        self.dither_rms_pos_thr_mm = float(self.get_parameter("dither_rms_pos_thr_mm").value)
        self.dither_rms_ang_thr_rad = float(self.get_parameter("dither_rms_ang_thr_rad").value)
        self.dither_decay = float(self.get_parameter("dither_decay").value)

        self.kick_max_before_recover = int(self.get_parameter("kick_max_before_recover").value)
        self.kick_reset_sec = float(self.get_parameter("kick_reset_sec").value)

        self.recover_check_lpf_tau_sec = float(self.get_parameter("recover_check_lpf_tau_sec").value)
        self.recover_use_overrides = bool(self.get_parameter("recover_use_overrides").value)
        self.recover_tau_sec = float(self.get_parameter("recover_tau_sec").value)
        self.recover_startup_ramp_sec = float(self.get_parameter("recover_startup_ramp_sec").value)
        self.recover_step_cap_pos_mm = float(self.get_parameter("recover_step_cap_pos_mm").value)
        self.recover_step_cap_ang_rad = float(self.get_parameter("recover_step_cap_ang_rad").value)
        self.recover_step_cap_fz = float(self.get_parameter("recover_step_cap_fz").value)
        self.recover_timeout_min_margin_sec = float(self.get_parameter("recover_timeout_min_margin_sec").value)
        self.recover_timeout_scale = float(self.get_parameter("recover_timeout_scale").value)

        # diffusion policy config
        self.diffusion_train_steps = int(self.get_parameter("diffusion_train_steps").value)
        self.diffusion_infer_steps = int(self.get_parameter("diffusion_infer_steps").value)
        self.diffusion_beta_start = float(self.get_parameter("diffusion_beta_start").value)
        self.diffusion_beta_end = float(self.get_parameter("diffusion_beta_end").value)
        self.diffusion_loss_type = str(self.get_parameter("diffusion_loss_type").value)

        # FLOW policy config
        self.flow_infer_steps = int(self.get_parameter("flow_infer_steps").value)
        self.flow_train_eps = float(self.get_parameter("flow_train_eps").value)
        self.flow_loss_type = str(self.get_parameter("flow_loss_type").value)
        self.flow_obs_hidden_dim = int(self.get_parameter("flow_obs_hidden_dim").value)
        self.flow_image_feature_dim = int(self.get_parameter("flow_image_feature_dim").value)
        self.flow_global_cond_dim = int(self.get_parameter("flow_global_cond_dim").value)
        self.flow_time_embed_dim = int(self.get_parameter("flow_time_embed_dim").value)
        self.flow_down_dims = str(self.get_parameter("flow_down_dims").value)
        self.flow_kernel_size = int(self.get_parameter("flow_kernel_size").value)
        self.flow_n_groups = int(self.get_parameter("flow_n_groups").value)
        self.flow_cond_predict_scale = bool(self.get_parameter("flow_cond_predict_scale").value)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[INFO] Using device: {self.device}")
        self.get_logger().info(
            f"[CAM] preprocess_mode={self.camera_preprocess_mode}, "
            f"alpha={self.camera_stabilize_alpha:.3f}, border={self.camera_stabilize_border_mode}, "
            f"jitter_report={self.camera_jitter_report_enable}, log_every={self.camera_jitter_log_every_n}"
        )

        # validate paths
        if not self.ckpt_dir or not os.path.isdir(self.ckpt_dir):
            raise RuntimeError(f"ckpt_dir invalid: {self.ckpt_dir}")
        if not self.act_root or not os.path.isdir(self.act_root):
            raise RuntimeError(f"act_root invalid: {self.act_root}")
        if self.policy_class not in ("ACT", "DIFFUSION", "FLOW"):
            raise RuntimeError(f"policy_class must be ACT, DIFFUSION, or FLOW, got: {self.policy_class}")

        # stats
        self.stats = _load_dataset_stats(self.ckpt_dir)
        if self.stats is None:
            self.get_logger().warn("[STATS] dataset_stats.pkl missing/invalid -> disable normalize/denorm.")
            self.normalize_qpos_enabled = False
            self.denorm_action_enabled = False
        else:
            self.get_logger().info(
                f"[STATS] Loaded dataset_stats.pkl from {self.ckpt_dir} | "
                f"qpos_mode={self.stats.qpos_mode}, act_mode={self.stats.act_mode}"
            )
            if self.stats.qpos_mode in ["minmax_01", "minmax_m11"]:
                self.get_logger().info(
                    f"[STATS] qpos_z_range=[{float(self.stats.qpos_a[2]):.3f},{float(self.stats.qpos_b[2]):.3f}] "
                    f"action_z_range=[{float(self.stats.act_a[2]):.3f},{float(self.stats.act_b[2]):.3f}] "
                    f"action_fz_range=[{float(self.stats.act_a[8]):.3f},{float(self.stats.act_b[8]):.3f}]"
                )

        # policy
        self.policy = self._load_policy_and_ckpt_from_act_root()

        # -----------------------------
        # State buffers
        # -----------------------------
        self._lock = threading.Lock()
        self._pose6: Optional[np.ndarray] = None
        self._force: Optional[np.ndarray] = None
        self._img_cam0: Optional[np.ndarray] = None

        # Online camera stabilization state. All jitter values are pixel units.
        self._cam_prev_raw_gray: Optional[np.ndarray] = None
        self._cam_prev_proc_gray: Optional[np.ndarray] = None
        self._cam_cum = np.zeros(3, dtype=np.float32)
        self._cam_smooth_cum = np.zeros(3, dtype=np.float32)
        self._cam_frame_count = 0
        self._cam_raw_jitter_ema = 0.0
        self._cam_proc_jitter_ema = 0.0

        self._force_hist: Deque[np.ndarray] = deque(maxlen=max(1, self.force_history_len))

        # baseline state
        self._sent_first_cmd = False
        self.prev_cmd: Optional[np.ndarray] = None
        self._t_start = _monotonic()
        self._t_first_pub = None

        self._start_pose6: Optional[np.ndarray] = None

        # contact state
        self._contact = False
        self._last_contact = False

        self.stage = Stage.APPROACH

        # anchor
        self._anchor_ready = False
        self._anchor_offset6 = np.zeros(6, dtype=np.float32)

        # plan buffer
        self.plans: Deque[Plan] = deque(maxlen=max(1, self.max_plans))

        # touch baseline
        self._fz_base = 0.0
        self._fz_base_init = False
        self._touch_ok = 0

        # preload
        self._preload_t0 = 0.0
        self._preload_ok = 0
        self._preload_hold_pose6 = None
        self._preload_target_N = max(self.preload_min_N, 10.0)

        # release
        self._release_t0 = 0.0
        self._release_start_fz_cmd = 0.0

        # stall / kick / recover
        self._stall_pose6_lpf: Optional[np.ndarray] = None
        self._stall_win_pose6: Optional[np.ndarray] = None
        self._stall_win_t0: float = _monotonic()

        self._fz_kick_active: bool = False
        self._fz_kick_t0: float = 0.0
        self._fz_kick_last_end_t: float = -1e9

        self._recover_t0: float = 0.0
        self._recover_ok: int = 0
        self._recover_last_end_t: float = -1e9
        self._recover_timeout_eff: float = self.recover_timeout_sec

        self._recover_pose6_lpf: Optional[np.ndarray] = None

        # dither
        self.dt_control = 1.0 / max(1e-6, self.control_hz)
        self.dt_infer = 1.0 / max(1e-6, self.infer_hz)

        hist_len = max(4, int(max(0.2, self.dither_win_sec) * self.control_hz) + 2)
        self._pose_hist6 = deque(maxlen=hist_len)
        self._dither_score = 0.0
        self._kick_count = 0
        self._kick_count_t0 = _monotonic()

        # -----------------------------
        # ROS I/O
        # -----------------------------
        img_rel = _reliability_from_str(self.image_qos_str)
        img_qos = _qos(depth=1, reliability=img_rel)
        vec_qos = _qos(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.create_subscription(Float64MultiArray, self.pose_topic, self._on_pose, vec_qos)
        self.create_subscription(Float64MultiArray, self.force_topic, self._on_force, vec_qos)
        self.create_subscription(Image, self.image_topic, self._on_img, img_qos)

        self.pub_cmd = self.create_publisher(Float64MultiArray, self.cmd_topic, 10)

        self.timer_control = self.create_timer(self.dt_control, self._on_control_timer)
        self.timer_infer = self.create_timer(self.dt_infer, self._on_infer_timer)

        self.get_logger().info(
            "[INFO] ✅ Ready.\n"
            f"  stage_start={self.stage.name}\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic}\n"
            f"  image_topic={self.image_topic}\n"
            f"  cmd_topic={self.cmd_topic}\n"
            f"  image_qos={self.image_qos_str}\n"
            f"  policy_class={self.policy_class}\n"
            f"  control_hz={self.control_hz} infer_hz={self.infer_hz}\n"
            f"  use_force_history={int(self.use_force_history)} force_history_len={self.force_history_len}\n"
            f"  diffusion_infer_steps={self.diffusion_infer_steps}\n"
            f"  tau_sec={self.tau_sec} startup_ramp_sec={self.startup_ramp_sec}\n"
            f"  step_caps(pos_mm={self.step_cap_pos_mm}, ang_rad={self.step_cap_ang_rad}, fz={self.step_cap_fz})\n"
            f"  temporal_agg={int(self.use_temporal_agg)} mode={self.temporal_agg_mode} tau_steps={self.temporal_agg_tau_steps} max_plans={self.max_plans}\n"
            f"  contact_gate(on={self.contact_on_thr}, off={self.contact_off_thr}) clear_on_change={int(self.clear_plans_on_contact_change)}\n"
            f"  touch(delta={int(self.touch_use_delta)}, thr={self.touch_fz_thr}, ok={self.touch_ok_count}, min_after={self.touch_min_after_start_sec}s, base_tau={self.touch_baseline_tau_sec}s)\n"
            f"  PRELOAD(removed: bypass APPROACH -> TRACK, nominal_src={self.preload_target_source}, nominal_min={self.preload_min_N}N)\n"
            f"  STALL(win_sec={self.stall_sec}, min_after={self.stall_min_after_start_sec}s, lpf_tau={self.stall_lpf_tau_sec}s, net_eps_pos={self.stall_window_net_pos_eps_mm}mm, net_eps_ang={self.stall_window_net_ang_eps_rad}rad)\n"
            f"  KICK(fz={self.fz_kick_N}N/{self.fz_kick_dur_sec}s, cooldown={self.fz_kick_cooldown_sec}s)\n"
            f"  RECOVER(removed)\n"
            f"  DITHER(enable={int(self.dither_enable)}, only_track={int(self.dither_only_track)}, min_after={self.dither_min_after_start_sec}s, win={self.dither_win_sec}s, dur={self.dither_sec}s, net_pos_thr={self.dither_net_pos_thr_mm}mm, ratio_thr={self.dither_path_ratio_thr}, rms_pos_thr={self.dither_rms_pos_thr_mm}mm)\n"
            f"  RELEASE(enable={int(self.release_assist_enable)}, ramp_sec={self.release_ramp_sec})\n"
        )

    # ------------------------------------------------------------
    # Small helpers (force extraction / history)
    # ------------------------------------------------------------
    def _extract_force3(self, raw_force: np.ndarray) -> np.ndarray:
        idx = list(self.force_indices)
        f3 = np.zeros(3, dtype=np.float32)
        for i, k in enumerate(idx):
            if k < raw_force.size:
                f3[i] = float(raw_force[k])
        return f3

    def _build_live_force_history(self, hist_list: List[np.ndarray], current_force3: np.ndarray) -> np.ndarray:
        """
        Returns (L,3), padded like dataset.py:
        if insufficient history, repeat the first available force on the left.
        """
        L = max(1, self.force_history_len)

        if len(hist_list) <= 0:
            hist = current_force3.reshape(1, 3).astype(np.float32)
        else:
            hist = np.stack(hist_list, axis=0).astype(np.float32)

        hist = hist[-L:]  # keep most recent L

        if hist.shape[0] < L:
            pad_count = L - hist.shape[0]
            pad_value = hist[0:1] if hist.shape[0] > 0 else current_force3.reshape(1, 3).astype(np.float32)
            pad = np.repeat(pad_value, pad_count, axis=0)
            hist = np.concatenate([pad, hist], axis=0)

        return hist.astype(np.float32)

    # ------------------------------------------------------------
    # Load policy (nrs_act/source/models/policy.py) + ckpt
    # ------------------------------------------------------------
    def _load_policy_and_ckpt_from_act_root(self):
        act_source = os.path.join(self.act_root, "source")
        if self.act_root not in sys.path:
            sys.path.insert(0, self.act_root)
        if act_source not in sys.path:
            sys.path.insert(0, act_source)

        try:
            from models.policy import ACTPolicy, DiffusionPolicy
        except Exception as e:
            raise RuntimeError(
                f"Failed to import ACT/Diffusion policy classes from {act_source}/models/policy.py : {e}"
            )

        try:
            from models.flow_core import FlowRGBPolicy
        except Exception as e:
            FlowRGBPolicy = None
            if str(self.policy_class).upper() == "FLOW":
                raise RuntimeError(
                    f"Failed to import FlowRGBPolicy from {act_source}/models/flow_core.py : {e}"
                )

        args_override = {
            "kl_weight": float(self.get_parameter("kl_weight").value),
            "num_queries": int(self.chunk_size),

            "lr": 1e-4,
            "hidden_dim": int(self.get_parameter("hidden_dim").value),
            "dim_feedforward": int(self.get_parameter("dim_feedforward").value),
            "lr_backbone": float(self.get_parameter("lr_backbone").value),
            "backbone": str(self.get_parameter("backbone").value),
            "enc_layers": int(self.get_parameter("enc_layers").value),
            "dec_layers": int(self.get_parameter("dec_layers").value),
            "nheads": int(self.get_parameter("nheads").value),

            "camera_names": ["cam0"],
            "state_dim": 9,
            "action_dim": 9,

            "image_resize_hw": int(self.get_parameter("image_resize_hw").value),
            "image_pool_hw": int(self.get_parameter("image_pool_hw").value),
            "pretrained_backbone": bool(self.get_parameter("pretrained_backbone").value),

            # observation encoder config
            "position_dim": self.position_dim,
            "force_dim": self.force_dim,
            "position_encoder_hidden_dim": self.position_encoder_hidden_dim,
            "force_encoder_hidden_dim": self.force_encoder_hidden_dim,
            "force_encoder_num_layers": self.force_encoder_num_layers,
            "force_encoder_dropout": self.force_encoder_dropout,
            "observation_encoder_activation": self.observation_encoder_activation,

            # diffusion config (ignored by ACTPolicy/FLOW)
            "diffusion_train_steps": self.diffusion_train_steps,
            "diffusion_infer_steps": self.diffusion_infer_steps,
            "diffusion_beta_start": self.diffusion_beta_start,
            "diffusion_beta_end": self.diffusion_beta_end,
            "diffusion_loss_type": self.diffusion_loss_type,

            # FLOW config
            "use_force_history": self.use_force_history,
            "force_history_len": self.force_history_len,
            "flow_infer_steps": self.flow_infer_steps,
            "flow_train_eps": self.flow_train_eps,
            "flow_loss_type": self.flow_loss_type,
            "flow_obs_hidden_dim": self.flow_obs_hidden_dim,
            "flow_image_feature_dim": self.flow_image_feature_dim,
            "flow_global_cond_dim": self.flow_global_cond_dim,
            "flow_time_embed_dim": self.flow_time_embed_dim,
            "flow_down_dims": self.flow_down_dims,
            "flow_kernel_size": self.flow_kernel_size,
            "flow_n_groups": self.flow_n_groups,
            "flow_cond_predict_scale": self.flow_cond_predict_scale,
        }

        policy_class = str(self.policy_class).upper()
        if policy_class == "ACT":
            self.get_logger().info("[INFO] Loading ACTPolicy from nrs_act/source/models/policy.py ...")
            policy = ACTPolicy(args_override).to(self.device)
        elif policy_class == "DIFFUSION":
            self.get_logger().info("[INFO] Loading DiffusionPolicy from nrs_act/source/models/policy.py ...")
            policy = DiffusionPolicy(args_override).to(self.device)
        elif policy_class == "FLOW":
            self.get_logger().info("[INFO] Loading FlowRGBPolicy from nrs_act/source/models/flow_core.py ...")
            if FlowRGBPolicy is None:
                raise RuntimeError("FlowRGBPolicy import failed.")
            policy = FlowRGBPolicy(args_override).to(self.device)
        else:
            raise RuntimeError(f"Unsupported policy_class: {self.policy_class}")

        policy.eval()

        ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"policy_best.ckpt not found: {ckpt_path}")

        ckpt_obj = torch.load(ckpt_path, map_location=self.device)

        if isinstance(ckpt_obj, dict):
            if "model_state_dict" in ckpt_obj:
                state_dict = ckpt_obj["model_state_dict"]
            elif "state_dict" in ckpt_obj:
                state_dict = ckpt_obj["state_dict"]
            else:
                state_dict = ckpt_obj
        else:
            state_dict = ckpt_obj

        missing, unexpected = _try_load_state_dict_compat(policy, state_dict)

        if (len(missing) + len(unexpected) > 0) and hasattr(policy, "model"):
            missing2, unexpected2 = _try_load_state_dict_compat(policy.model, state_dict)
            if (len(missing2) + len(unexpected2)) < (len(missing) + len(unexpected)):
                missing, unexpected = missing2, unexpected2

        self.get_logger().info(
            f"[INFO] Loaded ckpt from {ckpt_path}. missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(missing) > 0:
            self.get_logger().warn(f"[INFO] missing sample: {list(missing)[:10]}")
        if len(unexpected) > 0:
            self.get_logger().warn(f"[INFO] unexpected sample: {list(unexpected)[:10]}")
        self.get_logger().info(
            f"[INFO] policy_class={policy_class}, camera_names=['cam0'], "
            f"use_force_history={self.use_force_history}, force_history_len={self.force_history_len}"
        )
        return policy

    # ------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------
    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size >= 6:
            with self._lock:
                self._pose6 = arr[:6].copy()

    def _on_force(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        with self._lock:
            self._force = arr.copy()
            if arr.size >= 3:
                self._force_hist.append(self._extract_force3(arr))

    def _preprocess_live_image(self, rgb: np.ndarray) -> np.ndarray:
        """
        Causal online stabilization for inference-time camera observation.

        This does not crop or resize. Resizing remains handled by _to_tensor_image_stack().
        It estimates frame-to-frame global translation/rotation, smooths the cumulative
        camera trajectory by EMA, and applies the correction to the current RGB frame.
        """
        if self.camera_preprocess_mode in ("off", "none", "raw") or cv2 is None:
            return rgb.copy()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if self._cam_prev_raw_gray is None:
            self._cam_prev_raw_gray = gray
            self._cam_prev_proc_gray = gray
            return rgb.copy()

        dx, dy, da = _estimate_pair_transform(self._cam_prev_raw_gray, gray)
        raw_norm = float(math.sqrt(dx * dx + dy * dy))

        delta = np.array([dx, dy, da], dtype=np.float32)
        self._cam_cum = (self._cam_cum + delta).astype(np.float32)

        alpha = float(np.clip(self.camera_stabilize_alpha, 0.0, 0.999))
        self._cam_smooth_cum = (
            alpha * self._cam_smooth_cum + (1.0 - alpha) * self._cam_cum
        ).astype(np.float32)
        correction = self._cam_smooth_cum - self._cam_cum

        proc = _warp_rgb_affine(
            rgb,
            dx=float(correction[0]),
            dy=float(correction[1]),
            da=float(correction[2]),
            border_mode=self.camera_stabilize_border_mode,
        )
        proc_gray = cv2.cvtColor(proc, cv2.COLOR_RGB2GRAY)

        if self._cam_prev_proc_gray is not None:
            pdx, pdy, _ = _estimate_pair_transform(self._cam_prev_proc_gray, proc_gray)
            proc_norm = float(math.sqrt(pdx * pdx + pdy * pdy))
        else:
            proc_norm = raw_norm

        beta = 0.05
        if self._cam_frame_count <= 1:
            self._cam_raw_jitter_ema = raw_norm
            self._cam_proc_jitter_ema = proc_norm
        else:
            self._cam_raw_jitter_ema = (1.0 - beta) * self._cam_raw_jitter_ema + beta * raw_norm
            self._cam_proc_jitter_ema = (1.0 - beta) * self._cam_proc_jitter_ema + beta * proc_norm

        self._cam_frame_count += 1
        self._cam_prev_raw_gray = gray
        self._cam_prev_proc_gray = proc_gray

        if self.camera_jitter_report_enable and (self._cam_frame_count % self.camera_jitter_log_every_n == 0):
            reduction = 0.0
            if self._cam_raw_jitter_ema > 1e-9:
                reduction = 100.0 * (self._cam_raw_jitter_ema - self._cam_proc_jitter_ema) / self._cam_raw_jitter_ema
            self.get_logger().info(
                f"[CAM-JITTER] online EMA before={self._cam_raw_jitter_ema:.4f}px, "
                f"after={self._cam_proc_jitter_ema:.4f}px, "
                f"RMS_like_reduction={reduction:.2f}%, mode={self.camera_preprocess_mode}"
            )

        return proc

    def _on_img(self, msg: Image):
        try:
            rgb_raw = _img_to_rgb_numpy(msg)
            rgb = self._preprocess_live_image(rgb_raw)
            with self._lock:
                self._img_cam0 = rgb
        except Exception as e:
            self.get_logger().error(f"[CAM0 IMG] decode/preprocess failed: {e}")

    # ------------------------------------------------------------
    # Contact update
    # ------------------------------------------------------------
    def _update_contact(self, meas_fz: float) -> bool:
        prev = self._contact
        if (not self._contact) and (meas_fz >= self.contact_on_thr):
            self._contact = True
        elif self._contact and (meas_fz <= self.contact_off_thr):
            self._contact = False
        return (prev != self._contact)

    def _compute_preload_target(self) -> float:
        tgt = self.preload_fixed_N

        if (self.preload_target_source == "stats_mean") and (self.stats is not None):
            if getattr(self.stats, "qpos_mode", "zscore") == "zscore":
                mean_fz = float(self.stats.qpos_a[8])
                tgt = abs(mean_fz) * float(self.preload_target_scale)
            elif getattr(self.stats, "qpos_mode", "zscore") in ["minmax_01", "minmax_m11"]:
                qmin_fz = float(self.stats.qpos_a[8])
                qmax_fz = float(self.stats.qpos_b[8])
                mid_fz = 0.5 * (qmin_fz + qmax_fz)
                tgt = abs(mid_fz) * float(self.preload_target_scale)

        tgt = max(float(self.preload_min_N), float(tgt))
        return float(tgt)

    # ------------------------------------------------------------
    # Small helpers: reset dither / kick count
    # ------------------------------------------------------------
    def _reset_dither(self):
        self._pose_hist6.clear()
        self._dither_score = 0.0

    def _reset_kick_count(self):
        self._kick_count = 0
        self._kick_count_t0 = _monotonic()

    # ------------------------------------------------------------
    # Stage transitions
    # ------------------------------------------------------------
    def _enter_preload(self, pose6_now: np.ndarray):
        """
        PRELOAD removed (bypassed).
        Keep this function name so the rest of the code path stays unchanged,
        but transition directly from APPROACH to TRACK once touch is confirmed.
        """
        self._preload_t0 = _monotonic()
        self._preload_ok = 0
        self._preload_hold_pose6 = pose6_now.astype(np.float32).copy()
        self._preload_target_N = self._compute_preload_target()

        self.get_logger().warn(
            f"[STAGE] PRELOAD bypassed -> TRACK directly "
            f"(touch confirmed, nominal_target={self._preload_target_N:.2f}N)"
        )
        self._enter_track()

    def _enter_track(self):
        self.stage = Stage.TRACK
        self.plans.clear()
        self._anchor_ready = False

        self._reset_dither()
        self._reset_kick_count()

        self.get_logger().warn("[STAGE] -> TRACK (resume inference)")

    def _enter_release(self, fz_cmd_start: float):
        self.stage = Stage.RELEASE
        self._release_t0 = _monotonic()
        self._release_start_fz_cmd = float(max(0.0, fz_cmd_start))

        self._reset_dither()

        self.get_logger().warn(
            f"[STAGE] -> RELEASE (fz ramp {self._release_start_fz_cmd:.3f} -> 0 in {self.release_ramp_sec:.2f}s)"
        )

    def _soft_reset_to_approach(self, reason: str):
        self.stage = Stage.APPROACH
        self.plans.clear()
        self._anchor_ready = False
        self._touch_ok = 0
        self._fz_kick_active = False

        self._stall_win_pose6 = None
        self._stall_win_t0 = _monotonic()
        self._stall_pose6_lpf = None
        self._recover_pose6_lpf = None

        self._reset_dither()
        self._reset_kick_count()

        self.get_logger().warn(f"[APPROACH-RESET] {reason} -> clear plans/anchor and continue APPROACH (RECOVER removed)")

    # ------------------------------------------------------------
    # Infer timer
    # ------------------------------------------------------------
    def _on_infer_timer(self):
        if self.stage == Stage.PRELOAD:
            return

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()
            cam0 = None if self._img_cam0 is None else self._img_cam0.copy()
            force_hist_list = list(self._force_hist)

        if pose6 is None or force is None or cam0 is None:
            return
        if force.size < 3:
            return

        f3 = self._extract_force3(force)

        q_np = np.concatenate([pose6[:6], f3], axis=0).astype(np.float32)
        q_t = torch.from_numpy(q_np).unsqueeze(0).to(self.device, dtype=torch.float32)

        if self.normalize_qpos_enabled and self.stats is not None:
            q_t = _normalize_qpos(q_t, self.stats)

        force_hist_t = None
        if self.use_force_history:
            hist_np = self._build_live_force_history(force_hist_list, f3)  # (L,3)
            force_hist_t = torch.from_numpy(hist_np).unsqueeze(0).to(self.device, dtype=torch.float32)  # (1,L,3)
            if self.normalize_qpos_enabled and self.stats is not None:
                force_hist_t = _normalize_force_history(force_hist_t, self.stats)

        try:
            img_t = _to_tensor_image_stack(cam0, device=self.device, resize_hw=self.resize_hw)
        except Exception as e:
            self.get_logger().error(f"[INFER] image stack failed: {e}")
            return

        try:
            with torch.inference_mode():
                if self.use_force_history:
                    out = self.policy(q_t, img_t, force_history=force_hist_t)
                else:
                    out = self.policy(q_t, img_t)

            seq = _fix_policy_output_seq(out, self.chunk_size, self.policy_class)

            if self.denorm_action_enabled and self.stats is not None:
                seq = _denorm_action_seq(seq, self.stats)

            seq_den = seq.detach().cpu().numpy().astype(np.float32)
            seq_den[:, 8] = np.clip(seq_den[:, 8], -self.fz_hard_limit, self.fz_hard_limit)

        except Exception as e:
            self.get_logger().error(f"[INFER] policy forward failed: {e}")
            return

        self.plans.append(Plan(t0=_monotonic(), seq_den=seq_den))

    # ------------------------------------------------------------
    # Temporal aggregation
    # ------------------------------------------------------------
    def _temporal_agg_cmd(self, now_t: float) -> Optional[np.ndarray]:
        if not self.plans:
            return None

        vals: List[np.ndarray] = []
        wts: List[float] = []

        for p in list(self.plans):
            age_steps = int((now_t - p.t0) * self.control_hz)
            k = age_steps + int(self.pred_step_offset)
            if 0 <= k < p.seq_den.shape[0]:
                v = p.seq_den[k]
                if self.use_temporal_agg and self.temporal_agg_mode == "exp":
                    w = _exp_decay_weight(age_steps, self.temporal_agg_tau_steps)
                else:
                    w = 1.0
                vals.append(v.astype(np.float32))
                wts.append(float(w))

        if len(vals) == 0:
            p = self.plans[-1]
            age_steps = int((now_t - p.t0) * self.control_hz)
            k = int(np.clip(age_steps + int(self.pred_step_offset), 0, p.seq_den.shape[0] - 1))
            return p.seq_den[k].astype(np.float32)

        W = float(np.sum(wts))
        if W <= 1e-9:
            return vals[-1].astype(np.float32)

        acc = np.zeros(9, dtype=np.float32)
        for v, w in zip(vals, wts):
            acc += (w / W) * v
        return acc.astype(np.float32)

    # ------------------------------------------------------------
    # Publish helpers
    # ------------------------------------------------------------
    def _publish_cmd(self, cmd9: np.ndarray):
        m = Float64MultiArray()
        m.data = [float(x) for x in cmd9.reshape(-1).tolist()]
        self.pub_cmd.publish(m)

    def _ramp_from(self, t0: float, ramp_sec: float) -> float:
        if ramp_sec <= 1e-6:
            return 1.0
        t = _monotonic() - float(t0)
        return float(np.clip(t / float(ramp_sec), 0.0, 1.0))

    def _startup_ramp(self) -> float:
        return self._ramp_from(self._t_start, self.startup_ramp_sec)

    # ------------------------------------------------------------
    # PRELOAD control
    # ------------------------------------------------------------
    def _preload_control_step(self, pose6_now: np.ndarray, meas_fz: float) -> np.ndarray:
        hold = self._preload_hold_pose6 if self._preload_hold_pose6 is not None else pose6_now.astype(np.float32)

        cmd = np.zeros(9, dtype=np.float32)
        cmd[0:6] = pose6_now.astype(np.float32)
        cmd[6] = 0.0
        cmd[7] = 0.0
        cmd[8] = 0.0

        if self.press_hold_xy:
            cmd[0] = hold[0]
            cmd[1] = hold[1]
        if self.press_hold_rpy:
            cmd[3] = hold[3]
            cmd[4] = hold[4]
            cmd[5] = hold[5]

        target = float(self._preload_target_N)
        err = float(target - meas_fz)

        dz = self.preload_kp_mm_per_N * max(0.0, err)
        dz = float(np.clip(dz, 0.0, self.preload_dz_max_mm))
        cmd[2] = float(cmd[2] - dz)

        mode = self.press_force_cmd_mode
        if mode == "zero":
            cmd[8] = 0.0
        elif mode == "target":
            cmd[8] = float(target)
        else:
            prev_fz = float(self.prev_cmd[8]) if (self.prev_cmd is not None) else 0.0
            cmd[8] = float(prev_fz)

        cmd[8] = float(np.clip(cmd[8], 0.0, self.fz_hard_limit))
        return cmd

    # ------------------------------------------------------------
    # RELEASE force shaping
    # ------------------------------------------------------------
    def _release_force(self, cmd_target: np.ndarray) -> np.ndarray:
        cmd = cmd_target.astype(np.float32).copy()
        t = _monotonic() - self._release_t0
        if self.release_ramp_sec <= 1e-6:
            s = 1.0
        else:
            s = float(np.clip(t / self.release_ramp_sec, 0.0, 1.0))
        fz = (1.0 - s) * float(self._release_start_fz_cmd)
        cmd[6] = 0.0
        cmd[7] = 0.0
        cmd[8] = float(max(0.0, fz))
        return cmd

    # ------------------------------------------------------------
    # Stall LPF + Window update
    # ------------------------------------------------------------
    def _stall_update(self, pose6_now: np.ndarray) -> float:
        dt = self.dt_control
        beta = _beta_from_tau(dt, self.stall_lpf_tau_sec)

        if self._stall_pose6_lpf is None:
            self._stall_pose6_lpf = pose6_now.astype(np.float32).copy()
        else:
            self._stall_pose6_lpf = (
                self._stall_pose6_lpf + beta * (pose6_now.astype(np.float32) - self._stall_pose6_lpf)
            ).astype(np.float32)

        lp = self._stall_pose6_lpf

        if self._stall_win_pose6 is None:
            self._stall_win_pose6 = lp.copy()
            self._stall_win_t0 = _monotonic()
            return 0.0

        net_dp = float(np.linalg.norm(lp[:3] - self._stall_win_pose6[:3]))
        net_da = float(np.linalg.norm(lp[3:6] - self._stall_win_pose6[3:6]))

        if (net_dp >= self.stall_window_net_pos_eps_mm) or (net_da >= self.stall_window_net_ang_eps_rad):
            self._stall_win_pose6 = lp.copy()
            self._stall_win_t0 = _monotonic()
            return 0.0

        return float(_monotonic() - self._stall_win_t0)

    # ------------------------------------------------------------
    # DITHER update
    # ------------------------------------------------------------
    def _dither_update(self, pose6_now: np.ndarray) -> float:
        self._pose_hist6.append(pose6_now.astype(np.float32).copy())
        if len(self._pose_hist6) < 4:
            return 0.0

        arr = np.stack(self._pose_hist6, axis=0)
        P = arr[:, :3]
        A = arr[:, 3:6]

        net_p = float(np.linalg.norm(P[-1] - P[0]))
        net_a = float(np.linalg.norm(A[-1] - A[0]))

        dP = P[1:] - P[:-1]
        dA = A[1:] - A[:-1]
        path_p = float(np.sum(np.linalg.norm(dP, axis=1)))
        path_a = float(np.sum(np.linalg.norm(dA, axis=1)))

        ratio_p = path_p / max(net_p, 1e-9)
        ratio_a = path_a / max(net_a, 1e-9)

        Pm = np.mean(P, axis=0)
        Am = np.mean(A, axis=0)
        rms_p = float(np.sqrt(np.mean(np.sum((P - Pm) ** 2, axis=1))))
        rms_a = float(np.sqrt(np.mean(np.sum((A - Am) ** 2, axis=1))))

        small_net = (net_p <= self.dither_net_pos_thr_mm) and (net_a <= self.dither_net_ang_thr_rad)
        oscill = (
            (ratio_p >= self.dither_path_ratio_thr) or (ratio_a >= self.dither_path_ratio_thr) or
            (rms_p >= self.dither_rms_pos_thr_mm) or (rms_a >= self.dither_rms_ang_thr_rad)
        )

        inside = bool(small_net and oscill)

        if inside:
            self._dither_score += self.dt_control
        else:
            self._dither_score = max(0.0, self._dither_score - self.dt_control * float(self.dither_decay))

        return float(self._dither_score)

    def _dither_allowed(self, elapsed_since_start: float) -> bool:
        if not self.dither_enable:
            return False
        if elapsed_since_start < self.dither_min_after_start_sec:
            return False
        if self.stage in (Stage.PRELOAD, Stage.RELEASE):
            return False
        if self.dither_only_track and (self.stage != Stage.TRACK):
            return False
        return True

    # ------------------------------------------------------------
    # Kick helper
    # ------------------------------------------------------------
    def _try_start_kick(self, now_t: float, reason: str, age_sec: float):
        if self._fz_kick_active:
            return False
        if (now_t - self._fz_kick_last_end_t) < self.fz_kick_cooldown_sec:
            return False

        self._fz_kick_active = True
        self._fz_kick_t0 = now_t
        self._kick_count += 1
        self._kick_count_t0 = now_t

        self.get_logger().warn(
            f"[{reason}] (contact=1) age={age_sec:.2f}s -> FZ KICK start "
            f"(#{self._kick_count}/{self.kick_max_before_recover}, fz={self.fz_kick_N:.2f}N, dur={self.fz_kick_dur_sec:.2f}s)"
        )

        self._stall_win_pose6 = None
        self._stall_win_t0 = now_t
        return True

    # ------------------------------------------------------------
    # Control timer
    # ------------------------------------------------------------
    def _on_control_timer(self):
        now_t = _monotonic()

        with self._lock:
            pose6 = None if self._pose6 is None else self._pose6.copy()
            force = None if self._force is None else self._force.copy()

        if pose6 is None:
            return

        meas_fz = 0.0
        if force is not None and force.size >= 3:
            meas_fz = float(force[2])

        # (1) FIRST publish = current pose hold
        if not self._sent_first_cmd:
            cmd0 = np.zeros(9, dtype=np.float32)
            cmd0[0:6] = pose6.astype(np.float32)
            cmd0[6] = 0.0
            cmd0[7] = 0.0
            cmd0[8] = float(self.first_cmd_fz)

            self.prev_cmd = cmd0.copy()
            self._sent_first_cmd = True
            self._t_first_pub = now_t
            self._t_start = now_t

            self.stage = Stage.APPROACH
            self._start_pose6 = pose6.astype(np.float32).copy()

            self._fz_base = max(0.0, meas_fz)
            self._fz_base_init = True
            self._touch_ok = 0

            self._stall_pose6_lpf = None
            self._stall_win_pose6 = None
            self._stall_win_t0 = now_t

            self._fz_kick_active = False
            self._fz_kick_last_end_t = -1e9
            self._recover_last_end_t = -1e9
            self._recover_pose6_lpf = None

            self._reset_dither()
            self._reset_kick_count()

            self._publish_cmd(cmd0)
            self.get_logger().info("[START] First publish = current pose. stage=APPROACH")
            return

        if self.prev_cmd is None:
            return

        changed = self._update_contact(meas_fz)
        if changed:
            if self.clear_plans_on_contact_change:
                self.plans.clear()
                self._anchor_ready = False
            self.get_logger().warn(f"[CONTACT] changed -> {int(self._contact)} | meas_fz={meas_fz:.3f} | stage={self.stage.name}")

            self._reset_dither()

            if self.release_assist_enable:
                if (not self._contact) and self._last_contact and (self.stage == Stage.TRACK):
                    fz_start = float(self.prev_cmd[8]) if self.prev_cmd is not None else 0.0
                    self._enter_release(fz_start)

        self._last_contact = self._contact

        if (now_t - self._kick_count_t0) >= self.kick_reset_sec:
            self._reset_kick_count()

        # -----------------------------
        # Stage-dependent cmd_target
        # -----------------------------
        cmd_target = None

        if self.stage == Stage.PRELOAD:
            cmd_target = self._preload_control_step(pose6.astype(np.float32), meas_fz)

            if abs(meas_fz - self._preload_target_N) <= self.preload_tol_N:
                self._preload_ok += 1
            else:
                self._preload_ok = 0

            if self._preload_ok >= self.preload_ok_count:
                self.get_logger().warn(f"[PRELOAD] OK (meas_fz~{self._preload_target_N:.2f}N) for {self.preload_ok_count} ticks -> TRACK")
                self._enter_track()
            else:
                if (_monotonic() - self._preload_t0) >= self.preload_timeout_sec:
                    self.get_logger().warn(f"[PRELOAD] TIMEOUT {self.preload_timeout_sec:.2f}s (meas_fz={meas_fz:.2f}) -> TRACK anyway")
                    self._enter_track()

        else:
            cmd_pred = self._temporal_agg_cmd(now_t)

            if cmd_pred is None:
                self._publish_cmd(self.prev_cmd)
                return

            cmd_target = cmd_pred.astype(np.float32).copy()

            if self.action_type == "delta":
                cmd_target = (self.prev_cmd + cmd_target).astype(np.float32)

            if not self._anchor_ready:
                self._anchor_offset6 = (pose6.astype(np.float32) - cmd_target[0:6]).astype(np.float32)
                self._anchor_ready = True
                self.get_logger().info("[ANCHOR] initialized")

            cmd_target[0:6] = (cmd_target[0:6] + self._anchor_offset6).astype(np.float32)

            if self.stage == Stage.APPROACH:
                cmd_target[6] = 0.0
                cmd_target[7] = 0.0
                cmd_target[8] = 0.0

            if self.stage == Stage.RELEASE:
                cmd_target = self._release_force(cmd_target)
                if (_monotonic() - self._release_t0) >= max(1e-6, self.release_ramp_sec):
                    self.stage = Stage.APPROACH
                    self.plans.clear()
                    self._anchor_ready = False
                    self._touch_ok = 0
                    self._reset_dither()
                    self.get_logger().warn("[STAGE] RELEASE done -> APPROACH")

        cmd_target[8] = float(np.clip(cmd_target[8], 0.0, self.fz_hard_limit))

        # -----------------------------
        # STALL check
        # -----------------------------
        stall_win_age = 0.0
        elapsed_since_start = (now_t - self._t_first_pub) if (self._t_first_pub is not None) else 0.0

        if self._t_first_pub is not None:
            stall_win_age = self._stall_update(pose6.astype(np.float32))

            can_check_stall = (elapsed_since_start >= self.stall_min_after_start_sec)
            stalled = can_check_stall and (stall_win_age >= self.stall_sec)

            if stalled and (self.stage not in (Stage.PRELOAD, Stage.RELEASE)):
                if self._contact:
                    if self.recover_enable and (self._kick_count >= self.kick_max_before_recover) and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[STALL] contact=1 but kick_count={self._kick_count} >= {self.kick_max_before_recover} -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("STALL contact=1 kick limit")
                    else:
                        self._try_start_kick(now_t, reason="STALL", age_sec=stall_win_age)
                else:
                    if self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[STALL] (contact=0) window_age={stall_win_age:.2f}s -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("STALL contact=0")

            if self._fz_kick_active and ((now_t - self._fz_kick_t0) >= self.fz_kick_dur_sec):
                self._fz_kick_active = False
                self._fz_kick_last_end_t = now_t
                self.plans.clear()
                self._anchor_ready = False
                self.get_logger().warn("[STALL] FZ KICK end -> replan requested")

        # -----------------------------
        # DITHER check
        # -----------------------------
        dither_age = 0.0
        if self._t_first_pub is not None and self._dither_allowed(elapsed_since_start):
            dither_age = self._dither_update(pose6.astype(np.float32))

            if dither_age >= self.dither_sec:
                if self._contact:
                    if self.recover_enable and (self._kick_count >= self.kick_max_before_recover) and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[DITHER] contact=1 and kick_count={self._kick_count} >= {self.kick_max_before_recover} -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("DITHER contact=1 kick limit")
                    else:
                        started = self._try_start_kick(now_t, reason="DITHER", age_sec=dither_age)
                        if not started and self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec) and (self._kick_count >= self.kick_max_before_recover):
                            self.get_logger().warn("[DITHER] kick cooldown but kick limit reached -> APPROACH reset (RECOVER removed)")
                            self._soft_reset_to_approach("DITHER cooldown + kick limit")
                else:
                    if self.recover_enable and ((now_t - self._recover_last_end_t) >= self.recover_cooldown_sec):
                        self.get_logger().warn(f"[DITHER] contact=0 age={dither_age:.2f}s -> APPROACH reset (RECOVER removed)")
                        self._soft_reset_to_approach("DITHER contact=0")

                self._reset_dither()

        # -----------------------------
        # Touch detector
        # -----------------------------
        if self.stage == Stage.APPROACH and (not self._fz_kick_active):
            if not self._fz_base_init:
                self._fz_base = max(0.0, meas_fz)
                self._fz_base_init = True
            else:
                beta_base = _beta_from_tau(self.dt_control, self.touch_baseline_tau_sec)
                self._fz_base = float((1.0 - beta_base) * self._fz_base + beta_base * max(0.0, meas_fz))

            if self.touch_use_delta:
                touch_sig = max(0.0, meas_fz - self._fz_base)
            else:
                touch_sig = max(0.0, meas_fz)

            elapsed = elapsed_since_start
            allow_touch = (elapsed >= self.touch_min_after_start_sec)

            if allow_touch and (touch_sig >= self.touch_fz_thr):
                self._touch_ok += 1
            else:
                self._touch_ok = 0

            if self._touch_ok >= self.touch_ok_count:
                self._touch_ok = 0
                self._enter_preload(pose6.astype(np.float32))

        # -----------------------------
        # Apply kick
        # -----------------------------
        if self._fz_kick_active:
            cmd_target[8] = float(max(cmd_target[8], self.fz_kick_N))

        cmd_target[8] = float(np.clip(cmd_target[8], 0.0, self.fz_hard_limit))

        # -----------------------------
        # QP-safe slow-follow
        # -----------------------------
        dt = self.dt_control

        if (self.stage == Stage.RECOVER) and self.recover_use_overrides:
            tau = float(self.recover_tau_sec)
            beta = _beta_from_tau(dt, tau)
            ramp = self._ramp_from(self._recover_t0, self.recover_startup_ramp_sec)
            cap_pos = max(1e-9, self.recover_step_cap_pos_mm * ramp)
            cap_ang = max(1e-12, self.recover_step_cap_ang_rad * ramp)
            cap_fz = max(1e-9, self.recover_step_cap_fz * ramp)
        else:
            beta = _beta_from_tau(dt, self.tau_sec)
            ramp = self._startup_ramp()
            cap_pos = max(1e-9, self.step_cap_pos_mm * ramp)
            cap_ang = max(1e-12, self.step_cap_ang_rad * ramp)
            cap_fz = max(1e-9, self.step_cap_fz * ramp)

        d = (cmd_target - self.prev_cmd).astype(np.float32)
        d = (beta * d).astype(np.float32)

        for i in range(3):
            di = float(d[i])
            if abs(di) > cap_pos:
                d[i] = float(np.sign(di) * cap_pos)
        for i in range(3, 6):
            di = float(d[i])
            if abs(di) > cap_ang:
                d[i] = float(np.sign(di) * cap_ang)
        for i in (6, 7):
            di = float(d[i])
            if abs(di) > cap_pos:
                d[i] = float(np.sign(di) * cap_pos)
        di = float(d[8])
        if abs(di) > cap_fz:
            d[8] = float(np.sign(di) * cap_fz)

        cmd_next = (self.prev_cmd + d).astype(np.float32)

        for i in range(9):
            a0 = float(self.prev_cmd[i])
            a1 = float(cmd_next[i])
            tg = float(cmd_target[i])
            if (a0 - tg) * (a1 - tg) < 0.0:
                cmd_next[i] = tg

        self._publish_cmd(cmd_next)
        self.prev_cmd = cmd_next

        if (int(now_t * self.control_hz) % self.debug_every_n) == 0:
            base = self._fz_base if self._fz_base_init else 0.0
            touch_sig = max(0.0, meas_fz - base) if self.touch_use_delta else max(0.0, meas_fz)
            self.get_logger().info(
                f"[CTRL] stage={self.stage.name} contact={int(self._contact)} meas_fz={meas_fz:.3f} "
                f"fz_base={base:.3f} touch_sig={touch_sig:.3f} touch_ok={self._touch_ok} | "
                f"stall_win={stall_win_age:.2f}s dither={dither_age:.2f}s kickN={int(self._fz_kick_active)} kickCnt={self._kick_count} | "
                f"beta={beta:.4f} ramp={ramp:.3f} cap(pos={cap_pos:.4f}, ang={cap_ang:.6f}, fz={cap_fz:.4f}) | "
                f"cmd_xyz=[{cmd_next[0]:.3f},{cmd_next[1]:.3f},{cmd_next[2]:.3f}] cmd_fz={cmd_next[8]:.3f}"
            )


# ============================================================
# main
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCmdMotionInfer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
