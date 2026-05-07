#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
node_check_inference.py (FIXED / MINIMAL)

Goal:
- Subscribe ONLY 4 topics:
    /ur10skku/currentP                  (Float64MultiArray)  -> pose 6D
    /ur10skku/currentF                  (Float64MultiArray)  -> force 6D (use first 3 by default)
    /realsense/ee/color/image_raw       (sensor_msgs/Image)  -> ee cam
    /realsense/top/color/image_raw      (sensor_msgs/Image)  -> top cam
- Build ACT input exactly like training:
    qpos = [pose6 + force3] => (9,)
    image = stack([top, ee]) => (K=2,3,H,W) ; batch => (1,2,3,H,W) ; float in [0,1]
    camera_names MUST be ["cam_top","cam_ee"] (same order)
- Load ckpt_dir/policy_best.ckpt (model-only state_dict) using training-time policy.py in act_root
- When first (pose+force+both images) arrives, run one forward -> dump 100-step trajectory -> exit (default)

Why previous crash happened:
- all_cam_features empty => camera_names missing/empty in model construction OR wrong image format.
  This file hard-fixes both.

Usage:
ros2 run nrs_imitation node_check_inference --ros-args \
  -p ckpt_dir:=/home/eunseop/nrs_lab2/checkpoints/ur10e_swing/20260208_1536 \
  -p act_root:=/home/eunseop/nrs_lab2/nrs_imitation \
  -p chunk_size:=100 \
  -p image_qos:=reliable \
  -p dump_full:=False \
  -p dump_head_n:=10 \
  -p dump_tail_n:=10
"""

import os
import sys
import time
import pickle
from typing import Optional, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image


# -----------------------------
# Helpers
# -----------------------------
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


def _img_to_rgb_numpy(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image to np.uint8 (H,W,3) in RGB order.

    Supports:
    - rgb8
    - bgr8
    - rgba8 / bgra8 (alpha dropped)
    """
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()

    # data is bytes-like
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = buf.reshape((h, w, 3))
        return img
    if enc == "bgr8":
        img = buf.reshape((h, w, 3))
        return img[..., ::-1].copy()
    if enc == "rgba8":
        img = buf.reshape((h, w, 4))[..., :3]
        return img
    if enc == "bgra8":
        img = buf.reshape((h, w, 4))[..., :3]
        return img[..., ::-1].copy()

    # fallback attempt: assume 3-channel
    # (If your camera publishes something else, fix encoding upstream)
    try:
        img = buf.reshape((h, w, 3))
        return img
    except Exception as e:
        raise RuntimeError(f"Unsupported image encoding={msg.encoding}, size=({h},{w}), err={e}")


def _to_tensor_image_stack(top_rgb: np.ndarray, ee_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Make (B=1,K=2,3,H,W) float tensor in [0,1].
    Order MUST match training camera_names=["cam_top","cam_ee"] => [top, ee]
    """
    if top_rgb.shape != ee_rgb.shape:
        # You can resize here if needed, but training likely used same resolution.
        raise RuntimeError(f"Top/Ee image size mismatch: top={top_rgb.shape} ee={ee_rgb.shape}")

    # (H,W,3) -> (3,H,W)
    top = np.transpose(top_rgb, (2, 0, 1))
    ee = np.transpose(ee_rgb, (2, 0, 1))

    # stack (K,3,H,W)
    img = np.stack([top, ee], axis=0).astype(np.float32) / 255.0

    # (1,K,3,H,W)
    img_t = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)
    return img_t


def _to_tensor_qpos(pose6: np.ndarray, force6: np.ndarray, force_indices=(0, 1, 2),
                    device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    qpos = [pose6 + force3] => (1,9)
    """
    pose6 = np.asarray(pose6, dtype=np.float32).reshape(-1)
    force6 = np.asarray(force6, dtype=np.float32).reshape(-1)
    if pose6.size < 6:
        raise RuntimeError(f"pose6 size < 6: {pose6.size}")
    if force6.size < max(force_indices) + 1:
        raise RuntimeError(f"force6 size < needed idx: force size={force6.size}, idx={force_indices}")

    f3 = force6[list(force_indices)].astype(np.float32)
    q = np.concatenate([pose6[:6], f3], axis=0).astype(np.float32)  # (9,)
    q_t = torch.from_numpy(q).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,9)
    return q_t


def _load_stats_if_exists(ckpt_dir: str) -> Optional[dict]:
    p = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        stats = pickle.load(f)
    return stats


def _normalize_qpos(q_t: torch.Tensor, stats: dict) -> torch.Tensor:
    mu = torch.tensor(stats["qpos_mean"], dtype=torch.float32, device=q_t.device).view(1, 9)
    sd = torch.tensor(stats["qpos_std"], dtype=torch.float32, device=q_t.device).view(1, 9)
    return (q_t - mu) / sd


def _denormalize_action(a_t: torch.Tensor, stats: dict) -> torch.Tensor:
    """
    a_t: (...,9) normalized
    """
    mu = torch.tensor(stats["action_mean"], dtype=torch.float32, device=a_t.device).view(*(1,) * (a_t.dim() - 1), 9)
    sd = torch.tensor(stats["action_std"], dtype=torch.float32, device=a_t.device).view(*(1,) * (a_t.dim() - 1), 9)
    return a_t * sd + mu


def _fix_a_hat_shape(a_hat: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Standardize output to (T,9) with T=chunk_size.
    Handles common variants:
      - (B,T,9)
      - (T,B,9)
      - (T,9) already
    """
    if a_hat.dim() == 2:
        # (T,9) or (B,9) (unlikely)
        return a_hat

    if a_hat.dim() != 3:
        raise RuntimeError(f"Unexpected a_hat dim: {a_hat.shape}")

    B0, B1, B2 = a_hat.shape

    # case (B,T,9)
    if B0 == 1 and B1 == chunk_size and B2 == 9:
        return a_hat[0]  # (T,9)

    # case (T,B,9)
    if B0 == chunk_size and B1 == 1 and B2 == 9:
        return a_hat[:, 0, :]  # (T,9)

    # case (B,T,9) but other B
    if B2 == 9 and B1 == chunk_size:
        return a_hat[0]

    # last resort
    raise RuntimeError(f"Cannot interpret a_hat shape={a_hat.shape} with chunk_size={chunk_size}")


# -----------------------------
# Node
# -----------------------------
class NodeCheckInference(Node):
    def __init__(self):
        super().__init__("node_check_inference")

        # ---- parameters (minimal) ----
        self.declare_parameter("ckpt_dir", "")
        self.declare_parameter("act_root", "")
        self.declare_parameter("chunk_size", 100)

        self.declare_parameter("pose_topic", "/ur10skku/currentP")
        self.declare_parameter("force_topic", "/ur10skku/currentF")
        self.declare_parameter("ee_img_topic", "/realsense/ee/color/image_raw")
        self.declare_parameter("top_img_topic", "/realsense/top/color/image_raw")

        self.declare_parameter("image_qos", "best_effort")  # reliable | best_effort
        self.declare_parameter("dump_full", False)
        self.declare_parameter("dump_head_n", 10)
        self.declare_parameter("dump_tail_n", 10)
        self.declare_parameter("exit_after_dump", True)

        # model hyperparams (must match training)
        self.declare_parameter("kl_weight", 10.0)
        self.declare_parameter("hidden_dim", 512)
        self.declare_parameter("dim_feedforward", 3200)
        self.declare_parameter("nheads", 8)
        self.declare_parameter("enc_layers", 4)
        self.declare_parameter("dec_layers", 7)
        self.declare_parameter("backbone", "resnet18")
        self.declare_parameter("lr_backbone", 1e-5)
        self.declare_parameter("no_pretrained", False)
        self.declare_parameter("image_resize_hw", 256)
        self.declare_parameter("image_pool_hw", 4)

        # normalization handling
        self.declare_parameter("normalize_qpos", True)
        self.declare_parameter("denorm_action", True)
        self.declare_parameter("force_indices", [0, 1, 2])  # pick 3 from 6D currentF

        # ---- read params ----
        self.ckpt_dir = str(self.get_parameter("ckpt_dir").value)
        self.act_root = str(self.get_parameter("act_root").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.force_topic = str(self.get_parameter("force_topic").value)
        self.ee_img_topic = str(self.get_parameter("ee_img_topic").value)
        self.top_img_topic = str(self.get_parameter("top_img_topic").value)

        self.image_qos_str = str(self.get_parameter("image_qos").value)
        self.dump_full = bool(self.get_parameter("dump_full").value)
        self.dump_head_n = int(self.get_parameter("dump_head_n").value)
        self.dump_tail_n = int(self.get_parameter("dump_tail_n").value)
        self.exit_after_dump = bool(self.get_parameter("exit_after_dump").value)

        self.normalize_qpos_enabled = bool(self.get_parameter("normalize_qpos").value)
        self.denorm_action_enabled = bool(self.get_parameter("denorm_action").value)
        self.force_indices = tuple(int(x) for x in self.get_parameter("force_indices").value)

        # ---- device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"[INFO] Using device: {self.device}")

        # ---- validate paths ----
        if not self.ckpt_dir or not os.path.isdir(self.ckpt_dir):
            raise RuntimeError(f"ckpt_dir invalid: {self.ckpt_dir}")
        if not self.act_root or not os.path.isdir(self.act_root):
            raise RuntimeError(f"act_root invalid: {self.act_root}")

        # ---- load stats (optional but recommended) ----
        self.stats = _load_stats_if_exists(self.ckpt_dir)
        if self.stats is None:
            self.get_logger().warn("[STATS] dataset_stats.pkl not found. normalize_qpos/denorm_action will be disabled.")
            self.normalize_qpos_enabled = False
            self.denorm_action_enabled = False
        else:
            self.get_logger().info(f"[STATS] Loaded dataset stats: {os.path.join(self.ckpt_dir, 'dataset_stats.pkl')}")

        # ---- load policy ----
        self.policy = self._load_policy_and_ckpt()

        # ---- buffers for latest messages ----
        self._pose6: Optional[np.ndarray] = None
        self._force6: Optional[np.ndarray] = None
        self._img_top: Optional[np.ndarray] = None
        self._img_ee: Optional[np.ndarray] = None

        self._dumped_once = False

        # ---- QoS ----
        img_rel = _reliability_from_str(self.image_qos_str)
        img_qos = _qos(depth=1, reliability=img_rel)
        vec_qos = _qos(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # ---- subs ----
        self.create_subscription(Float64MultiArray, self.pose_topic, self._on_pose, vec_qos)
        self.create_subscription(Float64MultiArray, self.force_topic, self._on_force, vec_qos)
        self.create_subscription(Image, self.top_img_topic, self._on_top_img, img_qos)
        self.create_subscription(Image, self.ee_img_topic, self._on_ee_img, img_qos)

        self.get_logger().info(
            "[INFO] ✅ Ready. Waiting for pose+force+images...\n"
            f"  pose_topic={self.pose_topic}\n"
            f"  force_topic={self.force_topic}\n"
            f"  ee_img_topic={self.ee_img_topic}\n"
            f"  top_img_topic={self.top_img_topic}\n"
            f"  image_qos={self.image_qos_str}\n"
            f"  force_indices={self.force_indices}\n"
        )

        # ---- loop timer ----
        self.timer = self.create_timer(0.05, self._loop)  # 20 Hz


    def _load_policy_and_ckpt(self):
        # Make sure act_root is importable
        if self.act_root not in sys.path:
            sys.path.insert(0, self.act_root)

        # Import training-time policy.py
        try:
            from policy import ACTPolicy
        except Exception as e:
            raise RuntimeError(f"Failed to import ACTPolicy from {self.act_root}/policy.py : {e}")

        # Build args_override EXACTLY like training expects (dict, not namespace)
        args_override = {
            "lr": 1e-4,  # not used for inference but build function expects it
            "num_queries": int(self.chunk_size),
            "kl_weight": float(self.get_parameter("kl_weight").value),
            "hidden_dim": int(self.get_parameter("hidden_dim").value),
            "dim_feedforward": int(self.get_parameter("dim_feedforward").value),
            "lr_backbone": float(self.get_parameter("lr_backbone").value),
            "backbone": str(self.get_parameter("backbone").value),
            "enc_layers": int(self.get_parameter("enc_layers").value),
            "dec_layers": int(self.get_parameter("dec_layers").value),
            "nheads": int(self.get_parameter("nheads").value),

            # *** CRITICAL: must match training ***
            "camera_names": ["cam_top", "cam_ee"],

            # dims
            "state_dim": 9,
            "action_dim": 9,

            # perf knobs (safe defaults)
            "image_resize_hw": int(self.get_parameter("image_resize_hw").value),
            "image_pool_hw": int(self.get_parameter("image_pool_hw").value),
            "pretrained_backbone": (not bool(self.get_parameter("no_pretrained").value)),
        }

        self.get_logger().info("[INFO] Loading policy (training-time policy.py)...")
        policy = ACTPolicy(args_override).to(self.device)
        policy.eval()

        # Load ckpt (model-only)
        ckpt_path = os.path.join(self.ckpt_dir, "policy_best.ckpt")
        if not os.path.exists(ckpt_path):
            raise RuntimeError(f"policy_best.ckpt not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = policy.model.load_state_dict(ckpt, strict=False)
        self.get_logger().info(f"[INFO] Loaded ckpt. missing={len(missing)}, unexpected={len(unexpected)}")

        # sanity: camera_names must be non-empty
        try:
            cam_names = list(policy.model.camera_names)
        except Exception:
            cam_names = args_override["camera_names"]
        self.get_logger().info(f"[INFO] camera_names in model = {cam_names}")

        return policy


    # -----------------------------
    # Callbacks
    # -----------------------------
    def _on_pose(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size >= 6:
            self._pose6 = arr[:6].copy()

    def _on_force(self, msg: Float64MultiArray):
        arr = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            self._force6 = arr.copy()

    def _on_top_img(self, msg: Image):
        try:
            rgb = _img_to_rgb_numpy(msg)
            self._img_top = rgb
        except Exception as e:
            self.get_logger().error(f"[TOP IMG] decode failed: {e}")

    def _on_ee_img(self, msg: Image):
        try:
            rgb = _img_to_rgb_numpy(msg)
            self._img_ee = rgb
        except Exception as e:
            self.get_logger().error(f"[EE IMG] decode failed: {e}")


    # -----------------------------
    # Main loop
    # -----------------------------
    def _have_all(self) -> bool:
        return (self._pose6 is not None) and (self._force6 is not None) and (self._img_top is not None) and (self._img_ee is not None)

    def _loop(self):
        if self._dumped_once and self.exit_after_dump:
            return

        if not self._have_all():
            return

        try:
            # Build inputs
            q_t = _to_tensor_qpos(self._pose6, self._force6, force_indices=self.force_indices, device=self.device)
            img_t = _to_tensor_image_stack(self._img_top, self._img_ee, device=self.device)

            # Optional normalization
            if self.normalize_qpos_enabled and self.stats is not None:
                q_t = _normalize_qpos(q_t, self.stats)

            # Forward
            with torch.inference_mode():
                a_hat = self.policy(q_t, img_t)  # expected (1,T,9) or (T,1,9)

            seq = _fix_a_hat_shape(a_hat, self.chunk_size)  # (T,9)

            # Optional denormalize to real units
            if self.denorm_action_enabled and self.stats is not None:
                seq = _denormalize_action(seq, self.stats)

            seq_np = seq.detach().cpu().numpy()

            # Dump
            self._dump_sequence(seq_np)

            self._dumped_once = True
            if self.exit_after_dump:
                self.get_logger().info("[INFO] Done. Shutting down (exit_after_dump=True).")
                rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"[INFER] failed: {e}")
            # if it keeps failing, you can shut down or keep retrying; keep retrying is more useful


    def _dump_sequence(self, seq_np: np.ndarray):
        T = seq_np.shape[0]
        self.get_logger().info(f"[DUMP] inferred trajectory: shape={seq_np.shape} (T={T})")

        dump_full = self.dump_full
        hn = max(0, int(self.dump_head_n))
        tn = max(0, int(self.dump_tail_n))

        def fmt_row(i: int, row: np.ndarray) -> str:
            # row is (9,)
            return f"{i:03d}: " + " ".join([f"{x:+0.6f}" for x in row.tolist()])

        if dump_full:
            lines = [fmt_row(i, seq_np[i]) for i in range(T)]
            self.get_logger().info("[DUMP] FULL\n" + "\n".join(lines))
            return

        # head/tail
        head_idx = list(range(min(hn, T)))
        tail_idx = list(range(max(0, T - tn), T))

        lines = []
        if len(head_idx) > 0:
            lines.append("[HEAD]")
            lines += [fmt_row(i, seq_np[i]) for i in head_idx]
        if len(tail_idx) > 0:
            if len(head_idx) > 0:
                lines.append("...")
            lines.append("[TAIL]")
            lines += [fmt_row(i, seq_np[i]) for i in tail_idx]

        self.get_logger().info("[DUMP]\n" + "\n".join(lines))


# -----------------------------
# main
# -----------------------------
def main(args=None):
    rclpy.init(args=args)
    node = NodeCheckInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
    