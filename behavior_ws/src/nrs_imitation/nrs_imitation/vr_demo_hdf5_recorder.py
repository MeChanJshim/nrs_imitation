#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vr_demo_hdf5_recorder.py

Joystick-controlled single-camera ACT merged-HDF5 recorder.

This node supports two recording modes selected by ROS parameter:

  1) tracker mode
     - pose   : /calibrated_pose               Float64MultiArray [x y z wx wy wz]
     - force  : /ftsensor/measured_Cvalue      geometry_msgs/Wrench
     - image  : /realsense/vr/color/image_raw  sensor_msgs/Image

  2) robot mode
     - pose   : /ur10skku/currentP             Float64MultiArray [x y z wx wy wz]
     - force  : /ur10skku/currentF             Float64MultiArray [fx fy fz ...]
     - image  : /realsense/robot/color/image_raw sensor_msgs/Image

  - joystick commands  : /vr_demo_recorder/command     std_msgs/String

The old force-threshold start/end trigger has been removed.
Episode control is done by the joystick command node:
  A      -> start_recording
  B      -> end_recording
  X      -> erase_current_episode
  Y      -> terminate_node
  D-left -> prev_episode
  D-right-> next_episode

Important:
  - The recorder NO LONGER auto-stops when num_episodes is reached.
  - It keeps recording beyond 50 episodes if needed.
  - Shutdown happens only when:
      * joystick terminate_node command is received
      * backup keyboard quit key is pressed
      * Ctrl+C / explicit stop is requested

Output merged HDF5 layout:
  /home/eunseop/nrs_imitation/datasets/ACT/YYYYMMDD_HHMM/merged_hdf5/
    vr_demo_merged_YYYYMMDD_HHMM.hdf5

  episodes/
    ep_0000/
      position        (T, 6) float32  [x_mm y_mm z_mm wx wy wz]
      ft              (T, 3) float32  [fx fy fz]
      images/
        cam0          (T, H, W, 3) uint8 RGB

Force processing policy:
  - Fx, Fy -> 0 by default
  - Fz -> raw Fz + EMA
  - first/last force_edge_zero_sec -> all force zero

Designed to be converted by a single-camera version of demo_data_act_form.py.
"""

import os
import sys
import time
import json
import atexit
import threading
import select
import termios
import tty
from typing import Optional, List, Tuple, Set

import numpy as np
import h5py

try:
    import cv2
except Exception:
    cv2 = None

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Image


# ============================================================
# Image utilities
# ============================================================
def image_to_rgb_numpy(msg: Image) -> Optional[np.ndarray]:
    """
    Convert ROS sensor_msgs/Image to RGB uint8 numpy array.

    Supported encodings:
      rgb8, bgr8, rgba8, bgra8, mono8
    """
    enc = (msg.encoding or "").lower()
    h, w, step = int(msg.height), int(msg.width), int(msg.step)

    if h <= 0 or w <= 0 or step <= 0:
        return None

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if buf.size < h * step:
        return None

    row = buf[: h * step].reshape(h, step)

    if enc in ("rgb8", "bgr8"):
        need = w * 3
        if step < need:
            return None
        img = row[:, :need].reshape(h, w, 3)
        if enc == "bgr8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc in ("rgba8", "bgra8"):
        need = w * 4
        if step < need:
            return None
        img4 = row[:, :need].reshape(h, w, 4)
        img = img4[:, :, :3]
        if enc == "bgra8":
            img = img[:, :, ::-1]
        return img.copy()

    if enc == "mono8":
        need = w
        if step < need:
            return None
        gray = row[:, :need].reshape(h, w)
        return np.repeat(gray[:, :, None], 3, axis=2).copy()

    return None


def pick_image_shape(frames: List[Optional[np.ndarray]]) -> Tuple[int, int]:
    for im in frames:
        if im is not None and im.ndim == 3 and im.shape[0] > 1 and im.shape[1] > 1 and im.shape[2] == 3:
            return int(im.shape[0]), int(im.shape[1])
    return 0, 0


def stack_images_repeat_last(frames: List[Optional[np.ndarray]], logger=None) -> Optional[np.ndarray]:
    """
    Stack image frames into (T,H,W,3).
    Missing or shape-mismatched frames are replaced by the latest valid frame.
    If no valid frame exists, return None.
    """
    H, W = pick_image_shape(frames)
    if H <= 0 or W <= 0:
        return None

    T = len(frames)
    out = np.zeros((T, H, W, 3), dtype=np.uint8)
    last = np.zeros((H, W, 3), dtype=np.uint8)
    valid_count = 0
    repeated_count = 0

    for i, im in enumerate(frames):
        if im is not None and im.ndim == 3 and im.shape == (H, W, 3):
            out[i] = im
            last = im
            valid_count += 1
        else:
            out[i] = last
            repeated_count += 1

    if logger is not None:
        logger.info(
            f"[IMAGE] stacked: T={T}, H={H}, W={W}, "
            f"valid={valid_count}, repeated_or_invalid={repeated_count}"
        )

    return out




# ============================================================
# Camera stabilization / jitter diagnostics
# ============================================================
def moving_average_1d(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or x.size == 0:
        return x.astype(np.float32).copy()
    kernel = np.ones((2 * radius + 1,), dtype=np.float32) / float(2 * radius + 1)
    x_pad = np.pad(x.astype(np.float32), (radius, radius), mode="edge")
    y = np.convolve(x_pad, kernel, mode="same")
    return y[radius:-radius].astype(np.float32)


def estimate_pair_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
    """Estimate global dx, dy, dtheta from prev -> curr."""
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


def compute_jitter_metrics(images: np.ndarray, smoothing_radius: int = 15) -> dict:
    """
    Estimate high-frequency global camera jitter in pixel units.

    Jitter is defined as frame-to-frame global translation residual after
    removing low-frequency motion by moving-average smoothing.
    """
    imgs = np.asarray(images)
    T = int(imgs.shape[0]) if imgs.ndim >= 4 else 0
    if cv2 is None or T <= 2:
        return {
            "frames": T,
            "shape": tuple(imgs.shape[1:]) if imgs.ndim >= 4 else (),
            "rms_px": 0.0,
            "p95_px": 0.0,
            "max_px": 0.0,
            "mean_motion_px": 0.0,
        }

    transforms = np.zeros((T - 1, 3), dtype=np.float32)
    prev_gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
    for i in range(T - 1):
        curr_gray = cv2.cvtColor(imgs[i + 1], cv2.COLOR_RGB2GRAY)
        transforms[i] = np.asarray(estimate_pair_transform(prev_gray, curr_gray), dtype=np.float32)
        prev_gray = curr_gray

    trans_xy = transforms[:, :2].astype(np.float32)
    smooth_xy = np.zeros_like(trans_xy)
    for j in range(2):
        smooth_xy[:, j] = moving_average_1d(trans_xy[:, j], smoothing_radius)
    residual_xy = trans_xy - smooth_xy
    jitter = np.linalg.norm(residual_xy, axis=1)
    motion = np.linalg.norm(trans_xy, axis=1)

    return {
        "frames": T,
        "shape": tuple(imgs.shape[1:]),
        "rms_px": float(np.sqrt(np.mean(jitter ** 2))) if jitter.size else 0.0,
        "p95_px": float(np.percentile(jitter, 95)) if jitter.size else 0.0,
        "max_px": float(np.max(jitter)) if jitter.size else 0.0,
        "mean_motion_px": float(np.mean(motion)) if motion.size else 0.0,
    }


def stabilize_image_sequence(images: np.ndarray,
                             smoothing_radius: int = 15,
                             border_mode: str = "reflect") -> np.ndarray:
    """Offline whole-episode video stabilization. images: (T,H,W,3) RGB uint8."""
    if cv2 is None:
        return np.asarray(images, dtype=np.uint8).copy()

    imgs = np.asarray(images, dtype=np.uint8)
    T = int(imgs.shape[0])
    if T <= 1:
        return imgs.copy()

    prev_gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
    transforms = np.zeros((T - 1, 3), dtype=np.float32)

    for i in range(T - 1):
        curr_gray = cv2.cvtColor(imgs[i + 1], cv2.COLOR_RGB2GRAY)
        transforms[i] = np.asarray(estimate_pair_transform(prev_gray, curr_gray), dtype=np.float32)
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed = np.zeros_like(trajectory)
    for j in range(3):
        smoothed[:, j] = moving_average_1d(trajectory[:, j], smoothing_radius)

    diff = smoothed - trajectory
    transforms_smooth = transforms.copy()
    transforms_smooth[:, 0] += diff[:, 0]
    transforms_smooth[:, 1] += diff[:, 1]
    transforms_smooth[:, 2] += diff[:, 2]

    H, W = imgs.shape[1], imgs.shape[2]
    out = np.empty_like(imgs)
    out[0] = imgs[0]

    if str(border_mode).lower() == "constant":
        border_flag = cv2.BORDER_CONSTANT
    elif str(border_mode).lower() == "replicate":
        border_flag = cv2.BORDER_REPLICATE
    else:
        border_flag = cv2.BORDER_REFLECT

    for i in range(1, T):
        dx, dy, da = [float(x) for x in transforms_smooth[i - 1]]
        c = float(np.cos(da))
        s = float(np.sin(da))
        m = np.array([[c, -s, dx], [s, c, dy]], dtype=np.float32)
        out[i] = cv2.warpAffine(
            imgs[i],
            m,
            (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=border_flag,
        )
    return out.astype(np.uint8)


def center_crop_images(images: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.uint8)
    H, W = imgs.shape[1], imgs.shape[2]
    ch = int(min(max(1, crop_h), H))
    cw = int(min(max(1, crop_w), W))
    y0 = max(0, (H - ch) // 2)
    x0 = max(0, (W - cw) // 2)
    return imgs[:, y0:y0 + ch, x0:x0 + cw, :]


def resize_images(images: np.ndarray, resize_hw: int) -> np.ndarray:
    imgs = np.asarray(images, dtype=np.uint8)
    if resize_hw <= 0:
        return imgs.copy()
    if cv2 is None:
        return imgs.copy()
    out = []
    for im in imgs:
        out.append(cv2.resize(im, (resize_hw, resize_hw), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0).astype(np.uint8)


def preprocess_image_sequence(images: np.ndarray,
                              mode: str = "stabilize_crop",
                              crop_h: int = 384,
                              crop_w: int = 384,
                              resize_hw: int = 256,
                              smoothing_radius: int = 15,
                              border_mode: str = "reflect") -> np.ndarray:
    mode = str(mode).strip().lower()
    imgs = np.asarray(images, dtype=np.uint8)
    if mode in ("off", "none", "raw"):
        return imgs.copy()
    if mode not in ("stabilize", "stabilize_crop"):
        raise ValueError(f"Unsupported camera_preprocess_mode: {mode}")

    imgs = stabilize_image_sequence(
        imgs,
        smoothing_radius=int(smoothing_radius),
        border_mode=str(border_mode),
    )
    if mode == "stabilize_crop":
        imgs = center_crop_images(imgs, crop_h=int(crop_h), crop_w=int(crop_w))
        imgs = resize_images(imgs, resize_hw=int(resize_hw))
    return imgs.astype(np.uint8)


def format_jitter_metrics(m: dict) -> str:
    return (
        f"frames={m.get('frames', 0)}, shape={m.get('shape', ())}, "
        f"RMS={m.get('rms_px', 0.0):.4f}px, "
        f"P95={m.get('p95_px', 0.0):.4f}px, "
        f"max={m.get('max_px', 0.0):.4f}px, "
        f"mean_motion={m.get('mean_motion_px', 0.0):.4f}px"
    )

# ============================================================
# Filtering utilities
# ============================================================
def ema_1d(y: np.ndarray, alpha: float) -> np.ndarray:
    if y.size == 0:
        return y.astype(np.float64).copy()

    z = y.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return z

    for i in range(1, y.size):
        z[i] = alpha * y[i] + (1.0 - alpha) * z[i - 1]
    return z


def ema_nd(X: np.ndarray, alpha: float) -> np.ndarray:
    if X.size == 0:
        return X.astype(np.float64).copy()

    Z = X.astype(np.float64).copy()
    if alpha <= 0.0 or alpha >= 1.0:
        return Z

    for i in range(1, Z.shape[0]):
        Z[i] = alpha * X[i] + (1.0 - alpha) * Z[i - 1]
    return Z


def process_force_keep_fz_with_ema_and_edge_zero(
    Fraw: np.ndarray,
    fz_ema_alpha: float,
    edge_zero_sec: float,
    sample_hz: float,
    zero_xy: bool = True,
    logger=None,
) -> np.ndarray:
    """
    Latest filtering policy:
      - Fx, Fy -> 0
      - Fz -> raw Fz + EMA only
      - first edge_zero_sec and last edge_zero_sec -> all forces zero

    Fraw shape: (T, 3) = [Fx, Fy, Fz]
    """
    if Fraw.size == 0:
        return Fraw.astype(np.float64).copy()

    Fp = Fraw.astype(np.float64).copy()
    N = int(Fp.shape[0])

    if zero_xy:
        Fp[:, 0] = 0.0
        Fp[:, 1] = 0.0

    Fp[:, 2] = ema_1d(Fp[:, 2], alpha=float(fz_ema_alpha))

    edge_n = int(round(float(edge_zero_sec) * float(sample_hz)))
    edge_n = max(0, min(edge_n, N))

    if edge_n > 0:
        Fp[:edge_n, :] = 0.0
        Fp[max(0, N - edge_n):, :] = 0.0

    if logger is not None:
        raw_fz_abs_max = float(np.max(np.abs(Fraw[:, 2]))) if N > 0 else 0.0
        proc_fz_abs_max = float(np.max(np.abs(Fp[:, 2]))) if N > 0 else 0.0
        logger.info(
            f"[FORCE] zero_xy={zero_xy}, fz_ema_alpha={fz_ema_alpha}, "
            f"edge_zero_sec={edge_zero_sec}, edge_zero_samples={edge_n}, N={N}"
        )
        logger.info(
            f"[FORCE] raw |fz|max={raw_fz_abs_max:.3f} N, "
            f"processed |fz|max={proc_fz_abs_max:.3f} N"
        )

    return Fp


# ============================================================
# Keyboard backup quit
# ============================================================
class KeyboardQuitter:
    """Press quit_key without Enter to request graceful stop."""
    def __init__(self, quit_key: str = "q"):
        self.quit_key = (quit_key or "q").lower()
        self._stop_evt = threading.Event()
        self._hit_quit = threading.Event()
        self._thread = None
        self._enabled = False
        self._fd = None
        self._old_term = None

    def start(self) -> bool:
        if not sys.stdin.isatty():
            self._enabled = False
            return False
        self._enabled = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._restore_term()

    def hit(self) -> bool:
        return self._hit_quit.is_set()

    def _restore_term(self):
        try:
            if self._enabled and self._fd is not None and self._old_term is not None:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_term)
        except Exception:
            pass
        self._fd = None
        self._old_term = None

    def _loop(self):
        try:
            self._fd = sys.stdin.fileno()
            self._old_term = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)

            while not self._stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if not ch:
                    continue
                if ch.lower() == self.quit_key:
                    self._hit_quit.set()
                    break
        except Exception:
            pass
        finally:
            self._restore_term()


# ============================================================
# Main node
# ============================================================
class VRDemoHDF5Recorder(Node):
    def __init__(self):
        super().__init__("vr_demo_hdf5_recorder")

        # -------------------------
        # Save parameters
        # -------------------------
        self.declare_parameter("act_root_dir", "/home/eunseop/nrs_imitation/datasets/ACT")
        self.declare_parameter("merged_subdir", "merged_hdf5")
        self.declare_parameter("file_prefix", "vr_demo_merged")
        self.declare_parameter("overwrite_file", False)
        self.declare_parameter("allow_overwrite_episode", False)
        self.declare_parameter("flush_each_episode", True)

        self.declare_parameter("num_episodes", 50)
        self.declare_parameter("min_samples", 10)
        self.declare_parameter("quit_key", "q")

        # -------------------------
        # Topic parameters
        # -------------------------
        self.declare_parameter("recording_mode", "tracker")  # tracker | robot

        # Per-mode default topics
        self.declare_parameter("tracker_pose_topic", "/calibrated_pose")
        self.declare_parameter("tracker_force_topic", "/ftsensor/measured_Cvalue")
        self.declare_parameter("tracker_image_topic", "/realsense/vr/color/image_raw")

        self.declare_parameter("robot_pose_topic", "/ur10skku/currentP")
        self.declare_parameter("robot_force_topic", "/ur10skku/currentF")
        self.declare_parameter("robot_image_topic", "/realsense/robot/color/image_raw")

        # Optional manual override. If empty, mode defaults above are used.
        self.declare_parameter("pose_topic", "")
        self.declare_parameter("force_topic", "")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("command_topic", "/vr_demo_recorder/command")

        # -------------------------
        # Sampling / freshness
        # -------------------------
        self.declare_parameter("sample_hz", 30.0)
        self.declare_parameter("require_pose_fresh_sec", 0.20)
        self.declare_parameter("require_force_fresh_sec", 0.20)
        self.declare_parameter("require_image_fresh_sec", 0.50)
        self.declare_parameter("recording_status_period_sec", 1.0)
        self.declare_parameter("idle_status_period_sec", 0.0)  # 0 => print only on state change/command

        # -------------------------
        # Unit convention
        # -------------------------
        self.declare_parameter("pose_xyz_scale", 1000.0)  # m -> mm

        # -------------------------
        # Force processing
        # -------------------------
        self.declare_parameter("zero_xy_forces", True)
        self.declare_parameter("fz_ema_alpha", 0.2)
        self.declare_parameter("force_edge_zero_sec", 3.0)

        # -------------------------
        # Optional pose smoothing
        # Default OFF because this recorder is a synchronized multimodal ACT recorder.
        # -------------------------
        self.declare_parameter("pose_ema_enable", False)
        self.declare_parameter("pose_ema_alpha", 0.10)

        # -------------------------
        # Image save
        # -------------------------
        self.declare_parameter("image_dataset_name", "cam0")
        self.declare_parameter("image_compression", "gzip")  # gzip, lzf, none
        self.declare_parameter("image_gzip_level", 4)

        # -------------------------
        # Camera preprocessing / stabilization
        # Default ON: save stabilized/cropped RGB to reduce hand-shake noise.
        # Use camera_preprocess_mode:=off to save raw RGB.
        # -------------------------
        self.declare_parameter("camera_preprocess_mode", "stabilize_crop")  # off | stabilize | stabilize_crop
        self.declare_parameter("cam_crop_h", 384)
        self.declare_parameter("cam_crop_w", 384)
        self.declare_parameter("cam_resize_hw", 256)
        self.declare_parameter("cam_stab_smoothing_radius", 15)
        self.declare_parameter("cam_stab_border_mode", "reflect")
        self.declare_parameter("camera_jitter_report_enable", True)

        # -------------------------
        # Load parameters
        # -------------------------
        self.act_root_dir = str(self.get_parameter("act_root_dir").value)
        self.merged_subdir = str(self.get_parameter("merged_subdir").value)
        self.file_prefix = str(self.get_parameter("file_prefix").value)
        self.overwrite_file = bool(self.get_parameter("overwrite_file").value)
        self.allow_overwrite_episode = bool(self.get_parameter("allow_overwrite_episode").value)
        self.flush_each_episode = bool(self.get_parameter("flush_each_episode").value)

        self.num_episodes = int(self.get_parameter("num_episodes").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.quit_key = str(self.get_parameter("quit_key").value)

        self.recording_mode = str(self.get_parameter("recording_mode").value).strip().lower()

        self.tracker_pose_topic = str(self.get_parameter("tracker_pose_topic").value)
        self.tracker_force_topic = str(self.get_parameter("tracker_force_topic").value)
        self.tracker_image_topic = str(self.get_parameter("tracker_image_topic").value)

        self.robot_pose_topic = str(self.get_parameter("robot_pose_topic").value)
        self.robot_force_topic = str(self.get_parameter("robot_force_topic").value)
        self.robot_image_topic = str(self.get_parameter("robot_image_topic").value)

        pose_topic_override = str(self.get_parameter("pose_topic").value).strip()
        force_topic_override = str(self.get_parameter("force_topic").value).strip()
        image_topic_override = str(self.get_parameter("image_topic").value).strip()
        self.command_topic = str(self.get_parameter("command_topic").value)

        if self.recording_mode not in ("tracker", "robot"):
            raise RuntimeError(
                f"recording_mode must be 'tracker' or 'robot', got: {self.recording_mode}"
            )

        default_pose_topic = self.tracker_pose_topic if self.recording_mode == "tracker" else self.robot_pose_topic
        default_force_topic = self.tracker_force_topic if self.recording_mode == "tracker" else self.robot_force_topic
        default_image_topic = self.tracker_image_topic if self.recording_mode == "tracker" else self.robot_image_topic

        self.pose_topic = pose_topic_override if pose_topic_override else default_pose_topic
        self.force_topic = force_topic_override if force_topic_override else default_force_topic
        self.image_topic = image_topic_override if image_topic_override else default_image_topic

        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.dt = 1.0 / max(1e-9, self.sample_hz)

        self.require_pose_fresh_sec = float(self.get_parameter("require_pose_fresh_sec").value)
        self.require_force_fresh_sec = float(self.get_parameter("require_force_fresh_sec").value)
        self.require_image_fresh_sec = float(self.get_parameter("require_image_fresh_sec").value)
        self.recording_status_period_sec = float(self.get_parameter("recording_status_period_sec").value)
        self.idle_status_period_sec = float(self.get_parameter("idle_status_period_sec").value)

        self.pose_xyz_scale = float(self.get_parameter("pose_xyz_scale").value)

        self.zero_xy_forces = bool(self.get_parameter("zero_xy_forces").value)
        self.fz_ema_alpha = float(self.get_parameter("fz_ema_alpha").value)
        self.force_edge_zero_sec = float(self.get_parameter("force_edge_zero_sec").value)

        self.pose_ema_enable = bool(self.get_parameter("pose_ema_enable").value)
        self.pose_ema_alpha = float(self.get_parameter("pose_ema_alpha").value)

        self.image_dataset_name = str(self.get_parameter("image_dataset_name").value)
        self.image_compression = str(self.get_parameter("image_compression").value).lower()
        self.image_gzip_level = int(self.get_parameter("image_gzip_level").value)

        self.camera_preprocess_mode = str(self.get_parameter("camera_preprocess_mode").value).strip().lower()
        self.cam_crop_h = int(self.get_parameter("cam_crop_h").value)
        self.cam_crop_w = int(self.get_parameter("cam_crop_w").value)
        self.cam_resize_hw = int(self.get_parameter("cam_resize_hw").value)
        self.cam_stab_smoothing_radius = int(self.get_parameter("cam_stab_smoothing_radius").value)
        self.cam_stab_border_mode = str(self.get_parameter("cam_stab_border_mode").value).strip().lower()
        self.camera_jitter_report_enable = bool(self.get_parameter("camera_jitter_report_enable").value)
        if self.camera_preprocess_mode not in ("off", "none", "raw", "stabilize", "stabilize_crop"):
            raise RuntimeError(
                f"camera_preprocess_mode must be off/stabilize/stabilize_crop, got: {self.camera_preprocess_mode}"
            )

        # -------------------------
        # HDF5 lazy-open state
        # -------------------------
        self.h5_lock = threading.Lock()
        self.h5: Optional[h5py.File] = None
        self.grp_eps: Optional[h5py.Group] = None
        self.hdf5_path: Optional[str] = None
        self.run_stamp: Optional[str] = None

        # Selected episode index is controlled by D-pad.
        # If selected index already exists, start is blocked until erased or moved.
        self.selected_ep_idx = 0
        self.recording_ep_idx: Optional[int] = None
        self.last_saved_ep_idx: Optional[int] = None

        # -------------------------
        # Runtime state
        # -------------------------
        self.state_lock = threading.Lock()

        self.latest_pose6: Optional[np.ndarray] = None
        self.latest_force3: Optional[np.ndarray] = None
        self.latest_image: Optional[np.ndarray] = None

        self.latest_pose_t: float = 0.0
        self.latest_force_t: float = 0.0
        self.latest_image_t: float = 0.0

        self.episode_active = False
        self.finishing = False
        self.stop_requested = False
        self.stop_reason = ""
        self.recording_start_time: Optional[float] = None

        self.buf_position: List[np.ndarray] = []
        self.buf_ft_raw: List[np.ndarray] = []
        self.buf_images: List[Optional[np.ndarray]] = []

        self._last_recording_status_time = 0.0
        self._last_idle_status_time = 0.0

        # -------------------------
        # ROS IO
        # -------------------------
        self.create_subscription(Float64MultiArray, self.pose_topic, self.cb_pose, 100)

        if self.recording_mode == "tracker":
            self.create_subscription(Wrench, self.force_topic, self.cb_force_wrench, 200)
            self.force_msg_type = "geometry_msgs/Wrench"
        else:
            self.create_subscription(Float64MultiArray, self.force_topic, self.cb_force_array, 200)
            self.force_msg_type = "std_msgs/Float64MultiArray"

        self.create_subscription(Image, self.image_topic, self.cb_image, 10)
        self.create_subscription(String, self.command_topic, self.cb_command, 20)

        self.create_timer(self.dt, self.cb_sample_timer)
        self.create_timer(0.10, self.cb_status_timer)
        self.create_timer(0.05, self.cb_stop_timer)

        # -------------------------
        # Keyboard backup quit
        # -------------------------
        self.kb = KeyboardQuitter(quit_key=self.quit_key)
        kb_enabled = self.kb.start()
        atexit.register(self.kb.stop)

        # -------------------------
        # Logs
        # -------------------------
        self.get_logger().info("============================================================")
        self.get_logger().info("VRDemoHDF5Recorder initialized (joystick-controlled single-camera ACT recorder)")
        self.get_logger().info(f"  recording_mode: {self.recording_mode}")
        self.get_logger().info("HDF5 file is created lazily when the first episode is saved.")
        self.get_logger().info(f"  ACT root      : {self.act_root_dir}")
        self.get_logger().info(f"  merged dir    : <ACT root>/<YYYYMMDD_HHMM>/{self.merged_subdir}")
        self.get_logger().info(f"  filename      : {self.file_prefix}_YYYYMMDD_HHMM.hdf5")
        self.get_logger().info(f"  pose_topic    : {self.pose_topic}")
        self.get_logger().info(f"  force_topic   : {self.force_topic} [{self.force_msg_type}]")
        self.get_logger().info(f"  image_topic   : {self.image_topic}")
        self.get_logger().info(f"  command_topic : {self.command_topic}")
        self.get_logger().info(f"  image key     : images/{self.image_dataset_name}")
        self.get_logger().info(f"  sample_hz     : {self.sample_hz:.3f} Hz, dt={self.dt:.6f}s")
        self.get_logger().info(
            f"  freshness     : pose={self.require_pose_fresh_sec:.3f}s, "
            f"force={self.require_force_fresh_sec:.3f}s, image={self.require_image_fresh_sec:.3f}s"
        )
        self.get_logger().info("  trigger       : joystick command only; old Fx/Fy start/end trigger removed")
        self.get_logger().info("  joystick cmds : A=start, B=end/save, X=erase selected/current, Y=terminate, D-pad=select ep")
        self.get_logger().info("  stop policy   : terminate_node / backup quit key / Ctrl+C only (no auto-stop at num_episodes)")
        self.get_logger().info(
            f"  force proc    : zero_xy={self.zero_xy_forces}, "
            f"fz_ema_alpha={self.fz_ema_alpha}, edge_zero_sec={self.force_edge_zero_sec}"
        )
        self.get_logger().info(
            f"  pose          : xyz_scale={self.pose_xyz_scale}, "
            f"pose_ema={self.pose_ema_enable}(alpha={self.pose_ema_alpha})"
        )
        self.get_logger().info(
            f"  image save    : compression={self.image_compression}, gzip_level={self.image_gzip_level}"
        )
        self.get_logger().info(
            f"  cam preprocess: mode={self.camera_preprocess_mode}, "
            f"crop=({self.cam_crop_h},{self.cam_crop_w}), resize_hw={self.cam_resize_hw}, "
            f"stab_radius={self.cam_stab_smoothing_radius}, border={self.cam_stab_border_mode}, "
            f"jitter_report={self.camera_jitter_report_enable}"
        )
        self.get_logger().info(
            f"  target eps    : {self.num_episodes} (informational only; no auto-stop)"
        )
        if kb_enabled:
            self.get_logger().info(f"  Backup: press '{self.quit_key}' to stop gracefully. Ctrl+C also works.")
        else:
            self.get_logger().warn("  stdin is not a TTY -> backup key quit disabled. Use joystick Y or Ctrl+C.")
        self.get_logger().info("============================================================")
        self._print_status("READY")

    # ============================================================
    # HDF5 helpers
    # ============================================================
    @staticmethod
    def _stamp_now() -> str:
        return time.strftime("%Y%m%d_%H%M", time.localtime())

    def _build_hdf5_path(self, stamp: str) -> str:
        run_dir = os.path.join(self.act_root_dir, stamp, self.merged_subdir)
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"{self.file_prefix}_{stamp}.hdf5")

    @staticmethod
    def _ep_name(idx: int) -> str:
        return f"ep_{int(idx):04d}"

    @staticmethod
    def _parse_ep_idx(name: str) -> Optional[int]:
        if not name.startswith("ep_"):
            return None
        try:
            return int(name.split("_")[1])
        except Exception:
            return None

    def _existing_episode_indices_locked(self) -> Set[int]:
        if self.grp_eps is None:
            return set()
        out = set()
        for k in self.grp_eps.keys():
            idx = self._parse_ep_idx(k)
            if idx is not None:
                out.add(idx)
        return out

    def _existing_episode_count_locked(self) -> int:
        return len(self._existing_episode_indices_locked())

    def _max_existing_episode_idx_locked(self) -> int:
        eps = self._existing_episode_indices_locked()
        return max(eps) if eps else -1

    def _next_empty_index_locked(self, start: int = 0) -> int:
        eps = self._existing_episode_indices_locked()
        idx = max(0, int(start))
        while idx in eps:
            idx += 1
        return idx

    def _selected_exists_locked(self) -> bool:
        if self.grp_eps is None:
            return False
        return self._ep_name(self.selected_ep_idx) in self.grp_eps

    def _select_next_empty_after_save_locked(self, saved_idx: int):
        self.selected_ep_idx = self._next_empty_index_locked(start=int(saved_idx) + 1)

    def _ensure_hdf5_open(self):
        """Open HDF5 at first save/delete operation time."""
        if self.h5 is not None:
            return

        self.run_stamp = self._stamp_now()
        self.hdf5_path = self._build_hdf5_path(self.run_stamp)

        if self.overwrite_file and os.path.exists(self.hdf5_path):
            os.remove(self.hdf5_path)

        self.h5 = h5py.File(self.hdf5_path, "a")
        self.grp_eps = self.h5.require_group("episodes")
        self._write_root_meta()
        self.h5.flush()

        # If the file existed already, select the next empty episode.
        self.selected_ep_idx = self._next_empty_index_locked(start=self.selected_ep_idx)

        self.get_logger().info("============================================================")
        self.get_logger().info("[HDF5] opened")
        self.get_logger().info(f"  path         : {self.hdf5_path}")
        self.get_logger().info(f"  saved eps    : {self._existing_episode_count_locked()}")
        self.get_logger().info(f"  selected ep  : {self._ep_name(self.selected_ep_idx)}")
        self.get_logger().info("============================================================")

    def _write_root_meta(self):
        if self.h5 is None:
            return

        if "created_unix" not in self.h5.attrs:
            self.h5.attrs["created_unix"] = float(time.time())

        self.h5.attrs["format"] = np.string_("act_merged_hdf5_single_camera")
        self.h5.attrs["format_version"] = np.string_("1.1_joystick_control")
        self.h5.attrs["camera_names_json"] = np.string_(json.dumps([self.image_dataset_name]))
        self.h5.attrs["position_columns"] = np.string_("x_mm,y_mm,z_mm,wx,wy,wz")
        self.h5.attrs["ft_columns"] = np.string_("fx_N,fy_N,fz_N")
        self.h5.attrs["pose_note"] = np.string_(
            "pose xyz input meters -> stored millimeters by pose_xyz_scale; wx wy wz stored as rotation-vector radians"
        )
        self.h5.attrs["image_note"] = np.string_("RGB uint8, shape=(T,H,W,3)")
        self.h5.attrs["sample_hz"] = float(self.sample_hz)
        self.h5.attrs["dt"] = float(self.dt)
        self.h5.attrs["pose_topic"] = np.string_(self.pose_topic)
        self.h5.attrs["recording_mode"] = np.string_(self.recording_mode)
        self.h5.attrs["force_topic"] = np.string_(self.force_topic)
        self.h5.attrs["force_msg_type"] = np.string_(self.force_msg_type)
        self.h5.attrs["image_topic"] = np.string_(self.image_topic)
        self.h5.attrs["command_topic"] = np.string_(self.command_topic)
        self.h5.attrs["episode_control"] = np.string_(
            "joystick commands only: start_recording/end_recording/erase_current_episode/terminate_node/prev_episode/next_episode"
        )
        self.h5.attrs["force_processing"] = np.string_(
            f"zero_xy={self.zero_xy_forces}, fz_ema_alpha={self.fz_ema_alpha}, edge_zero_sec={self.force_edge_zero_sec}"
        )

    def _image_create_kwargs(self):
        if self.image_compression == "none":
            return {}
        if self.image_compression == "lzf":
            return {"compression": "lzf", "shuffle": True}
        if self.image_compression == "gzip":
            return {
                "compression": "gzip",
                "compression_opts": int(self.image_gzip_level),
                "shuffle": True,
            }

        self.get_logger().warn(f"[HDF5] unknown image_compression={self.image_compression}, using gzip")
        return {
            "compression": "gzip",
            "compression_opts": int(self.image_gzip_level),
            "shuffle": True,
        }

    def _save_episode_to_hdf5(
        self,
        ep_idx: int,
        position: np.ndarray,
        ft: np.ndarray,
        images: np.ndarray,
        reason: str,
    ):
        with self.h5_lock:
            self._ensure_hdf5_open()
            assert self.h5 is not None
            assert self.grp_eps is not None

            ep_name = self._ep_name(ep_idx)
            if ep_name in self.grp_eps:
                if not self.allow_overwrite_episode:
                    raise RuntimeError(
                        f"{ep_name} already exists. Select another index or erase it first."
                    )
                del self.grp_eps[ep_name]

            g = self.grp_eps.create_group(ep_name)
            g.attrs["saved_unix"] = float(time.time())
            g.attrs["reason"] = np.string_(str(reason))
            g.attrs["out_len"] = int(position.shape[0])
            g.attrs["sample_hz"] = float(self.sample_hz)
            g.attrs["dt"] = float(self.dt)
            g.attrs["pose_xyz_scale"] = float(self.pose_xyz_scale)
            g.attrs["zero_xy_forces"] = int(bool(self.zero_xy_forces))
            g.attrs["fz_ema_alpha"] = float(self.fz_ema_alpha)
            g.attrs["force_edge_zero_sec"] = float(self.force_edge_zero_sec)
            g.attrs["pose_ema_enable"] = int(bool(self.pose_ema_enable))
            g.attrs["pose_ema_alpha"] = float(self.pose_ema_alpha)
            g.attrs["image_dataset_name"] = np.string_(self.image_dataset_name)
            g.attrs["image_shape"] = np.array(images.shape[1:], dtype=np.int64)
            g.attrs["camera_preprocess_mode"] = np.string_(str(self.camera_preprocess_mode))
            g.attrs["cam_crop_h"] = int(self.cam_crop_h)
            g.attrs["cam_crop_w"] = int(self.cam_crop_w)
            g.attrs["cam_resize_hw"] = int(self.cam_resize_hw)
            g.attrs["cam_stab_smoothing_radius"] = int(self.cam_stab_smoothing_radius)
            g.attrs["cam_stab_border_mode"] = np.string_(str(self.cam_stab_border_mode))

            g.create_dataset("position", data=position.astype(np.float32), dtype="float32")
            g.create_dataset("ft", data=ft.astype(np.float32), dtype="float32")

            img_grp = g.create_group("images")
            img_grp.create_dataset(
                self.image_dataset_name,
                data=images.astype(np.uint8),
                dtype="uint8",
                chunks=(1, images.shape[1], images.shape[2], images.shape[3]),
                **self._image_create_kwargs(),
            )

            if self.flush_each_episode:
                self.h5.flush()

            self.last_saved_ep_idx = int(ep_idx)
            self._select_next_empty_after_save_locked(int(ep_idx))

    def _delete_episode_locked(self, ep_idx: int) -> bool:
        self._ensure_hdf5_open()
        assert self.h5 is not None
        assert self.grp_eps is not None

        ep_name = self._ep_name(ep_idx)
        if ep_name not in self.grp_eps:
            return False

        del self.grp_eps[ep_name]
        self.h5.flush()
        if self.last_saved_ep_idx == ep_idx:
            self.last_saved_ep_idx = None
        return True

    # ============================================================
    # Status / stop
    # ============================================================
    def _latest_flags_and_ages(self):
        now = time.time()
        has_pose = self.latest_pose6 is not None
        has_force = self.latest_force3 is not None
        has_image = self.latest_image is not None
        pose_age = now - self.latest_pose_t if has_pose else float("inf")
        force_age = now - self.latest_force_t if has_force else float("inf")
        image_age = now - self.latest_image_t if has_image else float("inf")
        pose_ok = has_pose and pose_age <= self.require_pose_fresh_sec
        force_ok = has_force and force_age <= self.require_force_fresh_sec
        image_ok = has_image and image_age <= self.require_image_fresh_sec
        return pose_ok, force_ok, image_ok, pose_age, force_age, image_age

    def _print_status(self, label: str = "STATUS"):
        with self.state_lock:
            steps = len(self.buf_position)
            active = bool(self.episode_active)
            finishing = bool(self.finishing)
            rec_idx = self.recording_ep_idx
            selected = self.selected_ep_idx
            pose_ok, force_ok, image_ok, pose_age, force_age, image_age = self._latest_flags_and_ages()
            elapsed = 0.0
            if self.recording_start_time is not None:
                elapsed = max(0.0, time.time() - self.recording_start_time)

        with self.h5_lock:
            saved_count = self._existing_episode_count_locked() if self.grp_eps is not None else 0
            max_existing = self._max_existing_episode_idx_locked() if self.grp_eps is not None else -1
            selected_exists = self._selected_exists_locked() if self.grp_eps is not None else False
            h5_path = self.hdf5_path or "not_opened"

        if self.stop_requested:
            mode = "STOP_REQUESTED"
        elif finishing:
            mode = "SAVING"
        elif active:
            mode = "RECORDING"
        else:
            mode = "IDLE"

        rec_name = "none" if rec_idx is None else self._ep_name(rec_idx)
        sel_name = self._ep_name(selected)
        self.get_logger().info(
            f"[{label}] mode={mode} | selected={sel_name}"
            f"{'(exists)' if selected_exists else '(empty)'} | recording={rec_name} | "
            f"steps={steps} | elapsed={elapsed:.1f}s | saved_count={saved_count} | max_ep={max_existing:04d} | "
            f"fresh(pose={pose_ok}, force={force_ok}, image={image_ok}) | "
            f"age(pose={pose_age:.2f}s, force={force_age:.2f}s, image={image_age:.2f}s) | "
            f"h5={h5_path}"
        )

    def request_stop(self, reason: str = "user_request"):
        if self.stop_requested:
            return
        self.stop_requested = True
        self.stop_reason = str(reason)
        self.get_logger().warn(f"[STOP REQUEST] reason={self.stop_reason}")
        self._print_status("STOP")

        if self.episode_active and not self.finishing:
            self.get_logger().warn("Stop requested while recording -> saving current episode before shutdown.")
            self._start_finish_thread(reason=self.stop_reason or "stop_requested")

    def cb_stop_timer(self):
        if self.kb.hit() and not self.stop_requested:
            self.request_stop(reason=f"keyboard_{self.quit_key}")

        if self.stop_requested and (not self.finishing) and (not self.episode_active):
            self.finalize_and_shutdown()

    def finalize_and_shutdown(self):
        self.get_logger().warn("Finalizing HDF5 and shutting down...")
        try:
            with self.h5_lock:
                if self.h5 is not None:
                    try:
                        self.h5.flush()
                    except Exception:
                        pass
                    try:
                        self.h5.close()
                    except Exception:
                        pass
                    self.h5 = None
                    self.grp_eps = None
        finally:
            try:
                self.kb.stop()
            except Exception:
                pass
            try:
                self.destroy_node()
            except Exception:
                pass
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    # ============================================================
    # ROS callbacks
    # ============================================================
    def cb_pose(self, msg: Float64MultiArray):
        if len(msg.data) < 6:
            return

        x, y, z, wx, wy, wz = msg.data[:6]
        pose = np.array(
            [
                self.pose_xyz_scale * float(x),
                self.pose_xyz_scale * float(y),
                self.pose_xyz_scale * float(z),
                float(wx),
                float(wy),
                float(wz),
            ],
            dtype=np.float64,
        )

        now = time.time()
        with self.state_lock:
            self.latest_pose6 = pose
            self.latest_pose_t = now

    def cb_force_wrench(self, msg: Wrench):
        F = np.array(
            [float(msg.force.x), float(msg.force.y), float(msg.force.z)],
            dtype=np.float64,
        )
        now = time.time()
        with self.state_lock:
            self.latest_force3 = F
            self.latest_force_t = now

    def cb_force_array(self, msg: Float64MultiArray):
        if len(msg.data) < 3:
            return
        F = np.array(
            [float(msg.data[0]), float(msg.data[1]), float(msg.data[2])],
            dtype=np.float64,
        )
        now = time.time()
        with self.state_lock:
            self.latest_force3 = F
            self.latest_force_t = now

    def cb_image(self, msg: Image):
        img = image_to_rgb_numpy(msg)
        if img is None:
            self.get_logger().warn(
                f"[IMAGE] unsupported or invalid image encoding='{msg.encoding}' "
                f"shape=({msg.height},{msg.width}), step={msg.step}",
                throttle_duration_sec=2.0,
            )
            return

        now = time.time()
        with self.state_lock:
            self.latest_image = img
            self.latest_image_t = now

    def cb_command(self, msg: String):
        cmd = (msg.data or "").strip().lower()
        if cmd == "":
            return

        self.get_logger().info(f"[CMD] received: {cmd}")

        if cmd == "start_recording":
            self._start_episode_from_command()
        elif cmd == "end_recording":
            self._end_episode_from_command()
        elif cmd == "erase_current_episode":
            self._erase_current_index_from_command()
        elif cmd == "terminate_node":
            self.request_stop(reason="joystick_terminate_node")
        elif cmd == "prev_episode":
            self._move_selected_episode(delta=-1)
        elif cmd == "next_episode":
            self._move_selected_episode(delta=+1)
        elif cmd in ("status", "print_status"):
            self._print_status("STATUS")
        else:
            self.get_logger().warn(f"[CMD] unknown command: {cmd}")
            self._print_status("UNKNOWN_CMD")

    def cb_sample_timer(self):
        if (not self.episode_active) or self.finishing or self.stop_requested:
            return

        now = time.time()

        with self.state_lock:
            if self.latest_pose6 is None or self.latest_force3 is None or self.latest_image is None:
                return

            pose_age = now - self.latest_pose_t
            force_age = now - self.latest_force_t
            image_age = now - self.latest_image_t

            if pose_age > self.require_pose_fresh_sec:
                self.get_logger().warn(f"[SKIP] pose stale: age={pose_age:.3f}s", throttle_duration_sec=2.0)
                return

            if force_age > self.require_force_fresh_sec:
                self.get_logger().warn(f"[SKIP] force stale: age={force_age:.3f}s", throttle_duration_sec=2.0)
                return

            if image_age > self.require_image_fresh_sec:
                self.get_logger().warn(f"[SKIP] image stale: age={image_age:.3f}s", throttle_duration_sec=2.0)
                return

            self.buf_position.append(self.latest_pose6.copy())
            self.buf_ft_raw.append(self.latest_force3.copy())
            self.buf_images.append(self.latest_image.copy())

    def cb_status_timer(self):
        now = time.time()
        with self.state_lock:
            active = self.episode_active
            finishing = self.finishing

        if active or finishing:
            if self.recording_status_period_sec > 0.0 and (
                now - self._last_recording_status_time >= self.recording_status_period_sec
            ):
                self._last_recording_status_time = now
                self._print_status("REC_STATUS")
        else:
            if self.idle_status_period_sec > 0.0 and (
                now - self._last_idle_status_time >= self.idle_status_period_sec
            ):
                self._last_idle_status_time = now
                self._print_status("IDLE_STATUS")

    # ============================================================
    # Command handlers
    # ============================================================
    def _start_episode_from_command(self):
        if self.stop_requested:
            self.get_logger().warn("[START] ignored: stop already requested")
            return
        if self.finishing:
            self.get_logger().warn("[START] ignored: previous episode is still saving")
            self._print_status("START_BLOCKED")
            return
        if self.episode_active:
            self.get_logger().warn("[START] ignored: already recording")
            self._print_status("START_BLOCKED")
            return

        # Require valid latest data before starting so the user gets immediate feedback.
        with self.state_lock:
            pose_ok, force_ok, image_ok, pose_age, force_age, image_age = self._latest_flags_and_ages()

        if not (pose_ok and force_ok and image_ok):
            self.get_logger().warn(
                f"[START] blocked: fresh data missing | "
                f"pose_ok={pose_ok}(age={pose_age:.2f}s), "
                f"force_ok={force_ok}(age={force_age:.2f}s), "
                f"image_ok={image_ok}(age={image_age:.2f}s)"
            )
            self._print_status("START_BLOCKED")
            return

        with self.h5_lock:
            if self.grp_eps is not None and self._selected_exists_locked() and not self.allow_overwrite_episode:
                self.get_logger().warn(
                    f"[START] blocked: selected {self._ep_name(self.selected_ep_idx)} already exists. "
                    "Move to an empty index or erase it first."
                )
                self._print_status("START_BLOCKED")
                return

        self._start_episode()

    def _end_episode_from_command(self):
        if self.finishing:
            self.get_logger().warn("[END] ignored: already saving")
            self._print_status("END_BLOCKED")
            return
        if not self.episode_active:
            self.get_logger().warn("[END] ignored: not recording")
            self._print_status("END_BLOCKED")
            return

        self.get_logger().info(f"=== EPISODE END REQUESTED ({self._ep_name(self.recording_ep_idx)}) by joystick B ===")
        self._start_finish_thread(reason="joystick_end_recording")

    def _erase_current_index_from_command(self):
        if self.finishing:
            self.get_logger().warn("[ERASE] ignored: episode is saving")
            self._print_status("ERASE_BLOCKED")
            return

        if self.episode_active:
            # While recording, X discards the current unsaved buffer.
            ep_name = self._ep_name(self.recording_ep_idx if self.recording_ep_idx is not None else self.selected_ep_idx)
            with self.state_lock:
                n = len(self.buf_position)
                self.buf_position.clear()
                self.buf_ft_raw.clear()
                self.buf_images.clear()
                self.episode_active = False
                self.recording_ep_idx = None
                self.recording_start_time = None
            self.get_logger().warn(f"[DISCARD] current recording {ep_name} discarded, dropped_steps={n}")
            self._print_status("DISCARDED")
            return

        # Idle: X deletes the selected saved episode from HDF5.
        with self.h5_lock:
            if self.h5 is None:
                self.get_logger().warn("[ERASE] ignored: HDF5 has not been opened yet; no saved episode exists")
                self._print_status("ERASE_BLOCKED")
                return

            ep_idx = int(self.selected_ep_idx)
            ep_name = self._ep_name(ep_idx)
            ok = self._delete_episode_locked(ep_idx)

        if ok:
            self.get_logger().warn(f"[ERASE] deleted {ep_name}")
            self._print_status("ERASED")
        else:
            self.get_logger().warn(f"[ERASE] ignored: {ep_name} does not exist")
            self._print_status("ERASE_BLOCKED")

    def _move_selected_episode(self, delta: int):
        if self.episode_active or self.finishing:
            self.get_logger().warn("[SELECT] ignored while recording/saving")
            self._print_status("SELECT_BLOCKED")
            return

        old_idx = int(self.selected_ep_idx)

        with self.h5_lock:
            if self.grp_eps is None:
                max_selectable = 0
            else:
                max_existing = self._max_existing_episode_idx_locked()
                # Allow selecting one empty slot after the last existing episode.
                max_selectable = max(0, max_existing + 1)

            new_idx = int(np.clip(old_idx + int(delta), 0, max_selectable))
            self.selected_ep_idx = new_idx

        if new_idx == old_idx:
            self.get_logger().info(f"[SELECT] stay at {self._ep_name(self.selected_ep_idx)}")
        else:
            self.get_logger().info(f"[SELECT] {self._ep_name(old_idx)} -> {self._ep_name(new_idx)}")
        self._print_status("SELECT")

    # ============================================================
    # Episode control
    # ============================================================
    def _start_episode(self):
        with self.state_lock:
            self.recording_ep_idx = int(self.selected_ep_idx)
            self.episode_active = True
            self.recording_start_time = time.time()
            self.buf_position.clear()
            self.buf_ft_raw.clear()
            self.buf_images.clear()

        self.get_logger().info(f"=== EPISODE STARTED ({self._ep_name(self.recording_ep_idx)}) by joystick A ===")
        self._print_status("STARTED")

    def _start_finish_thread(self, reason: str):
        if self.finishing:
            return

        self.finishing = True

        with self.state_lock:
            ep_idx = int(self.recording_ep_idx if self.recording_ep_idx is not None else self.selected_ep_idx)
            self.episode_active = False
            self.recording_start_time = None

            P_list = self.buf_position.copy()
            F_list = self.buf_ft_raw.copy()
            I_list = self.buf_images.copy()

            self.buf_position.clear()
            self.buf_ft_raw.clear()
            self.buf_images.clear()
            self.recording_ep_idx = None

        self._print_status("SAVING")

        th = threading.Thread(
            target=self._finish_episode_worker,
            args=(ep_idx, P_list, F_list, I_list, reason),
            daemon=True,
        )
        th.start()

    def _finish_episode_worker(
        self,
        ep_idx: int,
        P_list: List[np.ndarray],
        F_list: List[np.ndarray],
        I_list: List[Optional[np.ndarray]],
        reason: str,
    ):
        try:
            Np = len(P_list)
            Nf = len(F_list)
            Ni = len(I_list)
            N = min(Np, Nf, Ni)

            if N < max(1, self.min_samples):
                self.get_logger().warn(
                    f"Episode dropped: too short. "
                    f"N={N}, min_samples={self.min_samples}, "
                    f"Np={Np}, Nf={Nf}, Ni={Ni}, reason={reason}"
                )
                return

            P = np.asarray(P_list[:N], dtype=np.float64)
            Fraw = np.asarray(F_list[:N], dtype=np.float64)
            images = stack_images_repeat_last(I_list[:N], logger=self.get_logger())

            if images is None:
                self.get_logger().warn(f"Episode dropped: no valid image frames. N={N}, reason={reason}")
                return

            raw_jitter = compute_jitter_metrics(
                images,
                smoothing_radius=max(1, int(self.cam_stab_smoothing_radius)),
            )
            images_proc = preprocess_image_sequence(
                images,
                mode=self.camera_preprocess_mode,
                crop_h=self.cam_crop_h,
                crop_w=self.cam_crop_w,
                resize_hw=self.cam_resize_hw,
                smoothing_radius=max(1, int(self.cam_stab_smoothing_radius)),
                border_mode=self.cam_stab_border_mode,
            )
            proc_jitter = compute_jitter_metrics(
                images_proc,
                smoothing_radius=max(1, int(self.cam_stab_smoothing_radius)),
            )
            if self.camera_jitter_report_enable:
                before = float(raw_jitter.get("rms_px", 0.0))
                after = float(proc_jitter.get("rms_px", 0.0))
                reduction = 0.0 if before <= 1e-9 else 100.0 * (before - after) / before
                self.get_logger().info(
                    f"[CAM-JITTER] before: {format_jitter_metrics(raw_jitter)}"
                )
                self.get_logger().info(
                    f"[CAM-JITTER] after : {format_jitter_metrics(proc_jitter)} | "
                    f"RMS_reduction={reduction:.2f}% | mode={self.camera_preprocess_mode}"
                )
            images = images_proc

            if self.pose_ema_enable:
                P_out = ema_nd(P, alpha=self.pose_ema_alpha)
            else:
                P_out = P.copy()

            F_out = process_force_keep_fz_with_ema_and_edge_zero(
                Fraw,
                fz_ema_alpha=self.fz_ema_alpha,
                edge_zero_sec=self.force_edge_zero_sec,
                sample_hz=self.sample_hz,
                zero_xy=self.zero_xy_forces,
                logger=self.get_logger(),
            )

            self._save_episode_to_hdf5(
                ep_idx=ep_idx,
                position=P_out,
                ft=F_out,
                images=images,
                reason=reason,
            )

            self.get_logger().info(
                f"=== EPISODE SAVED ({self._ep_name(ep_idx)}) "
                f"N={N}, position={P_out.shape}, ft={F_out.shape}, images={images.shape}, "
                f"reason={reason} ==="
            )

            self._print_status("SAVED")

            # No auto-stop on num_episodes anymore.
            # The recorder keeps running until an explicit termination request arrives
            # (joystick terminate_node / backup keyboard quit / Ctrl+C).

        except Exception as e:
            self.get_logger().error(f"Episode processing failed: {repr(e)}")
        finally:
            self.finishing = False
            if self.stop_requested and not self.episode_active:
                self._print_status("SAVE_DONE_STOP_PENDING")


def main(args=None):
    rclpy.init(args=args)
    node = VRDemoHDF5Recorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        try:
            node.request_stop(reason="KeyboardInterrupt")
        except Exception:
            pass
        time.sleep(0.1)
        try:
            if rclpy.ok():
                node.finalize_and_shutdown()
        except Exception:
            pass
    finally:
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()