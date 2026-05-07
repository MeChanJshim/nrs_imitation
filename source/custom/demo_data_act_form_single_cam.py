#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_data_act_form_single_cam.py

Single-camera ACT dataset converter for the current VR demo pipeline.

Input merged HDF5 format:
  <ACT_ROOT>/<RUN_ID>/merged_hdf5/vr_demo_merged_<RUN_ID>.hdf5
    /episodes/ep_xxxx/position        (T, 6)
    /episodes/ep_xxxx/ft              (T, 3)
    /episodes/ep_xxxx/images/cam0     (T, H, W, 3), uint8 RGB

Output ACT episode format:
  <ACT_ROOT>/<RUN_ID>/<episodes_subdir>/episode_0.hdf5
    /observations/position            (T_pad, 6)
    /observations/force               (T_pad, 3)
    /observations/images/cam0         (T_pad, H, W, 3), uint8 RGB
    /observations/is_pad              (T_pad,)
    /action/position                  (T_pad, 6), next-step hold
    /action/force                     (T_pad, 3), next-step hold
    /meta/orig_len
    /meta/T_pad
    /meta/pad_starts_at
    /meta/truncated
    /meta/camera_name
    /meta/cam_preprocess_mode

After successful conversion, the input merged HDF5 file is deleted by default
to save disk space. Use --keep-merged to preserve it.

Recommended camera preprocessing mode:
  --cam_preprocess stabilize_crop
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

try:
    import cv2
except Exception:
    cv2 = None


ROOT_DEFAULT = "/home/eunseop/nrs_imitation/datasets/ACT"
MERGED_SUBDIR = "merged_hdf5"
EPISODES_SUBDIR_RAW = "episodes_ft"
EPISODES_SUBDIR_CAMPROC = "episodes_ft_camproc"


# ============================================================
# Path utilities
# ============================================================
def _is_probably_timestamp_name(name: str) -> bool:
    stem = Path(name).stem
    candidates = [name, stem]
    if stem.startswith("vr_demo_merged_"):
        candidates.append(stem.replace("vr_demo_merged_", "", 1))

    for s in candidates:
        for fmt in ("%Y%m%d%H%M", "%Y%m%d_%H%M", "%m%d_%H%M"):
            try:
                datetime.strptime(s, fmt)
                return True
            except ValueError:
                pass
    return False


def _find_hdf5_files_under(directory: str, recursive: bool = False) -> List[str]:
    d = Path(os.path.expanduser(directory))
    if not d.exists() or not d.is_dir():
        return []

    files: List[str] = []
    for pat in ("*.hdf5", "*.h5"):
        iterator = d.rglob(pat) if recursive else d.glob(pat)
        for p in iterator:
            if p.is_file():
                files.append(str(p))

    return sorted(set(files))


def _pick_latest_file(files: List[str]) -> str:
    if not files:
        raise FileNotFoundError("No HDF5 file candidates found.")

    def score(path: str):
        p = Path(path)
        st = p.stat()
        timestamp_bonus = 1 if _is_probably_timestamp_name(p.name) else 0
        return (timestamp_bonus, p.stem, st.st_mtime)

    return sorted(files, key=score, reverse=True)[0]


def resolve_input_path(user_input: Optional[str], root_dir: str) -> str:
    root_dir = os.path.expanduser(root_dir)

    if user_input is not None and str(user_input).strip() != "":
        p = Path(os.path.expanduser(user_input))

        if p.is_file():
            return str(p)

        if p.is_dir():
            direct = _find_hdf5_files_under(str(p), recursive=False)
            if direct:
                return _pick_latest_file(direct)

            rec = _find_hdf5_files_under(str(p), recursive=True)
            if rec:
                merged = [x for x in rec if MERGED_SUBDIR in Path(x).parts]
                return _pick_latest_file(merged if merged else rec)

            raise FileNotFoundError(f"No .hdf5/.h5 file found under input directory: {p}")

        raise FileNotFoundError(f"Input path not found: {p}")

    root = Path(root_dir)
    candidates: List[str] = []

    if root.exists():
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            merged_dir = run_dir / MERGED_SUBDIR
            if merged_dir.is_dir():
                candidates.extend(_find_hdf5_files_under(str(merged_dir), recursive=False))

        legacy_merged = root / MERGED_SUBDIR
        if legacy_merged.is_dir():
            candidates.extend(_find_hdf5_files_under(str(legacy_merged), recursive=False))

    if not candidates:
        raise FileNotFoundError(
            f"No merged HDF5 found. Expected under:\n"
            f"  {root_dir}/<RUN_ID>/{MERGED_SUBDIR}/*.hdf5\n"
            f"or legacy:\n"
            f"  {root_dir}/{MERGED_SUBDIR}/*.hdf5"
        )

    return _pick_latest_file(candidates)


def infer_run_dir_from_input(input_path: str, root_dir: str, run_dir_arg: Optional[str]) -> str:
    root_dir = os.path.expanduser(root_dir)

    if run_dir_arg is not None and str(run_dir_arg).strip() != "":
        run_dir = os.path.join(root_dir, str(run_dir_arg).strip())
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    p = Path(os.path.expanduser(input_path)).resolve()

    if p.parent.name == MERGED_SUBDIR:
        run_dir = str(p.parent.parent)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    run_dir = os.path.join(root_dir, datetime.now().strftime("%Y%m%d_%H%M"))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _default_output_subdir(cam_preprocess: str) -> str:
    if str(cam_preprocess).strip().lower() == "stabilize_crop":
        return EPISODES_SUBDIR_CAMPROC
    return EPISODES_SUBDIR_RAW


def resolve_output_dir(input_path: str,
                       root_dir: str,
                       output_arg: Optional[str],
                       run_dir_arg: Optional[str],
                       cam_preprocess: str) -> str:
    if output_arg is not None and str(output_arg).strip() != "":
        out = os.path.expanduser(output_arg)
        os.makedirs(out, exist_ok=True)
        return out

    run_dir = infer_run_dir_from_input(input_path, root_dir, run_dir_arg)
    out = os.path.join(run_dir, _default_output_subdir(cam_preprocess))
    os.makedirs(out, exist_ok=True)
    return out


# ============================================================
# Array utilities
# ============================================================
def pad_repeat_last_small(arr: np.ndarray, target_len: int) -> np.ndarray:
    T = int(arr.shape[0])
    if T == target_len:
        return arr
    if T <= 0:
        raise ValueError("Cannot pad empty array.")
    if T > target_len:
        return arr[:target_len]

    pad_n = target_len - T
    last = arr[-1:, ...]
    pad_block = np.repeat(last, pad_n, axis=0)
    return np.concatenate([arr, pad_block], axis=0)


def shift_next_hold(x: np.ndarray) -> np.ndarray:
    T = int(x.shape[0])
    if T <= 1:
        return x.copy()
    return np.concatenate([x[1:], x[-1:]], axis=0)


# ============================================================
# Camera preprocessing
# ============================================================
def _moving_average_1d(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x.copy()
    kernel = np.ones((2 * radius + 1,), dtype=np.float32) / float(2 * radius + 1)
    x_pad = np.pad(x.astype(np.float32), (radius, radius), mode="edge")
    y = np.convolve(x_pad, kernel, mode="same")
    return y[radius:-radius]


def _estimate_pair_transform(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
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


def stabilize_image_sequence(images: np.ndarray,
                             smoothing_radius: int = 15,
                             border_mode: str = "reflect") -> np.ndarray:
    """
    images: (T,H,W,3) uint8 RGB
    returns stabilized images with the same shape.
    """
    if cv2 is None:
        print("[WARN] OpenCV is not available. camera preprocessing falls back to raw images.")
        return images.copy()

    imgs = np.asarray(images)
    T = int(imgs.shape[0])
    if T <= 1:
        return imgs.copy()

    prev_gray = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)
    transforms = np.zeros((T - 1, 3), dtype=np.float32)

    for i in range(T - 1):
        curr_gray = cv2.cvtColor(imgs[i + 1], cv2.COLOR_RGB2GRAY)
        transforms[i] = np.asarray(_estimate_pair_transform(prev_gray, curr_gray), dtype=np.float32)
        prev_gray = curr_gray

    trajectory = np.cumsum(transforms, axis=0)
    smoothed = np.zeros_like(trajectory)
    for j in range(3):
        smoothed[:, j] = _moving_average_1d(trajectory[:, j], smoothing_radius)

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

    return out


def center_crop_images(images: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    imgs = np.asarray(images)
    H, W = imgs.shape[1], imgs.shape[2]
    ch = int(min(max(1, crop_h), H))
    cw = int(min(max(1, crop_w), W))
    y0 = max(0, (H - ch) // 2)
    x0 = max(0, (W - cw) // 2)
    return imgs[:, y0:y0 + ch, x0:x0 + cw, :]


def resize_images(images: np.ndarray, resize_hw: int) -> np.ndarray:
    if resize_hw <= 0:
        return images.copy()
    if cv2 is None:
        print("[WARN] OpenCV is not available. resize is skipped.")
        return images.copy()

    out = []
    for img in images:
        out.append(cv2.resize(img, (resize_hw, resize_hw), interpolation=cv2.INTER_LINEAR))
    return np.stack(out, axis=0).astype(np.uint8)


def preprocess_cam_sequence(images: np.ndarray,
                            mode: str = "off",
                            crop_h: int = 384,
                            crop_w: int = 384,
                            resize_hw: int = 256,
                            stab_smoothing_radius: int = 15,
                            stab_border_mode: str = "reflect") -> np.ndarray:
    mode = str(mode).strip().lower()
    imgs = np.asarray(images, dtype=np.uint8)

    if mode == "off":
        return imgs.copy()
    if mode != "stabilize_crop":
        raise ValueError(f"Unsupported cam_preprocess mode: {mode}")

    imgs = stabilize_image_sequence(
        imgs,
        smoothing_radius=int(stab_smoothing_radius),
        border_mode=stab_border_mode,
    )
    imgs = center_crop_images(imgs, crop_h=int(crop_h), crop_w=int(crop_w))
    imgs = resize_images(imgs, resize_hw=int(resize_hw))
    return imgs.astype(np.uint8)




def compute_jitter_metrics(images: np.ndarray, smoothing_radius: int = 15) -> dict:
    imgs = np.asarray(images, dtype=np.uint8)
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
        transforms[i] = np.asarray(_estimate_pair_transform(prev_gray, curr_gray), dtype=np.float32)
        prev_gray = curr_gray

    trans_xy = transforms[:, :2].astype(np.float32)
    smooth_xy = np.zeros_like(trans_xy)
    for j in range(2):
        smooth_xy[:, j] = _moving_average_1d(trans_xy[:, j], smoothing_radius)
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


def format_jitter_metrics(m: dict) -> str:
    return (
        f"frames={m.get('frames', 0)}, shape={m.get('shape', ())}, "
        f"RMS={m.get('rms_px', 0.0):.4f}px, "
        f"P95={m.get('p95_px', 0.0):.4f}px, "
        f"max={m.get('max_px', 0.0):.4f}px, "
        f"mean_motion={m.get('mean_motion_px', 0.0):.4f}px"
    )


def copy_or_write_images_streaming(img_source,
                                   out_ds: h5py.Dataset,
                                   T_orig: int,
                                   T_pad: int,
                                   block: int = 8):
    """
    img_source may be either:
      - h5py dataset-like object
      - numpy array (T,H,W,3)
    """
    if T_orig <= 0:
        raise ValueError("T_orig must be > 0")

    t = 0
    while t < T_orig:
        n = min(block, T_orig - t)
        out_ds[t:t + n, ...] = img_source[t:t + n, ...]
        t += n

    remain = T_pad - T_orig
    if remain <= 0:
        return

    last = img_source[T_orig - 1, ...]
    t = T_orig
    while remain > 0:
        n = min(block, remain)
        out_ds[t:t + n, ...] = np.repeat(last[None, ...], n, axis=0)
        t += n
        remain -= n


# ============================================================
# Merged HDF5 format helpers
# ============================================================
def detect_format(h5: h5py.File) -> str:
    if "episodes" in h5:
        return "episodes_group"
    raise KeyError("Unsupported input: expected top-level group 'episodes'.")


def list_episode_keys(ep_grp: h5py.Group) -> List[str]:
    keys = sorted(list(ep_grp.keys()))

    def _keynum(k: str) -> int:
        digits = "".join([c for c in k if c.isdigit()])
        return int(digits) if digits else 10**9

    keys.sort(key=_keynum)
    return keys


def pick_img_key(img_grp: h5py.Group, candidates: List[str]) -> str:
    for k in candidates:
        if k in img_grp:
            return k
    raise KeyError(f"Missing image dataset. tried={candidates}, available={list(img_grp.keys())}")


def read_episode_single_camera(grp: h5py.Group,
                               camera_name: str = "cam0") -> Tuple[np.ndarray, np.ndarray, h5py.Dataset, int, str]:
    if "position" not in grp:
        raise KeyError(f"Missing 'position' in episode. available={list(grp.keys())}")
    if "ft" not in grp:
        raise KeyError(f"Missing 'ft' in episode. available={list(grp.keys())}")
    if "images" not in grp:
        raise KeyError(f"Missing 'images' in episode. available={list(grp.keys())}")

    pos = np.asarray(grp["position"][()], dtype=np.float64)
    ft = np.asarray(grp["ft"][()], dtype=np.float64)

    if pos.ndim != 2 or pos.shape[1] != 6:
        raise ValueError(f"position must be (T,6). got {pos.shape}")
    if ft.ndim != 2 or ft.shape[1] != 3:
        raise ValueError(f"ft must be (T,3). got {ft.shape}")

    img_grp = grp["images"]
    candidates = [
        camera_name,
        "cam0",
        "camera0",
        "cam_vr",
        "vr",
        "top",
        "cam_top",
        "front",
        "cam_front",
        "ee",
        "cam_ee",
    ]
    candidates = list(dict.fromkeys(candidates))
    img_key = pick_img_key(img_grp, candidates)
    img_ds = img_grp[img_key]

    if img_ds.ndim != 4 or img_ds.shape[-1] != 3:
        raise ValueError(f"image dataset must be (T,H,W,3). got {img_ds.shape}")

    T = int(min(pos.shape[0], ft.shape[0], img_ds.shape[0]))
    if T <= 0:
        raise ValueError("Episode too short.")

    return pos[:T], ft[:T], img_ds, T, img_key


# ============================================================
# Writer
# ============================================================
def write_episode_clean_single_camera(out_path: str,
                                      obs_pos: np.ndarray,
                                      obs_force: np.ndarray,
                                      img_source,
                                      T_orig: int,
                                      T_pad: int,
                                      out_camera_name: str = "cam0",
                                      image_compression: str = "lzf",
                                      image_copy_block: int = 8,
                                      cam_preprocess_mode: str = "off"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    act_pos_next = shift_next_hold(obs_pos)
    act_force_next = shift_next_hold(obs_force)

    obs_pos_p = pad_repeat_last_small(obs_pos, T_pad)
    obs_force_p = pad_repeat_last_small(obs_force, T_pad)
    act_pos_next_p = pad_repeat_last_small(act_pos_next, T_pad)
    act_force_next_p = pad_repeat_last_small(act_force_next, T_pad)

    is_pad = np.zeros((T_pad,), dtype=bool)
    pad_starts_at = -1
    if T_orig < T_pad:
        is_pad[T_orig:] = True
        pad_starts_at = int(T_orig)

    H, W, C = int(img_source.shape[1]), int(img_source.shape[2]), int(img_source.shape[3])
    chunks = (1, H, W, C)
    compression_arg = None if str(image_compression).lower() == "none" else image_compression

    with h5py.File(out_path, "w") as h:
        obs_grp = h.create_group("observations")
        obs_grp.create_dataset("position", data=obs_pos_p, dtype="float64")
        obs_grp.create_dataset("force", data=obs_force_p, dtype="float64")

        img_grp = obs_grp.create_group("images")
        out_img = img_grp.create_dataset(
            out_camera_name,
            shape=(T_pad, H, W, C),
            dtype="uint8",
            chunks=chunks,
            compression=compression_arg,
        )
        obs_grp.create_dataset("is_pad", data=is_pad, dtype="bool")

        act_grp = h.create_group("action")
        act_grp.create_dataset("position", data=act_pos_next_p, dtype="float64")
        act_grp.create_dataset("force", data=act_force_next_p, dtype="float64")

        meta = h.create_group("meta")
        meta.create_dataset("orig_len", data=np.array(int(T_orig), dtype=np.int64))
        meta.create_dataset("T_pad", data=np.array(int(T_pad), dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(int(pad_starts_at), dtype=np.int64))
        meta.create_dataset("truncated", data=np.array(bool(T_orig > T_pad), dtype=np.bool_))
        meta.create_dataset("camera_name", data=np.bytes_(out_camera_name))
        meta.create_dataset("cam_preprocess_mode", data=np.bytes_(str(cam_preprocess_mode)))

        copy_or_write_images_streaming(img_source, out_img, T_orig=T_orig, T_pad=T_pad, block=image_copy_block)


# ============================================================
# Convert
# ============================================================
def convert_merged_hdf5(input_path: str,
                        output_dir: str,
                        target_len: Optional[int] = None,
                        truncate: bool = False,
                        ep_prefix: str = "episode",
                        input_camera_name: str = "cam0",
                        output_camera_name: str = "cam0",
                        image_compression: str = "lzf",
                        image_copy_block: int = 8,
                        cam_preprocess: str = "stabilize_crop",
                        cam_crop_h: int = 384,
                        cam_crop_w: int = 384,
                        cam_resize_hw: int = 256,
                        cam_stab_smoothing_radius: int = 15,
                        cam_stab_border_mode: str = "reflect") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "input": input_path,
        "output_dir": output_dir,
        "format": "merged_hdf5_episodes_group_single_camera",
        "camera": {
            "input_camera_name": input_camera_name,
            "output_camera_name": output_camera_name,
            "cam_preprocess": cam_preprocess,
            "cam_crop_h": int(cam_crop_h),
            "cam_crop_w": int(cam_crop_w),
            "cam_resize_hw": int(cam_resize_hw),
            "cam_stab_smoothing_radius": int(cam_stab_smoothing_radius),
            "cam_stab_border_mode": str(cam_stab_border_mode),
        },
        "pad_mode": "repeat_last",
        "truncate": bool(truncate),
        "episodes": [],
    }

    cam_mode = str(cam_preprocess).strip().lower()

    with h5py.File(input_path, "r") as f:
        ep_grp = f["episodes"] if detect_format(f) == "episodes_group" else None
        ep_keys = list_episode_keys(ep_grp)
        print(f"[INFO] episodes found = {len(ep_keys)}")

        lengths: List[int] = []
        valid_keys: List[str] = []
        for k in ep_keys:
            try:
                grp = ep_grp[k]
                _, _, _, T, img_key = read_episode_single_camera(grp, camera_name=input_camera_name)
                lengths.append(int(T))
                valid_keys.append(k)
                print(f"[SCAN] {k}: T={T}, image_key={img_key}")
            except Exception as e:
                print(f"[WARN] {k}: skip length scan ({e})")

        if len(lengths) == 0:
            raise ValueError("All episodes unreadable.")

        T_max = int(max(lengths))
        T_pad = int(T_max if target_len is None else (target_len if truncate else max(T_max, target_len)))
        manifest["T_pad"] = int(T_pad)

        out_idx = 0
        for k in valid_keys:
            grp = ep_grp[k]
            try:
                pos, ft, img_ds, T_orig, img_key = read_episode_single_camera(grp, camera_name=input_camera_name)
            except Exception as e:
                print(f"[SKIP] {k}: cannot read ({e})")
                continue

            if truncate and T_orig > T_pad:
                pos_use = pos[:T_pad]
                ft_use = ft[:T_pad]
                T_orig_use = T_pad
            else:
                pos_use = pos
                ft_use = ft
                T_orig_use = T_orig

            if cam_mode == "off":
                raw_imgs_for_jitter = np.asarray(img_ds[:T_orig_use], dtype=np.uint8)
                before_jitter = compute_jitter_metrics(raw_imgs_for_jitter, smoothing_radius=int(cam_stab_smoothing_radius))
                after_jitter = before_jitter
                img_source = img_ds
            else:
                raw_imgs = np.asarray(img_ds[:T_orig_use], dtype=np.uint8)
                before_jitter = compute_jitter_metrics(raw_imgs, smoothing_radius=int(cam_stab_smoothing_radius))
                img_source = preprocess_cam_sequence(
                    raw_imgs,
                    mode=cam_mode,
                    crop_h=int(cam_crop_h),
                    crop_w=int(cam_crop_w),
                    resize_hw=int(cam_resize_hw),
                    stab_smoothing_radius=int(cam_stab_smoothing_radius),
                    stab_border_mode=str(cam_stab_border_mode),
                )
                after_jitter = compute_jitter_metrics(img_source, smoothing_radius=int(cam_stab_smoothing_radius))

            before_rms = float(before_jitter.get("rms_px", 0.0))
            after_rms = float(after_jitter.get("rms_px", 0.0))
            reduction = 0.0 if before_rms <= 1e-9 else 100.0 * (before_rms - after_rms) / before_rms
            print(f"[CAM-JITTER] {k} before: {format_jitter_metrics(before_jitter)}")
            print(f"[CAM-JITTER] {k} after : {format_jitter_metrics(after_jitter)} | RMS_reduction={reduction:.2f}% | mode={cam_mode}")

            out_path = os.path.join(output_dir, f"{ep_prefix}_{out_idx}.hdf5")
            write_episode_clean_single_camera(
                out_path=out_path,
                obs_pos=pos_use[:T_orig_use],
                obs_force=ft_use[:T_orig_use],
                img_source=img_source,
                T_orig=T_orig_use,
                T_pad=T_pad,
                out_camera_name=output_camera_name,
                image_compression=image_compression,
                image_copy_block=image_copy_block,
                cam_preprocess_mode=cam_mode,
            )

            print(f"[OK] {k} -> {out_path} (orig={T_orig}, final={T_pad}, image_key={img_key}, cam_preprocess={cam_mode})")
            manifest["episodes"].append({
                "episode_key": k,
                "episode_file": out_path,
                "orig_T": int(T_orig),
                "T_used": int(T_orig_use),
                "T_pad": int(T_pad),
                "input_image_key": img_key,
                "output_image_key": output_camera_name,
                "cam_preprocess": cam_mode,
                "jitter_before": before_jitter,
                "jitter_after": after_jitter,
                "jitter_rms_reduction_percent": reduction,
            })
            out_idx += 1

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[DONE] conversion complete. episodes={len(manifest['episodes'])}, T_pad={manifest['T_pad']}")
    print(f"[DONE] manifest = {manifest_path}")
    return manifest


def delete_merged_file(input_path: str, delete_empty_merged_dir: bool = True):
    input_path = os.path.expanduser(input_path)
    merged_dir = os.path.dirname(input_path)

    if not os.path.isfile(input_path):
        print(f"[WARN] merged file already missing, skip delete: {input_path}")
        return

    os.remove(input_path)
    print(f"[DELETE] merged HDF5 removed: {input_path}")

    if delete_empty_merged_dir and os.path.isdir(merged_dir):
        try:
            if len(os.listdir(merged_dir)) == 0:
                os.rmdir(merged_dir)
                print(f"[DELETE] empty merged_hdf5 dir removed: {merged_dir}")
        except Exception as e:
            print(f"[WARN] failed to remove empty merged_hdf5 dir: {e}")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert single-camera merged_hdf5 episodes into ACT episode_*.hdf5 files."
    )
    parser.add_argument("--root", default=ROOT_DEFAULT)
    parser.add_argument("--input", "-i", default=None,
                        help="Merged HDF5 file, merged_hdf5 directory, or run directory. If omitted, latest is selected.")
    parser.add_argument("--output", "-o", default=None,
                        help="Output episode directory. If omitted, use <run_dir>/<episodes_subdir>.")
    parser.add_argument("--run-dir", default=None,
                        help="Output run folder name under root. Usually not needed when input is under a run folder.")
    parser.add_argument("--ep-prefix", default="episode")
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--truncate", action="store_true")

    parser.add_argument("--input-camera-name", default="cam0",
                        help="Camera dataset name to read under episodes/ep_xxxx/images/.")
    parser.add_argument("--output-camera-name", default="cam0",
                        help="Camera dataset name to write under observations/images/.")
    parser.add_argument("--image-compression", default="lzf",
                        choices=["lzf", "gzip", "none"],
                        help="HDF5 compression for output image datasets.")
    parser.add_argument("--image-copy-block", type=int, default=8,
                        help="Block size for image copy/pad.")

    parser.add_argument("--cam_preprocess", default="stabilize_crop", choices=["off", "stabilize_crop"],
                        help="Camera preprocessing mode applied in the converter.")
    parser.add_argument("--cam_crop_h", type=int, default=384,
                        help="Crop height after stabilization.")
    parser.add_argument("--cam_crop_w", type=int, default=384,
                        help="Crop width after stabilization.")
    parser.add_argument("--cam_resize_hw", type=int, default=256,
                        help="Final square resize after crop. <=0 disables resize.")
    parser.add_argument("--cam_stab_smoothing_radius", type=int, default=15,
                        help="Smoothing radius for global camera stabilization.")
    parser.add_argument("--cam_stab_border_mode", default="reflect", choices=["reflect", "replicate", "constant"],
                        help="Border mode for warpAffine during stabilization.")

    parser.add_argument("--keep-merged", action="store_true",
                        help="Do not delete merged HDF5 after successful conversion.")
    parser.add_argument("--keep-merged-dir", action="store_true",
                        help="Do not remove empty merged_hdf5 directory after deleting merged file.")

    args = parser.parse_args()

    input_path = resolve_input_path(args.input, args.root)
    output_dir = resolve_output_dir(input_path, args.root, args.output, args.run_dir, args.cam_preprocess)

    print(f"[INFO] input  = {input_path}")
    print(f"[INFO] output = {output_dir}")
    print(f"[INFO] cam_preprocess = {args.cam_preprocess}")

    manifest = convert_merged_hdf5(
        input_path=input_path,
        output_dir=output_dir,
        target_len=args.target_len,
        truncate=args.truncate,
        ep_prefix=args.ep_prefix,
        input_camera_name=args.input_camera_name,
        output_camera_name=args.output_camera_name,
        image_compression=args.image_compression,
        image_copy_block=max(1, int(args.image_copy_block)),
        cam_preprocess=args.cam_preprocess,
        cam_crop_h=args.cam_crop_h,
        cam_crop_w=args.cam_crop_w,
        cam_resize_hw=args.cam_resize_hw,
        cam_stab_smoothing_radius=max(0, int(args.cam_stab_smoothing_radius)),
        cam_stab_border_mode=args.cam_stab_border_mode,
    )

    if not args.keep_merged:
        if len(manifest.get("episodes", [])) <= 0:
            raise RuntimeError("No episodes converted. Refusing to delete merged HDF5.")
        delete_merged_file(input_path, delete_empty_merged_dir=(not args.keep_merged_dir))
    else:
        print(f"[KEEP] merged HDF5 preserved: {input_path}")


if __name__ == "__main__":
    main()