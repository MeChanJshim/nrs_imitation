#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo_data_act_form.py

Single-camera ACT dataset converter for the new VR demo pipeline.

Input merged HDF5 format:
  <ACT_ROOT>/<RUN_ID>/merged_hdf5/vr_demo_merged_<RUN_ID>.hdf5
    /episodes/ep_xxxx/position        (T, 6)
    /episodes/ep_xxxx/ft              (T, 3)
    /episodes/ep_xxxx/images/cam0     (T, H, W, 3), uint8 RGB

Output ACT episode format:
  <ACT_ROOT>/<RUN_ID>/episodes_ft/episode_0.hdf5
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

After successful conversion, the input merged HDF5 file is deleted by default
to save disk space. Use --keep-merged to preserve it.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np


ROOT_DEFAULT = "/home/eunseop/nrs_act/datasets/ACT"
MERGED_SUBDIR = "merged_hdf5"
EPISODES_SUBDIR = "episodes_ft"


# ============================================================
# Path utilities
# ============================================================
def _is_probably_timestamp_name(name: str) -> bool:
    """
    Accept common run/file timestamp patterns.
      20260421_1509
      202604211509
      0421_1509
      vr_demo_merged_20260421_1509.hdf5
    """
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
    """
    Resolve merged HDF5 input path.

    Supports:
      1) --input /path/to/file.hdf5
      2) --input /path/to/merged_hdf5/
      3) --input /path/to/run_dir/
      4) no --input:
         search latest under:
           root/*/merged_hdf5/*.hdf5
           root/merged_hdf5/*.hdf5   legacy fallback
    """
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
        # New layout: ACT/<RUN_ID>/merged_hdf5/*.hdf5
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            merged_dir = run_dir / MERGED_SUBDIR
            if merged_dir.is_dir():
                candidates.extend(_find_hdf5_files_under(str(merged_dir), recursive=False))

        # Legacy fallback: ACT/merged_hdf5/*.hdf5
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
    """
    For new layout:
      input = root/RUN_ID/merged_hdf5/file.hdf5
      run_dir = root/RUN_ID
    """
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


def resolve_output_dir(input_path: str,
                       root_dir: str,
                       output_arg: Optional[str],
                       run_dir_arg: Optional[str]) -> str:
    if output_arg is not None and str(output_arg).strip() != "":
        out = os.path.expanduser(output_arg)
        os.makedirs(out, exist_ok=True)
        return out

    run_dir = infer_run_dir_from_input(input_path, root_dir, run_dir_arg)
    out = os.path.join(run_dir, EPISODES_SUBDIR)
    os.makedirs(out, exist_ok=True)
    return out


# ============================================================
# Array utilities
# ============================================================
def pad_repeat_last_small(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad/truncate small arrays such as position or force by repeating last row."""
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
    """action(t) = observation(t+1), final action holds final observation."""
    T = int(x.shape[0])
    if T <= 1:
        return x.copy()
    return np.concatenate([x[1:], x[-1:]], axis=0)


def copy_images_streaming(in_ds: h5py.Dataset,
                          out_ds: h5py.Dataset,
                          T_orig: int,
                          T_pad: int,
                          block: int = 8):
    """Stream-copy image dataset (T,H,W,3) with repeat-last padding."""
    if T_orig <= 0:
        raise ValueError("T_orig must be > 0")

    t = 0
    while t < T_orig:
        n = min(block, T_orig - t)
        out_ds[t:t + n, ...] = in_ds[t:t + n, ...]
        t += n

    remain = T_pad - T_orig
    if remain <= 0:
        return

    last = in_ds[T_orig - 1, ...]
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
    """
    Expected new input:
      grp['position']              : (T,6)
      grp['ft']                    : (T,3)
      grp['images'][camera_name]   : (T,H,W,3)
    """
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
                                      img_ds: h5py.Dataset,
                                      T_orig: int,
                                      T_pad: int,
                                      out_camera_name: str = "cam0",
                                      image_compression: str = "lzf",
                                      image_copy_block: int = 8):
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

    H, W, C = int(img_ds.shape[1]), int(img_ds.shape[2]), int(img_ds.shape[3])
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

        T_img = int(min(T_orig, img_ds.shape[0]))
        copy_images_streaming(img_ds, out_img, T_orig=T_img, T_pad=T_pad, block=image_copy_block)


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
                        image_copy_block: int = 8) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "input": input_path,
        "output_dir": output_dir,
        "format": "merged_hdf5_episodes_group_single_camera",
        "camera": {
            "input_camera_name": input_camera_name,
            "output_camera_name": output_camera_name,
        },
        "pad_mode": "repeat_last",
        "truncate": bool(truncate),
        "episodes": [],
    }

    with h5py.File(input_path, "r") as f:
        fmt = detect_format(f)
        if fmt != "episodes_group":
            raise RuntimeError("Unexpected format.")

        ep_grp = f["episodes"]
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
        if target_len is None:
            T_pad = int(T_max)
        else:
            T_pad = int(target_len if truncate else max(T_max, target_len))

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

            out_path = os.path.join(output_dir, f"{ep_prefix}_{out_idx}.hdf5")
            write_episode_clean_single_camera(
                out_path=out_path,
                obs_pos=pos_use[:T_orig_use],
                obs_force=ft_use[:T_orig_use],
                img_ds=img_ds,
                T_orig=T_orig_use,
                T_pad=T_pad,
                out_camera_name=output_camera_name,
                image_compression=image_compression,
                image_copy_block=image_copy_block,
            )

            print(f"[OK] {k} -> {out_path} (orig={T_orig}, final={T_pad}, image_key={img_key})")
            manifest["episodes"].append({
                "episode_key": k,
                "episode_file": out_path,
                "orig_T": int(T_orig),
                "T_used": int(T_orig_use),
                "T_pad": int(T_pad),
                "input_image_key": img_key,
                "output_image_key": output_camera_name,
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
                        help="Output episodes_ft directory. If omitted, use <run_dir>/episodes_ft.")
    parser.add_argument("--run-dir", default=None,
                        help="Output run folder name under root. Usually not needed when input is under a run folder.")
    parser.add_argument("--ep-prefix", default="episode")
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--truncate", action="store_true")

    parser.add_argument("--input-camera-name", default="cam0",
                        help="Camera dataset name to read under episodes/ep_xxxx/images/.")
    parser.add_argument("--output-camera-name", default="cam0",
                        help="Camera dataset name to write under observations/images/.")
    parser.add_argument("--image-compression", default="lzf", choices=["lzf", "gzip", "none"],
                        help="HDF5 compression for output image datasets.")
    parser.add_argument("--image-copy-block", type=int, default=8,
                        help="Block size for streaming image copy.")

    parser.add_argument("--keep-merged", action="store_true",
                        help="Do not delete merged HDF5 after successful conversion.")
    parser.add_argument("--keep-merged-dir", action="store_true",
                        help="Do not remove empty merged_hdf5 directory after deleting merged file.")

    args = parser.parse_args()

    input_path = resolve_input_path(args.input, args.root)
    output_dir = resolve_output_dir(input_path, args.root, args.output, args.run_dir)

    print(f"[INFO] input  = {input_path}")
    print(f"[INFO] output = {output_dir}")

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
    )

    if not args.keep_merged:
        if len(manifest.get("episodes", [])) <= 0:
            raise RuntimeError("No episodes converted. Refusing to delete merged HDF5.")
        delete_merged_file(input_path, delete_empty_merged_dir=(not args.keep_merged_dir))
    else:
        print(f"[KEEP] merged HDF5 preserved: {input_path}")


if __name__ == "__main__":
    main()
