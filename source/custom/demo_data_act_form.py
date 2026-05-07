#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import h5py
import numpy as np
from typing import Optional, List, Tuple
from datetime import datetime
from pathlib import Path

# ROOT_DEFAULT = "/home/eunseop/nrs_lab2/datasets/ACT"
ROOT_DEFAULT = "/home/eunseop/nrs_imitation/datasets/ACT"
MERGED_SUBDIR = "merged_hdf5"

# ---------------------------
# Utils: path resolve (merged_hdf5 latest)
# ---------------------------
def _is_probably_timestamp_name(name: str) -> bool:
    # examples: 202601241646 (12 digits), 20260124_1646, 0124_1813 (legacy)
    for fmt in ("%Y%m%d%H%M", "%Y%m%d_%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            pass
    return False


def resolve_input_path(user_input: Optional[str], root_dir: str) -> str:
    """
    merged_hdf5 전용:
    - user_input이 file이면 그대로 사용 (확장자 없어도 OK)
    - user_input이 dir이면 dir 내부에서 최신 파일 선택
    - user_input이 None이면 root/merged_hdf5 내부에서 최신 파일 선택
    """
    merged_dir = os.path.join(root_dir, MERGED_SUBDIR)

    def pick_latest_file_in_dir(d: str) -> str:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")

        files = []
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                # 확장자 있든 없든 다 후보로 넣되,
                # timestamp 형태면 우선순위 높이고, 아니면 mtime으로라도 최신을 고른다.
                st = os.stat(p)
                files.append((name, p, st.st_mtime))

        if not files:
            raise FileNotFoundError(f"No file found under: {d}")

        # 1) timestamp 이름 후보가 있으면 그 중 name 기준 최신
        ts = [x for x in files if _is_probably_timestamp_name(x[0])]
        if ts:
            ts.sort(key=lambda x: x[0], reverse=True)
            return ts[0][1]

        # 2) 아니면 mtime 최신
        files.sort(key=lambda x: x[2], reverse=True)
        return files[0][1]

    if user_input is None:
        return pick_latest_file_in_dir(merged_dir)

    user_input = os.path.expanduser(user_input)
    if os.path.isfile(user_input):
        return user_input
    if os.path.isdir(user_input):
        return pick_latest_file_in_dir(user_input)

    raise FileNotFoundError(f"Input path not found: {user_input}")


def make_run_dir(root_dir: str, run_dir: Optional[str] = None) -> str:
    os.makedirs(root_dir, exist_ok=True)
    if run_dir is None or str(run_dir).strip() == "":
        run_dir = datetime.now().strftime("%Y%m%d_%H%M")
    out = os.path.join(root_dir, run_dir)
    os.makedirs(out, exist_ok=True)
    return out


def pad_repeat_last_small(arr: np.ndarray, target_len: int) -> np.ndarray:
    """position/force 같은 작은 배열만 안전하게 pad."""
    T = arr.shape[0]
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
    """action(t)=x(t+1), 마지막은 hold"""
    T = x.shape[0]
    if T <= 1:
        return x.copy()
    return np.concatenate([x[1:], x[-1:]], axis=0)


def copy_images_streaming(in_ds: h5py.Dataset,
                          out_ds: h5py.Dataset,
                          T_orig: int,
                          T_pad: int,
                          block: int = 8):
    """
    (T,H,W,3) uint8 를 block 단위로 복사 + repeat_last 패딩.
    """
    if T_orig <= 0:
        raise ValueError("T_orig must be > 0")

    # copy original
    t = 0
    while t < T_orig:
        n = min(block, T_orig - t)
        out_ds[t:t+n, ...] = in_ds[t:t+n, ...]
        t += n

    # pad with last frame
    remain = T_pad - T_orig
    if remain <= 0:
        return
    last = in_ds[T_orig - 1, ...]  # (H,W,3) 한 장만 로드
    t = T_orig
    while remain > 0:
        n = min(block, remain)
        out_ds[t:t+n, ...] = np.repeat(last[None, ...], n, axis=0)
        t += n
        remain -= n


# ---------------------------
# merged_hdf5 format helpers (episodes group)
# ---------------------------
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


def read_episode_small(grp: h5py.Group) -> Tuple[np.ndarray, np.ndarray, h5py.Dataset, h5py.Dataset, int]:
    """
    expected:
      grp['position'] : (T,6)
      grp['ft']       : (T,3)
      grp['images'][top/ee] : (T,H,W,3)
    """
    if "position" not in grp:
        raise KeyError(f"Missing 'position' in episode. available={list(grp.keys())}")
    if "ft" not in grp:
        raise KeyError(f"Missing 'ft' in episode. available={list(grp.keys())}")
    if "images" not in grp:
        raise KeyError(f"Missing 'images' in episode. available={list(grp.keys())}")

    pos = np.asarray(grp["position"][()], dtype=np.float64)
    ft  = np.asarray(grp["ft"][()], dtype=np.float64)

    img_grp = grp["images"]
    k_top = pick_img_key(img_grp, ["top", "cam_top", "camera_top", "front", "cam_front", "camera_front"])
    k_ee  = pick_img_key(img_grp, ["ee", "cam_ee", "camera_ee", "head", "cam_head", "camera_head"])

    img_top_ds = img_grp[k_top]
    img_ee_ds  = img_grp[k_ee]

    T = int(min(pos.shape[0], ft.shape[0], img_top_ds.shape[0], img_ee_ds.shape[0]))
    if T <= 0:
        raise ValueError("Episode too short.")
    return pos[:T], ft[:T], img_top_ds, img_ee_ds, T


# ---------------------------
# writer (clean keys + T_pad meta)
# ---------------------------
def write_episode_clean(out_path: str,
                        obs_pos: np.ndarray,       # (T,6)
                        obs_force: np.ndarray,     # (T,3)
                        img_top_ds: h5py.Dataset,  # (T,H,W,3)
                        img_ee_ds: h5py.Dataset,   # (T,H,W,3)
                        T_orig: int,
                        T_pad: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    act_pos_next   = shift_next_hold(obs_pos)
    act_force_next = shift_next_hold(obs_force)

    obs_pos_p        = pad_repeat_last_small(obs_pos,        T_pad)
    obs_force_p      = pad_repeat_last_small(obs_force,      T_pad)
    act_pos_next_p   = pad_repeat_last_small(act_pos_next,   T_pad)
    act_force_next_p = pad_repeat_last_small(act_force_next, T_pad)

    is_pad = np.zeros((T_pad,), dtype=bool)
    pad_starts_at = -1
    if T_orig < T_pad:
        is_pad[T_orig:] = True
        pad_starts_at = int(T_orig)

    # image shape
    H, W, C = img_top_ds.shape[1], img_top_ds.shape[2], img_top_ds.shape[3]
    chunks = (1, H, W, C)

    with h5py.File(out_path, "w") as h:
        obs_grp = h.create_group("observations")
        obs_grp.create_dataset("position", data=obs_pos_p, dtype="float64")
        obs_grp.create_dataset("force",    data=obs_force_p, dtype="float64")

        img_grp = obs_grp.create_group("images")
        out_top = img_grp.create_dataset("cam_top",
                                         shape=(T_pad, H, W, C),
                                         dtype="uint8",
                                         chunks=chunks,
                                         compression="lzf")
        out_ee  = img_grp.create_dataset("cam_ee",
                                         shape=(T_pad, H, W, C),
                                         dtype="uint8",
                                         chunks=chunks,
                                         compression="lzf")

        obs_grp.create_dataset("is_pad", data=is_pad, dtype="bool")

        act_grp = h.create_group("action")
        act_grp.create_dataset("position", data=act_pos_next_p, dtype="float64")
        act_grp.create_dataset("force",    data=act_force_next_p, dtype="float64")

        meta = h.create_group("meta")
        meta.create_dataset("orig_len",      data=np.array(int(T_orig), dtype=np.int64))
        meta.create_dataset("T_pad",         data=np.array(int(T_pad), dtype=np.int64))
        meta.create_dataset("pad_starts_at", data=np.array(int(pad_starts_at), dtype=np.int64))
        meta.create_dataset("truncated",     data=np.array(bool(T_orig > T_pad), dtype=np.bool_))

        # stream copy images (repeat_last pad)
        T_img = int(min(T_orig, img_top_ds.shape[0], img_ee_ds.shape[0]))
        copy_images_streaming(img_top_ds, out_top, T_orig=T_img, T_pad=T_pad, block=8)
        copy_images_streaming(img_ee_ds,  out_ee,  T_orig=T_img, T_pad=T_pad, block=8)


# ---------------------------
# main convert
# ---------------------------
def convert_merged_hdf5(input_path: str,
                        output_dir: str,
                        target_len: Optional[int] = None,
                        truncate: bool = False,
                        ep_prefix: str = "episode"):
    os.makedirs(output_dir, exist_ok=True)

    manifest = {
        "input": input_path,
        "output_dir": output_dir,
        "format": "merged_hdf5_episodes_group",
        "pad_mode": "repeat_last",
        "truncate": truncate,
        "episodes": []
    }

    with h5py.File(input_path, "r") as f:
        fmt = detect_format(f)
        if fmt != "episodes_group":
            raise RuntimeError("Unexpected format.")

        ep_grp = f["episodes"]
        ep_keys = list_episode_keys(ep_grp)
        print(f"[INFO] episodes found = {len(ep_keys)}")

        # 1) length scan (shape only / small read)
        lengths = []
        for k in ep_keys:
            try:
                grp = ep_grp[k]
                # small read + dataset refs
                pos, ft, img_top_ds, img_ee_ds, T = read_episode_small(grp)
                lengths.append(int(T))
            except Exception as e:
                print(f"[WARN] {k}: skip length scan ({e})")
                lengths.append(0)

        T_max = max(lengths)
        if T_max <= 0:
            raise ValueError("All episodes unreadable.")

        if target_len is None:
            T_pad = int(T_max)
        else:
            T_pad = int(target_len if truncate else max(T_max, target_len))
        manifest["T_pad"] = int(T_pad)

        # 2) convert
        out_idx = 0
        for k in ep_keys:
            grp = ep_grp[k]
            try:
                pos, ft, img_top_ds, img_ee_ds, T_orig = read_episode_small(grp)
            except Exception as e:
                print(f"[SKIP] {k}: cannot read ({e})")
                continue

            if truncate and T_orig > T_pad:
                pos = pos[:T_pad]
                ft  = ft[:T_pad]
                T_orig_use = T_pad
            else:
                T_orig_use = T_orig

            out_path = os.path.join(output_dir, f"{ep_prefix}_{out_idx}.hdf5")
            write_episode_clean(out_path,
                                obs_pos=pos[:T_orig_use],
                                obs_force=ft[:T_orig_use],
                                img_top_ds=img_top_ds,
                                img_ee_ds=img_ee_ds,
                                T_orig=T_orig_use,
                                T_pad=T_pad)

            print(f"[OK] {k} -> {out_path} (orig={T_orig}, final={T_pad})")
            manifest["episodes"].append({
                "episode_key": k,
                "episode_file": out_path,
                "orig_T": int(T_orig),
                "T_pad": int(T_pad),
            })
            out_idx += 1

    with open(os.path.join(output_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp, indent=2)

    print("[DONE] conversion complete. T_pad =", manifest["T_pad"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT_DEFAULT)
    parser.add_argument("--input", "-i", default=None,
                        help="merged_hdf5 파일(확장자 없어도 됨) 또는 merged_hdf5 디렉터리")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--run-dir", default=None,
                        help="출력 run 폴더 이름(없으면 YYYYMMDD_HHMM)")
    parser.add_argument("--ep-prefix", default="episode")
    parser.add_argument("--target-len", type=int, default=None)
    parser.add_argument("--truncate", action="store_true")
    args = parser.parse_args()

    input_path = resolve_input_path(args.input, args.root)

    if args.output is not None:
        output_dir = args.output
    else:
        run_base = make_run_dir(args.root, args.run_dir)
        output_dir = os.path.join(run_base, "episodes_ft")

    print(f"[INFO] input  = {input_path}")
    print(f"[INFO] output = {output_dir}")

    convert_merged_hdf5(input_path, output_dir,
                        target_len=args.target_len,
                        truncate=args.truncate,
                        ep_prefix=args.ep_prefix)


if __name__ == "__main__":
    main()
