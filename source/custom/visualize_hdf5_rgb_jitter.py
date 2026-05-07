#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_hdf5_rgb_jitter.py

Analyze and visualize single-camera RGB hand-shake / jitter in ACT-format HDF5 episodes.

Default target:
    /home/eunseop/nrs_imitation/datasets/ACT/<latest_timestamp>/episodes_ft/*.hdf5

Expected HDF5 image key candidates:
    /observations/images/cam0
    /observations/images/<camera_name>
    /images/cam0
    /cam0

Outputs:
    /home/eunseop/nrs_imitation/analysis_logs/camera_jitter/<timestamp_dataset>/
        jitter_summary.csv
        jitter_summary_by_index.png
        jitter_summary_ranking.png
        episode_XX_jitter.png
        best_worst_report.txt

Core idea:
    1. Estimate frame-to-frame global translation using phase correlation.
    2. Smooth the estimated motion to approximate intentional low-frequency camera motion.
    3. Treat high-frequency residual as hand-shake jitter.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except Exception as e:
    cv2 = None


# =============================================================================
# Utilities
# =============================================================================

def natural_key(path: Path):
    """Sort episode_2 before episode_10."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def is_timestamp_like(name: str) -> bool:
    for fmt in ("%Y%m%d_%H%M", "%Y%m%d%H%M", "%m%d_%H%M"):
        try:
            datetime.strptime(name, fmt)
            return True
        except ValueError:
            continue
    return False


def find_latest_dataset_dir(
    act_root: Path,
    subdir_name: str = "episodes_ft",
) -> Path:
    """
    Find the latest ACT dataset folder that contains episode_*.hdf5.
    Priority:
      1. timestamp-like folder name
      2. lexicographic timestamp
      3. modification time
    """
    if not act_root.exists():
        raise FileNotFoundError(f"ACT root does not exist: {act_root}")

    candidates = []
    for run_dir in act_root.iterdir():
        if not run_dir.is_dir():
            continue
        ep_dir = run_dir / subdir_name
        if not ep_dir.is_dir():
            continue
        files = sorted(ep_dir.glob("episode_*.hdf5"), key=natural_key)
        if len(files) == 0:
            continue
        timestamp_bonus = 1 if is_timestamp_like(run_dir.name) else 0
        candidates.append((timestamp_bonus, run_dir.name, ep_dir.stat().st_mtime, ep_dir, len(files)))

    if len(candidates) == 0:
        # Fallback recursive search
        for ep_dir in act_root.rglob(subdir_name):
            if not ep_dir.is_dir():
                continue
            files = sorted(ep_dir.glob("episode_*.hdf5"), key=natural_key)
            if len(files) == 0:
                continue
            parent_name = ep_dir.parent.name
            timestamp_bonus = 1 if is_timestamp_like(parent_name) else 0
            candidates.append((timestamp_bonus, parent_name, ep_dir.stat().st_mtime, ep_dir, len(files)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No dataset directory found under {act_root} with subdir={subdir_name} and episode_*.hdf5"
        )

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


def list_hdf5_keys(h5obj, prefix: str = "") -> List[str]:
    keys = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            keys.append("/" + name)
    h5obj.visititems(visitor)
    return keys


def resolve_image_dataset(f: h5py.File, camera_name: str = "cam0") -> Tuple[h5py.Dataset, str]:
    candidates = [
        f"observations/images/{camera_name}",
        f"/observations/images/{camera_name}",
        f"images/{camera_name}",
        f"/images/{camera_name}",
        camera_name,
        f"/{camera_name}",
    ]

    for key in candidates:
        key2 = key[1:] if key.startswith("/") else key
        if key2 in f and isinstance(f[key2], h5py.Dataset):
            return f[key2], "/" + key2

    # Last resort: find any dataset ending with camera_name or containing images.
    all_keys = list_hdf5_keys(f)
    preferred = []
    for key in all_keys:
        k = key.lower()
        if k.endswith("/" + camera_name.lower()):
            preferred.append(key)
        elif "image" in k or "cam" in k:
            preferred.append(key)

    if preferred:
        key = preferred[0]
        return f[key], key

    raise KeyError(
        f"Could not find image dataset for camera_name={camera_name}. "
        f"Available datasets: {all_keys[:30]}"
    )


def decode_frame(frame: np.ndarray) -> np.ndarray:
    """
    Return RGB uint8 image with shape (H,W,3).
    Supports:
      - raw RGB H,W,3
      - CHW 3,H,W
      - grayscale H,W
      - encoded bytes 1D uint8
    """
    arr = np.asarray(frame)

    # Encoded image bytes
    if arr.ndim == 1 and arr.dtype == np.uint8:
        if cv2 is None:
            raise RuntimeError("cv2 is required to decode compressed image bytes.")
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2.imdecode failed.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if arr.ndim == 3:
        # CHW -> HWC
        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))

        # RGBA -> RGB
        if arr.shape[-1] == 4:
            arr = arr[..., :3]

        if arr.shape[-1] != 3:
            raise RuntimeError(f"Unsupported 3D frame shape: {arr.shape}")

        if arr.dtype != np.uint8:
            # support [0,1] float or wider int
            if np.issubdtype(arr.dtype, np.floating):
                if arr.max() <= 1.5:
                    arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    if arr.ndim == 2:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=-1)

    raise RuntimeError(f"Unsupported frame shape: {arr.shape}, dtype={arr.dtype}")


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return x
    window = int(max(1, window))
    if window <= 1 or n < 3:
        return x.copy()

    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window <= 1:
        return x.copy()

    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def phase_correlation_motion(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    downsample: float = 0.5,
    hann_cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> Tuple[float, float, float]:
    """
    Estimate translation from prev to curr.
    Returns:
        dx_px, dy_px, response
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: sudo apt install python3-opencv")

    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

    ds = float(np.clip(downsample, 0.1, 1.0))
    if ds < 0.999:
        prev_gray = cv2.resize(prev_gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
        curr_gray = cv2.resize(curr_gray, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)

    prev_f = prev_gray.astype(np.float32)
    curr_f = curr_gray.astype(np.float32)

    # Remove brightness offset to make phase correlation more stable.
    prev_f -= float(prev_f.mean())
    curr_f -= float(curr_f.mean())

    shape = prev_f.shape
    hann = None
    if hann_cache is not None:
        if shape not in hann_cache:
            hann_cache[shape] = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
        hann = hann_cache[shape]

    if hann is not None:
        shift, response = cv2.phaseCorrelate(prev_f, curr_f, hann)
    else:
        shift, response = cv2.phaseCorrelate(prev_f, curr_f)

    dx = float(shift[0]) / ds
    dy = float(shift[1]) / ds
    return dx, dy, float(response)


@dataclass
class EpisodeJitterResult:
    episode_path: Path
    image_key: str
    num_frames: int
    num_pairs: int
    mean_motion_px: float
    rms_motion_px: float
    mean_jitter_px: float
    rms_jitter_px: float
    p95_jitter_px: float
    max_jitter_px: float
    mean_response: float
    plot_path: Path


def analyze_episode(
    episode_path: Path,
    out_dir: Path,
    camera_name: str = "cam0",
    stride: int = 1,
    max_frames: int = 0,
    downsample: float = 0.5,
    smooth_window: int = 21,
    max_shift_clip_px: float = 80.0,
    save_plot: bool = True,
) -> EpisodeJitterResult:
    stride = max(1, int(stride))
    max_frames = int(max_frames)
    hann_cache = {}

    with h5py.File(episode_path, "r") as f:
        dset, image_key = resolve_image_dataset(f, camera_name=camera_name)
        n_total = int(dset.shape[0])

        indices = list(range(0, n_total, stride))
        if max_frames > 0:
            indices = indices[:max_frames]

        if len(indices) < 3:
            raise RuntimeError(f"Not enough frames in {episode_path}: selected={len(indices)}")

        dxs, dys, responses = [], [], []
        prev_rgb = decode_frame(dset[indices[0]])

        for idx in indices[1:]:
            curr_rgb = decode_frame(dset[idx])
            dx, dy, resp = phase_correlation_motion(
                prev_rgb,
                curr_rgb,
                downsample=downsample,
                hann_cache=hann_cache,
            )

            # Reject extreme phase-correlation wrap/outlier for visualization stability.
            dx = float(np.clip(dx, -max_shift_clip_px, max_shift_clip_px))
            dy = float(np.clip(dy, -max_shift_clip_px, max_shift_clip_px))

            dxs.append(dx)
            dys.append(dy)
            responses.append(resp)
            prev_rgb = curr_rgb

    dx = np.asarray(dxs, dtype=np.float64)
    dy = np.asarray(dys, dtype=np.float64)
    response = np.asarray(responses, dtype=np.float64)
    t = np.arange(len(dx), dtype=np.float64)

    motion_norm = np.sqrt(dx ** 2 + dy ** 2)

    sx = moving_average(dx, smooth_window)
    sy = moving_average(dy, smooth_window)

    jx = dx - sx
    jy = dy - sy
    jitter_norm = np.sqrt(jx ** 2 + jy ** 2)

    cum_x = np.cumsum(dx)
    cum_y = np.cumsum(dy)
    cum_sx = np.cumsum(sx)
    cum_sy = np.cumsum(sy)

    mean_motion = float(np.mean(motion_norm))
    rms_motion = float(np.sqrt(np.mean(motion_norm ** 2)))
    mean_jitter = float(np.mean(jitter_norm))
    rms_jitter = float(np.sqrt(np.mean(jitter_norm ** 2)))
    p95_jitter = float(np.percentile(jitter_norm, 95))
    max_jitter = float(np.max(jitter_norm))
    mean_response = float(np.mean(response))

    plot_path = out_dir / f"{episode_path.stem}_rgb_jitter.png"

    if save_plot:
        fig = plt.figure(figsize=(14, 10))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(t, dx, label="dx raw", linewidth=0.8)
        ax1.plot(t, dy, label="dy raw", linewidth=0.8)
        ax1.plot(t, sx, label="dx smooth", linewidth=1.2)
        ax1.plot(t, sy, label="dy smooth", linewidth=1.2)
        ax1.set_title("Frame-to-frame global translation")
        ax1.set_xlabel("frame pair index")
        ax1.set_ylabel("translation [px]")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(t, motion_norm, label="motion norm", linewidth=0.9)
        ax2.plot(t, jitter_norm, label="jitter residual norm", linewidth=0.9)
        ax2.set_title("Motion vs. high-frequency jitter")
        ax2.set_xlabel("frame pair index")
        ax2.set_ylabel("magnitude [px]")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(cum_x, cum_y, label="raw cumulative motion", linewidth=0.9)
        ax3.plot(cum_sx, cum_sy, label="smoothed low-frequency motion", linewidth=1.2)
        ax3.set_title("Cumulative camera motion trajectory")
        ax3.set_xlabel("cumulative x [px]")
        ax3.set_ylabel("cumulative y [px]")
        ax3.axis("equal")
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.hist(jitter_norm, bins=40, alpha=0.85)
        ax4.axvline(rms_jitter, linestyle="--", linewidth=1.2, label=f"RMS={rms_jitter:.3f}px")
        ax4.axvline(p95_jitter, linestyle="--", linewidth=1.2, label=f"P95={p95_jitter:.3f}px")
        ax4.set_title("Jitter residual distribution")
        ax4.set_xlabel("jitter residual norm [px]")
        ax4.set_ylabel("count")
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        fig.suptitle(
            f"{episode_path.name} | frames={len(indices)} | key={image_key}\n"
            f"RMS jitter={rms_jitter:.3f}px, P95={p95_jitter:.3f}px, "
            f"mean response={mean_response:.3f}",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)

    return EpisodeJitterResult(
        episode_path=episode_path,
        image_key=image_key,
        num_frames=len(indices),
        num_pairs=len(dx),
        mean_motion_px=mean_motion,
        rms_motion_px=rms_motion,
        mean_jitter_px=mean_jitter,
        rms_jitter_px=rms_jitter,
        p95_jitter_px=p95_jitter,
        max_jitter_px=max_jitter,
        mean_response=mean_response,
        plot_path=plot_path,
    )


def save_summary_csv(results: Sequence[EpisodeJitterResult], csv_path: Path):
    fields = [
        "episode",
        "image_key",
        "num_frames",
        "num_pairs",
        "mean_motion_px",
        "rms_motion_px",
        "mean_jitter_px",
        "rms_jitter_px",
        "p95_jitter_px",
        "max_jitter_px",
        "mean_response",
        "plot_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "episode": str(r.episode_path),
                "image_key": r.image_key,
                "num_frames": r.num_frames,
                "num_pairs": r.num_pairs,
                "mean_motion_px": f"{r.mean_motion_px:.6f}",
                "rms_motion_px": f"{r.rms_motion_px:.6f}",
                "mean_jitter_px": f"{r.mean_jitter_px:.6f}",
                "rms_jitter_px": f"{r.rms_jitter_px:.6f}",
                "p95_jitter_px": f"{r.p95_jitter_px:.6f}",
                "max_jitter_px": f"{r.max_jitter_px:.6f}",
                "mean_response": f"{r.mean_response:.6f}",
                "plot_path": str(r.plot_path),
            })


def _plot_summary_bars(
    results: Sequence[EpisodeJitterResult],
    out_path: Path,
    sort_mode: str = "index",
):
    if len(results) == 0:
        return

    if sort_mode == "rms_desc":
        plot_results = sorted(results, key=lambda r: r.rms_jitter_px, reverse=True)
        title = "RGB hand-shake / jitter ranking by episode"
        xlabel = "episode, sorted by RMS jitter descending"
    elif sort_mode == "index":
        plot_results = sorted(results, key=lambda r: natural_key(r.episode_path))
        title = "RGB hand-shake / jitter by episode index"
        xlabel = "episode index"
    else:
        raise ValueError(f"Unsupported sort_mode: {sort_mode}")

    labels = [r.episode_path.stem.replace("episode_", "ep") for r in plot_results]
    rms = [r.rms_jitter_px for r in plot_results]
    p95 = [r.p95_jitter_px for r in plot_results]

    x = np.arange(len(plot_results))
    width = 0.42

    fig, ax = plt.subplots(figsize=(max(12, len(results) * 0.25), 6))
    ax.bar(x - width / 2, rms, width, label="RMS jitter [px]")
    ax.bar(x + width / 2, p95, width, label="P95 jitter [px]")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("jitter residual [px]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_summary_plot_by_index(results: Sequence[EpisodeJitterResult], out_path: Path):
    _plot_summary_bars(results, out_path, sort_mode="index")


def save_summary_plot_ranking(results: Sequence[EpisodeJitterResult], out_path: Path):
    _plot_summary_bars(results, out_path, sort_mode="rms_desc")

def save_report(results: Sequence[EpisodeJitterResult], out_path: Path):
    if len(results) == 0:
        return

    sorted_by_rms = sorted(results, key=lambda r: r.rms_jitter_px, reverse=True)
    worst = sorted_by_rms[:5]
    best = list(reversed(sorted_by_rms[-5:]))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("RGB Jitter Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total episodes analyzed: {len(results)}\n")
        f.write(f"Mean RMS jitter: {np.mean([r.rms_jitter_px for r in results]):.4f} px\n")
        f.write(f"Median RMS jitter: {np.median([r.rms_jitter_px for r in results]):.4f} px\n")
        f.write(f"Mean P95 jitter: {np.mean([r.p95_jitter_px for r in results]):.4f} px\n\n")

        f.write("[Worst episodes by RMS jitter]\n")
        for r in worst:
            f.write(
                f"- {r.episode_path.name}: RMS={r.rms_jitter_px:.4f}px, "
                f"P95={r.p95_jitter_px:.4f}px, max={r.max_jitter_px:.4f}px, "
                f"response={r.mean_response:.4f}\n"
            )

        f.write("\n[Best episodes by RMS jitter]\n")
        for r in best:
            f.write(
                f"- {r.episode_path.name}: RMS={r.rms_jitter_px:.4f}px, "
                f"P95={r.p95_jitter_px:.4f}px, max={r.max_jitter_px:.4f}px, "
                f"response={r.mean_response:.4f}\n"
            )

        f.write("\nInterpretation\n")
        f.write("- RMS jitter: high-frequency residual magnitude after smoothing global motion.\n")
        f.write("- P95 jitter: robust high-jitter threshold; useful for detecting shaky demonstrations.\n")
        f.write("- Mean response: phase correlation confidence. Very low values may indicate weak texture or unreliable estimation.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_root", type=str, default="/home/eunseop/nrs_imitation/datasets/ACT")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Direct path to episodes_ft. If omitted, latest ACT/<timestamp>/episodes_ft is used.")
    parser.add_argument("--subdir_name", type=str, default="episodes_ft",
                        help="Usually episodes_ft or episodes_ft_camproc.")
    parser.add_argument("--camera_name", type=str, default="cam0")
    parser.add_argument("--output_root", type=str, default="/home/eunseop/nrs_imitation/analysis_logs/camera_jitter")
    parser.add_argument("--max_episodes", type=int, default=0,
                        help="0 means all episodes.")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="0 means all frames in each episode.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every N-th frame.")
    parser.add_argument("--downsample", type=float, default=0.5,
                        help="Downsample ratio for phase correlation.")
    parser.add_argument("--smooth_window", type=int, default=21,
                        help="Moving-average window for low-frequency motion. Odd number recommended.")
    parser.add_argument("--max_shift_clip_px", type=float, default=80.0,
                        help="Clip extreme per-frame shift estimates for robust plotting.")
    parser.add_argument("--no_episode_plots", action="store_true",
                        help="Only save CSV and summary ranking plot.")
    args = parser.parse_args()

    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: sudo apt install python3-opencv")

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir).expanduser()
    else:
        dataset_dir = find_latest_dataset_dir(Path(args.act_root).expanduser(), subdir_name=args.subdir_name)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    episode_files = sorted(dataset_dir.glob("episode_*.hdf5"), key=natural_key)
    if args.max_episodes > 0:
        episode_files = episode_files[:args.max_episodes]

    if len(episode_files) == 0:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {dataset_dir}")

    run_name = dataset_dir.parent.name + "_" + dataset_dir.name
    out_dir = Path(args.output_root).expanduser() / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("[RGB JITTER ANALYSIS]")
    print(f"dataset_dir       : {dataset_dir}")
    print(f"num episodes      : {len(episode_files)}")
    print(f"camera_name       : {args.camera_name}")
    print(f"output_dir        : {out_dir}")
    print(f"stride            : {args.stride}")
    print(f"max_frames        : {args.max_frames if args.max_frames > 0 else 'ALL'}")
    print(f"downsample        : {args.downsample}")
    print(f"smooth_window     : {args.smooth_window}")
    print("=" * 80)

    results: List[EpisodeJitterResult] = []
    failed = []

    for i, ep in enumerate(episode_files):
        print(f"[{i+1:03d}/{len(episode_files):03d}] {ep.name} ...", end=" ", flush=True)
        try:
            r = analyze_episode(
                episode_path=ep,
                out_dir=out_dir,
                camera_name=args.camera_name,
                stride=args.stride,
                max_frames=args.max_frames,
                downsample=args.downsample,
                smooth_window=args.smooth_window,
                max_shift_clip_px=args.max_shift_clip_px,
                save_plot=(not args.no_episode_plots),
            )
            results.append(r)
            print(f"RMS={r.rms_jitter_px:.3f}px P95={r.p95_jitter_px:.3f}px response={r.mean_response:.3f}")
        except Exception as e:
            failed.append((ep, str(e)))
            print(f"FAILED: {e}")

    if len(results) == 0:
        raise RuntimeError("All episodes failed. Check image key / HDF5 structure.")

    csv_path = out_dir / "jitter_summary.csv"
    index_plot_path = out_dir / "jitter_summary_by_index.png"
    ranking_path = out_dir / "jitter_summary_ranking.png"
    report_path = out_dir / "best_worst_report.txt"

    save_summary_csv(results, csv_path)
    save_summary_plot_by_index(results, index_plot_path)
    save_summary_plot_ranking(results, ranking_path)
    save_report(results, report_path)

    if failed:
        fail_path = out_dir / "failed_episodes.txt"
        with fail_path.open("w", encoding="utf-8") as f:
            for ep, err in failed:
                f.write(f"{ep}: {err}\n")
        print(f"[WARN] failed episodes saved -> {fail_path}")

    print("\n[DONE]")
    print(f"summary csv       : {csv_path}")
    print(f"index plot        : {index_plot_path}")
    print(f"ranking plot      : {ranking_path}")
    print(f"report            : {report_path}")
    if not args.no_episode_plots:
        print(f"episode plots     : {out_dir}/episode_*_rgb_jitter.png")


if __name__ == "__main__":
    main()