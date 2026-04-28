#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Dict, List

import torch
from tqdm import tqdm

from common.utils import set_seed
from models.policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from training.debug import debug_norm_once, make_grad_scaler, autocast_context

try:
    from training.plotting import plot_history
except Exception:
    plot_history = None


def _scalarize_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for k, v in loss_dict.items():
        out[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) else float(v)
    return out


def _mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if len(dicts) == 0:
        return {}
    keys = dicts[0].keys()
    return {k: sum(d[k] for d in dicts) / len(dicts) for k in keys}


def make_policy(policy_class: str, policy_config: Dict):
    policy_class = str(policy_class).upper()
    if policy_class == "ACT":
        return ACTPolicy(policy_config)
    if policy_class == "CNNMLP":
        return CNNMLPPolicy(policy_config)
    if policy_class == "DIFFUSION":
        return DiffusionPolicy(policy_config)
    raise ValueError(f"Unsupported policy_class: {policy_class}")


def _unpack_batch(batch, device: torch.device):
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"batch must be tuple/list, got {type(batch)}")
    if len(batch) == 4:
        image, qpos, action, is_pad = batch
        force_history = None
    elif len(batch) == 5:
        image, qpos, action, is_pad, force_history = batch
    else:
        raise ValueError(f"Unexpected batch length: {len(batch)}")

    image = image.to(device, non_blocking=True)
    qpos = qpos.to(device, non_blocking=True)
    action = action.to(device, non_blocking=True)
    is_pad = is_pad.to(device, non_blocking=True)
    if force_history is not None:
        force_history = force_history.to(device, non_blocking=True)
    return image, qpos, action, is_pad, force_history


def forward_pass(batch, policy, device):
    image, qpos, action, is_pad, force_history = _unpack_batch(batch, device)
    if force_history is None:
        return policy(qpos, image, action, is_pad)
    return policy(qpos, image, action, is_pad, force_history=force_history)


@torch.no_grad()
def _run_validation(val_loader, policy, device, amp_enabled: bool = False) -> Dict[str, float]:
    policy.eval()
    val_dicts = []
    for batch in val_loader:
        with autocast_context(amp_enabled, device):
            out = forward_pass(batch, policy, device)
        val_dicts.append(_scalarize_loss_dict(out))
    return _mean_dict(val_dicts)


def _save_checkpoint(ckpt_path: str, epoch: int, policy, optimizer, train_summary, val_summary, config):
    payload = {
        "epoch": int(epoch),
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "config": config,
    }
    torch.save(payload, ckpt_path)


def _plot_history_if_possible(history, ckpt_dir: str, policy_class: str, seed: int):
    if plot_history is None:
        return
    try:
        plot_history(history, ckpt_dir=ckpt_dir, policy_class=policy_class, seed=seed)
    except Exception:
        pass


def train_bc(train_loader, val_loader, config):
    device = config["device"]
    seed = int(config.get("seed", 0))
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    num_epochs = int(config["num_epochs"])
    ckpt_dir = config["ckpt_dir"]

    amp_enabled = bool(config.get("amp", False))
    debug_norm = bool(config.get("debug_norm", False))
    debug_norm_batches = int(config.get("debug_norm_batches", 1))
    debug_batches = int(config.get("debug_batches", 3))
    save_every = int(config.get("save_every", 0))

    os.makedirs(ckpt_dir, exist_ok=True)
    set_seed(seed)

    if debug_norm:
        print("[INFO] debug_norm enabled: printing post-normalization stats for TRAIN and VAL.")
        debug_norm_once(train_loader, tag="TRAIN", max_batches=debug_norm_batches)
        debug_norm_once(val_loader, tag="VAL", max_batches=debug_norm_batches)

    policy = make_policy(policy_class, policy_config).to(device)
    optimizer = policy.configure_optimizers()
    scaler = make_grad_scaler(amp_enabled, device)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[DEBUG] Policy class = {policy_class}, trainable params = {n_params / 1e6:.2f}M")

    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(ckpt_dir, "policy_best.ckpt")
    last_ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
    history = {"train": [], "val": []}
    epoch_pbar = tqdm(range(num_epochs))

    for epoch in epoch_pbar:
        print(f"Epoch {epoch}")
        val_summary = _run_validation(val_loader, policy, device, amp_enabled)
        history["val"].append({"epoch": epoch, **val_summary})

        if len(val_summary) > 0:
            val_msg = " | ".join(f"{k}:{v:.6f}" for k, v in sorted(val_summary.items()))
            print(f"Val: {val_msg}")
            current_val_loss = val_summary.get("loss", None)
            if current_val_loss is not None and current_val_loss < best_val_loss:
                best_val_loss = float(current_val_loss)
                best_epoch = int(epoch)
                _save_checkpoint(best_ckpt_path, epoch, policy, optimizer, None, val_summary, config)

        policy.train()
        train_dicts = []
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(amp_enabled, device):
                out = forward_pass(batch, policy, device)
                loss = out["loss"]
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scalars = _scalarize_loss_dict(out)
            train_dicts.append(scalars)
            if batch_idx < debug_batches:
                print(f"[DEBUG] Epoch {epoch}, batch {batch_idx}, train loss = {scalars['loss']:.6f}")

        train_summary = _mean_dict(train_dicts)
        history["train"].append({"epoch": epoch, **train_summary})

        train_loss = train_summary.get("loss", float("nan"))
        val_loss = val_summary.get("loss", float("nan")) if len(val_summary) > 0 else float("nan")
        epoch_pbar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}")

        if save_every > 0 and (epoch % save_every == 0):
            periodic_ckpt = os.path.join(ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt")
            _save_checkpoint(periodic_ckpt, epoch, policy, optimizer, train_summary, val_summary, config)

        _save_checkpoint(last_ckpt_path, epoch, policy, optimizer, train_summary, val_summary, config)

    _plot_history_if_possible(history, ckpt_dir, policy_class, seed)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_ckpt_path": best_ckpt_path,
        "last_ckpt_path": last_ckpt_path,
        "history": history,
    }