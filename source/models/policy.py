#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from .act_core import (
    build_ACT_model_and_optimizer,
    build_CNNMLP_model_and_optimizer,
)
from .diffusion_core import build_DIFFUSION_model_and_optimizer


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        print(f"[ACTPolicy] KL Weight = {self.kl_weight}")

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        env_state = None
        image = self._normalize(image)
        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos=qpos, image=image, env_state=env_state,
                actions=actions, is_pad=is_pad, force_history=force_history,
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            valid_mask = (~is_pad).unsqueeze(-1).float()
            valid_count = valid_mask.sum().clamp_min(1.0) * actions.shape[-1]
            l1 = (all_l1 * valid_mask).sum() / valid_count
            loss_dict = {"l1": l1, "kl": total_kld[0]}
            loss_dict["loss"] = loss_dict["l1"] + self.kl_weight * loss_dict["kl"]
            return loss_dict

        a_hat, _, _ = self.model(qpos=qpos, image=image, env_state=env_state, force_history=force_history)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        env_state = None
        image = self._normalize(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(
                qpos=qpos, image=image, env_state=env_state,
                actions=actions, force_history=force_history,
            )
            mse = F.mse_loss(actions, a_hat)
            return {"mse": mse, "loss": mse}

        a_hat = self.model(qpos=qpos, image=image, env_state=env_state, force_history=force_history)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_DIFFUSION_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer
        self.num_queries = int(args_override["num_queries"])
        self.action_dim = int(args_override.get("action_dim", 9))
        self.diffusion_train_steps = int(args_override.get("diffusion_train_steps", 100))
        self.diffusion_infer_steps = int(args_override.get("diffusion_infer_steps", 10))
        self.beta_start = float(args_override.get("diffusion_beta_start", 1e-4))
        self.beta_end = float(args_override.get("diffusion_beta_end", 2e-2))
        self.loss_type = str(args_override.get("diffusion_loss_type", "mse")).strip().lower()

        betas = torch.linspace(self.beta_start, self.beta_end, self.diffusion_train_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_bars", alpha_bars, persistent=False)

        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[DiffusionPolicy] train_steps={self.diffusion_train_steps} infer_steps={self.diffusion_infer_steps}")
        print(f"[DiffusionPolicy] params={n_params / 1e6:.2f}M")

    def configure_optimizers(self):
        return self.optimizer

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, x_shape):
        out = arr.gather(0, t)
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def _q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_ab = torch.sqrt(self._extract(self.alpha_bars, t, x_start.shape))
        sqrt_one_minus_ab = torch.sqrt(1.0 - self._extract(self.alpha_bars, t, x_start.shape))
        return sqrt_ab * x_start + sqrt_one_minus_ab * noise

    def _loss(self, pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor):
        per = torch.abs(pred - target) if self.loss_type == "l1" else (pred - target) ** 2
        valid_mask = (~is_pad).unsqueeze(-1).float()
        valid_count = valid_mask.sum().clamp_min(1.0) * pred.shape[-1]
        return (per * valid_mask).sum() / valid_count

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        B, K, C, H, W = image.shape
        flat = image.reshape(B * K, C, H, W)
        flat = self._normalize(flat)
        return flat.reshape(B, K, C, H, W)

    @torch.no_grad()
    def _ddim_sample(self, qpos, image, force_history=None):
        device = qpos.device
        B = qpos.shape[0]
        T = self.num_queries
        Da = self.action_dim
        x = torch.randn(B, T, Da, device=device, dtype=qpos.dtype)

        if self.diffusion_infer_steps >= self.diffusion_train_steps:
            time_seq = list(range(self.diffusion_train_steps - 1, -1, -1))
        else:
            idx = torch.linspace(self.diffusion_train_steps - 1, 0, steps=self.diffusion_infer_steps, device=device)
            time_seq = sorted(set(int(round(v.item())) for v in idx), reverse=True)

        for i, t_int in enumerate(time_seq):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps = self.model(noisy_actions=x, timesteps=t, qpos=qpos, image=image, force_history=force_history)
            ab_t = self.alpha_bars[t_int]
            ab_prev = torch.tensor(1.0, device=device, dtype=x.dtype) if i == len(time_seq) - 1 else self.alpha_bars[time_seq[i + 1]]
            sqrt_ab_t = torch.sqrt(ab_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t)
            x0 = (x - sqrt_one_minus_ab_t * eps) / torch.clamp(sqrt_ab_t, min=1e-8)
            x = torch.sqrt(ab_prev) * x0 + torch.sqrt(torch.clamp(1.0 - ab_prev, min=0.0)) * eps

        return x

    def forward(self, qpos, image, actions=None, is_pad=None, force_history=None):
        image = self._normalize_image(image)
        if actions is not None:
            actions = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]
            B = actions.shape[0]
            t = torch.randint(0, self.diffusion_train_steps, (B,), device=actions.device, dtype=torch.long)
            noise = torch.randn_like(actions)
            noisy_actions = self._q_sample(actions, t, noise)
            pred_noise = self.model(
                noisy_actions=noisy_actions, timesteps=t,
                qpos=qpos, image=image, force_history=force_history,
            )
            diffusion_loss = self._loss(pred_noise, noise, is_pad)
            return {"diffusion": diffusion_loss, "loss": diffusion_loss}

        return self._ddim_sample(qpos=qpos, image=image, force_history=force_history)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    return total_kld, dimension_wise_kld, mean_kld