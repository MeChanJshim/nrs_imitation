#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_core.py

RGB-conditioned Flow Matching policy core for nrs_imitation.

References
----------
[1] Flow Matching original generative modeling framework:
    Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le,
    "Flow Matching for Generative Modeling", ICLR 2023.
    Role:
        General generative modeling / continuous normalizing flow training framework.
    arXiv:
        https://arxiv.org/abs/2210.02747
    OpenReview:
        https://openreview.net/forum?id=PqvMRDCJT9t

[2] Rectified Flow, closely related formulation:
    Xingchao Liu, Chengyue Gong, Qiang Liu,
    "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow",
    ICLR 2023.
    Role:
        Straight-line probability path / velocity-field learning formulation closely
        related to the objective used in this implementation.
    arXiv:
        https://arxiv.org/abs/2209.03003
    OpenReview:
        https://openreview.net/forum?id=XVjTT1nw5z

[3] Early core robot imitation learning application:
    Eugenio Chisari, Nick Heppert, Max Argus, Tim Welschehold,
    Thomas Brox, Abhinav Valada,
    "Learning Robotic Manipulation Policies from Point Clouds with Conditional Flow Matching",
    CoRL 2024.
    Also known as:
        PointFlowMatch.
    Role:
        One of the early key works that applies Conditional Flow Matching to
        robot manipulation imitation learning, especially with point-cloud observations.
    arXiv:
        https://arxiv.org/abs/2409.07343
    Project:
        https://pointflowmatch.cs.uni-freiburg.de/
    Code:
        https://github.com/robot-learning-freiburg/PointFlowMatch

[4] Fast robot flow policy / inference efficiency:
    Qinglun Zhang, Zhen Liu, Haoqiang Fan, Guanghui Liu, Bing Zeng, Shuaicheng Liu,
    "FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via
    Consistency Flow Matching for Robot Manipulation", AAAI 2025.
    Role:
        Robot manipulation policy using 3D point-cloud observations and
        consistency flow matching, emphasizing fast / nearly one-step policy generation.
    arXiv:
        https://arxiv.org/abs/2412.04987
    Code:
        https://github.com/zql-kk/FlowPolicy

[5] Force-aware / contact-rich robot policy application:
    Tianyu Li, Yihan Li, Zizhe Zhang, Nadia Figueroa,
    "Flow with the Force Field: Learning 3D Compliant Flow Matching Policies
    from Force and Demonstration-Guided Simulation Data", arXiv 2025 / 2026.
    Role:
        Contact-rich robot policy using point cloud + force input and compliant
        flow matching. This paper motivates the force-aware extension direction,
        but this file intentionally keeps the current ACT-compatible 9D action
        and does not use virtual pose or impedance output.
    arXiv:
        https://arxiv.org/abs/2510.02738
    Project:
        https://flow-with-the-force-field.github.io/

Purpose
-------
First-stage Flow RGB baseline:
    observation:
        - cam0 RGB image
        - qpos/state 9D or 6/9D depending on dataset stats
        - optional force_history (L, 3)
    action:
        - current ACT-compatible 9D action chunk
          [x, y, z, wx, wy, wz, fx, fy, fz]

No virtual pose is used.
No impedance/compliance output is used.
Low-level admittance controller remains responsible for target-force tracking.

Training objective
------------------
Conditional Flow Matching / Rectified Flow style objective:

    z0 ~ N(0, I)
    z1 = action chunk
    t  ~ U(0, 1)
    zt = (1 - t) z0 + t z1
    u  = z1 - z0

    L = || v_theta(zt, t, obs) - u ||^2

Inference
---------
Euler integration:

    z <- N(0, I)
    for k in 0 ... K-1:
        z <- z + dt * v_theta(z, t_k, obs)

    predicted action chunk = z

Implementation note
-------------------
This file is a custom adaptation for the nrs_imitation HDF5 / ACT pipeline.
It is not a direct copy of any official Flow Matching, Rectified Flow,
PointFlowMatch, FlowPolicy, or Flow-with-the-Force-Field repository.

In this first-stage implementation, the goal is to compare:
    ACT-RGB-force baseline
    vs.
    FLOW-RGB-force baseline

under the same HDF5 data format, same 9D action space, and same low-level
admittance controller.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:
    from torchvision.models import resnet18  # type: ignore
    ResNet18_Weights = None  # type: ignore

import torchvision.transforms as T


# =============================================================================
# Small utilities
# =============================================================================

def _parse_down_dims(v) -> Tuple[int, ...]:
    if isinstance(v, str):
        return tuple(int(x.strip()) for x in v.split(",") if x.strip())
    if isinstance(v, Iterable):
        return tuple(int(x) for x in v)
    raise TypeError(f"Unsupported down_dims type: {type(v)}")


def _make_group_count(channels: int, requested_groups: int) -> int:
    g = min(int(requested_groups), int(channels))
    while g > 1 and (channels % g != 0):
        g -= 1
    return max(1, g)


# =============================================================================
# Time embedding and 1D U-Net blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float in [0, 1]
        return: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]

        scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Conv1dBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        g = _make_group_count(out_ch, n_groups)

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(g, out_ch),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """
    FiLM-conditioned residual block.

    x    : (B, C, T)
    cond : (B, cond_dim)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)

        cond_out = out_channels * 2 if self.cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_out),
        )

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        emb = self.cond_encoder(cond)

        if self.cond_predict_scale:
            emb = emb.view(emb.shape[0], 2, self.out_channels, 1)
            scale = emb[:, 0]
            bias = emb[:, 1]
            out = scale * out + bias
        else:
            out = out + emb[:, :, None]

        out = self.blocks[1](out)
        return out + self.residual_conv(x)


# =============================================================================
# Observation encoder
# =============================================================================

class FlowRGBObservationEncoder(nn.Module):
    """
    Encodes current observation into a global condition vector.

    qpos:
        (B, state_dim)
    image:
        (B, K, 3, H, W), cam0 is used
    force_history:
        optional (B, L, 3)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        state_dim = int(cfg.get("state_dim", 9))
        force_dim = int(cfg.get("force_dim", 3))
        obs_hidden_dim = int(cfg.get("flow_obs_hidden_dim", 256))
        image_feature_dim = int(cfg.get("flow_image_feature_dim", 512))
        global_cond_dim = int(cfg.get("flow_global_cond_dim", 256))

        pretrained_backbone = bool(cfg.get("pretrained_backbone", True))
        self.use_force_history = bool(cfg.get("use_force_history", True))

        self.qpos_encoder = nn.Sequential(
            nn.Linear(state_dim, obs_hidden_dim),
            nn.LayerNorm(obs_hidden_dim),
            nn.Mish(),
            nn.Linear(obs_hidden_dim, obs_hidden_dim),
            nn.Mish(),
        )

        if self.use_force_history:
            force_hidden_dim = int(cfg.get("force_encoder_hidden_dim", 64))
            force_num_layers = int(cfg.get("force_encoder_num_layers", 1))
            force_dropout = float(cfg.get("force_encoder_dropout", 0.0))
            self.force_gru = nn.GRU(
                input_size=force_dim,
                hidden_size=force_hidden_dim,
                num_layers=force_num_layers,
                batch_first=True,
                dropout=force_dropout if force_num_layers > 1 else 0.0,
            )
            force_out_dim = force_hidden_dim
        else:
            self.force_gru = None
            force_out_dim = 0

        if ResNet18_Weights is not None:
            weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
            backbone = resnet18(weights=weights)
        else:
            backbone = resnet18(pretrained=pretrained_backbone)

        backbone.fc = nn.Identity()
        self.image_backbone = backbone

        self.image_proj = nn.Sequential(
            nn.Linear(512, image_feature_dim),
            nn.LayerNorm(image_feature_dim),
            nn.Mish(),
        )

        fuse_dim = obs_hidden_dim + image_feature_dim + force_out_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_dim, global_cond_dim),
            nn.LayerNorm(global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim),
        )

        self.global_cond_dim = global_cond_dim

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image.dim() != 5:
            raise RuntimeError(f"Expected image shape (B,K,3,H,W), got {tuple(image.shape)}")

        q_feat = self.qpos_encoder(qpos)

        cam0 = image[:, 0]  # (B,3,H,W)
        img_feat = self.image_backbone(cam0)
        img_feat = self.image_proj(img_feat)

        feats = [q_feat, img_feat]

        if self.use_force_history and force_history is not None:
            _, h = self.force_gru(force_history)
            feats.append(h[-1])

        return self.fuse(torch.cat(feats, dim=-1))


# =============================================================================
# Conditional U-Net velocity field
# =============================================================================

class ConditionalUnet1D(nn.Module):
    """
    Conditional 1D U-Net for velocity field prediction.

    sample:
        (B, H, action_dim)
    t:
        (B,) float in [0,1]
    global_cond:
        (B, global_cond_dim)
    output:
        (B, H, action_dim)
    """

    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        time_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()

        down_dims = list(down_dims)
        if len(down_dims) < 2:
            raise ValueError("down_dims should contain at least 2 levels, e.g. 256,512,1024")

        cond_dim = time_embed_dim + global_cond_dim

        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        start_dim = down_dims[0]
        self.input_proj = Conv1dBlock(input_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups)

        # Down path.
        # For default down_dims=[256,512,1024] and horizon=200:
        # T: 200 -> 100 -> 50
        self.down_modules = nn.ModuleList()
        prev_dim = start_dim
        for i, dim_out in enumerate(down_dims):
            is_last = i == len(down_dims) - 1
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(prev_dim, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
            prev_dim = dim_out

        mid_dim = down_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        # Up path.
        # With skips [256,512,1024], bottleneck is 1024.
        # Up blocks:
        #   concat 1024 + 1024 -> 512, upsample 50->100
        #   concat 512 + 512   -> 256, upsample 100->200
        up_pairs = list(reversed(list(zip(down_dims[:-1], down_dims[1:]))))  # [(512,1024), (256,512)]

        self.up_modules = nn.ModuleList()
        for dim_out, dim_in in up_pairs:
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in * 2, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_out),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(self, sample: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = sample.moveaxis(-1, -2)  # (B,C,T)
        x = self.input_proj(x)

        t_emb = self.time_encoder(t)
        cond = torch.cat([t_emb, global_cond], dim=-1)

        skips = []
        for res1, res2, down in self.down_modules:
            x = res1(x, cond)
            x = res2(x, cond)
            skips.append(x)
            x = down(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for res1, res2, up in self.up_modules:
            skip = skips.pop()
            if x.shape[-1] != skip.shape[-1]:
                m = min(x.shape[-1], skip.shape[-1])
                x = x[..., :m]
                skip = skip[..., :m]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond)
            x = res2(x, cond)
            x = up(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


# =============================================================================
# Full Flow Matching Policy
# =============================================================================

class FlowRGBPolicy(nn.Module):
    """
    RGB-conditioned Flow Matching Policy.

    This class is directly usable by train_flow.py.
    """

    def __init__(self, cfg: dict):
        super().__init__()

        self.cfg = dict(cfg)
        self.num_queries = int(cfg.get("num_queries", 200))
        self.action_dim = int(cfg.get("action_dim", 9))
        self.flow_train_eps = float(cfg.get("flow_train_eps", 1e-4))
        self.flow_infer_steps = int(cfg.get("flow_infer_steps", 10))
        self.flow_loss_type = str(cfg.get("flow_loss_type", "mse")).lower()

        self.obs_encoder = FlowRGBObservationEncoder(cfg)

        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_encoder.global_cond_dim,
            time_embed_dim=int(cfg.get("flow_time_embed_dim", 256)),
            down_dims=_parse_down_dims(cfg.get("flow_down_dims", "256,512,1024")),
            kernel_size=int(cfg.get("flow_kernel_size", 5)),
            n_groups=int(cfg.get("flow_n_groups", 8)),
            cond_predict_scale=bool(cfg.get("flow_cond_predict_scale", False)),
        )

        self.image_normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: (B,K,3,H,W), usually already in [0,1].
        """
        B, K, C, H, W = image.shape
        flat = image.reshape(B * K, C, H, W)
        flat = self.image_normalize(flat)
        return flat.reshape(B, K, C, H, W)

    def _condition(self, qpos: torch.Tensor, image: torch.Tensor, force_history: Optional[torch.Tensor]):
        image = self._normalize_image(image)
        return self.obs_encoder(qpos=qpos, image=image, force_history=force_history)

    def predict_velocity(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cond = self._condition(qpos=qpos, image=image, force_history=force_history)
        return self.velocity_net(sample=z_t, t=t, global_cond=cond)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        if self.flow_loss_type == "l1":
            per = torch.abs(pred - target)
        else:
            per = (pred - target) ** 2

        valid_mask = (~is_pad).unsqueeze(-1).float()
        denom = valid_mask.sum().clamp_min(1.0) * pred.shape[-1]
        return (per * valid_mask).sum() / denom

    def forward(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
        force_history: Optional[torch.Tensor] = None,
    ):
        """
        Training:
            return loss dict.
        Inference:
            return sampled action chunk.
        """
        if actions is not None:
            assert is_pad is not None, "is_pad is required for training"

            z1 = actions[:, : self.num_queries]
            is_pad = is_pad[:, : self.num_queries]

            B = z1.shape[0]
            z0 = torch.randn_like(z1)

            # Avoid exact 0/1 for numerical stability.
            eps = self.flow_train_eps
            t = torch.rand(B, device=z1.device, dtype=z1.dtype) * (1.0 - 2.0 * eps) + eps

            t_view = t.view(B, 1, 1)
            z_t = (1.0 - t_view) * z0 + t_view * z1
            target_velocity = z1 - z0

            pred_velocity = self.predict_velocity(
                z_t=z_t,
                t=t,
                qpos=qpos,
                image=image,
                force_history=force_history,
            )

            loss = self._masked_loss(pred_velocity, target_velocity, is_pad)
            return {
                "flow": loss,
                "loss": loss,
            }

        return self.sample_action(
            qpos=qpos,
            image=image,
            force_history=force_history,
            num_steps=self.flow_infer_steps,
        )

    @torch.no_grad()
    def sample_action(
        self,
        qpos: torch.Tensor,
        image: torch.Tensor,
        force_history: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = int(num_steps or self.flow_infer_steps)
        steps = max(1, steps)

        B = qpos.shape[0]
        device = qpos.device
        dtype = qpos.dtype

        z = torch.randn(B, self.num_queries, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / float(steps)

        for k in range(steps):
            # midpoint Euler tends to be slightly more stable than using t=k/steps.
            t_val = (k + 0.5) / float(steps)
            t = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.predict_velocity(
                z_t=z,
                t=t,
                qpos=qpos,
                image=image,
                force_history=force_history,
            )
            z = z + dt * v

        return z


def build_flow_rgb_policy_and_optimizer(cfg: dict):
    model = FlowRGBPolicy(cfg)
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 1e-6))
    beta1 = float(cfg.get("beta1", 0.95))
    beta2 = float(cfg.get("beta2", 0.999))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))
    return model, optimizer
