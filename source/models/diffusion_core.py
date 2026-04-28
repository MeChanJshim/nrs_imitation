#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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


def _parse_down_dims(v) -> Tuple[int, ...]:
    if isinstance(v, str):
        return tuple(int(x.strip()) for x in v.split(",") if x.strip())
    if isinstance(v, Iterable):
        return tuple(int(x) for x in v)
    raise TypeError(f"Unsupported down_dims type: {type(v)}")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return x[:, None]
        emb_scale = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Conv1dBlock(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int, kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        padding = kernel_size // 2
        g = min(n_groups, out_dim)
        while out_dim % g != 0 and g > 1:
            g -= 1
        self.block = nn.Sequential(
            nn.Conv1d(inp_dim, out_dim, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(g, out_dim),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8, cond_predict_scale: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = bool(cond_predict_scale)
        self.out_channels = int(out_channels)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.view(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed[:, :, None]
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class DiffusionObservationEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        qpos_dim = int(cfg.get("state_dim", 9))
        force_dim = int(cfg.get("force_dim", 3))
        force_hidden_dim = int(cfg.get("force_encoder_hidden_dim", 64))
        force_num_layers = int(cfg.get("force_encoder_num_layers", 1))
        force_dropout = float(cfg.get("force_encoder_dropout", 0.0))
        obs_hidden_dim = int(cfg.get("diffusion_obs_hidden_dim", 256))
        image_feature_dim = int(cfg.get("diffusion_image_feature_dim", 512))
        global_cond_dim = int(cfg.get("diffusion_global_cond_dim", 256))
        pretrained_backbone = bool(cfg.get("pretrained_backbone", True))

        self.qpos_encoder = nn.Sequential(
            nn.Linear(qpos_dim, obs_hidden_dim),
            nn.LayerNorm(obs_hidden_dim),
            nn.Mish(),
            nn.Linear(obs_hidden_dim, obs_hidden_dim),
            nn.Mish(),
        )

        self.use_force_history = bool(cfg.get("use_force_history", False))
        if self.use_force_history:
            self.force_gru = nn.GRU(
                input_size=force_dim,
                hidden_size=force_hidden_dim,
                num_layers=force_num_layers,
                batch_first=True,
                dropout=(force_dropout if force_num_layers > 1 else 0.0),
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

        in_dim = obs_hidden_dim + force_out_dim + image_feature_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, global_cond_dim),
            nn.LayerNorm(global_cond_dim),
            nn.Mish(),
            nn.Linear(global_cond_dim, global_cond_dim),
        )
        self.global_cond_dim = global_cond_dim

    def forward(self, qpos: torch.Tensor, image: torch.Tensor, force_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.qpos_encoder(qpos)
        if image.dim() != 5:
            raise RuntimeError(f"Expected image (B,K,3,H,W), got {tuple(image.shape)}")
        cam0 = image[:, 0]
        img_feat = self.image_proj(self.image_backbone(cam0))
        feats = [q, img_feat]
        if self.use_force_history and force_history is not None:
            _, h = self.force_gru(force_history)
            feats.append(h[-1])
        return self.fuse(torch.cat(feats, dim=-1))


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, diffusion_step_embed_dim: int = 256,
                 down_dims: Sequence[int] = (256, 512, 1024), kernel_size: int = 5,
                 n_groups: int = 8, cond_predict_scale: bool = False):
        super().__init__()
        down_dims = list(down_dims)
        all_dims = [input_dim] + down_dims
        start_dim = down_dims[0]

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        self.input_proj = Conv1dBlock(input_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups)

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList()
        prev_dim = start_dim
        for ind, (_, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(prev_dim, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))
            prev_dim = dim_out

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale),
        ])

        rev_in_out = list(zip(reversed(all_dims[1:]), reversed(all_dims[:-1])))
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(rev_in_out[1:]):
            is_last = ind >= (len(rev_in_out[1:]) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in * 2, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale),
                Upsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = sample.moveaxis(-1, -2)
        x = self.input_proj(x)
        t_emb = self.diffusion_step_encoder(timestep)
        cond = torch.cat([t_emb, global_cond], dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            h.append(x)
            x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for resnet, resnet2, upsample in self.up_modules:
            skip = h.pop()
            if x.shape[-1] != skip.shape[-1]:
                m = min(x.shape[-1], skip.shape[-1])
                x = x[..., :m]
                skip = skip[..., :m]
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = upsample(x)

        return self.final_conv(x).moveaxis(-1, -2)


class DiffusionPolicyCore(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        action_dim = int(cfg.get("action_dim", 9))
        time_dim = int(cfg.get("diffusion_time_embed_dim", 256))
        down_dims = _parse_down_dims(cfg.get("diffusion_down_dims", (256, 512, 1024)))
        kernel_size = int(cfg.get("diffusion_kernel_size", 5))
        n_groups = int(cfg.get("diffusion_n_groups", 8))
        cond_predict_scale = bool(cfg.get("diffusion_cond_predict_scale", False))

        self.obs_encoder = DiffusionObservationEncoder(cfg)
        self.unet = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=self.obs_encoder.global_cond_dim,
            diffusion_step_embed_dim=time_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor,
                qpos: torch.Tensor, image: torch.Tensor,
                force_history: Optional[torch.Tensor] = None) -> torch.Tensor:
        gc = self.obs_encoder(qpos=qpos, image=image, force_history=force_history)
        return self.unet(sample=noisy_actions, timestep=timesteps, global_cond=gc)


def build_DIFFUSION_model_and_optimizer(args_override: dict):
    model = DiffusionPolicyCore(args_override)
    lr = float(args_override.get("lr", 1e-4))
    weight_decay = float(args_override.get("weight_decay", 1e-6))
    betas = tuple(args_override.get("optimizer_betas", (0.95, 0.999)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    return model, optimizer