from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from hdbo_benchmark.utils.constants import MODELS_DIR


@dataclass(slots=True)
class LatentDiffusionConfig:
    z_dim: int
    T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    hidden_size: int = 256
    depth: int = 3
    time_embedding_dim: int = 64


def _extract(buffer: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    values = buffer.gather(0, timesteps.long())
    while values.ndim < len(shape):
        values = values.unsqueeze(-1)
    return values


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        device = timesteps.device
        dtype = timesteps.dtype if timesteps.is_floating_point() else torch.float32
        frequencies = torch.arange(half_dim, device=device, dtype=dtype)
        frequencies = torch.exp(
            -math.log(10_000.0) * frequencies / max(half_dim - 1, 1)
        )
        angles = timesteps.to(dtype).unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])],
                dim=-1,
            )
        return embedding


class LatentDiffusionDenoiser(nn.Module):
    def __init__(self, config: LatentDiffusionConfig) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.time_embedding_dim),
        )

        layers: list[nn.Module] = []
        in_features = config.z_dim + config.time_embedding_dim
        for _ in range(config.depth):
            layers.append(nn.Linear(in_features, config.hidden_size))
            layers.append(nn.SiLU())
            in_features = config.hidden_size
        layers.append(nn.Linear(in_features, config.z_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_projection(self.time_embedding(timesteps))
        inputs = torch.cat([noisy_latents, time_embedding], dim=-1)
        return self.network(inputs)


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        config: LatentDiffusionConfig,
        z_mean: torch.Tensor | None = None,
        z_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.denoiser = LatentDiffusionDenoiser(config)

        betas = torch.linspace(config.beta_start, config.beta_end, config.T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer(
            "sqrt_one_minus_alpha_bars",
            torch.sqrt(1.0 - alpha_bars),
        )
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(1e-12))

        default_mean = torch.zeros(config.z_dim, dtype=torch.float32)
        default_std = torch.ones(config.z_dim, dtype=torch.float32)
        self.register_buffer("z_mean", default_mean)
        self.register_buffer("z_std", default_std)
        if z_mean is not None and z_std is not None:
            self.set_latent_stats(z_mean, z_std)

    @property
    def z_dim(self) -> int:
        return self.config.z_dim

    def set_latent_stats(self, z_mean: torch.Tensor, z_std: torch.Tensor) -> None:
        self.z_mean.copy_(z_mean.detach().to(device=self.z_mean.device, dtype=self.z_mean.dtype))
        self.z_std.copy_(
            z_std.detach()
            .to(device=self.z_std.device, dtype=self.z_std.dtype)
            .clamp_min(1e-6)
        )

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.z_mean) / self.z_std.clamp_min(1e-6)

    def unnormalize_latents(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        return normalized_latents * self.z_std + self.z_mean

    def forward(self, noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.denoiser(noisy_latents, timesteps)

    def q_sample(
        self,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_latents)
        return _extract(self.sqrt_alpha_bars, timesteps, clean_latents.shape) * clean_latents + _extract(
            self.sqrt_one_minus_alpha_bars, timesteps, clean_latents.shape
        ) * noise

    def predict_start_from_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar = _extract(self.sqrt_alpha_bars, timesteps, noisy_latents.shape)
        sqrt_one_minus = _extract(
            self.sqrt_one_minus_alpha_bars, timesteps, noisy_latents.shape
        )
        return (noisy_latents - sqrt_one_minus * predicted_noise) / sqrt_alpha_bar.clamp_min(
            1e-6
        )

    def training_loss(self, clean_latents: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(
            0,
            self.config.T,
            (clean_latents.shape[0],),
            device=clean_latents.device,
        )
        noise = torch.randn_like(clean_latents)
        noisy_latents = self.q_sample(clean_latents, timesteps, noise=noise)
        predicted_noise = self(noisy_latents, timesteps)
        return F.mse_loss(predicted_noise, noise)

    def build_sampling_schedule(self, num_steps: int | None = None) -> list[int]:
        if num_steps is None or num_steps >= self.config.T:
            return list(range(self.config.T - 1, -1, -1))

        steps = torch.linspace(self.config.T - 1, 0, num_steps)
        unique_steps = []
        for step in steps.round().to(dtype=torch.long).tolist():
            if step not in unique_steps:
                unique_steps.append(step)
        if unique_steps[-1] != 0:
            unique_steps.append(0)
        return unique_steps

    def _clip_guidance(self, guidance: torch.Tensor, clip_guidance: float | None) -> torch.Tensor:
        if clip_guidance is None or clip_guidance <= 0.0:
            return guidance

        norms = guidance.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        scales = torch.clamp(clip_guidance / norms, max=1.0)
        return guidance * scales

    def sample(
        self,
        n_samples: int,
        device: torch.device | None = None,
        num_steps: int | None = None,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        guidance_scale: float = 1.0,
        clip_guidance: float | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = device or self.z_mean.device
        dtype = self.z_mean.dtype
        if initial_noise is None:
            current = torch.randn((n_samples, self.z_dim), device=device, dtype=dtype)
        else:
            current = initial_noise.to(device=device, dtype=dtype)

        schedule = self.build_sampling_schedule(num_steps=num_steps)
        for step_idx, timestep in enumerate(schedule):
            prev_timestep = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
            timestep_batch = torch.full(
                (current.shape[0],),
                timestep,
                device=device,
                dtype=torch.long,
            )
            with torch.no_grad():
                predicted_noise = self(current, timestep_batch)
                predicted_start = self.predict_start_from_noise(
                    current,
                    timestep_batch,
                    predicted_noise,
                )

            if guidance_fn is not None and guidance_scale != 0.0:
                guidance = guidance_fn(predicted_start.detach())
                guidance = torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)
                guidance = self._clip_guidance(guidance, clip_guidance)
                predicted_start = predicted_start + guidance_scale * guidance

            if prev_timestep < 0:
                current = predicted_start
                continue

            alpha_bar_prev = self.alpha_bars[prev_timestep]
            current = torch.sqrt(alpha_bar_prev.clamp_min(1e-12)) * predicted_start + torch.sqrt(
                (1.0 - alpha_bar_prev).clamp_min(0.0)
            ) * predicted_noise

        return current

    def checkpoint_dict(self) -> dict[str, object]:
        return {
            "state_dict": self.state_dict(),
            "z_dim": self.config.z_dim,
            "T": self.config.T,
            "beta_start": self.config.beta_start,
            "beta_end": self.config.beta_end,
            "hidden_size": self.config.hidden_size,
            "depth": self.config.depth,
            "time_embedding_dim": self.config.time_embedding_dim,
            "z_mean": self.z_mean.detach().cpu(),
            "z_std": self.z_std.detach().cpu(),
        }


def save_latent_diffusion_checkpoint(
    model: LatentDiffusionModel,
    checkpoint_path: str | Path,
    json_path: str | Path | None = None,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = model.checkpoint_dict()
    torch.save(checkpoint, checkpoint_path)

    config_path = Path(json_path) if json_path is not None else checkpoint_path.with_suffix(".json")
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                key: value
                for key, value in {
                    **asdict(model.config),
                    "checkpoint_path": str(checkpoint_path),
                    "z_mean": model.z_mean.detach().cpu().tolist(),
                    "z_std": model.z_std.detach().cpu().tolist(),
                }.items()
            },
            fp,
            indent=2,
        )


def _strip_compiled_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }


def load_latent_diffusion_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> LatentDiffusionModel:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = LatentDiffusionConfig(
        z_dim=int(checkpoint["z_dim"]),
        T=int(checkpoint["T"]),
        beta_start=float(checkpoint["beta_start"]),
        beta_end=float(checkpoint["beta_end"]),
        hidden_size=int(checkpoint["hidden_size"]),
        depth=int(checkpoint["depth"]),
        time_embedding_dim=int(checkpoint["time_embedding_dim"]),
    )

    model = LatentDiffusionModel(
        config=config,
        z_mean=checkpoint["z_mean"].to(dtype=torch.float32),
        z_std=checkpoint["z_std"].to(dtype=torch.float32),
    )
    state_dict = _strip_compiled_prefix_from_state_dict(checkpoint["state_dict"])
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()
    return model


def resolve_default_latent_diffusion_checkpoint(
    latent_dim: int,
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir) if output_dir is not None else MODELS_DIR / "training_diffusion_on_zinc_250k"
    pattern = f"latent_diffusion_latent_dim-{latent_dim}-seed-*.pt"
    matching_checkpoints = sorted(output_dir.glob(pattern))
    if not matching_checkpoints:
        raise FileNotFoundError(
            "Could not find a pretrained latent diffusion checkpoint matching "
            f"{pattern} in {output_dir}."
        )

    return matching_checkpoints[0]
