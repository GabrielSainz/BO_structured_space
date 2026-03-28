from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from hdbo_benchmark.generative_models.vae_factory import load_zinc_vae_from_checkpoint
from hdbo_benchmark.utils.constants import DEVICE
from hdbo_benchmark.utils.data.zinc_250k import load_zinc_250k_dataset


@dataclass(slots=True)
class LatentDatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    z_mean: torch.Tensor
    z_std: torch.Tensor
    cache_path: Path | None
    num_examples: int


def resolve_default_latent_cache_path(
    vae_checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Path:
    vae_checkpoint_path = Path(vae_checkpoint_path)
    output_dir = Path(output_dir)
    return output_dir / f"latent_cache_{vae_checkpoint_path.stem}.npz"


def _encode_dataset_to_latents(
    vae_checkpoint_path: str | Path,
    dataset_root: str | Path | None,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    vae = load_zinc_vae_from_checkpoint(vae_checkpoint_path, device=device)
    vae.eval()

    onehot_dataset = load_zinc_250k_dataset(dataset_root=dataset_root)
    latent_batches = []
    for start_idx in range(0, len(onehot_dataset), batch_size):
        batch = torch.from_numpy(onehot_dataset[start_idx : start_idx + batch_size]).to(
            device=device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            latent_batches.append(vae.encode(batch).mean.detach().cpu())

    return torch.cat(latent_batches, dim=0)


def _load_or_build_latents(
    vae_checkpoint_path: str | Path,
    dataset_root: str | Path | None,
    output_dir: str | Path,
    batch_size: int,
    device: torch.device,
    cache_latents: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Path | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = resolve_default_latent_cache_path(vae_checkpoint_path, output_dir)

    if cache_latents and cache_path.exists():
        cached = np.load(cache_path)
        latents = torch.from_numpy(cached["latents"]).to(dtype=torch.float32)
        z_mean = torch.from_numpy(cached["z_mean"]).to(dtype=torch.float32)
        z_std = torch.from_numpy(cached["z_std"]).to(dtype=torch.float32)
        return latents, z_mean, z_std, cache_path

    latents = _encode_dataset_to_latents(
        vae_checkpoint_path=vae_checkpoint_path,
        dataset_root=dataset_root,
        batch_size=batch_size,
        device=device,
    ).to(dtype=torch.float32)
    z_mean = latents.mean(dim=0)
    z_std = latents.std(dim=0).clamp_min(1e-6)

    if cache_latents:
        np.savez_compressed(
            cache_path,
            latents=latents.numpy(),
            z_mean=z_mean.numpy(),
            z_std=z_std.numpy(),
            vae_checkpoint_path=str(Path(vae_checkpoint_path)),
        )
        return latents, z_mean, z_std, cache_path

    return latents, z_mean, z_std, None


def build_latent_diffusion_dataloaders(
    vae_checkpoint_path: str | Path,
    dataset_root: str | Path | None = None,
    output_dir: str | Path = "data/trained_models/training_diffusion_on_zinc_250k",
    seed: int = 0,
    batch_size: int = 512,
    device: torch.device = DEVICE,
    cache_latents: bool = True,
    validation_fraction: float = 0.1,
) -> LatentDatasetBundle:
    latents, z_mean, z_std, cache_path = _load_or_build_latents(
        vae_checkpoint_path=vae_checkpoint_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        batch_size=batch_size,
        device=device,
        cache_latents=cache_latents,
    )
    normalized_latents = (latents - z_mean) / z_std.clamp_min(1e-6)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(normalized_latents.shape[0])
    n_val = int(validation_fraction * normalized_latents.shape[0])
    n_val = min(max(n_val, 1), max(normalized_latents.shape[0] - 1, 1))

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    if len(train_indices) == 0:
        train_indices = val_indices

    train_dataset = TensorDataset(normalized_latents[train_indices])
    val_dataset = TensorDataset(normalized_latents[val_indices])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return LatentDatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        z_mean=z_mean,
        z_std=z_std,
        cache_path=cache_path,
        num_examples=int(normalized_latents.shape[0]),
    )
