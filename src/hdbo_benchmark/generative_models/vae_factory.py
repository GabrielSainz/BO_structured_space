"""
Defines a common way of loading VAEs from just the name
of the experiment and the latent dimension.
"""

from pathlib import Path
from typing import Literal

import torch

from hdbo_benchmark.assets import __file__ as ASSETS_PATH
from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.generative_models.vae_mario import VAEMario
from hdbo_benchmark.generative_models.vae_selfies import VAESelfies
from hdbo_benchmark.utils.constants import DEVICE, MODELS_DIR

ASSETS_DIR = Path(ASSETS_PATH).parent


def _strip_compiled_prefix_from_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict

    return {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }


def infer_zinc_vae_latent_dim_from_checkpoint(
    checkpoint_path: str | Path,
    map_location: torch.device | str = "cpu",
) -> int:
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"The VAE checkpoint at {checkpoint_path} is not a state_dict checkpoint."
        )

    cleaned_state_dict = _strip_compiled_prefix_from_state_dict(state_dict)
    if "encoder_mu.weight" not in cleaned_state_dict:
        raise ValueError(
            "Could not infer the Zinc VAE latent dimensionality from checkpoint "
            f"{checkpoint_path}."
        )

    latent_dim = int(cleaned_state_dict["encoder_mu.weight"].shape[0])
    return latent_dim


def load_zinc_vae_from_checkpoint(
    checkpoint_path: str | Path,
    latent_dim: int | None = None,
    device: torch.device = DEVICE,
) -> VAESelfies:
    checkpoint_path = Path(checkpoint_path)
    resolved_latent_dim = latent_dim or infer_zinc_vae_latent_dim_from_checkpoint(
        checkpoint_path, map_location=device
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"The VAE checkpoint at {checkpoint_path} is not a state_dict checkpoint."
        )

    vae = VAESelfies(latent_dim=resolved_latent_dim, device=device)
    vae.load_state_dict(_strip_compiled_prefix_from_state_dict(state_dict))
    vae.eval()
    return vae


class VAEFactory:
    def create(
        self,
        experiment_name: Literal["benchmark_on_mario", "benchmark_on_pmo"],
        latent_dim: int,
    ) -> VAE:
        match experiment_name:
            case "benchmark_on_pmo":
                vae = self._create_vae_on_molecules(latent_dim)
            case "benchmark_on_mario":
                vae = self._create_vae_on_mario(latent_dim)
            case _:
                raise ValueError("...")

        return vae

    def _create_vae_on_mario(self, latent_dim: int) -> VAEMario:
        MODELS_DIR = ASSETS_DIR / "training_vae_on_mario"
        match latent_dim:
            case 2:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-2-seed-2"
                    / "latent_dim-2-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 16:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-16-seed-2"
                    / "latent_dim-16-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 64:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-64-seed-3"
                    / "latent_dim-64-batch_size-256-lr-0.001-seed-3.pt"
                )
            case 256:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-256-seed-3"
                    / "latent_dim-256-batch_size-256-lr-0.001-seed-3.pt"
                )
            case 512:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-512-seed-2"
                    / "latent_dim-512-batch_size-256-lr-0.001-seed-2.pt"
                )
            case 1024:
                weights_path = (
                    MODELS_DIR
                    / "vae_mario-latent_dim-1024-seed-1"
                    / "latent_dim-1024-batch_size-256-lr-0.001-seed-1.pt"
                )
            case _:
                raise NotImplementedError

        vae = VAEMario(latent_dim=latent_dim, device=DEVICE)
        opt_vae: VAEMario = torch.compile(vae)
        opt_vae.load_state_dict(torch.load(weights_path, map_location=DEVICE))
        return opt_vae

    def _create_vae_on_molecules(self, latent_dim: int) -> VAESelfies:
        candidate_dirs = [
            MODELS_DIR / "training_vae_on_zinc_250k",
            ASSETS_DIR / "training_vae_on_zinc_250k",
        ]
        match latent_dim:
            case 2:
                filename = "latent_dim-2-batch_size-512-lr-0.0005-seed-1.pt"
            case 64:
                filename = "latent_dim-64-batch_size-512-lr-0.0005-seed-0.pt"
            case 128:
                filename = "latent_dim-128-batch_size-512-lr-0.0005-seed-1.pt"
            case _:
                raise NotImplementedError

        weights_path = next(
            (models_dir / filename for models_dir in candidate_dirs if (models_dir / filename).exists()),
            None,
        )
        if weights_path is None:
            raise FileNotFoundError(
                f"Could not find a pretrained Zinc VAE checkpoint for latent_dim={latent_dim}."
            )

        return load_zinc_vae_from_checkpoint(
            checkpoint_path=weights_path,
            latent_dim=latent_dim,
            device=DEVICE,
        )
