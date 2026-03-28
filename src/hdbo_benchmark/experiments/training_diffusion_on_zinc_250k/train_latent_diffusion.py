from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import torch

from hdbo_benchmark.generative_models.latent_diffusion import (
    LatentDiffusionConfig,
    LatentDiffusionModel,
    save_latent_diffusion_checkpoint,
)
from hdbo_benchmark.generative_models.latent_diffusion_dataset import (
    build_latent_diffusion_dataloaders,
)
from hdbo_benchmark.generative_models.vae_factory import (
    infer_zinc_vae_latent_dim_from_checkpoint,
)
from hdbo_benchmark.utils.constants import DEVICE, MODELS_DIR, ROOT_DIR


def _validation_loss(
    model: LatentDiffusionModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for (batch,) in dataloader:
            batch = batch.to(device=device, dtype=torch.float32)
            losses.append(float(model.training_loss(batch).item()))
    return float(np.mean(losses))


@click.command()
@click.option(
    "--vae-checkpoint-path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
)
@click.option(
    "--dataset-root",
    type=click.Path(path_type=Path),
    default=ROOT_DIR / "data" / "small_molecule_datasets" / "processed",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=MODELS_DIR / "training_diffusion_on_zinc_250k",
)
@click.option("--seed", type=int, default=0)
@click.option("--batch-size", type=int, default=512)
@click.option("--epochs", type=int, default=200)
@click.option("--learning-rate", type=float, default=1e-3)
@click.option("--use-wandb/--no-use-wandb", default=False)
@click.option("--cache-latents/--no-cache-latents", default=True)
def main(
    vae_checkpoint_path: Path,
    dataset_root: Path,
    output_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    use_wandb: bool,
    cache_latents: bool,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    z_dim = infer_zinc_vae_latent_dim_from_checkpoint(vae_checkpoint_path)
    run_name = f"latent_diffusion_latent_dim-{z_dim}-seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_bundle = build_latent_diffusion_dataloaders(
        vae_checkpoint_path=vae_checkpoint_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
        seed=seed,
        batch_size=batch_size,
        device=DEVICE,
        cache_latents=cache_latents,
    )

    diffusion_config = LatentDiffusionConfig(
        z_dim=z_dim,
        hidden_size=min(256, max(64, z_dim)),
    )
    model = LatentDiffusionModel(
        config=diffusion_config,
        z_mean=data_bundle.z_mean,
        z_std=data_bundle.z_std,
    ).to(device=DEVICE, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = output_dir / f"{run_name}.pt"
    json_path = output_dir / f"{run_name}.json"

    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project="training_diffusion_on_zinc_250k",
            name=run_name,
            config={
                "vae_checkpoint_path": str(vae_checkpoint_path),
                "dataset_root": str(dataset_root),
                "output_dir": str(output_dir),
                "seed": seed,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "z_dim": z_dim,
                "T": diffusion_config.T,
                "beta_start": diffusion_config.beta_start,
                "beta_end": diffusion_config.beta_end,
                "hidden_size": diffusion_config.hidden_size,
                "depth": diffusion_config.depth,
                "time_embedding_dim": diffusion_config.time_embedding_dim,
                "cache_path": str(data_bundle.cache_path) if data_bundle.cache_path else None,
            },
        )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in data_bundle.train_loader:
            batch = batch.to(device=DEVICE, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))
        val_loss = _validation_loss(model, data_bundle.val_loader, DEVICE)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            model.eval()
            save_latent_diffusion_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                json_path=json_path,
            )

        print(
            f"Epoch {epoch}/{epochs}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, "
            f"best_val_loss={best_val_loss:.6f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "checkpoint_path": str(checkpoint_path),
                }
            )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
