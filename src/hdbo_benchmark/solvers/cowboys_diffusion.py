from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gpytorch
import numpy as np
import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement, LogProbabilityOfImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf_discrete
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

from hdbo_benchmark.generative_models.latent_diffusion import (
    LatentDiffusionModel,
    load_latent_diffusion_checkpoint,
)
from hdbo_benchmark.generative_models.vae import VAE
from hdbo_benchmark.utils.experiments.normalization import from_range_to_unit_cube

try:
    from botorch.acquisition.analytic import LogExpectedImprovement
except ImportError:
    LogExpectedImprovement = None


warnings.filterwarnings("ignore", message=".*contained to the unit cube")
RDLogger.DisableLog("rdApp.*")


@dataclass
class StructuredBOState:
    model: SingleTaskGP
    best_y: torch.Tensor
    acquisition: nn.Module
    observed_structures: list[str]


@dataclass
class DiffusionSamplingResult:
    sampled_latents: torch.Tensor
    num_rounds: int
    num_unique: int


class SafeLogExpectedImprovement(nn.Module):
    def __init__(self, model: SingleTaskGP, best_f: torch.Tensor, dtype: torch.dtype) -> None:
        super().__init__()
        self.base_acquisition = ExpectedImprovement(model, best_f=best_f)
        self.minimum = torch.finfo(dtype).tiny

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.base_acquisition(x).clamp_min(self.minimum))


class LatentCritic(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.network(latents).squeeze(-1)


class COWBOYSDiffusion(BaseBayesianOptimization):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        penalize_nans_with: float = -10.0,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        guide_mode: str = "distill",
        weight_type: str = "pi",
        num_candidates: int = 1000,
        num_diffusion_steps: int = 100,
        guidance_scale: float = 1.0,
        clip_guidance: float = 1.0,
    ) -> None:
        super().__init__(black_box, x0, y0)
        if guide_mode not in {"real", "distill"}:
            raise ValueError("guide_mode must be either 'real' or 'distill'.")
        if weight_type not in {"pi", "ei"}:
            raise ValueError("weight_type must be either 'pi' or 'ei'.")

        self.device = device
        self.dtype = torch.float64
        self.batch_size = batch_size
        self.guide_mode = guide_mode
        self.weight_type = weight_type
        self.num_candidates = num_candidates
        self.num_diffusion_steps = num_diffusion_steps
        self.guidance_scale = guidance_scale
        self.clip_guidance = clip_guidance
        self.penalize_nans_with = penalize_nans_with
        self.min_log_value = -1e8

        self._given_vae = False
        self._given_diffusion = False
        self.diffusion_model: LatentDiffusionModel | None = None

        self.critic: LatentCritic | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self._critic_learning_rate = 1e-3
        self._critic_steps = 64
        self._critic_batch_size = max(32, 2 * num_candidates)
        self._critic_aux_samples = max(32, 2 * num_candidates)

        self._sampling_batch_size = max(self.num_candidates, 4 * self.batch_size)
        self._max_sampling_rounds = 8
        self._real_guidance_directions = 4
        self._real_guidance_step_size = 0.15

    def _fit_model(
        self,
        model: type[SingleTaskGP],
        x: np.ndarray,
        y: np.ndarray,
    ) -> SingleTaskGP:
        x_tensor = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        y_tensor = torch.from_numpy(y).to(dtype=self.dtype, device=self.device)

        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ) and gpytorch.settings.fast_pred_var(state=False):
            model_instance = model(
                x_tensor,
                y_tensor,
                mean_module=None,
                covar_module=ScaleKernel(TanimotoKernel()),
            )
            mll = ExactMarginalLogLikelihood(model_instance.likelihood, model_instance)
            fit_gpytorch_mll(mll)
            model_instance.eval()

        self.gp_model_of_objective = model_instance
        return model_instance

    def next_candidate(self) -> np.ndarray:
        if not self._given_vae:
            raise ValueError("Need to pass in a VAE using `set_vae_and_bounds`.")
        if not self._given_diffusion or self.diffusion_model is None:
            raise ValueError(
                "Need to pass in a latent diffusion model using `set_diffusion_model` "
                "or `load_diffusion_model_from_checkpoint`."
            )

        observed_unit_latents, y = self.get_history_as_arrays()
        observed_latents = self._to_tensor(self._unit_to_latent(observed_unit_latents))
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        best_score_so_far = (
            float(np.nanmax(y)) if np.isfinite(y).any() else self.penalize_nans_with
        )
        print(
            f"collected at {len(observed_unit_latents)} points so far, "
            f"with best score so far as {best_score_so_far}"
        )

        bo_state = self._fit_structured_bo_state(observed_latents, y)
        if self.guide_mode == "distill":
            self._update_distill_critic(observed_latents, bo_state)

        sampling_result = self._sample_guided_latents(bo_state)
        print(
            f"did {sampling_result.num_rounds} diffusion rounds "
            f"({self.num_diffusion_steps} reverse steps each) and found "
            f"{sampling_result.num_unique} unique"
        )
        selected_latents = self._score_and_filter_candidates(
            sampling_result.sampled_latents,
            bo_state,
        )
        self._log_selected_candidates(selected_latents, bo_state)

        return self._latent_to_unit(selected_latents.detach().cpu().numpy())

    def set_vae_and_bounds(self, vae: VAE, vae_bounds: tuple[float, float]) -> None:
        self.vae = vae
        self.vae_bounds = vae_bounds
        self.bounds = vae_bounds
        self._given_vae = True

    def set_diffusion_model(self, diffusion_model: LatentDiffusionModel) -> None:
        self.diffusion_model = diffusion_model.to(device=self.device, dtype=self.dtype)
        self.diffusion_model.eval()
        self._given_diffusion = True
        self.critic = None
        self.critic_optimizer = None

    def load_diffusion_model_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        diffusion_model = load_latent_diffusion_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.set_diffusion_model(diffusion_model)

    def _fit_structured_bo_state(
        self,
        observed_latents: torch.Tensor,
        y: np.ndarray,
    ) -> StructuredBOState:
        sanitized_y = np.asarray(y, dtype=float).copy()
        sanitized_y[np.isnan(sanitized_y)] = self.penalize_nans_with

        observed_structures = self._latent_array_to_structure_list(observed_latents)
        observed_features, _ = self._structure_list_to_fingerprint_tensors(observed_structures)
        model = self._fit_model(SingleTaskGP, observed_features.cpu().numpy(), sanitized_y)
        best_y = torch.tensor(sanitized_y.max(), dtype=self.dtype, device=self.device)
        acquisition = self._build_acquisition(model, best_y)

        return StructuredBOState(
            model=model,
            best_y=best_y,
            acquisition=acquisition,
            observed_structures=observed_structures,
        )

    def _build_acquisition(self, model: SingleTaskGP, best_y: torch.Tensor) -> nn.Module:
        if self.weight_type == "pi":
            return LogProbabilityOfImprovement(model, best_f=best_y)

        if LogExpectedImprovement is not None:
            return LogExpectedImprovement(model, best_f=best_y)
        return SafeLogExpectedImprovement(model, best_f=best_y, dtype=self.dtype)

    def _update_distill_critic(
        self,
        observed_latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> None:
        if self.diffusion_model is None:
            return

        self._ensure_critic()
        assert self.critic is not None
        assert self.critic_optimizer is not None

        prior_latents = self.diffusion_model.unnormalize_latents(
            torch.randn(
                (self._critic_aux_samples, self.diffusion_model.z_dim),
                device=self.device,
                dtype=self.dtype,
            )
        )
        training_latents = torch.cat([observed_latents, prior_latents], dim=0)
        training_inputs = self.diffusion_model.normalize_latents(training_latents).detach()
        targets, _, _, _ = self._evaluate_log_weight(training_latents, bo_state)
        targets = targets.detach()

        self.critic.train()
        for _ in range(self._critic_steps):
            batch_size = min(self._critic_batch_size, training_inputs.shape[0])
            indices = torch.randint(
                training_inputs.shape[0],
                (batch_size,),
                device=self.device,
            )
            predictions = self.critic(training_inputs[indices])
            loss = F.mse_loss(predictions, targets[indices])
            self.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.critic_optimizer.step()
        self.critic.eval()

    def _ensure_critic(self) -> None:
        if self.diffusion_model is None:
            return
        if self.critic is not None:
            return

        hidden_dim = min(256, max(64, self.diffusion_model.z_dim))
        self.critic = LatentCritic(self.diffusion_model.z_dim, hidden_dim).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self._critic_learning_rate,
        )

    def _sample_guided_latents(
        self,
        bo_state: StructuredBOState,
    ) -> DiffusionSamplingResult:
        assert self.diffusion_model is not None
        observed_structures = set(bo_state.observed_structures)
        guidance_fn = self._build_guidance_function(bo_state)

        sampled_batches: list[torch.Tensor] = []
        num_rounds = 0
        for _ in range(self._max_sampling_rounds):
            num_rounds += 1
            normalized_latents = self.diffusion_model.sample(
                n_samples=self._sampling_batch_size,
                device=self.device,
                num_steps=self.num_diffusion_steps,
                guidance_fn=guidance_fn,
                guidance_scale=self.guidance_scale,
                clip_guidance=self.clip_guidance,
            )
            sampled_batches.append(
                self.diffusion_model.unnormalize_latents(normalized_latents).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )
            combined_latents = torch.cat(sampled_batches, dim=0)
            if (
                self._count_unique_valid_structures(
                    combined_latents,
                    observed_structures,
                )
                >= max(self.num_candidates, self.batch_size)
            ):
                return DiffusionSamplingResult(
                    sampled_latents=combined_latents,
                    num_rounds=num_rounds,
                    num_unique=self._count_unique_valid_structures(
                        combined_latents,
                        observed_structures,
                    ),
                )

        for _ in range(2):
            num_rounds += 1
            normalized_latents = self.diffusion_model.sample(
                n_samples=self._sampling_batch_size,
                device=self.device,
                num_steps=self.num_diffusion_steps,
                guidance_fn=None,
            )
            sampled_batches.append(
                self.diffusion_model.unnormalize_latents(normalized_latents).to(
                    device=self.device,
                    dtype=self.dtype,
                )
            )

        combined_latents = torch.cat(sampled_batches, dim=0)
        return DiffusionSamplingResult(
            sampled_latents=combined_latents,
            num_rounds=num_rounds,
            num_unique=self._count_unique_valid_structures(
                combined_latents,
                observed_structures,
            ),
        )

    def _build_guidance_function(self, bo_state: StructuredBOState) -> Any:
        if self.guide_mode == "real":
            return lambda normalized_latents: self._estimate_real_guidance(
                normalized_latents,
                bo_state,
            )
        return lambda normalized_latents: self._estimate_distill_guidance(
            normalized_latents
        )

    def _estimate_real_guidance(
        self,
        normalized_latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> torch.Tensor:
        assert self.diffusion_model is not None
        # The SELFIES -> SMILES -> RDKit fingerprint path is discrete, so we guide
        # with finite differences of the true structure-space acquisition instead
        # of pretending the decoder/fingerprint map is end-to-end differentiable.
        n_points, latent_dim = normalized_latents.shape
        directions = torch.randn(
            (n_points, self._real_guidance_directions, latent_dim),
            device=self.device,
            dtype=self.dtype,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        step_size = self._real_guidance_step_size

        plus = normalized_latents[:, None, :] + step_size * directions
        minus = normalized_latents[:, None, :] - step_size * directions
        plus_latents = self.diffusion_model.unnormalize_latents(
            plus.reshape(-1, latent_dim)
        ).to(device=self.device, dtype=self.dtype)
        minus_latents = self.diffusion_model.unnormalize_latents(
            minus.reshape(-1, latent_dim)
        ).to(device=self.device, dtype=self.dtype)

        log_weight_plus, _, _, _ = self._evaluate_log_weight(plus_latents, bo_state)
        log_weight_minus, _, _, _ = self._evaluate_log_weight(minus_latents, bo_state)
        directional_derivatives = (log_weight_plus - log_weight_minus).reshape(
            n_points, self._real_guidance_directions
        ) / (2.0 * step_size)

        guidance = (
            directional_derivatives.unsqueeze(-1) * directions
        ).mean(dim=1)
        return torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

    def _estimate_distill_guidance(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        if self.critic is None:
            return torch.zeros_like(normalized_latents)

        critic_inputs = normalized_latents.detach().clone().requires_grad_(True)
        critic_values = self.critic(critic_inputs).sum()
        guidance = torch.autograd.grad(critic_values, critic_inputs)[0]
        return torch.nan_to_num(guidance, nan=0.0, posinf=0.0, neginf=0.0)

    def _score_and_filter_candidates(
        self,
        candidate_latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> torch.Tensor:
        if candidate_latents.numel() == 0:
            return candidate_latents

        log_weight, structures, fingerprints, valid_mask = self._evaluate_log_weight(
            candidate_latents,
            bo_state,
        )
        seen_structures = set(bo_state.observed_structures)
        unique_indices: list[int] = []

        for idx, structure in enumerate(structures):
            if not bool(valid_mask[idx].item()):
                continue
            if structure in seen_structures:
                continue
            seen_structures.add(structure)
            unique_indices.append(idx)

        if not unique_indices:
            return candidate_latents[:1]

        unique_latents = candidate_latents[unique_indices]
        unique_features = fingerprints[unique_indices]
        unique_log_weight = log_weight[unique_indices]

        if unique_latents.shape[0] <= self.batch_size:
            return unique_latents

        if self.batch_size == 1:
            best_idx = int(torch.argmax(unique_log_weight).item())
            return unique_latents[best_idx : best_idx + 1]

        acquisition = qLogExpectedImprovement(bo_state.model, best_f=bo_state.best_y)
        chosen_features, _ = optimize_acqf_discrete(
            acquisition,
            min(self.batch_size, unique_features.shape[0]),
            unique_features,
            max_batch_size=1_000,
        )
        chosen_indices = self._match_selected_features(unique_features, chosen_features)
        return unique_latents[chosen_indices]

    def _evaluate_log_weight(
        self,
        latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
        structures = self._latent_array_to_structure_list(latents)
        fingerprints, valid_mask = self._structure_list_to_fingerprint_tensors(structures)
        with torch.no_grad():
            log_weight = bo_state.acquisition(fingerprints[:, None, :]).view(-1)

        log_weight = torch.nan_to_num(
            log_weight,
            nan=self.min_log_value,
            neginf=self.min_log_value,
            posinf=0.0,
        )
        log_weight = log_weight.to(dtype=self.dtype, device=self.device)
        log_weight[~valid_mask] = self.min_log_value
        return log_weight, structures, fingerprints, valid_mask

    def _log_selected_candidates(
        self,
        selected_latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> None:
        if selected_latents.numel() == 0:
            print("selected_candidate: none")
            return

        log_weight, structures, _, valid_mask = self._evaluate_log_weight(
            selected_latents,
            bo_state,
        )
        for idx, structure in enumerate(structures):
            validity = "valid" if bool(valid_mask[idx].item()) else "fallback_invalid"
            print(
                f"selected_candidate_{idx}: "
                f"log_weight={float(log_weight[idx].item()):.3f}, "
                f"status={validity}"
                #f"structure={self._structure_preview(structure)}"
            )

    def _latent_array_to_structure_list(self, latents: torch.Tensor) -> list[str]:
        decoded = self.vae.decode_to_string_array(latents.detach().cpu().numpy())
        return self._normalize_structure_array(decoded)

    def _normalize_structure_array(self, structures: np.ndarray) -> list[str]:
        normalized: list[str] = []
        for structure in np.asarray(structures, dtype=object):
            if isinstance(structure, str):
                normalized.append(structure)
                continue

            tokens = [str(token) for token in np.asarray(structure, dtype=object).tolist()]
            cleaned_tokens = [
                token for token in tokens if token not in {"<pad>", "<cls>", "<eos>"}
            ]
            normalized.append("".join(cleaned_tokens))
        return normalized

    def _structure_list_to_fingerprint_tensors(
        self,
        structures: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fallback_mol = Chem.MolFromSmiles("Cc1ccccc1")
        fingerprints = np.zeros((len(structures), 2048))
        valid_mask = torch.ones(len(structures), device=self.device, dtype=torch.bool)

        for idx, structure in enumerate(structures):
            mol = self._structure_to_mol(structure)
            if mol is None:
                valid_mask[idx] = False
                mol = fallback_mol

            fingerprint = rdMolDescriptors.GetMorganFingerprint(
                mol,
                radius=3,
                useCounts=True,
            )
            for key, value in fingerprint.GetNonzeroElements().items():
                fingerprints[idx, key % 2048] += value

        return (
            torch.tensor(fingerprints, device=self.device, dtype=self.dtype),
            valid_mask,
        )

    def _structure_to_mol(self, structure: str) -> Any:
        try:
            smiles = sf.decoder(structure)
        except Exception:
            return None
        return Chem.MolFromSmiles(smiles)

    def _count_unique_valid_structures(
        self,
        latents: torch.Tensor,
        observed_structures: set[str],
    ) -> int:
        if latents.numel() == 0:
            return 0

        structures = self._latent_array_to_structure_list(latents)
        n_unique = 0
        seen_structures = set(observed_structures)
        for structure in structures:
            if structure in seen_structures:
                continue
            if self._structure_to_mol(structure) is None:
                continue
            seen_structures.add(structure)
            n_unique += 1
        return n_unique

    def _structure_preview(self, structure: str, max_length: int = 80) -> str:
        if len(structure) <= max_length:
            return structure
        return f"{structure[: max_length - 3]}..."

    def _match_selected_features(
        self,
        candidate_features: torch.Tensor,
        chosen_features: torch.Tensor,
    ) -> list[int]:
        chosen_indices: list[int] = []
        available = torch.ones(
            candidate_features.shape[0],
            device=self.device,
            dtype=torch.bool,
        )

        for chosen_feature in chosen_features:
            distances = (candidate_features - chosen_feature.unsqueeze(0)).abs().sum(dim=1)
            distances[~available] = torch.inf
            chosen_index = int(torch.argmin(distances).item())
            chosen_indices.append(chosen_index)
            available[chosen_index] = False

        return chosen_indices

    def _unit_to_latent(self, unit_latents: np.ndarray) -> np.ndarray:
        lower, upper = self.vae_bounds
        return unit_latents * (upper - lower) + lower

    def _latent_to_unit(self, latents: np.ndarray) -> np.ndarray:
        return from_range_to_unit_cube(latents, self.vae_bounds)

    def _to_tensor(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return torch.tensor(values, device=self.device, dtype=self.dtype)
