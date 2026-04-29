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
from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_CANDIDATES,
    build_top_candidate_diagnostics,
    preserve_rng_state,
)

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
        distillation_n: int = 1024,
        guidance_scale: float = 1.0,
        clip_guidance: float = 1.0,
        guide_every: int = 1,
        guidance_alpha_bar_lower: float = 1e-4,
        guidance_alpha_bar_upper: float = 0.999,
        eta: float = 0.0,
    ) -> None:
        super().__init__(black_box, x0, y0)
        if guide_mode not in {"real", "distill"}:
            raise ValueError("guide_mode must be either 'real' or 'distill'.")
        if weight_type not in {"pi", "ei"}:
            raise ValueError("weight_type must be either 'pi' or 'ei'.")
        if distillation_n <= 0:
            raise ValueError("distillation_n must be a positive integer.")
        if eta < 0.0:
            raise ValueError("eta must be non-negative.")
        if guide_every <= 0:
            raise ValueError("guide_every must be a positive integer.")
        if not 0.0 <= guidance_alpha_bar_lower < guidance_alpha_bar_upper <= 1.0:
            raise ValueError(
                "guidance_alpha_bar_lower and guidance_alpha_bar_upper must satisfy "
                "0 <= lower < upper <= 1."
            )

        self.device = device
        self.dtype = torch.float64
        self.batch_size = batch_size
        self.guide_mode = guide_mode
        self.weight_type = weight_type
        self.num_candidates = num_candidates
        self.distillation_n = distillation_n
        self.guidance_scale = guidance_scale
        self.clip_guidance = clip_guidance
        self.guide_every = guide_every
        self.guidance_alpha_bar_lower = guidance_alpha_bar_lower
        self.guidance_alpha_bar_upper = guidance_alpha_bar_upper
        self.eta = eta
        self.penalize_nans_with = penalize_nans_with
        self.min_log_value = -1e8

        self._given_vae = False
        self._given_diffusion = False
        self.diffusion_model: LatentDiffusionModel | None = None

        self.critic: LatentCritic | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self._critic_learning_rate = 1e-3
        self._critic_steps = 64
        self._critic_batch_size = max(32, self.distillation_n)
        self._critic_aux_samples = self.distillation_n

        self._sampling_batch_size = max(self.num_candidates, self.batch_size)
        self._max_sampling_rounds = 8
        self._real_guidance_directions = 4
        self._real_guidance_step_size = 0.15
        self.last_selected_candidates: list[dict[str, Any]] = []
        self.pending_selected_candidates: list[dict[str, Any]] = []
        self._last_reported_history_size = int(np.asarray(x0).shape[0])
        self.iteration_sampling_metrics_history: list[dict[str, int]] = []
        self.iteration_candidate_diagnostics_history: list[dict[str, Any]] = []
        self._seen_sampled_decoded_structures: set[str] = set()

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
        self._report_pending_candidate_observations(observed_unit_latents, y)
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
        unique_valid_sampled_structures = self._unique_valid_structures(
            sampling_result.sampled_latents
        )
        unique_new_valid_structures = self._unique_valid_structures(
            sampling_result.sampled_latents,
            set(bo_state.observed_structures),
        )
        total_reverse_steps = int(self.diffusion_model.config.T) if self.diffusion_model is not None else -1
        print(
            f"did {sampling_result.num_rounds} diffusion rounds "
            f"({total_reverse_steps} reverse steps each, guide_every={self.guide_every}, "
            f"alpha_bar in ({self.guidance_alpha_bar_lower:g}, {self.guidance_alpha_bar_upper:g})) "
            f"and found "
            f"{len(unique_new_valid_structures)} unique"
        )
        self._record_sampling_metrics(
            sampled_structures=unique_valid_sampled_structures,
            unique_structures_not_in_observed_history=unique_new_valid_structures,
        )
        selected_latents, ranked_diagnostic_candidates = self._score_and_filter_candidates(
            sampling_result.sampled_latents,
            bo_state,
        )
        self._record_top_candidate_diagnostics(ranked_diagnostic_candidates)
        selected_candidate_metadata = self._build_selected_candidate_metadata(
            selected_latents,
            bo_state,
        )
        self._log_selected_candidates(selected_candidate_metadata)
        self.last_selected_candidates = selected_candidate_metadata
        self.pending_selected_candidates = [
            candidate.copy() for candidate in selected_candidate_metadata
        ]
        self._last_reported_history_size = int(len(observed_unit_latents))

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
                guidance_fn=guidance_fn,
                guidance_scale=self.guidance_scale,
                clip_guidance=self.clip_guidance,
                guide_every=self.guide_every,
                guidance_alpha_bar_lower=self.guidance_alpha_bar_lower,
                guidance_alpha_bar_upper=self.guidance_alpha_bar_upper,
                eta=self.eta,
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
                >= self.batch_size
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
                guidance_fn=None,
                guide_every=self.guide_every,
                guidance_alpha_bar_lower=self.guidance_alpha_bar_lower,
                guidance_alpha_bar_upper=self.guidance_alpha_bar_upper,
                eta=self.eta,
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
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        if candidate_latents.numel() == 0:
            return candidate_latents, []

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
            return candidate_latents[:1], []

        unique_latents = candidate_latents[unique_indices]
        unique_features = fingerprints[unique_indices]
        unique_log_weight = log_weight[unique_indices]
        unique_structures = [structures[idx] for idx in unique_indices]

        if unique_latents.shape[0] <= self.batch_size:
            selection_scores = self._evaluate_selection_scores(unique_features, bo_state)
            ranked_diagnostic_candidates = self._build_ranked_candidate_diagnostics(
                unique_latents,
                unique_structures,
                selection_scores,
                bo_state.observed_structures,
            )
            return unique_latents, ranked_diagnostic_candidates

        if self.batch_size == 1:
            best_idx = int(torch.argmax(unique_log_weight).item())
            selection_scores = self._evaluate_selection_scores(unique_features, bo_state)
            ranked_diagnostic_candidates = self._build_ranked_candidate_diagnostics(
                unique_latents,
                unique_structures,
                selection_scores,
                bo_state.observed_structures,
            )
            return unique_latents[best_idx : best_idx + 1], ranked_diagnostic_candidates

        acquisition = qLogExpectedImprovement(bo_state.model, best_f=bo_state.best_y)
        chosen_features, _ = optimize_acqf_discrete(
            acquisition,
            min(self.batch_size, unique_features.shape[0]),
            unique_features,
            max_batch_size=1_000,
        )
        chosen_indices = self._match_selected_features(unique_features, chosen_features)
        selection_scores = self._evaluate_selection_scores(unique_features, bo_state)
        ranked_diagnostic_candidates = self._build_ranked_candidate_diagnostics(
            unique_latents,
            unique_structures,
            selection_scores,
            bo_state.observed_structures,
        )
        return unique_latents[chosen_indices], ranked_diagnostic_candidates

    def _evaluate_selection_scores(
        self,
        candidate_features: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> torch.Tensor:
        with preserve_rng_state():
            acquisition = qLogExpectedImprovement(bo_state.model, best_f=bo_state.best_y)
            with torch.no_grad():
                selection_scores = acquisition(candidate_features[:, None, :]).view(-1)
        return torch.nan_to_num(
            selection_scores,
            nan=self.min_log_value,
            neginf=self.min_log_value,
            posinf=0.0,
        )

    def _build_ranked_candidate_diagnostics(
        self,
        candidate_latents: torch.Tensor,
        candidate_structures: list[str],
        selection_scores: torch.Tensor,
        observed_structures: list[str],
    ) -> list[dict[str, Any]]:
        if candidate_latents.numel() == 0 or not candidate_structures:
            return []

        unit_latents = self._latent_to_unit(candidate_latents.detach().cpu().numpy())
        ranked_indices = torch.argsort(selection_scores, descending=True).tolist()
        ranked_candidates: list[dict[str, Any]] = []
        seen_molecules = {
            canonical_smiles
            for structure in observed_structures
            if (canonical_smiles := self._canonical_smiles(structure)) is not None
        }

        for idx in ranked_indices:
            canonical_smiles = self._canonical_smiles(candidate_structures[idx])
            if canonical_smiles is None:
                continue
            if canonical_smiles in seen_molecules:
                continue

            seen_molecules.add(canonical_smiles)
            ranked_candidates.append(
                {
                    "source": "diffusion",
                    "structure": candidate_structures[idx],
                    "canonical_smiles": canonical_smiles,
                    "selection_score": float(selection_scores[idx].detach().cpu().item()),
                    "unit_latent": unit_latents[idx].tolist(),
                }
            )

        return ranked_candidates

    def _record_top_candidate_diagnostics(
        self,
        ranked_candidates: list[dict[str, Any]],
    ) -> None:
        try:
            diagnostics = build_top_candidate_diagnostics(
                self.black_box,
                ranked_candidates,
                top_k=TOP_K_CANDIDATES,
                evaluate_objectives=False,
            )
        except Exception as exc:
            print(f"Warning: could not record top-k diagnostic candidates: {exc}")
            diagnostics = {
                "top_k": int(TOP_K_CANDIDATES),
                "available_unique_top_10_candidate_count": 0,
                "mean_available_unique_top_10_candidate_objective": np.nan,
                "top_candidates": [],
            }

        diagnostics["iteration"] = len(self.iteration_candidate_diagnostics_history) + 1
        self.iteration_candidate_diagnostics_history.append(diagnostics)

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

    def _build_selected_candidate_metadata(
        self,
        selected_latents: torch.Tensor,
        bo_state: StructuredBOState,
    ) -> list[dict[str, Any]]:
        if selected_latents.numel() == 0:
            return []

        log_weight, structures, _, valid_mask = self._evaluate_log_weight(
            selected_latents,
            bo_state,
        )
        selected_unit_latents = self._latent_to_unit(selected_latents.detach().cpu().numpy())
        selected_candidates: list[dict[str, Any]] = []

        for idx, structure in enumerate(structures):
            validity = "valid" if bool(valid_mask[idx].item()) else "fallback_invalid"
            selected_candidates.append(
                {
                    "structure": structure,
                    "status": validity,
                    "log_weight": float(log_weight[idx].detach().cpu().item()),
                    "latent": selected_latents[idx].detach().cpu().numpy().tolist(),
                    "unit_latent": selected_unit_latents[idx].tolist(),
                }
            )

        return selected_candidates

    def _log_selected_candidates(self, selected_candidates: list[dict[str, Any]]) -> None:
        if not selected_candidates:
            print("selected_candidate: none")
            return

        for idx, candidate in enumerate(selected_candidates):
            print(
                f"selected_candidate_{idx}: "
                f"log_weight={candidate['log_weight']:.3f}, "
                f"status={candidate['status']}"
                #f"structure={self._structure_preview(candidate['structure'])}"
            )

    def _report_pending_candidate_observations(
        self,
        observed_unit_latents: np.ndarray,
        y: np.ndarray,
    ) -> None:
        if not self.pending_selected_candidates:
            self._last_reported_history_size = int(len(observed_unit_latents))
            return

        start_idx = min(self._last_reported_history_size, len(observed_unit_latents))
        new_unit_latents = np.asarray(observed_unit_latents[start_idx:], dtype=float)
        new_y = np.asarray(y, dtype=float).reshape(-1)[start_idx:]
        matched_new_indices: set[int] = set()
        remaining_candidates: list[dict[str, Any]] = []

        for candidate in self.pending_selected_candidates:
            matched_index = self._match_unit_latent(
                candidate["unit_latent"],
                new_unit_latents,
                matched_new_indices,
            )
            if matched_index is None:
                remaining_candidates.append(candidate)
                continue

            matched_new_indices.add(matched_index)
            observed_value = float(new_y[matched_index])
            candidate["observed_value"] = observed_value
            print(
                "evaluated_candidate: "
                f"observed_value={observed_value:.6g}, "
                f"log_weight={candidate['log_weight']:.3f}, "
                f"status={candidate['status']}"
                #f"structure={self._structure_preview(candidate['structure'])}"
            )

        self.pending_selected_candidates = remaining_candidates
        self._last_reported_history_size = int(len(observed_unit_latents))

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

    def _canonical_smiles(self, structure: str) -> str | None:
        mol = self._structure_to_mol(structure)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _count_unique_valid_structures(
        self,
        latents: torch.Tensor,
        observed_structures: set[str],
    ) -> int:
        return len(self._unique_valid_structures(latents, observed_structures))

    def _unique_valid_structures(
        self,
        latents: torch.Tensor,
        observed_structures: set[str] | None = None,
    ) -> list[str]:
        if latents.numel() == 0:
            return []

        structures = self._latent_array_to_structure_list(latents)
        unique_structures: list[str] = []
        seen_structures = set() if observed_structures is None else set(observed_structures)
        for structure in structures:
            if structure in seen_structures:
                continue
            if self._structure_to_mol(structure) is None:
                continue
            seen_structures.add(structure)
            unique_structures.append(structure)
        return unique_structures

    def _record_sampling_metrics(
        self,
        sampled_structures: list[str],
        unique_structures_not_in_observed_history: list[str],
    ) -> None:
        new_distinct_sampled_structures = [
            structure
            for structure in sampled_structures
            if structure not in self._seen_sampled_decoded_structures
        ]
        self._seen_sampled_decoded_structures.update(sampled_structures)
        self.iteration_sampling_metrics_history.append(
            {
                "iteration": len(self.iteration_sampling_metrics_history) + 1,
                "sample_unique_decoded_molecules_in_iteration": len(sampled_structures),
                "sample_new_distinct_decoded_molecules": len(
                    new_distinct_sampled_structures
                ),
                "sample_cumulative_distinct_decoded_molecules": len(
                    self._seen_sampled_decoded_structures
                ),
                "sample_unique_decoded_molecules_not_in_observed_history": len(
                    unique_structures_not_in_observed_history
                ),
            }
        )

    def _structure_preview(self, structure: str, max_length: int = 80) -> str:
        if len(structure) <= max_length:
            return structure
        return f"{structure[: max_length - 3]}..."

    def _match_unit_latent(
        self,
        unit_latent: list[float],
        observed_unit_latents: np.ndarray,
        excluded_indices: set[int],
    ) -> int | None:
        if observed_unit_latents.size == 0:
            return None

        target = np.asarray(unit_latent, dtype=float)
        for idx, observed in enumerate(observed_unit_latents):
            if idx in excluded_indices:
                continue
            if np.allclose(observed, target, atol=1e-10, rtol=1e-8):
                return idx
        return None

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
