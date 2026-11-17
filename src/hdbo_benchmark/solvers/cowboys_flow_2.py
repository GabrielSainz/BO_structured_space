"""
Flow-assisted COWBOYS solver.
"""

from __future__ import annotations

import math
import warnings
import zlib
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import selfies as sf
import torch
import torch.nn as nn
from botorch.acquisition import qLogExpectedImprovement
from botorch.acquisition.analytic import LogProbabilityOfImprovement
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

from hdbo_benchmark.utils.experiments.normalization import (
    from_range_to_unit_cube,
)
from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_CANDIDATES,
    build_top_candidate_diagnostics,
    preserve_rng_state,
)

warnings.filterwarnings("ignore", message=".*contained to the unit cube")
RDLogger.DisableLog("rdApp.*")


@dataclass
class StructuredBOState:
    model: SingleTaskGP
    best_y: torch.Tensor
    acquisition: LogProbabilityOfImprovement
    observed_structures: list[str]


@dataclass
class LatentTargetEvaluation:
    log_target: torch.Tensor
    log_prior: torch.Tensor
    log_weight: torch.Tensor
    structures: list[str]
    fingerprints: torch.Tensor


@dataclass
class WeightedLatentPool:
    latents: torch.Tensor
    normalized_weights: torch.Tensor
    log_weights: torch.Tensor
    structures: list[str]
    fingerprints: torch.Tensor


@dataclass
class MHSamplingResult:
    sampled_latents: torch.Tensor
    sampled_sources: list[str]
    accepted_latents: torch.Tensor
    acceptance_rate: float
    proposal_diagnostics: "ProposalDiagnostics"


@dataclass
class ProposalDiagnostics:
    n_local_proposals: int
    n_global_proposals: int
    n_local_accepted: int
    n_global_accepted: int
    n_unique_local_proposed_structures: int
    n_unique_global_proposed_structures: int
    n_unique_local_accepted_structures: int
    n_unique_global_accepted_structures: int

    @property
    def local_acceptance_rate(self) -> float:
        return self.n_local_accepted / max(self.n_local_proposals, 1)

    @property
    def global_acceptance_rate(self) -> float:
        return self.n_global_accepted / max(self.n_global_proposals, 1)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n_local_proposals": self.n_local_proposals,
            "n_global_proposals": self.n_global_proposals,
            "n_local_accepted": self.n_local_accepted,
            "n_global_accepted": self.n_global_accepted,
            "local_acceptance_rate": self.local_acceptance_rate,
            "global_acceptance_rate": self.global_acceptance_rate,
            "n_unique_local_proposed_structures": self.n_unique_local_proposed_structures,
            "n_unique_global_proposed_structures": self.n_unique_global_proposed_structures,
            "n_unique_local_accepted_structures": self.n_unique_local_accepted_structures,
            "n_unique_global_accepted_structures": self.n_unique_global_accepted_structures,
        }


class AffineCouplingLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )

    def _scale_and_shift(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale, shift = self.net(inputs).chunk(2, dim=-1)
        scale = 0.8 * torch.tanh(scale)
        scale = scale * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        return scale, shift

    def forward_to_base(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = z * self.mask
        scale, shift = self._scale_and_shift(masked)
        transformed = masked + (1.0 - self.mask) * (z * torch.exp(scale) + shift)
        log_det = scale.sum(dim=-1)
        return transformed, log_det

    def inverse_from_base(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = u * self.mask
        scale, shift = self._scale_and_shift(masked)
        transformed = masked + (1.0 - self.mask) * ((u - shift) * torch.exp(-scale))
        log_det = (-scale).sum(dim=-1)
        return transformed, log_det


class WeightedRealNVPProposal(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        depth: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        base_mask = (torch.arange(latent_dim) % 2 == 0).to(device=device, dtype=dtype)
        masks = [base_mask if i % 2 == 0 else 1.0 - base_mask for i in range(depth)]
        self.layers = nn.ModuleList(
            [AffineCouplingLayer(latent_dim, hidden_dim, mask) for mask in masks]
        )
        self.register_buffer("base_loc", torch.zeros(latent_dim, device=device, dtype=dtype))
        self.register_buffer("base_scale", torch.ones(latent_dim, device=device, dtype=dtype))
        self.to(device=device, dtype=dtype)
        self.eval()

    def _base_dist(self) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.base_loc, self.base_scale)

    def to_base(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = z
        total_log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for layer in self.layers:
            transformed, log_det = layer.forward_to_base(transformed)
            total_log_det = total_log_det + log_det
        return transformed, total_log_det

    def from_base(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        transformed = u
        total_log_det = torch.zeros(u.shape[0], device=u.device, dtype=u.dtype)
        for layer in reversed(self.layers):
            transformed, log_det = layer.inverse_from_base(transformed)
            total_log_det = total_log_det + log_det
        return transformed, total_log_det

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        base_inputs, log_det = self.to_base(z)
        return self._base_dist().log_prob(base_inputs).sum(dim=-1) + log_det

    def sample(self, n_samples: int) -> torch.Tensor:
        base_samples = self._base_dist().sample((n_samples,))
        transformed, _ = self.from_base(base_samples)
        return transformed

    def fit_weighted(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
        n_steps: int,
        learning_rate: float = 1e-3,
    ) -> None:
        if points.numel() == 0:
            return

        normalized_weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train()

        for _ in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            # Weighted MLE matches KL(\hat{pi}_t || q_phi) up to a phi-independent constant.
            loss = -(normalized_weights * self.log_prob(points)).sum()
            loss.backward()
            optimizer.step()

        self.eval()


class FlowMixtureProposal:
    def __init__(
        self,
        flow: WeightedRealNVPProposal,
        local_step_size: float,
        local_weight: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.flow = flow
        self.local_step_size = local_step_size
        self.local_weight = float(np.clip(local_weight, 0.0, 1.0))
        self.device = device
        self.dtype = dtype

    def propose(self, current: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_chains, latent_dim = current.shape
        proposals = self.flow.sample(n_chains)
        if self.local_weight <= 0.0:
            return proposals, torch.zeros(n_chains, device=self.device, dtype=torch.bool)

        use_local = torch.rand(n_chains, device=self.device) < self.local_weight
        if use_local.any():
            local_noise = torch.randn(
                (int(use_local.sum().item()), latent_dim),
                device=self.device,
                dtype=self.dtype,
            )
            proposals[use_local] = current[use_local] + self.local_step_size * local_noise
        return proposals, use_local

    def log_prob(self, proposed: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        local_log_prob = self._local_log_prob(proposed, current)
        global_log_prob = self.flow.log_prob(proposed)

        if self.local_weight <= 0.0:
            return global_log_prob
        if self.local_weight >= 1.0:
            return local_log_prob

        log_local_weight = torch.full_like(local_log_prob, math.log(self.local_weight))
        log_global_weight = torch.full_like(
            global_log_prob, math.log(1.0 - self.local_weight)
        )
        return torch.logaddexp(
            log_local_weight + local_log_prob,
            log_global_weight + global_log_prob,
        )

    def _local_log_prob(self, proposed: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        diff = (proposed - current) / self.local_step_size
        component_log_prob = (
            -0.5 * diff.pow(2)
            - math.log(self.local_step_size)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return component_log_prob.sum(dim=-1)

class COWBOYSFlow(BaseBayesianOptimization):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        batch_size: int = 1,
        num_chains: int = 32,
        n_mh_steps: int = 30,
        burn_in: int = 0, # 0 , 50
        local_step_size: float = 0.15,
        local_proposal_weight: float = 0.5, # 0.3 , 0.5
        pool_size: int = 512, # 256, 128
        flow_training_steps: int = 50, # 100, 50
        flow_hidden_dim: int | None = None,
        flow_depth: int = 4, # 6, 4
        penalize_nans_with: float = -10.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(black_box, x0, y0)
        self.device = device
        self.dtype = torch.float64
        self.batch_size = batch_size # how many candidates are returned per BO iteration
        self.num_chains = num_chains # number of parallel MH chains to run
        self.n_mh_steps = n_mh_steps # total MH proposal budget per BO iteration
        self.burn_in = burn_in # number of initial MH steps to discard as burn-in
        self.local_step_size = local_step_size # scale of the exact local Gaussian proposal - sigma (z' = z + sigma * eps)
        self.local_proposal_weight = local_proposal_weight # local/global balance in the mixture proposal
        self.pool_size = pool_size # how many latent points are used to build the weighted pool for flow fitting
        self.flow_training_steps = flow_training_steps # how long the global proposal is fit to the weighted pool each iteration
        self.flow_hidden_dim = flow_hidden_dim
        self.flow_depth = flow_depth
        self.penalize_nans_with = penalize_nans_with
        self.min_log_value = -1e8
        self._given_vae = False
        self._flow_is_initialized = False
        self.accepted_latent_history = torch.zeros((0, 0), device=self.device, dtype=self.dtype)
        self.last_sampling_diagnostics: dict[str, float | int] = {}
        self.last_selected_candidates: list[dict[str, Any]] = []
        self.pending_selected_candidates: list[dict[str, Any]] = []
        self.iteration_sampling_metrics_history: list[dict[str, int]] = []
        self.iteration_candidate_diagnostics_history: list[dict[str, Any]] = []
        self._seen_sampled_decoded_structures: set[str] = set()
        self._last_reported_history_size = int(np.asarray(x0).shape[0])

    def _fit_model(
        self, model: type[SingleTaskGP], x: np.ndarray, y: np.ndarray
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
            raise ValueError("Need to pass in a generative model using `set_vae_and_bounds`.")

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

        if not self._flow_is_initialized:
            pool_latents = self._build_step_zero_pool(observed_latents, y.reshape(-1))
        else:
            pool_latents = self._build_iteration_pool(observed_latents, y.reshape(-1))

        weighted_pool = self._weight_latent_pool(pool_latents, bo_state)
        self._fit_flow_from_weighted_pool(weighted_pool)
        self._flow_is_initialized = True

        initial_states = self._initialize_chains(observed_latents, y.reshape(-1))
        mh_result = self._run_mh_sampling(bo_state, initial_states)
        sampled_structures = self._decode_latents(mh_result.sampled_latents)
        unique_new_structures = self._unique_new_structures(
            sampled_structures,
            bo_state.observed_structures,
        )
        n_unique = len(unique_new_structures)
        print(f"did {self.n_mh_steps} steps and found {n_unique} unique")
        self._record_sampling_metrics(
            sampled_structures=sampled_structures,
            unique_structures_not_in_observed_history=unique_new_structures,
        )
        self._update_replay_buffer(mh_result.accepted_latents)
        print(
            "mcmc diagnostics: "
            f"acceptance_rate={mh_result.acceptance_rate:.3f}, "
            f"accepted_moves={mh_result.accepted_latents.shape[0]}, "
            f"post_burn_samples={mh_result.sampled_latents.shape[0]}, "
            f"replay_buffer={self.accepted_latent_history.shape[0]}"
        )
        print(
            "proposal diagnostics: "
            f"local_proposals={mh_result.proposal_diagnostics.n_local_proposals}, "
            f"global_proposals={mh_result.proposal_diagnostics.n_global_proposals}, "
            f"local_accepted={mh_result.proposal_diagnostics.n_local_accepted}, "
            f"global_accepted={mh_result.proposal_diagnostics.n_global_accepted}, "
            f"local_acceptance_rate={mh_result.proposal_diagnostics.local_acceptance_rate:.3f}, "
            f"global_acceptance_rate={mh_result.proposal_diagnostics.global_acceptance_rate:.3f}, "
            f"unique_local_proposed={mh_result.proposal_diagnostics.n_unique_local_proposed_structures}, "
            f"unique_global_proposed={mh_result.proposal_diagnostics.n_unique_global_proposed_structures}, "
            f"unique_local_accepted={mh_result.proposal_diagnostics.n_unique_local_accepted_structures}, "
            f"unique_global_accepted={mh_result.proposal_diagnostics.n_unique_global_accepted_structures}"
        )

        (
            selected_latents,
            selected_candidate_metadata,
            ranked_diagnostic_candidates,
        ) = self._select_candidates(
            mh_result.sampled_latents,
            mh_result.sampled_sources,
            weighted_pool,
            bo_state,
        )
        self._record_top_candidate_diagnostics(ranked_diagnostic_candidates)
        selected_candidate_metadata = self._build_selected_candidate_metadata(
            selected_latents,
            selected_candidate_metadata,
            bo_state,
        )
        self._log_selected_candidates(selected_candidate_metadata)
        self.last_selected_candidates = selected_candidate_metadata
        self.pending_selected_candidates = [candidate.copy() for candidate in selected_candidate_metadata]

        self.last_sampling_diagnostics = {
            "acceptance_rate": mh_result.acceptance_rate,
            "accepted_moves": int(mh_result.accepted_latents.shape[0]),
            "num_pool_latents": int(weighted_pool.latents.shape[0]),
            "num_mh_samples": int(mh_result.sampled_latents.shape[0]),
            "num_replay_latents": int(self.accepted_latent_history.shape[0]),
            **mh_result.proposal_diagnostics.to_dict(),
        }

        self._last_reported_history_size = int(len(observed_unit_latents))
        return self._latent_to_unit(selected_latents.detach().cpu().numpy())

    def set_vae_and_bounds(self, vae: Any, vae_bounds: tuple[float, float]) -> None:
        self.vae = vae
        self.vae_bounds = vae_bounds
        self.bounds = vae_bounds
        self._given_vae = True
        self._flow_is_initialized = False
        self._initialize_flow_objects()

    def _initialize_flow_objects(self) -> None:
        latent_dim = int(self.vae.latent_dim)
        hidden_dim = self.flow_hidden_dim or min(256, max(64, latent_dim))
        self.latent_prior = torch.distributions.Normal(
            torch.zeros(latent_dim, device=self.device, dtype=self.dtype),
            torch.ones(latent_dim, device=self.device, dtype=self.dtype),
        )
        self.flow = WeightedRealNVPProposal(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=self.flow_depth,
            device=self.device,
            dtype=self.dtype,
        )
        self.proposal = FlowMixtureProposal(
            flow=self.flow,
            local_step_size=self.local_step_size,
            local_weight=self.local_proposal_weight,
            device=self.device,
            dtype=self.dtype,
        )
        self.accepted_latent_history = torch.zeros(
            (0, latent_dim), device=self.device, dtype=self.dtype
        )

    def _fit_structured_bo_state(
        self, observed_latents: torch.Tensor, y: np.ndarray
    ) -> StructuredBOState:
        sanitized_y = np.asarray(y, dtype=float).copy()
        sanitized_y[np.isnan(sanitized_y)] = self.penalize_nans_with

        observed_structures = self._decode_latents(observed_latents)
        observed_features = self._structures_to_feature_tensors(observed_structures).cpu().numpy()
        model = self._fit_model(SingleTaskGP, observed_features, sanitized_y)
        best_y = torch.tensor(sanitized_y.max(), dtype=self.dtype, device=self.device)
        acquisition = LogProbabilityOfImprovement(model, best_f=best_y)

        return StructuredBOState(
            model=model,
            best_y=best_y,
            acquisition=acquisition,
            observed_structures=observed_structures,
        )

    def _evaluate_latent_target(
        self, latents: torch.Tensor, bo_state: StructuredBOState
    ) -> LatentTargetEvaluation:
        structures = self._decode_latents(latents)
        fingerprints = self._structures_to_feature_tensors(structures)

        with torch.no_grad():
            log_weight = bo_state.acquisition(fingerprints[:, None, :]).view(-1)

        log_weight = torch.nan_to_num(
            log_weight,
            nan=self.min_log_value,
            neginf=self.min_log_value,
            posinf=0.0,
        )
        log_prior = self.latent_prior.log_prob(latents).sum(dim=-1)
        log_target = log_prior + log_weight

        return LatentTargetEvaluation(
            log_target=log_target,
            log_prior=log_prior,
            log_weight=log_weight,
            structures=structures,
            fingerprints=fingerprints,
        )

    def _build_step_zero_pool(
        self, observed_latents: torch.Tensor, y: np.ndarray
    ) -> torch.Tensor:
        target_size = max(self.pool_size, observed_latents.shape[0])
        remaining = max(target_size - observed_latents.shape[0], 0)
        n_prior = math.ceil(0.6 * remaining)
        n_local = remaining - n_prior
        anchors = self._top_observed_latents(observed_latents, y)

        parts = [observed_latents]
        if n_prior > 0:
            parts.append(self._sample_prior_latents(n_prior))
        if n_local > 0:
            parts.append(self._sample_local_perturbations(anchors, n_local))

        return torch.cat(parts, dim=0)

    def _build_iteration_pool(
        self, observed_latents: torch.Tensor, y: np.ndarray
    ) -> torch.Tensor:
        replay_budget = min(self.accepted_latent_history.shape[0], self.pool_size // 3)
        replay_latents = self._subsample_latents(self.accepted_latent_history, replay_budget)
        current_size = observed_latents.shape[0] + replay_latents.shape[0]
        remaining = max(self.pool_size - current_size, 0)
        n_prior = remaining // 2
        n_local = remaining - n_prior
        anchors = torch.cat(
            [
                self._top_observed_latents(observed_latents, y),
                self._subsample_latents(replay_latents, min(replay_latents.shape[0], 8)),
            ],
            dim=0,
        )

        parts = [observed_latents]
        if replay_latents.numel() > 0:
            parts.append(replay_latents)
        if n_prior > 0:
            parts.append(self._sample_prior_latents(n_prior))
        if n_local > 0:
            parts.append(self._sample_local_perturbations(anchors, n_local))

        return torch.cat(parts, dim=0)

    def _weight_latent_pool(
        self, pool_latents: torch.Tensor, bo_state: StructuredBOState
    ) -> WeightedLatentPool:
        evaluation = self._evaluate_latent_target(pool_latents, bo_state)
        normalized_weights = self._normalize_log_weights(evaluation.log_weight)
        return WeightedLatentPool(
            latents=pool_latents,
            normalized_weights=normalized_weights,
            log_weights=evaluation.log_weight,
            structures=evaluation.structures,
            fingerprints=evaluation.fingerprints,
        )

    def _fit_flow_from_weighted_pool(self, weighted_pool: WeightedLatentPool) -> None:
        self.flow.fit_weighted(
            points=weighted_pool.latents,
            weights=weighted_pool.normalized_weights,
            n_steps=self.flow_training_steps,
        )

    def _initialize_chains(
        self, observed_latents: torch.Tensor, y: np.ndarray
    ) -> torch.Tensor:
        anchors = [self._top_observed_latents(observed_latents, y)]
        if self.accepted_latent_history.numel() > 0:
            anchors.append(
                self._subsample_latents(
                    self.accepted_latent_history,
                    min(self.num_chains, self.accepted_latent_history.shape[0]),
                )
            )
        anchor_latents = torch.cat(anchors, dim=0)
        if anchor_latents.numel() == 0:
            return self._sample_prior_latents(self.num_chains)

        indices = torch.randint(
            anchor_latents.shape[0],
            (self.num_chains,),
            device=self.device,
        )
        initial_states = anchor_latents[indices].clone()
        initial_states = initial_states + 0.1 * self.local_step_size * torch.randn_like(
            initial_states
        )
        return initial_states

    def _run_mh_sampling(
        self, bo_state: StructuredBOState, initial_states: torch.Tensor
    ) -> MHSamplingResult:
        current_states = initial_states.clone()
        current_log_target = self._evaluate_latent_target(current_states, bo_state).log_target
        accepted_latents: list[torch.Tensor] = []
        sampled_latents: list[torch.Tensor] = []
        sampled_sources: list[str] = []
        total_accepts = 0
        n_local_proposals = 0
        n_global_proposals = 0
        n_local_accepted = 0
        n_global_accepted = 0
        local_proposed_structures: set[str] = set()
        global_proposed_structures: set[str] = set()
        local_accepted_structures: set[str] = set()
        global_accepted_structures: set[str] = set()
        current_sources = ["initial"] * self.num_chains
        burn_in = min(self.burn_in, max(self.n_mh_steps - 1, 0))

        for step in range(self.n_mh_steps):
            proposed_states, use_local = self.proposal.propose(current_states)
            proposed_evaluation = self._evaluate_latent_target(proposed_states, bo_state)
            proposed_log_target = proposed_evaluation.log_target
            use_global = ~use_local
            n_local_proposals += int(use_local.sum().item())
            n_global_proposals += int(use_global.sum().item())
            self._extend_structure_set(
                local_proposed_structures, proposed_evaluation.structures, use_local
            )
            self._extend_structure_set(
                global_proposed_structures, proposed_evaluation.structures, use_global
            )
            log_q_forward = self.proposal.log_prob(proposed_states, current_states)
            log_q_reverse = self.proposal.log_prob(current_states, proposed_states)
            log_acceptance = proposed_log_target + log_q_reverse
            log_acceptance = log_acceptance - current_log_target - log_q_forward
            log_acceptance = torch.nan_to_num(
                log_acceptance,
                nan=self.min_log_value,
                neginf=self.min_log_value,
                posinf=0.0,
            )
            log_acceptance = torch.clamp(log_acceptance, max=0.0)

            uniforms = torch.log(
                torch.rand(self.num_chains, device=self.device, dtype=self.dtype).clamp_min(
                    torch.finfo(self.dtype).tiny
                )
            )
            accept = uniforms < log_acceptance

            if accept.any():
                accepted_indices = torch.where(accept)[0]
                current_states[accepted_indices] = proposed_states[accepted_indices]
                current_log_target[accepted_indices] = proposed_log_target[accepted_indices]
                accepted_latents.append(proposed_states[accepted_indices].detach().clone())
                total_accepts += int(accepted_indices.numel())
                accepted_local = accept & use_local
                accepted_global = accept & use_global
                n_local_accepted += int(accepted_local.sum().item())
                n_global_accepted += int(accepted_global.sum().item())
                self._extend_structure_set(
                    local_accepted_structures, proposed_evaluation.structures, accepted_local
                )
                self._extend_structure_set(
                    global_accepted_structures, proposed_evaluation.structures, accepted_global
                )
                for idx in accepted_indices.tolist():
                    current_sources[idx] = "local" if bool(use_local[idx].item()) else "global"

            if step >= burn_in:
                sampled_latents.append(current_states.detach().clone())
                sampled_sources.extend(current_sources)

        if sampled_latents:
            stacked_samples = torch.cat(sampled_latents, dim=0)
        else:
            stacked_samples = current_states.detach().clone()
            sampled_sources = current_sources.copy()

        if accepted_latents:
            stacked_accepts = torch.cat(accepted_latents, dim=0)
        else:
            stacked_accepts = torch.zeros(
                (0, current_states.shape[1]), device=self.device, dtype=self.dtype
            )

        proposal_diagnostics = ProposalDiagnostics(
            n_local_proposals=n_local_proposals,
            n_global_proposals=n_global_proposals,
            n_local_accepted=n_local_accepted,
            n_global_accepted=n_global_accepted,
            n_unique_local_proposed_structures=len(local_proposed_structures),
            n_unique_global_proposed_structures=len(global_proposed_structures),
            n_unique_local_accepted_structures=len(local_accepted_structures),
            n_unique_global_accepted_structures=len(global_accepted_structures),
        )

        return MHSamplingResult(
            sampled_latents=stacked_samples,
            sampled_sources=sampled_sources,
            accepted_latents=stacked_accepts,
            acceptance_rate=total_accepts / max(self.n_mh_steps * self.num_chains, 1),
            proposal_diagnostics=proposal_diagnostics,
        )

    def _select_candidates(
        self,
        sampled_latents: torch.Tensor,
        sampled_sources: list[str],
        weighted_pool: WeightedLatentPool,
        bo_state: StructuredBOState,
    ) -> tuple[torch.Tensor, list[dict[str, str]], list[dict[str, Any]]]:
        observed_structures = set(bo_state.observed_structures)
        candidate_latents = sampled_latents

        candidate_structures = self._decode_latents(candidate_latents)
        unique_latents: list[torch.Tensor] = []
        unique_metadata: list[dict[str, str]] = []
        seen_structures = set(observed_structures)

        for latent, structure, source in zip(candidate_latents, candidate_structures, sampled_sources):
            if structure in seen_structures:
                continue
            seen_structures.add(structure)
            unique_latents.append(latent.unsqueeze(0))
            unique_metadata.append(
                {
                    "source": source,
                    "structure": structure,
                }
            )

        if len(unique_latents) < self.batch_size:
            pool_order = torch.argsort(weighted_pool.normalized_weights, descending=True)
            for idx in pool_order.tolist():
                structure = weighted_pool.structures[idx]
                if structure in seen_structures:
                    continue
                seen_structures.add(structure)
                unique_latents.append(weighted_pool.latents[idx : idx + 1])
                unique_metadata.append(
                    {
                        "source": "pool_fallback",
                        "structure": structure,
                    }
                )
                if len(unique_latents) >= self.batch_size:
                    break

        if not unique_latents:
            best_index = int(torch.argmax(weighted_pool.normalized_weights).item())
            fallback = weighted_pool.latents[best_index : best_index + 1]
            fallback_metadata = [
                {
                    "source": "pool_fallback",
                    "structure": weighted_pool.structures[best_index],
                }
            ]
            ranked_fallback_candidates = self._build_ranked_candidate_diagnostics(
                fallback,
                fallback_metadata,
                torch.tensor(
                    [weighted_pool.log_weights[best_index].detach().cpu().item()],
                    device=self.device,
                    dtype=self.dtype,
                ),
                bo_state.observed_structures,
            )
            return fallback, fallback_metadata, ranked_fallback_candidates

        stacked_latents = torch.cat(unique_latents, dim=0)
        candidate_features = self._structures_to_feature_tensors(
            [candidate["structure"] for candidate in unique_metadata]
        )
        if stacked_latents.shape[0] <= self.batch_size:
            selection_scores = self._evaluate_selection_scores(candidate_features, bo_state)
            ranked_diagnostic_candidates = self._build_ranked_candidate_diagnostics(
                stacked_latents,
                unique_metadata,
                selection_scores,
                bo_state.observed_structures,
            )
            return stacked_latents, unique_metadata, ranked_diagnostic_candidates

        acquisition = qLogExpectedImprovement(bo_state.model, best_f=bo_state.best_y)
        chosen_features, _ = optimize_acqf_discrete(
            acquisition,
            min(self.batch_size, stacked_latents.shape[0]),
            candidate_features,
            max_batch_size=1_000,
        )
        chosen_indices = self._match_selected_features(candidate_features, chosen_features)
        selection_scores = self._evaluate_selection_scores(candidate_features, bo_state)
        ranked_diagnostic_candidates = self._build_ranked_candidate_diagnostics(
            stacked_latents,
            unique_metadata,
            selection_scores,
            bo_state.observed_structures,
        )
        return (
            stacked_latents[chosen_indices],
            [unique_metadata[idx].copy() for idx in chosen_indices],
            ranked_diagnostic_candidates,
        )

    def _update_replay_buffer(self, accepted_latents: torch.Tensor) -> None:
        if accepted_latents.numel() == 0:
            return

        updated_buffer = torch.cat([self.accepted_latent_history, accepted_latents], dim=0)
        max_buffer_size = max(4 * self.pool_size, self.num_chains * self.batch_size)
        if updated_buffer.shape[0] > max_buffer_size:
            updated_buffer = self._subsample_latents(updated_buffer, max_buffer_size)
        self.accepted_latent_history = updated_buffer.detach()

    def _extend_structure_set(
        self, destination: set[str], structures: list[str], mask: torch.Tensor
    ) -> None:
        if mask.numel() == 0 or not mask.any():
            return

        destination.update(structures[idx] for idx in torch.where(mask)[0].tolist())

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
        candidate_metadata: list[dict[str, str]],
        selection_scores: torch.Tensor,
        observed_structures: list[str],
    ) -> list[dict[str, Any]]:
        if candidate_latents.numel() == 0 or not candidate_metadata:
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
            canonical_smiles = self._canonical_smiles(candidate_metadata[idx]["structure"])
            if canonical_smiles is None:
                continue
            if canonical_smiles in seen_molecules:
                continue

            seen_molecules.add(canonical_smiles)
            ranked_candidates.append(
                {
                    **candidate_metadata[idx],
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

    def _build_selected_candidate_metadata(
        self,
        selected_latents: torch.Tensor,
        selected_candidate_metadata: list[dict[str, str]],
        bo_state: StructuredBOState,
    ) -> list[dict[str, Any]]:
        if selected_latents.numel() == 0:
            return []

        evaluation = self._evaluate_latent_target(selected_latents, bo_state)
        selected_unit_latents = self._latent_to_unit(selected_latents.detach().cpu().numpy())
        enriched_metadata: list[dict[str, Any]] = []

        for idx, candidate in enumerate(selected_candidate_metadata):
            enriched_metadata.append(
                {
                    **candidate,
                    "structure": evaluation.structures[idx],
                    "log_probability_of_improvement": float(
                        evaluation.log_weight[idx].detach().cpu().item()
                    ),
                    "log_target": float(evaluation.log_target[idx].detach().cpu().item()),
                    "latent": selected_latents[idx].detach().cpu().numpy().tolist(),
                    "unit_latent": selected_unit_latents[idx].tolist(),
                }
            )

        return enriched_metadata

    def _log_selected_candidates(self, selected_candidates: list[dict[str, Any]]) -> None:
        for idx, candidate in enumerate(selected_candidates):
            print(
                f"selected_candidate_{idx}: "
                f"source={candidate['source']}, "
                f"log_p_improvement={candidate['log_probability_of_improvement']:.3f}, "
                f"log_target={candidate['log_target']:.3f} "
                #f"structure={self._structure_preview(candidate['structure'])}"
            )

    def _report_pending_candidate_observations(
        self, observed_unit_latents: np.ndarray, y: np.ndarray
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
            matched_index = self._match_unit_latent(candidate["unit_latent"], new_unit_latents, matched_new_indices)
            if matched_index is None:
                remaining_candidates.append(candidate)
                continue

            matched_new_indices.add(matched_index)
            observed_value = float(new_y[matched_index])
            candidate["observed_value"] = observed_value
            print(
                "evaluated_candidate: "
                f"source={candidate['source']}, "
                f"observed_value={observed_value:.6g}"
                #f"structure={self._structure_preview(candidate['structure'])}"
            )

        self.pending_selected_candidates = remaining_candidates
        self._last_reported_history_size = int(len(observed_unit_latents))

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

    def _structure_preview(self, structure: str, max_length: int = 80) -> str:
        if len(structure) <= max_length:
            return structure
        return f"{structure[: max_length - 3]}..."

    def _decode_latents(self, latents: torch.Tensor) -> list[str]:
        decoded = self.vae.decode_to_string_array(latents.detach().cpu().numpy())
        return self._normalize_structure_array(decoded)

    def _structure_to_mol(self, structure: str) -> Any:
        candidate_smiles = structure
        try:
            candidate_smiles = sf.decoder(structure)
        except Exception:
            candidate_smiles = structure

        if not candidate_smiles:
            return None
        return Chem.MolFromSmiles(candidate_smiles)

    def _canonical_smiles(self, structure: str) -> str | None:
        mol = self._structure_to_mol(structure)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _normalize_structure_array(self, structures: np.ndarray) -> list[str]:
        normalized: list[str] = []
        for structure in np.asarray(structures, dtype=object):
            if isinstance(structure, str):
                normalized.append(structure)
                continue

            tokens = [str(token) for token in np.asarray(structure, dtype=object).tolist()]
            cleaned_tokens = [token for token in tokens if token not in {"<pad>", "<cls>", "<eos>"}]
            normalized.append("".join(cleaned_tokens))
        return normalized

    def _structures_to_feature_tensors(self, structures: list[str]) -> torch.Tensor:
        if not structures:
            return torch.zeros((0, 2048), device=self.device, dtype=self.dtype)

        if self._looks_like_selfies(structures):
            return self._selfies_list_to_fingerprint_tensors(structures)
        return self._hashed_string_feature_tensors(structures)

    def _looks_like_selfies(self, structures: list[str]) -> bool:
        if not structures:
            return False
        n_selfies_like = sum(("[" in structure and "]" in structure) for structure in structures)
        return n_selfies_like >= max(1, len(structures) // 2)

    def _selfies_list_to_fingerprint_tensors(self, selfies_strings: list[str]) -> torch.Tensor:
        mols = []
        fallback_mol = Chem.MolFromSmiles("Cc1ccccc1")

        for structure in selfies_strings:
            try:
                smiles = sf.decoder(structure)
                mol = Chem.MolFromSmiles(smiles)
            except Exception:
                mol = None
            mols.append(mol if mol is not None else fallback_mol)

        fingerprints = np.zeros((len(mols), 2048))
        for i, mol in enumerate(mols):
            fingerprint = rdMolDescriptors.GetMorganFingerprint(
                mol, radius=3, useCounts=True
            )
            for key, value in fingerprint.GetNonzeroElements().items():
                fingerprints[i, key % 2048] += value

        return torch.tensor(fingerprints, device=self.device, dtype=self.dtype)

    def _hashed_string_feature_tensors(self, structures: list[str]) -> torch.Tensor:
        features = np.zeros((len(structures), 2048))
        for i, structure in enumerate(structures):
            if not structure:
                features[i, 0] = 1.0
                continue
            for n_gram_size in (1, 2, 3):
                for start in range(max(len(structure) - n_gram_size + 1, 0)):
                    n_gram = structure[start : start + n_gram_size]
                    bucket = zlib.crc32(n_gram.encode("utf-8")) % 2048
                    features[i, bucket] += 1.0
        return torch.tensor(features, device=self.device, dtype=self.dtype)

    def _sample_prior_latents(self, n_samples: int) -> torch.Tensor:
        return torch.randn(
            (n_samples, int(self.vae.latent_dim)),
            device=self.device,
            dtype=self.dtype,
        )

    def _sample_local_perturbations(
        self, anchors: torch.Tensor, n_samples: int
    ) -> torch.Tensor:
        if n_samples <= 0:
            return torch.zeros((0, int(self.vae.latent_dim)), device=self.device, dtype=self.dtype)
        if anchors.numel() == 0:
            return self._sample_prior_latents(n_samples)

        indices = torch.randint(anchors.shape[0], (n_samples,), device=self.device)
        perturbations = self.local_step_size * torch.randn(
            (n_samples, anchors.shape[1]),
            device=self.device,
            dtype=self.dtype,
        )
        return anchors[indices] + perturbations

    def _top_observed_latents(self, latents: torch.Tensor, y: np.ndarray) -> torch.Tensor:
        y_vector = np.asarray(y, dtype=float).reshape(-1)
        y_vector[np.isnan(y_vector)] = self.penalize_nans_with
        n_keep = min(max(1, self.batch_size * 4), latents.shape[0])
        top_indices = np.argsort(y_vector)[-n_keep:]
        return latents[top_indices]

    def _subsample_latents(self, latents: torch.Tensor, n_samples: int) -> torch.Tensor:
        if n_samples <= 0 or latents.numel() == 0:
            return torch.zeros((0, latents.shape[-1]), device=self.device, dtype=self.dtype)
        if latents.shape[0] <= n_samples:
            return latents

        indices = torch.randperm(latents.shape[0], device=self.device)[:n_samples]
        return latents[indices]

    def _normalize_log_weights(self, log_weights: torch.Tensor) -> torch.Tensor:
        finite_mask = torch.isfinite(log_weights)
        if not finite_mask.any():
            return torch.full_like(log_weights, 1.0 / max(log_weights.numel(), 1))

        normalized = torch.zeros_like(log_weights)
        normalized[finite_mask] = torch.softmax(log_weights[finite_mask], dim=0)
        total_weight = normalized.sum().clamp_min(torch.finfo(normalized.dtype).tiny)
        return normalized / total_weight

    def _match_selected_features(
        self, candidate_features: torch.Tensor, chosen_features: torch.Tensor
    ) -> list[int]:
        chosen_indices: list[int] = []
        available = torch.ones(candidate_features.shape[0], device=self.device, dtype=torch.bool)

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

    def _unique_new_structures(
        self, sampled_structures: list[str], observed_structures: list[str]
    ) -> list[str]:
        observed = set(observed_structures)
        return list(set(sampled_structures).difference(observed))

    def _record_sampling_metrics(
        self,
        sampled_structures: list[str],
        unique_structures_not_in_observed_history: list[str],
    ) -> None:
        unique_sampled_structures = list(dict.fromkeys(sampled_structures))
        new_distinct_sampled_structures = [
            structure
            for structure in unique_sampled_structures
            if structure not in self._seen_sampled_decoded_structures
        ]
        self._seen_sampled_decoded_structures.update(unique_sampled_structures)
        self.iteration_sampling_metrics_history.append(
            {
                "iteration": len(self.iteration_sampling_metrics_history) + 1,
                "sample_unique_decoded_molecules_in_iteration": len(unique_sampled_structures),
                "sample_new_distinct_decoded_molecules": len(
                    new_distinct_sampled_structures
                ),
                "sample_cumulative_distinct_decoded_molecules": len(
                    self._seen_sampled_decoded_structures
                ),
                "sample_unique_decoded_molecules_not_in_observed_history": len(
                    set(unique_structures_not_in_observed_history)
                ),
            }
        )

    def _to_tensor(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return torch.tensor(values, device=self.device, dtype=self.dtype)
