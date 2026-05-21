from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import gpytorch
import numpy as np
import selfies as sf
import torch
import torch.nn as nn
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox
from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)
from rdkit import Chem
from rdkit import RDLogger

from hdbo_benchmark.utils.experiments.normalization import from_range_to_unit_cube
from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_CANDIDATES,
    build_top_candidate_diagnostics,
    preserve_rng_state,
)

warnings.filterwarnings("ignore", message=".*contained to the unit cube")
RDLogger.DisableLog("rdApp.*")


@dataclass
class LatentBOState:
    model: SingleTaskGP
    best_y: torch.Tensor
    acquisition: nn.Module
    observed_structures: list[str]


class LatentSpaceBayesianOptimization(BaseBayesianOptimization):
    """Original-style LSBO: GP surrogate and EI acquisition in VAE latent space.

    The benchmark transforms molecular inputs to normalized VAE latents before
    constructing this solver. This class therefore fits its surrogate directly
    on the normalized latent coordinates; the affine map back to the VAE domain
    is used only for decoding and metric diagnostics.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        batch_size: int = 1,
        num_candidates: int = 1000,
        raw_samples: int = 512,
        num_restarts: int = 10,
        penalize_nans_with: float = -10.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(black_box, x0, y0)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive.")
        if raw_samples <= 0:
            raise ValueError("raw_samples must be positive.")
        if num_restarts <= 0:
            raise ValueError("num_restarts must be positive.")

        self.device = device
        self.dtype = torch.float64
        self.batch_size = batch_size
        self.num_candidates = num_candidates
        self.raw_samples = raw_samples
        self.num_restarts = num_restarts
        self.penalize_nans_with = penalize_nans_with
        self.min_value = -1e8

        self._given_vae = False
        self.last_selected_candidates: list[dict[str, Any]] = []
        self.pending_selected_candidates: list[dict[str, Any]] = []
        self.iteration_sampling_metrics_history: list[dict[str, int]] = []
        self.iteration_candidate_diagnostics_history: list[dict[str, Any]] = []
        self._seen_sampled_decoded_structures: set[str] = set()
        self._last_reported_history_size = int(np.asarray(x0).shape[0])

    def next_candidate(self) -> np.ndarray:
        if not self._given_vae:
            raise ValueError("Need to pass in a VAE using `set_vae_and_bounds`.")

        observed_unit_latents, y = self.get_history_as_arrays()
        self._report_pending_candidate_observations(observed_unit_latents, y)
        observed_unit_latents = np.asarray(observed_unit_latents, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        best_score_so_far = (
            float(np.nanmax(y)) if np.isfinite(y).any() else self.penalize_nans_with
        )
        print(
            f"collected at {len(observed_unit_latents)} points so far, "
            f"with best score so far as {best_score_so_far}"
        )

        bo_state = self._fit_latent_bo_state(observed_unit_latents, y)
        optimized_unit_latents = self._optimize_acquisition(
            bo_state.acquisition,
            int(observed_unit_latents.shape[-1]),
        )
        diagnostic_unit_latents = self._build_diagnostic_unit_latents(
            observed_unit_latents,
            optimized_unit_latents,
        )
        diagnostic_scores = self._evaluate_acquisition(
            bo_state.acquisition,
            diagnostic_unit_latents,
        )

        candidate_latents = self._to_tensor(
            self._unit_to_latent(diagnostic_unit_latents.detach().cpu().numpy())
        )
        candidate_structures = self._decode_latents(candidate_latents)
        unique_new_structures = self._unique_valid_structures(
            candidate_structures,
            set(bo_state.observed_structures),
        )
        print(
            f"ranked {diagnostic_unit_latents.shape[0]} diagnostic latent candidates "
            f"and found {len(unique_new_structures)} unique valid new molecules"
        )
        self._record_sampling_metrics(
            sampled_structures=candidate_structures,
            unique_structures_not_in_observed_history=unique_new_structures,
        )

        ranked_diagnostic_candidates = self._rank_diagnostic_candidates(
            diagnostic_unit_latents,
            diagnostic_scores,
            candidate_structures,
            bo_state.observed_structures,
        )
        selected_unit_latents = self._select_candidate(
            optimized_unit_latents,
            ranked_diagnostic_candidates,
            bo_state.observed_structures,
        )
        self._record_top_candidate_diagnostics(ranked_diagnostic_candidates)
        selected_candidate_metadata = self._build_selected_candidate_metadata(
            selected_unit_latents,
            bo_state,
        )
        self._log_selected_candidates(selected_candidate_metadata)
        self.last_selected_candidates = selected_candidate_metadata
        self.pending_selected_candidates = [
            candidate.copy() for candidate in selected_candidate_metadata
        ]
        self._last_reported_history_size = int(len(observed_unit_latents))

        return selected_unit_latents.detach().cpu().numpy()

    def set_vae_and_bounds(self, vae: Any, vae_bounds: tuple[float, float]) -> None:
        self.vae = vae
        self.vae_bounds = vae_bounds
        self.bounds = vae_bounds
        self._given_vae = True

    def _fit_latent_bo_state(
        self,
        observed_unit_latents: np.ndarray,
        y: np.ndarray,
    ) -> LatentBOState:
        sanitized_y = np.asarray(y, dtype=float).copy()
        sanitized_y[np.isnan(sanitized_y)] = self.penalize_nans_with

        x_tensor = torch.tensor(
            observed_unit_latents,
            device=self.device,
            dtype=self.dtype,
        )
        y_tensor = torch.tensor(sanitized_y, device=self.device, dtype=self.dtype)

        latent_dim = int(x_tensor.shape[-1])
        covar_module = ScaleKernel(RBFKernel()).to(device=self.device, dtype=self.dtype)
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False,
            log_prob=False,
            solves=False,
        ) and gpytorch.settings.fast_pred_var(state=False):
            model = SingleTaskGP(
                x_tensor,
                y_tensor,
                mean_module=None,
                covar_module=covar_module,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

        self.gp_model_of_objective = model
        best_y = torch.tensor(
            float(np.nanmax(sanitized_y)),
            dtype=self.dtype,
            device=self.device,
        )
        acquisition = ExpectedImprovement(model, best_f=best_y)
        observed_latents = self._to_tensor(self._unit_to_latent(observed_unit_latents))
        observed_structures = self._decode_latents(observed_latents)

        # Store for debugging and for a quick sanity check in logs.
        self.latent_dim = latent_dim

        return LatentBOState(
            model=model,
            best_y=best_y,
            acquisition=acquisition,
            observed_structures=observed_structures,
        )

    def _build_diagnostic_unit_latents(
        self,
        observed_unit_latents: np.ndarray,
        optimized_unit_latents: torch.Tensor,
    ) -> torch.Tensor:
        latent_dim = int(observed_unit_latents.shape[-1])
        sobol_engine = torch.quasirandom.SobolEngine(
            dimension=latent_dim,
            scramble=True,
        )
        sobol_candidates = sobol_engine.draw(self.num_candidates).to(
            device=self.device,
            dtype=self.dtype,
        )
        return torch.cat(
            [optimized_unit_latents, sobol_candidates],
            dim=0,
        ).clamp(0.0, 1.0)

    def _optimize_acquisition(
        self,
        acquisition: nn.Module,
        latent_dim: int,
    ) -> torch.Tensor:
        bounds = torch.stack(
            [
                torch.zeros(latent_dim, device=self.device, dtype=self.dtype),
                torch.ones(latent_dim, device=self.device, dtype=self.dtype),
            ]
        )
        try:
            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=self.batch_size,
                num_restarts=self.num_restarts,
                raw_samples=self.raw_samples,
            )
            return candidate.detach().reshape(self.batch_size, latent_dim).clamp(0.0, 1.0)
        except Exception as exc:
            print(f"Warning: LSBO acquisition optimization failed: {exc}")
            sobol_engine = torch.quasirandom.SobolEngine(
                dimension=latent_dim,
                scramble=True,
            )
            return sobol_engine.draw(self.batch_size).to(
                device=self.device,
                dtype=self.dtype,
            )

    def _evaluate_acquisition(
        self,
        acquisition: nn.Module,
        unit_latents: torch.Tensor,
    ) -> torch.Tensor:
        if unit_latents.numel() == 0:
            return torch.zeros((0,), device=self.device, dtype=self.dtype)

        with preserve_rng_state():
            with torch.no_grad():
                scores = acquisition(unit_latents[:, None, :]).view(-1)
        return torch.nan_to_num(
            scores.to(dtype=self.dtype, device=self.device),
            nan=self.min_value,
            neginf=self.min_value,
            posinf=0.0,
        )

    def _rank_diagnostic_candidates(
        self,
        candidate_unit_latents: torch.Tensor,
        candidate_scores: torch.Tensor,
        candidate_structures: list[str],
        observed_structures: list[str],
    ) -> list[dict[str, Any]]:
        ranked_indices = torch.argsort(candidate_scores, descending=True).tolist()
        observed_smiles = {
            canonical_smiles
            for structure in observed_structures
            if (canonical_smiles := self._canonical_smiles(structure)) is not None
        }
        seen_smiles = set(observed_smiles)

        ranked_candidates: list[dict[str, Any]] = []
        for idx in ranked_indices:
            structure = candidate_structures[idx]
            canonical_smiles = self._canonical_smiles(structure)
            if canonical_smiles is None:
                continue
            if canonical_smiles in seen_smiles:
                continue

            seen_smiles.add(canonical_smiles)
            candidate = {
                "source": "latent_ei",
                "structure": structure,
                "canonical_smiles": canonical_smiles,
                "selection_score": float(candidate_scores[idx].detach().cpu().item()),
                "unit_latent": candidate_unit_latents[idx].detach().cpu().tolist(),
            }
            ranked_candidates.append(candidate)

        return ranked_candidates

    def _select_candidate(
        self,
        optimized_unit_latents: torch.Tensor,
        ranked_diagnostic_candidates: list[dict[str, Any]],
        observed_structures: list[str],
    ) -> torch.Tensor:
        optimized_latents = self._to_tensor(
            self._unit_to_latent(optimized_unit_latents.detach().cpu().numpy())
        )
        optimized_structures = self._decode_latents(optimized_latents)
        observed_smiles = {
            canonical_smiles
            for structure in observed_structures
            if (canonical_smiles := self._canonical_smiles(structure)) is not None
        }

        for idx, structure in enumerate(optimized_structures):
            canonical_smiles = self._canonical_smiles(structure)
            if canonical_smiles is None:
                continue
            if canonical_smiles in observed_smiles:
                continue
            return optimized_unit_latents[idx : idx + 1]

        if ranked_diagnostic_candidates:
            return torch.tensor(
                [ranked_diagnostic_candidates[0]["unit_latent"]],
                device=self.device,
                dtype=self.dtype,
            )

        return optimized_unit_latents[:1]

    def _record_top_candidate_diagnostics(
        self,
        ranked_candidates: list[dict[str, Any]],
    ) -> None:
        try:
            diagnostics = build_top_candidate_diagnostics(
                self.black_box,
                ranked_candidates,
                top_k=TOP_K_CANDIDATES,
                evaluate_objectives=True,
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
        selected_unit_latents: torch.Tensor,
        bo_state: LatentBOState,
    ) -> list[dict[str, Any]]:
        selected_latents = self._to_tensor(
            self._unit_to_latent(selected_unit_latents.detach().cpu().numpy())
        )
        structures = self._decode_latents(selected_latents)
        selection_scores = self._evaluate_acquisition(
            bo_state.acquisition,
            selected_unit_latents,
        )

        selected_candidates: list[dict[str, Any]] = []
        for idx, structure in enumerate(structures):
            selected_candidates.append(
                {
                    "source": "latent_ei",
                    "structure": structure,
                    "canonical_smiles": self._canonical_smiles(structure),
                    "selection_score": float(
                        selection_scores[idx].detach().cpu().item()
                    ),
                    "latent": selected_latents[idx].detach().cpu().numpy().tolist(),
                    "unit_latent": selected_unit_latents[idx].detach().cpu().tolist(),
                }
            )
        return selected_candidates

    def _log_selected_candidates(self, selected_candidates: list[dict[str, Any]]) -> None:
        for idx, candidate in enumerate(selected_candidates):
            print(
                f"selected_candidate_{idx}: "
                f"source={candidate['source']}, "
                f"ei={candidate['selection_score']:.6g}, "
                f"valid={candidate['canonical_smiles'] is not None}"
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
                f"ei={candidate['selection_score']:.6g}"
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

    def _decode_latents(self, latents: torch.Tensor) -> list[str]:
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

    def _canonical_smiles(self, structure: str) -> str | None:
        mol = self._structure_to_mol(structure)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _structure_to_mol(self, structure: str) -> Any:
        candidate_smiles = structure
        try:
            candidate_smiles = sf.decoder(structure)
        except Exception:
            candidate_smiles = structure

        if not candidate_smiles:
            return None
        return Chem.MolFromSmiles(candidate_smiles)

    def _unique_valid_structures(
        self,
        structures: list[str],
        observed_structures: set[str] | None = None,
    ) -> list[str]:
        seen_structures = set() if observed_structures is None else set(observed_structures)
        unique_structures: list[str] = []
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
                "sample_unique_decoded_molecules_in_iteration": len(
                    unique_sampled_structures
                ),
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

    def _unit_to_latent(self, unit_latents: np.ndarray) -> np.ndarray:
        lower, upper = self.vae_bounds
        return unit_latents * (upper - lower) + lower

    def _latent_to_unit(self, latents: np.ndarray) -> np.ndarray:
        return from_range_to_unit_cube(latents, self.vae_bounds)

    def _to_tensor(self, values: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return torch.tensor(values, device=self.device, dtype=self.dtype)
