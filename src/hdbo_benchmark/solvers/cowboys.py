"""
This module provides functionality to run COWBOYS
"""

from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox
from hdbo_benchmark.generative_models.vae_factory import VAE
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.acquisition import qLogExpectedImprovement
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import selfies as sf
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.kernels import ScaleKernel
from hdbo_benchmark.utils.experiments.normalization import (
    from_range_to_unit_cube,
    from_unit_cube_to_range,
)
from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_CANDIDATES,
    build_top_candidate_diagnostics,
    preserve_rng_state,
)
import gpytorch

import warnings
warnings.filterwarnings("ignore", message='.*contained to the unit cube')


from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')     


from poli_baselines.solvers.bayesian_optimization.base_bayesian_optimization.base_bayesian_optimization import (
    BaseBayesianOptimization,
)


class COWBOYS(BaseBayesianOptimization):
    """Molecular COWBOYS: structure-space GP and raw-latent pCN.

    History/returned candidates use normalized coordinates for the poli wrapper.
    Sampling uses the VAE's unbounded standard-normal latent coordinates.
    See docs/COWBOYS_IMPLEMENTATION.md for intentional upstream differences.
    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        penalize_nans_with: float = -10.0,
        device: torch.device = torch.device("cpu"),
        num_chains: int = 1,
        n_steps: int = 100,
        max_sampling_steps: int = 50_000,
    ):
        """
        Initialize the VanillaBayesianOptimization class.

        Parameters:
        ----------
        black_box : AbstractBlackBox
            The black box function to be optimized.
        x0 : np.ndarray
            The initial input samples.
        y0 : np.ndarray
            The initial output values.
        mean : Mean, optional
        penalize_nans_with : float, optional
            The value to penalize NaN values with, by default -10.0.
        """
        super().__init__(black_box, x0, y0)
        self.penalize_nans_with = penalize_nans_with
        self.device = device
        self._given_vae = False # will be updated once VAE passed in using `set_vae` after init
        self.batch_size = 1 # number of molecules to choose at each iteration
        if num_chains < 1 or n_steps < 1 or max_sampling_steps < n_steps:
            raise ValueError("Require num_chains >= 1 and max_sampling_steps >= n_steps >= 1")
        self.num_chains = num_chains
        self.n_steps = n_steps
        self.max_sampling_steps = max_sampling_steps
        self.betas = torch.tensor([0.1]).to(dtype=torch.float64, device=self.device)[None,:].repeat(self.num_chains,1) # keep track of beta param for each MCMC chain
        self.iteration_sampling_metrics_history: list[dict[str, int]] = []
        self.iteration_candidate_diagnostics_history: list[dict[str, Any]] = []
        self._seen_sampled_decoded_structures: set[str] = set()
        initial_structures = getattr(black_box, "initial_observed_structures", None)
        self.observed_structures = (
            self._structure_strings(initial_structures)
            if initial_structures is not None else None
        )

    def _fit_model(
        self, model: SingleTaskGP, x: np.ndarray, y: np.ndarray
    ) -> SingleTaskGP: # fit a GP model to the data
        x = torch.from_numpy(x).to(dtype=torch.float64, device=self.device)
        y = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)

        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ), gpytorch.settings.fast_pred_var(state=False):
            model_instance = model(
                x,
                y,
                mean_module=None,
                covar_module=ScaleKernel(TanimotoKernel()),
            )
            mll = ExactMarginalLogLikelihood(model_instance.likelihood, model_instance)
            fit_gpytorch_mll(mll)
            model_instance.eval()

        self.gp_model_of_objective = model_instance
        return model_instance

    @staticmethod
    def _structure_strings(structures: np.ndarray) -> list[str]:
        return ["".join(np.asarray(row).reshape(-1).tolist()) for row in structures]

    def _unit_to_latent(self, values: np.ndarray) -> np.ndarray:
        return from_unit_cube_to_range(values, self.vae_bounds)

    def _latent_to_unit(self, values: np.ndarray) -> np.ndarray:
        return from_range_to_unit_cube(values, self.vae_bounds)

    def set_vae_and_bounds(
        self, vae: VAE, vae_bounds: tuple[float, float],
        initial_structures: np.ndarray | None = None,
    ) -> None:
        """Attach the decoder and the original, actually evaluated structures.

        The molecular problem wrapper supplies initial structures automatically.
        Direct users must supply them here; reconstructions cannot replace them.
        """
        if not np.isfinite(vae_bounds).all() or vae_bounds[1] <= vae_bounds[0]:
            raise ValueError("VAE bounds must be finite and increasing")
        structures = (self._structure_strings(initial_structures)
                      if initial_structures is not None else self.observed_structures)
        x, _ = self.get_history_as_arrays()
        if structures is None or len(structures) != len(x):
            raise ValueError("Pass the original evaluated initial_structures, aligned with x0/y0")
        if any(self._canonical_smiles(s) is None for s in structures):
            raise ValueError("Initial observed structures must be valid nonempty molecules")
        self.vae = vae
        self.vae.eval()
        self.vae_bounds = tuple(vae_bounds)
        self.bounds = (0.0, 1.0)
        self.observed_structures = list(structures)
        self._given_vae = True

    def get_run_configuration(self) -> dict[str, Any]:
        return {
            "implementation": "cowboys_corrected_pcn_v1",
            "latent_sampling_coordinates": "raw_vae_standard_normal",
            "history_coordinates": "normalized",
            "gp_observations": "actual_evaluated_structures",
            "novelty": "canonical_smiles",
            "threshold_policy": "fixed_incumbent",
            "batch_size": self.batch_size,
            "latent_dim": self.vae.latent_dim,
            "num_chains": self.num_chains,
            "n_steps": self.n_steps,
            "max_sampling_steps": self.max_sampling_steps,
            "beta_adaptation": "upstream_diminishing_log_beta_target_0.234",
            "vae_bounds": list(self.vae_bounds),
            "vae_checkpoint": getattr(self.vae, "checkpoint_path", None),
            "vae_checkpoint_sha256": getattr(self.vae, "checkpoint_sha256", None),
        }

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Keep actual decoded observations aligned with the base solver history."""
        structures = self._structure_strings(
            self.vae.decode_to_string_array(self._unit_to_latent(np.asarray(x)))
        )
        if len(structures) != len(x):
            raise ValueError("Decoder returned a different number of structures")
        super().update(x, y)
        self.observed_structures.extend(structures)

    def next_candidate(self) -> np.ndarray:
        """Sample PI-weighted raw latents and select a new molecule using log-EI."""
        if not self._given_vae:
            raise ValueError("Pass the VAE and original observations using set_vae_and_bounds")
        if self.batch_size != 1:
            raise ValueError("This baseline implements the paper's batch_size=1 protocol")
        x, y = self.get_history_as_arrays()
        if len(self.observed_structures) != len(x):
            raise ValueError("Observed structures and objective history are misaligned")
        y = np.asarray(y, dtype=float).copy()
        y[np.isnan(y)] = self.penalize_nans_with
        if not np.isfinite(y).all():
            raise ValueError("Objective history contains infinite values")
        previous = list(self.observed_structures)
        observed_molecules = {self._canonical_smiles(s) for s in previous}
        features = self.selfies_list_to_fingerprint_tensors(previous)
        model = self._fit_model(SingleTaskGP, features.cpu().numpy(), y)
        best_so_far = torch.as_tensor(y.max(), dtype=torch.float64, device=self.device)
        log_pi = LogProbabilityOfImprovement(model, best_f=best_so_far)

        def log_likelihood(raw_latents):
            structures = self._structure_strings(
                self.vae.decode_to_string_array(raw_latents.cpu().numpy())
            )
            valid = [i for i, s in enumerate(structures) if self._canonical_smiles(s) is not None]
            values = torch.full((len(structures),), -torch.inf,
                                dtype=torch.float64, device=self.device)
            if valid:
                fp = self.selfies_list_to_fingerprint_tensors([structures[i] for i in valid])
                values[valid] = log_pi(fp[:, None, :]).reshape(-1)
            return values, structures

        with torch.no_grad():
            # x is normalized, but the VAE prior and pCN proposal are in raw z.
            start = self._unit_to_latent(x[int(y.argmax())][None, :])
            currents = torch.as_tensor(start, dtype=torch.float64, device=self.device).repeat(self.num_chains, 1)
            current_log_pi, _ = log_likelihood(currents)
            sampled_structures = []
            eligible: dict[str, tuple[torch.Tensor, str]] = {}
            for counter in range(self.max_sampling_steps):
                noise = torch.randn(currents.shape, dtype=torch.float64, device=self.device)
                proposals = torch.sqrt(1 - self.betas.square()) * currents + self.betas * noise
                proposal_log_pi, structures = log_likelihood(proposals)
                # Gaussian prior cancels against the pCN Hastings proposal ratio.
                log_alpha = torch.nan_to_num(proposal_log_pi - current_log_pi,
                                            nan=-torch.inf, posinf=0.0, neginf=-torch.inf).clamp(max=0)
                accept = log_alpha > torch.log(torch.rand(self.num_chains, dtype=torch.float64, device=self.device))
                for i in torch.where(accept)[0].tolist():
                    currents[i] = proposals[i]
                    current_log_pi[i] = proposal_log_pi[i]
                    structure = structures[i]
                    canonical = self._canonical_smiles(structure)
                    if canonical is None:
                        continue
                    sampled_structures.append(structure)
                    if canonical not in observed_molecules and canonical not in eligible:
                        eligible[canonical] = (proposals[i].clone(), structure)
                # Retain the released implementation's diminishing log-beta rule.
                log_betas = (torch.log(self.betas)
                             + 0.5 * (counter + 1) ** (-0.6)
                             * (log_alpha.exp() - 0.234)[:, None])
                self.betas = log_betas.exp().clamp(1e-6, 1.0)
                if counter + 1 >= self.n_steps and len(eligible) >= self.batch_size:
                    break
            else:
                raise RuntimeError(
                    f"COWBOYS found no new valid molecule after {self.max_sampling_steps} "
                    "pCN steps; threshold was kept fixed. Increase max_sampling_steps explicitly."
                )
            self._record_sampling_metrics(
                sampled_structures, [entry[1] for entry in eligible.values()]
            )
            candidate_latents = torch.stack([entry[0] for entry in eligible.values()])
            candidate_structures = [entry[1] for entry in eligible.values()]
            candidate_features = self.selfies_list_to_fingerprint_tensors(candidate_structures)
            acq = qLogExpectedImprovement(model, best_f=best_so_far)
            # Select indices directly: fingerprint equality does not identify a molecule.
            scores = torch.cat([
                acq(chunk[:, None, :]).reshape(-1)
                for chunk in candidate_features.split(1000)
            ])
            if not torch.isfinite(scores).any():
                raise RuntimeError("All candidate EI values are nonfinite")
            scores = torch.where(torch.isfinite(scores), scores, -torch.inf)
            chosen = int(scores.argmax())
            unit_latents = self._latent_to_unit(candidate_latents.cpu().numpy())
            ranked = self._build_ranked_diagnostic_candidates(
                torch.as_tensor(unit_latents, dtype=torch.float64, device=self.device),
                candidate_structures, candidate_features, acq, previous,
            )
            self._record_top_candidate_diagnostics(ranked)
        return unit_latents[chosen:chosen + 1]


    def selfies_list_to_fingerprint_tensors(self, selfies: list):

        smiles_list = [sf.decoder(s) for s in selfies]
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        if any(m is None or m.GetNumAtoms() == 0 for m in mols):
            raise ValueError("Cannot fingerprint an invalid or empty observed molecule")

        # convert mols into Morgan fingerprints
        fp_dicts = []
        for mol in mols:
            fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=3, useCounts=True)
            fp_dicts.append(fp.GetNonzeroElements())

        # Fold fingerprints into array
        out = np.zeros((len(fp_dicts), 2048))
        for i, fp in enumerate(fp_dicts):
            for k, v in fp.items():
                out[i, k % 2048] += v

        return torch.tensor(out).to(dtype=torch.float64, device=self.device)

    def _build_ranked_diagnostic_candidates(
        self,
        candidate_latents: torch.Tensor,
        candidate_structures: list[str],
        candidate_features: torch.Tensor,
        acquisition: qLogExpectedImprovement,
        observed_structures: list[str],
    ) -> list[dict[str, Any]]:
        if candidate_latents.numel() == 0 or not candidate_structures:
            return []

        with preserve_rng_state():
            with torch.no_grad():
                selection_scores = acquisition(candidate_features[:, None, :]).view(-1)
        selection_scores = torch.nan_to_num(
            selection_scores,
            nan=-1e8,
            neginf=-1e8,
            posinf=0.0,
        )

        ranked_indices = torch.argsort(selection_scores, descending=True).tolist()
        ranked_candidates: list[dict[str, Any]] = []
        seen_molecules = {
            canonical_smiles
            for structure in observed_structures
            if (canonical_smiles := self._canonical_smiles(structure)) is not None
        }

        for idx in ranked_indices:
            structure = candidate_structures[idx]
            canonical_smiles = self._canonical_smiles(structure)
            if canonical_smiles is None:
                continue
            if canonical_smiles in seen_molecules:
                continue

            seen_molecules.add(canonical_smiles)
            ranked_candidates.append(
                {
                    "source": "mcmc",
                    "structure": structure,
                    "canonical_smiles": canonical_smiles,
                    "selection_score": float(selection_scores[idx].detach().cpu().item()),
                    "unit_latent": candidate_latents[idx].detach().cpu().numpy().tolist(),
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

    def _selfies_to_mol(self, structure: str):
        try:
            smiles = sf.decoder(structure)
        except Exception:
            return None
        return Chem.MolFromSmiles(smiles)

    def _canonical_smiles(self, structure: str) -> str | None:
        mol = self._selfies_to_mol(structure)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

    def _record_sampling_metrics(
        self,
        sampled_structures: list[str],
        unique_structures_not_in_observed_history: list[str],
    ) -> None:
        unique_sampled_structures = list(dict.fromkeys(
            self._canonical_smiles(s) for s in sampled_structures
            if self._canonical_smiles(s) is not None
        ))
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
