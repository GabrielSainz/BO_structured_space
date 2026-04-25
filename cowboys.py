"""
This module provides functionality to run COWBOYS
"""

from typing import Any, Tuple, Type

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from poli.core.abstract_black_box import AbstractBlackBox
from hdbo_benchmark.generative_models.vae_factory import VAE
from botorch.acquisition.analytic import LogProbabilityOfImprovement
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import selfies as sf
from gauche.kernels.fingerprint_kernels import TanimotoKernel
from gpytorch.kernels import ScaleKernel
from hdbo_benchmark.utils.experiments.normalization import from_unit_cube_to_range
from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    TOP_K_CANDIDATES,
    build_top_candidate_diagnostics,
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
    """
    TODO

    """

    def __init__(
        self,
        black_box: AbstractBlackBox,
        x0: np.ndarray,
        y0: np.ndarray,
        penalize_nans_with: float = -10.0,
        device: torch.device = torch.device("cpu"),
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
        self.num_chains = 10 # number of MCMC chains
        self.n_steps = 100 # minimum number of MCMC steps
        self.betas = torch.tensor([0.1]).to(dtype=torch.float64, device=self.device)[None,:].repeat(self.num_chains,1) # keep track of beta param for each MCMC chain
        self.iteration_sampling_metrics_history: list[dict[str, int]] = []
        self.iteration_candidate_diagnostics_history: list[dict[str, Any]] = []
        self._seen_sampled_decoded_structures: set[str] = set()

    def _fit_model(
        self, model: SingleTaskGP, x: np.ndarray, y: np.ndarray
    ) -> SingleTaskGP: # fit a GP model to the data
        x = torch.from_numpy(x).to(dtype=torch.float64, device=self.device)
        y = torch.from_numpy(y).to(dtype=torch.float64, device=self.device)

        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False) and gpytorch.settings.fast_pred_var(state=False):
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

    def next_candidate(self) -> np.ndarray:
        """Runs one loop of COWBOYS.

        Returns
        -------
        candidate : np.ndarray
            The next candidate(s) to evaluate.

        Notes
        -----
        We penalize the NaNs in the objective function by
        assigning them a value stored in self.penalize_nans_with.
        """
        
        
        if not self._given_vae:
            raise ValueError("need to pass in VAE to solver using its `set_vae` method")
        
        def latent_array_to_selfies_list(z: np.ndarray) -> list:
            # helper function to convert latent array to selfies strings via the VAE decoder
            z = from_unit_cube_to_range(z, self.bounds)
            selfies_strings = self.vae.decode_to_string_array(z)
            return selfies_strings.tolist()


        # Build up the history
        x, y = self.get_history_as_arrays()
        print(f"collected at {len(x)} points so far, with best score so far as {y.max()}") 
        prev_selfies = latent_array_to_selfies_list(x)
        prev_fingerprints = self.selfies_list_to_fingerprint_tensors(prev_selfies).cpu().numpy()

        # Penalize NaNs
        y[np.isnan(y)] = self.penalize_nans_with

        # Fit a GP
        model = self._fit_model(SingleTaskGP, prev_fingerprints, y)


        with torch.no_grad():
            best_so_far = torch.tensor(y).to(torch.float64).max() # starting location for MCMC chains
            log_prob_improvement = LogProbabilityOfImprovement(model, best_f=best_so_far)
            counter = 0 


            def log_lik(s):  # eval log likelihood from GP
                x_selfies = latent_array_to_selfies_list(s.cpu().numpy())
                x_fingerprints = self.selfies_list_to_fingerprint_tensors(x_selfies)
                return log_prob_improvement(x_fingerprints[:,None,:]), x_selfies


            # initialise everything ready for MCMC
            s_currents = torch.tensor(x[y.argmax(), :][None, :]).to(dtype=torch.float64, device=self.device).repeat(self.num_chains, 1) # [C, d]
            log_lik_currents, _= log_lik(s_currents)
            s_samples = torch.zeros((0,self.vae.latent_dim)).to(device=self.device)
            selfies_samples =[]
   

            while True:
                if counter>0 and counter%self.n_steps==0:
                    #if still looking for a new point, reduce the temperature (NOTE THAT THIS IS NEVER ACTUALLY TRIGGERED IN PRACTICE)
                    best_so_far *= 0.95
                    log_prob_improvement = LogProbabilityOfImprovement(model, best_f=best_so_far)
                
        
                # perfom CN MCMC for each chain in parallel
                nu = torch.randn((self.num_chains,self.vae.latent_dim)).to(device=self.device).to(dtype=torch.float64)
                s_candidates = torch.sqrt(1-self.betas**2)*s_currents + self.betas*nu
                log_lik_candidate, selfies_candidates = log_lik(
                    s_candidates
                )
                log_acceptance_prob = log_lik_candidate - log_lik_currents
                log_acceptance_prob =  torch.clip(log_acceptance_prob, -100000000, 0)
                accept = log_acceptance_prob > torch.log(torch.rand(self.num_chains).to(device=self.device))
                if torch.sum(accept)>0: # if any chains accepted, update thhose chains
                    accepted_idx = torch.where(accept)[0]
                    s_currents[accepted_idx] = s_candidates[accepted_idx]
                    log_lik_currents[accepted_idx] = log_lik_candidate[accepted_idx]
                    selfies_samples = selfies_samples + [selfies_candidates[i] for i in accepted_idx]
                    s_samples = torch.vstack([s_samples, s_candidates[accepted_idx]])


                # do adaptive MCMC on the CN's Beta param
                target_acceptance_prob = 0.234
                log_betas = torch.log(self.betas) + 0.5*(torch.pow(counter+1,torch.tensor([-0.6], dtype=torch.float64).to(device=self.device)))*(torch.exp(log_acceptance_prob) - target_acceptance_prob)[:,None]
                self.betas = torch.clip(torch.exp(log_betas),0.000001,1.0)

                counter+=1
                # keep track of the unique samples found so far
                unique_new_structures = list(set(selfies_samples).difference(set(prev_selfies)))
                unique_idx = [selfies_samples.index(x) for x in unique_new_structures]
                if counter>=self.n_steps and len(unique_idx)>=self.batch_size:
                    # end the loop if we have done enough steps and found enough unique samples
                    break
            print(f"did {counter} steps and found {len(unique_idx)} unique")
            self._record_sampling_metrics(
                sampled_structures=selfies_samples,
                unique_structures_not_in_observed_history=unique_new_structures,
            )


        # filter the unique samples found so far if there are more than than desired using EI heuristic
        fingerprints_samples =  self.selfies_list_to_fingerprint_tensors(selfies_samples)
        acq = qLogExpectedImprovement(model, best_f=best_so_far)
        ranked_diagnostic_candidates = self._build_ranked_diagnostic_candidates(
            candidate_latents=s_samples,
            candidate_structures=selfies_samples,
            candidate_features=fingerprints_samples,
            acquisition=acq,
            observed_structures=prev_selfies,
        )
        self._record_top_candidate_diagnostics(ranked_diagnostic_candidates)
        fingerprints_chosen_for_model, _ = optimize_acqf_discrete(acq, min(self.batch_size, len(selfies_samples)), fingerprints_samples, max_batch_size=1_000)
        chosen_idx = [fingerprints_samples.tolist().index(x) for x in fingerprints_chosen_for_model.tolist()]
        return s_samples[chosen_idx].cpu().numpy()



    def set_vae_and_bounds(self, vae: VAE, vae_bounds: tuple[float, float]):
        self.vae = vae
        self.vae_bounds = vae_bounds
        self._given_vae = True


    def selfies_list_to_fingerprint_tensors(self, selfies: list):

        smiles_list = [sf.decoder(s) for s in selfies]
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mols = [m if m is not None else Chem.MolFromSmiles('Cc1ccccc1') for m in mols]
        assert None not in mols, "Invalid SMILES found"

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
            )
        except Exception as exc:
            print(f"Warning: could not evaluate top-k diagnostic candidates: {exc}")
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
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)

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
