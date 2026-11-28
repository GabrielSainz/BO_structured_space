"""Sequential AFGG with the released SeqFlow sampler and ROTLSC artifacts."""

import hashlib
import importlib.metadata
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import selfies as sf
import torch
from rdkit import Chem

from hdbo_benchmark.utils.logging.candidate_diagnostics import (
    build_top_candidate_diagnostics,
    preserve_rng_state,
)
from hdbo_benchmark.utils.logging.gp_predictions import analytic_expected_improvement


N_INITIAL = 10
SEQUENCE_LENGTH = 110
BRIDGE_LEVELS = 11
SNIS_SAMPLES = 64
RHO_MIN, RHO_MAX = 0.90, 0.9999
TEMPERATURE = 2.0
POOL_SIZE, N_ANCHORS = 10, 5
ACQ_CHUNK, DECODE_CHUNK = 512, 1024
WEIGHT_START, WEIGHT_MIN, WEIGHT_MAX = 0.95, 0.30, 0.99
WEIGHT_FACTOR, WEIGHT_RESET = 1.015, 0.935
FAILURE_TOL, SUCCESS_TOL = 4, 10


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(molecule) if molecule is not None else None


class SeqFlowReference:
    """Load the authors' inference code from local files without invoking training."""

    def __init__(self, root, nfbo_root, checkpoint, device):
        if root is None or nfbo_root is None:
            raise ValueError("AFGG requires --afgg-root and --afgg-nfbo-root (or AFGG_ROOT and NFBO_ROOT)")
        root, nfbo_root = Path(root).resolve(), Path(nfbo_root).resolve()
        checkpoint = Path(checkpoint).resolve() if checkpoint else nfbo_root / "model_weight/seqflow_1.27M_3.pt"
        reference_files = (
            "afgg/seqflow_prior.py", "afgg/seqflow_rollout.py", "afgg/guaca_init.py",
            "afgg/bo_module.py", "afgg/d_af.py", "afgg/process_input.py",
            "data/guacamol_vocab.json", "data/guacamol_train_data_first_20k.csv",
        )
        for path in (*[root / name for name in reference_files], checkpoint,
                     nfbo_root / "generative_model/model/seqflow/lstm_flow.py"):
            if not path.is_file():
                raise FileNotFoundError(f"AFGG input is missing: {path}")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("AFGG was configured for CUDA, but PyTorch cannot see a CUDA GPU")

        sys.path.insert(0, str(root))
        import afgg
        from afgg.bo_module import build_gp_trainer
        from afgg.guaca_init import draw_initial_molecules
        from afgg.process_input import ProcessInputMolecule
        from afgg.seqflow_prior import SeqFlowPrior
        from afgg.seqflow_rollout import vp_guided_rollout

        if Path(afgg.__file__).resolve().parent != root / "afgg":
            raise ValueError("The imported afgg package does not match --afgg-root")

        # d_af sets Torch's global default dtype when imported. Keep that side effect
        # within the AFGG run and build the reference GP's parameters in float64.
        with _double_precision_defaults():
            from afgg.d_af import EIWrapper

        # Start any fingerprint workers before creating a CUDA context for the flow.
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        self.process = ProcessInputMolecule(root / "data/guacamol_vocab.json", device=self.device,
                                            n_workers=n_workers)
        with _double_precision_defaults():
            self.flow = SeqFlowPrior(nfbo_root=nfbo_root, ckpt_path=checkpoint,
                                     seq_len=SEQUENCE_LENGTH, device=self.device, dtype=torch.float32)
        self.fit_gp = build_gp_trainer("tanimoto", self.device, noise_min=1e-4).fit
        self.acquisition = EIWrapper
        self.rollout = vp_guided_rollout
        self.draw_initial_molecules = draw_initial_molecules
        self.provenance = {
            "checkpoint_sha256": _digest(checkpoint),
            "reference_sha256": {name: _digest(root / name) for name in reference_files},
            "nfbo_flow_sha256": _digest(nfbo_root / "generative_model/model/seqflow/lstm_flow.py"),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("numpy", "torch", "botorch", "gpytorch", "gauche", "rdkit",
                          "selfies", "poli-core", "PyTDC")},
            "python": sys.version.split()[0],
            "device": str(self.device),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }

    def initial_design(self, seed):
        # Mirror the reference permutation prefix, dequantization seed and round-trip filter.
        smiles, selfies = self.draw_initial_molecules(int(N_INITIAL * 1.5) + 32, seed)
        tokens = self.flow.selfies_to_tokens(selfies).to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.flow.encode(tokens, dequantise=True, generator=generator)
        recovered, _ = self.flow.decode(latents, max_batch=DECODE_CHUNK)
        keep = (recovered == tokens).all(dim=1).nonzero().flatten()[:N_INITIAL].tolist()
        if len(keep) != N_INITIAL:
            raise RuntimeError(f"AFGG seed {seed}: only {len(keep)} initial molecules round-tripped through SeqFlow")
        return [smiles[i] for i in keep], [selfies[i] for i in keep], latents[keep]


@contextmanager
def _double_precision_defaults():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@dataclass
class Anchor:
    smiles: str
    score: float
    latent: torch.Tensor
    weight: float = WEIGHT_START
    successes: int = 0
    failures: int = 0

    def update(self, score):
        if score > self.score:
            self.failures = 0
            self.successes += 1
            if self.successes == SUCCESS_TOL:
                self.weight = max(self.weight / WEIGHT_FACTOR, WEIGHT_MIN)
                self.successes = 0
        else:
            self.successes = 0
            self.failures += 1
            if self.failures == FAILURE_TOL:
                self.weight *= WEIGHT_FACTOR
                if self.weight >= WEIGHT_MAX - 1e-9:
                    self.weight = WEIGHT_RESET
                self.failures = 0


class AFGGSeqFlow:
    batch_size = 1

    def __init__(self, black_box, x0, y0, *, reference, initial_selfies, initial_latents):
        self.black_box = black_box
        self.reference = reference
        self.observed_structures = list(initial_selfies)
        initial_smiles = np.asarray(x0).reshape(-1).tolist()
        self._y = np.asarray(y0, dtype=float).reshape(-1).tolist()
        if not np.isfinite(self._y).all():
            raise ValueError("AFGG needs finite initial objective scores")
        self._initial_size = len(self._y)
        self._best = [max(self._y)]
        self._seen = set(initial_smiles)
        self._generated = set()
        self._anchors = []
        self._features, _ = reference.process.selfies_list_to_fingerprint_tensors(initial_selfies)
        for i in torch.argsort(torch.as_tensor(self._y, dtype=torch.float64), descending=True).tolist():
            self._offer_anchor(initial_smiles[i], self._y[i], initial_latents[i])
        self.iteration_sampling_metrics_history = []
        self.iteration_candidate_diagnostics_history = []
        self.pre_evaluation_prediction_history = []

    def _offer_anchor(self, smiles, score, latent):
        if any(anchor.smiles == smiles for anchor in self._anchors):
            return
        if len(self._anchors) < N_ANCHORS or score > self._anchors[-1].score:
            self._anchors.append(Anchor(smiles, score, latent.detach()))
            self._anchors.sort(key=lambda anchor: anchor.score, reverse=True)
            del self._anchors[N_ANCHORS:]

    def _features_for_tokens(self, tokens):
        selfies = self.reference.flow.tokens_to_selfies(tokens)
        features, smiles = self.reference.process.selfies_list_to_fingerprint_tensors(selfies)
        return selfies, features, smiles

    @torch.no_grad()
    def _propose(self, acquisition):
        ref = self.reference
        start = ref.flow.sample_prior(POOL_SIZE)
        slots = [slot for slot in range(N_ANCHORS) for _ in range(POOL_SIZE // N_ANCHORS)]
        for row, slot in enumerate(slots):
            if slot < len(self._anchors):
                anchor = self._anchors[slot]
                weight = start.new_tensor(anchor.weight)
                start[row] = weight * anchor.latent + (1 - weight.square()).sqrt() * start[row]

        def log_tilt(tokens):
            _, features, _ = self._features_for_tokens(tokens)
            return torch.cat([acquisition(chunk).reshape(-1) for chunk in features.split(ACQ_CHUNK)])

        latents, logits, stats = ref.rollout(
            ref.flow, start, n_steps=BRIDGE_LEVELS - 1, n_samples=SNIS_SAMPLES,
            a_min=RHO_MIN, a_max=RHO_MAX, log_tilt_fn=log_tilt, gamma_fn=lambda _: TEMPERATURE,
            update="mean", renorm=True, decode_max_batch=DECODE_CHUNK, q=self.batch_size,
        )
        selfies, features, smiles = self._features_for_tokens(logits.argmax(dim=-1))
        seen_pool = set()
        eligible = []
        for i, smile in enumerate(smiles):
            if smile not in self._seen and smile not in seen_pool:
                eligible.append(i)
            seen_pool.add(smile)
        choices = eligible or list(range(len(smiles)))  # Released code's raw-pool fallback.
        scores = acquisition(features[choices]).reshape(-1)
        ranked = torch.argsort(scores, descending=True, stable=True).tolist()
        selected = choices[ranked[0]]
        canonical = [_canonical(smile) for smile in smiles]
        # Use the exact SELFIES readout when it produced the oracle input. The
        # reference processor substitutes toluene when RDKit rejects a decode.
        for i, smile in enumerate(smiles):
            if smile == "Cc1ccccc1" and sf.decoder(selfies[i]) != smile:
                selfies[i] = sf.encoder(smile)
        candidates = []
        ranked_canonical = set()
        for j in ranked:
            i = choices[j]
            identity = canonical[i]
            if identity is not None and identity not in ranked_canonical:
                ranked_canonical.add(identity)
                candidates.append({"structure": selfies[i], "canonical_smiles": identity,
                                   "selection_score": float(scores[j])})
        return selected, slots, latents, selfies, features, smiles, canonical, candidates, stats

    def step(self):
        targets = self._features.new_tensor(self._y)
        incumbent = max(self._y)
        with _double_precision_defaults():
            model = self.reference.fit_gp(self._features, targets).eval()
            acquisition = self.reference.acquisition(model, self.reference.device, incumbent)
        selected, slots, latents, selfies, features, smiles, canonical, candidates, stats = self._propose(acquisition)
        query = features[selected:selected + 1]
        prediction = self._prediction(model, query, canonical[selected], incumbent)
        score = float(np.asarray(self.black_box(np.asarray([smiles[selected]]))).item())
        if not np.isfinite(score):
            raise ValueError("AFGG received a nonfinite objective score")
        slot = slots[selected]
        if slot < len(self._anchors):
            self._anchors[slot].update(score)
        self._offer_anchor(smiles[selected], score, latents[selected])
        observed = {_canonical(smile) for smile in self._seen}
        pool = set(canonical) - {None}
        self.iteration_sampling_metrics_history.append({
            "sample_unique_decoded_molecules_in_iteration": len(pool),
            "sample_new_distinct_decoded_molecules": len(pool - self._generated),
            "sample_cumulative_distinct_decoded_molecules": len(pool | self._generated),
            "sample_unique_decoded_molecules_not_in_observed_history": len(pool - observed),
        })
        self._generated.update(pool)
        self.iteration_candidate_diagnostics_history.append(
            build_top_candidate_diagnostics(self.black_box, candidates, evaluate_objectives=False)
        )
        self.pre_evaluation_prediction_history.append(prediction)
        self.observed_structures.append(selfies[selected])
        self._seen.add(smiles[selected])
        self._features = torch.cat((self._features, query))
        self._y.append(score)
        self._best.append(max(incumbent, score))
        print(f"AFGG step {len(self._best) - 1}: score={score:.6g}, best={self._best[-1]:.6g}, "
              f"pool={len(pool)}, SNIS ESS={np.mean(stats['ess']):.2f}/{SNIS_SAMPLES}")

    def _prediction(self, model, query, canonical, incumbent):
        row = {"observation_index": len(self._y), "bo_step": len(self._best),
               "captured": "after_candidate_selection_before_oracle_evaluation",
               "n_training_observations": len(self._y), "canonical_smiles": canonical,
               "mean": None, "latent_std": None, "predictive_std": None,
               "incumbent_for_acquisition": incumbent, "analytic_expected_improvement": None,
               "missing_reason": {}}
        try:
            with preserve_rng_state(), torch.no_grad():
                latent = model.posterior(query, observation_noise=False)
                noisy = model.posterior(query, observation_noise=True)
                mean, std = latent.mean.item(), latent.variance.sqrt().item()
                row.update(mean=mean, latent_std=std, predictive_std=noisy.variance.sqrt().item(),
                           analytic_expected_improvement=analytic_expected_improvement(mean, std, incumbent))
        except Exception as exc:
            row["missing_reason"]["prediction"] = f"{type(exc).__name__}: {exc}"
        return row

    def get_history_as_arrays(self):
        return self._features.cpu().numpy(), np.asarray(self._y).reshape(-1, 1)

    def get_best_performance(self):
        return np.asarray(self._best)

    def get_run_configuration(self):
        return {
            "implementation": "afgg_seqflow_reference_sequential_v1", "batch_size": self.batch_size,
            "gp_observations": "actual_evaluated_structures", "selection_acquisition": "LogExpectedImprovement",
            "surrogate": "float64 exact standardized GP, scaled Tanimoto, Morgan counts radius 3 folded to 2048",
            "gp_noise_floor": 1e-4, "initialization_policy": "GuacaMol first 20k, seeded permutation, SeqFlow round-trip filter",
            "n_initial": self._initial_size, "generator_dtype": "float32", "sequence_length": SEQUENCE_LENGTH,
            "bridge_levels": BRIDGE_LEVELS, "rho_min": RHO_MIN, "rho_max": RHO_MAX,
            "snis_samples": SNIS_SAMPLES, "snis_update": "mean", "snis_renormalize": True,
            "temperature": TEMPERATURE, "pool_size": POOL_SIZE, "anchors": N_ANCHORS,
            "anchor_weight": WEIGHT_START, "anchor_weight_min": WEIGHT_MIN, "anchor_weight_max": WEIGHT_MAX,
            "anchor_weight_factor": WEIGHT_FACTOR, "anchor_weight_reset": WEIGHT_RESET,
            "anchor_success_tolerance": SUCCESS_TOL, "anchor_failure_tolerance": FAILURE_TOL,
            "acquisition_chunk": ACQ_CHUNK, "decode_chunk": DECODE_CHUNK,
            **self.reference.provenance,
        }
