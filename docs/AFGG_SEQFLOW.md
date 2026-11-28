# Sequential AFGG with SeqFlow

`afgg_seqflow` uses the released AFGG SeqFlow inference code and NFBO's frozen
GuacaMol checkpoint. The [paper](https://arxiv.org/pdf/2610.22763) describes the
method. The [reference release](https://anonymous.4open.science/r/afgg/README.md)
supplies the sampler, surrogate, acquisition and initialization code. The
[NFBO repository](https://github.com/mlvlab/NFBO) supplies the public weights.
The AFGG release does not include its trained CDCD weights.

## Experiment protocol

| Setting | Released sequential SeqFlow configuration |
| --- | --- |
| Initialization | 10 molecules from the released GuacaMol first-20k pool, seeded permutation with noisy SeqFlow encode/decode round-trip filtering |
| Oracle budget | One selected molecule per step, 300 BO steps after initialization |
| Generator | Frozen float32 SeqFlow, 110 positions × 40 latent dimensions |
| Surrogate | Float64 exact GP with standardized targets, scaled Tanimoto kernel and `1e-4` likelihood noise floor |
| Guidance and selection | Pointwise analytic LogEI, temperature 2.0 for the power tilt, 10 final candidates |
| Variance-preserving bridge | 11 levels and 10 updates, correlation 0.90 to 0.9999, 64 SNIS draws per trajectory and level, weighted-mean update with prior-shell renormalization |
| Trust regions | Five top-scoring anchors, interpolation weight 0.95, factor 1.015, failure/success tolerances 4/10, bounds 0.30/0.99, reset to 0.935 at saturation |
| Batch limits | 512 candidates per acquisition call and 1,024 latents per decode call |

The reference's SeqFlow fingerprint path uses radius-3 Morgan counts folded to
2,048 dimensions. The paper describes binary Morgan fingerprints. The runner
follows the released implementation. The 110 positions were verified using the
reference tokenizer rule on the MD5-verified GuacaMol train split. The runner
uses the shipped initialization CSV and vocabulary, so a run needs no full
dataset download.

The Hendrix grid uses seeds 1–5 and the five tasks in the local COWBOYS report.

- `albuterol_similarity`
- `drd2_docking`
- `osimetrinib_mpo`
- `ranolazine_mpo`
- `scaffold_hop`

The two methods share the local poli/PMO objective implementations, score
direction, seed IDs and evaluation budget. AFGG uses SMILES at the oracle
boundary and its GuacaMol initialization. Local COWBOYS uses its existing
initialization, so the starting molecules can differ by method for a given
seed. The archived local COWBOYS runs use a ZINC 250K VAE. The AFGG paper's
COWBOYS baseline used a VAE trained on GuacaMol. The repository's Python and
package versions can differ from the authors' environment. AFGG follows the
released raw-SMILES filter and raw-pool
fallback, and directly retains the chosen candidate index when fingerprints
collide. The run status records the effective settings, dependency versions,
and SHA-256 hashes of the checkpoint and reference inputs.

## Run the smoke tests

Point `AFGG_ROOT` at the extracted [reference archive](https://anonymous.4open.science/api/repo/afgg/zip)
and `NFBO_ROOT` at a local checkout of NFBO. The pinned NFBO revision used for
local verification is `8f260f89626ddffc567555929f8c45afbe8c48ae`. Follow the
NFBO README to combine and extract its split checkpoint archive. The runner
looks for `model_weight/seqflow_1.27M_3.pt` in that checkout. Use
`AFGG_SEQFLOW_CKPT` or `--afgg-checkpoint` for a checkpoint stored elsewhere.
The public checkpoint's SHA-256 is
`2d9423e7f6a50c6241df6e5e65fbbe61574df1aeba46e23d4f8a2c6ce85d0fe1`.

From the repository root, run one full-settings BO step locally. This evaluates
10 initial molecules and one generated molecule.

```powershell
$env:AFGG_ROOT = '<extracted AFGG source directory>'
$env:NFBO_ROOT = '<NFBO checkout>'
$env:HDBO_TORCH_DEVICE = 'cpu'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
conda run --no-capture-output -n hdbo_benchmark python -u run.py --solver-name afgg_seqflow --function-name albuterol_similarity --seed 1 --max-iter 1 --no-strict-on-hash --wandb-mode disabled --sufix local-smoke
```

On Hendrix, make the reference sources and checkpoint visible to the compute
nodes. Use the CUDA-enabled `hdbo_benchmark` environment and submit from the
repository root with the paths exported. The local Hendrix runbook covers SSH,
Conda and Slurm status. The [DIKU Slurm guide](https://diku-dk.github.io/wiki/slurm-cluster)
lists available GPU types and cluster limits.
The existing CPU smoke environment was set up with CPU-only Torch. Use
`HDBO_CONDA_ENV` to select a separate environment with a CUDA build of Torch
if that CPU setup is still in use.

```bash
mkdir -p slurm_logs
sbatch scripts/slurm/hendrix_afgg.sh smoke
```

Check `sacct` for `COMPLETED` with exit code `0:0`. Check
`results/afgg_seqflow/preflight/hendrix-JOB_ID/` for status `completed`, one BO
iteration, and 11 aligned objective observations. Export its metrics with
`--planned-bo-steps 1`. After validating the smoke run, submit the separate
25-run grid when authorized. The array limits concurrency to five GPUs.

```bash
sbatch --array=0-24%5 --time=12:00:00 scripts/slurm/hendrix_afgg.sh grid
```

## Export evaluation metrics

Each run writes the standard best-score NPY, history NPZ, status and iteration
JSON, and selected-candidate pre-evaluation GP predictions. The history `x`
contains the GP's fingerprints, and `structures` contains the corresponding
SELFIES for the shared molecular evaluation. Iteration sampling counts describe
the final guided pool. COWBOYS counts describe its accepted pCN decodings.
Top-candidate LogEI scores and the selected point's GP diagnostics require no
extra oracle calls.

```bash
python -m hdbo_benchmark.evaluation export --solver afgg_seqflow --results-dir results/afgg_seqflow/hendrix-ARRAY_JOB_ID --planned-bo-steps 300
```

This uses the same [offline molecular metrics](MOLECULAR_METRICS.md) as COWBOYS,
including Top-K quality, AUC, diversity, novelty and GP diagnostics. Apply the
same task-specific hit thresholds to both exports when using hit-based metrics.
The existing aggregate command checks initialization and oracle provenance
before treating runs as one homogeneous comparison group.
