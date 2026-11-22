#!/bin/bash
#SBATCH --job-name=hdbo-cpu
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/hendrix-%j.log
#SBATCH --open-mode=append

set -euo pipefail

if [[ $# -ne 5 ]]; then
    printf 'Usage: sbatch %s FUNCTION SOLVER DIMENSIONS BO_STEPS SEED\n' "$0" >&2
    exit 2
fi

if [[ ! -f run.py || ! -d src/hdbo_benchmark ]]; then
    printf 'Submit this job from the ROTLSC repository root.\n' >&2
    exit 2
fi

if ! command -v conda >/dev/null; then
    printf 'Load your Conda module or put its executable on PATH before submitting.\n' >&2
    exit 2
fi

environment="${HDBO_CONDA_ENV:-hdbo_benchmark}"
run_id="hendrix-${SLURM_JOB_ID:?Submit this script using sbatch}"
if [[ ${SLURM_RESTART_COUNT:-0} != 0 ]]; then
    run_id+="-r${SLURM_RESTART_COUNT}"
fi

export HDBO_TORCH_DEVICE=cpu
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

printf 'Job %s on %s\nWorking directory %s\nRun suffix %s\nConda environment %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD" "$run_id" "$environment"
if revision=$(git rev-parse --verify HEAD 2>/dev/null); then
    printf 'Git commit %s\n' "$revision"
    git status --short
fi

conda run --no-capture-output -n "$environment" python -u run.py \
    --function-name "$1" \
    --solver-name "$2" \
    --n-dimensions "$3" \
    --max-iter "$4" \
    --seed "$5" \
    --wandb-mode disabled \
    --no-strict-on-hash \
    --no-evaluate-top-candidate-objectives \
    --tag hendrix \
    --sufix "$run_id"
