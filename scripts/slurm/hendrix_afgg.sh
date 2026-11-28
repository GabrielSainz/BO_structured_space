#!/bin/bash
#SBATCH --job-name=afgg-seqflow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/afgg-%j.log
#SBATCH --open-mode=append

set -euo pipefail

if [[ ! -f run.py || ! -d src/hdbo_benchmark ]]; then
    printf 'Submit this job from the ROTLSC repository root.\n' >&2
    exit 2
fi

case "${1:-}" in
    smoke)
        function_name=albuterol_similarity
        seed=1
        steps=1
        run_id="preflight/hendrix-${SLURM_JOB_ID:?Submit with sbatch}"
        ;;
    grid)
        # Task and seed identities from the archived five-task COWBOYS runs.
        tasks=(albuterol_similarity drd2_docking osimetrinib_mpo ranolazine_mpo scaffold_hop)
        index="${SLURM_ARRAY_TASK_ID:?Submit grid as an array with indexes 0-24}"
        if [[ ! "$index" =~ ^([0-9]|1[0-9]|2[0-4])$ ]]; then
            printf 'Invalid AFGG array index %s. Expected 0-24.\n' "$index" >&2
            exit 2
        fi
        function_name="${tasks[index / 5]}"
        seed=$((index % 5 + 1))
        steps=300
        run_id="hendrix-${SLURM_ARRAY_JOB_ID:?Submit with sbatch}"
        ;;
    *)
        printf 'Usage: sbatch %s smoke | grid\n' "$0" >&2
        exit 2
        ;;
esac

if [[ ${SLURM_RESTART_COUNT:-0} != 0 ]]; then
    run_id+="-r${SLURM_RESTART_COUNT}"
fi

: "${AFGG_ROOT:?Set AFGG_ROOT to the local AFGG reference source directory}"
: "${NFBO_ROOT:?Set NFBO_ROOT to the local NFBO checkout}"

export HDBO_TORCH_DEVICE=cuda
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

printf 'AFGG %s | task %s | seed %s | BO steps %s | results/afgg_seqflow/%s\n' \
    "$1" "$function_name" "$seed" "$steps" "$run_id"
if revision=$(git rev-parse --verify HEAD 2>/dev/null); then
    printf 'Repository revision %s\n' "$revision"
    git status --short
fi
conda run --no-capture-output -n "${HDBO_CONDA_ENV:-hdbo_benchmark}" python -u run.py \
    --solver-name afgg_seqflow --function-name "$function_name" --seed "$seed" \
    --max-iter "$steps" --sufix "$run_id" --no-strict-on-hash \
    --wandb-mode disabled --no-evaluate-top-candidate-objectives --tag hendrix
