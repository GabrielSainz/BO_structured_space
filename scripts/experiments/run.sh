#!/usr/bin/env bash
# Corrected COWBOYS baseline; protocol and deviations: docs/COWBOYS_IMPLEMENTATION.md
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Activate the hdbo_benchmark Python 3.10 environment"' || exit 1
export HDBO_TORCH_DEVICE="${HDBO_TORCH_DEVICE:-cpu}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

RUN_ID="${RUN_ID:-corrected_pcn_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
MAX_ITER="${MAX_ITER:-300}"
read -r -a seeds <<< "${SEEDS:-1 2 3 4 5}"
if [[ -z "${FUNCTIONS:-}" ]]; then
    FUNCTIONS=$("$PYTHON_BIN" -c 'import json; print(" ".join(json.load(open("configs/cowboys_paper_table1.json"))["tasks"]))') || exit 1
fi
read -r -a functions <<< "$FUNCTIONS"
output_dir="results/cowboys/$RUN_ID"
if [[ -e "$output_dir" ]]; then
    printf 'Output directory already exists: %s. Choose a fresh RUN_ID.\n' "$output_dir" >&2
    exit 1
fi
mkdir -p "$output_dir" || exit 1
printf 'task\tseed\texit_code\n' > "$output_dir/sweep_status.tsv"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    "$PYTHON_BIN" -m unittest tests.test_structure tests.test_cowboys_fidelity tests.test_cowboys_paper_report -v || exit 1
    # Stop before an expensive sweep if the actual GP/VAE/oracle path fails.
    "$PYTHON_BIN" run.py --function-name albuterol_similarity --solver-name cowboys \
        --n-dimensions 128 --max-iter 1 --seed 1 --no-strict-on-hash \
        --wandb-mode disabled --no-evaluate-top-candidate-objectives \
        --sufix "$RUN_ID/preflight" > "$output_dir/preflight.log" 2>&1 || {
        printf 'Preflight failed; see %s/preflight.log. No sweep was launched.\n' "$output_dir" >&2
        exit 1
    }
fi
failures=0
for seed in "${seeds[@]}"; do
    for function_name in "${functions[@]}"; do
        printf 'COWBOYS: %s, seed %s\n' "$function_name" "$seed"
        if "$PYTHON_BIN" run.py --function-name "$function_name" --solver-name cowboys \
            --n-dimensions 128 --max-iter "$MAX_ITER" --seed "$seed" \
            --no-strict-on-hash --wandb-mode disabled \
            --no-evaluate-top-candidate-objectives --tag cowboys-corrected-pcn-v1 \
            --sufix "$RUN_ID" > "$output_dir/${function_name}_${seed}.log" 2>&1; then
            code=0
        else
            code=$?
            failures=$((failures + 1))
            printf 'FAILED: %s seed %s; inspect its log.\n' "$function_name" "$seed" >&2
        fi
        printf '%s\t%s\t%s\n' "$function_name" "$seed" "$code" >> "$output_dir/sweep_status.tsv"
    done
done
"$PYTHON_BIN" -m hdbo_benchmark.evaluation.compare_cowboys_paper "$output_dir" || exit 1
printf 'Finished with %s failed runs. Outputs: %s\n' "$failures" "$output_dir"
[[ "$failures" -eq 0 ]]
