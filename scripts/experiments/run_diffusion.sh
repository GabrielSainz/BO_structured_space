#!/usr/bin/env bash

set -u -o pipefail

CHECKPOINT_PATH="data/trained_models/training_diffusion_on_zinc_250k/latent_diffusion_latent_dim-128-seed-0.pt"

LOCAL_RESULTS="/content/BO_structured_space/results"
DRIVE_ROOT="/content/drive/MyDrive/university_of_copenhagen/master_thesis/poli_benchmark/results_diffusion_gs20_clip40"
DRIVE_RESULTS="${DRIVE_ROOT}/results"
DRIVE_LOGS="${DRIVE_ROOT}/logs"

mkdir -p "${DRIVE_RESULTS}"
mkdir -p "${DRIVE_LOGS}"

sync_results() {
  echo "Syncing results to Drive..."
  mkdir -p "${DRIVE_RESULTS}"
  rsync -avh "${LOCAL_RESULTS}/" "${DRIVE_RESULTS}/"
  sync
  echo "Sync finished."
}

trap sync_results EXIT

run_one() {
  local function_name="$1"
  local seed="$2"
  local guidance_scale="$3"
  local clip_guidance="$4"

  local run_name="${function_name}__seed-${seed}__gs-${guidance_scale}__clip-${clip_guidance}"
  local log_file="${DRIVE_LOGS}/${run_name}.log"

  echo
  echo "============================================================"
  echo "Running function=${function_name} seed=${seed} guidance_scale=${guidance_scale} clip_guidance=${clip_guidance}"
  echo "Log file: ${log_file}"
  echo "============================================================"

  if python run.py \
    --function-name "${function_name}" \
    --solver-name cowboys_diffusion \
    --n-dimensions 128 \
    --max-iter 300 \
    --seed "${seed}" \
    --diffusion-checkpoint-path "${CHECKPOINT_PATH}" \
    --num-candidates 1000 \
    --guidance-scale "${guidance_scale}" \
    --clip-guidance "${clip_guidance}" \
    --guide-every 1 \
    --guidance-alpha-bar-lower 1e-4 \
    --guidance-alpha-bar-upper 0.999 \
    --diffusion-eta 0 \
    --no-strict-on-hash \
    --tag diffusion-test \
    --sufix colab-diffusion \
    2>&1 | tee "${log_file}"
  then
    echo "Run finished successfully." | tee -a "${log_file}"
    return 0
  else
    echo "Run failed: ${run_name}" | tee -a "${log_file}"
    return 1
  fi
}

status=0

for seed in {1..5}; do
  run_one "albuterol_similarity"   "${seed}" 15 40 || status=1
  sync_results

  run_one "amlodipine_mpo"         "${seed}" 15 40 || status=1
  sync_results

  run_one "celecoxib_rediscovery"  "${seed}" 15 40 || status=1
  sync_results

  run_one "deco_hop"               "${seed}" 15 40 || status=1
  sync_results
done

exit "${status}"