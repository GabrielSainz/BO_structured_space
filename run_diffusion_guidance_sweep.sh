#!/usr/bin/env bash

set -u

SEED=1
FUNCTION_NAME="albuterol_similarity"
CHECKPOINT_PATH="data/trained_models/training_diffusion_on_zinc_250k/latent_diffusion_latent_dim-128-seed-0.pt"

# This grid stays close to the currently best-looking region around
# guidance_scale=10 and clip_guidance=30 instead of pushing harder.
COMBINATIONS=(
  "5 30"
  "10 20"
  "10 30"
  "10 40"
  "15 30"
  "20 30"
)

status=0

for combo in "${COMBINATIONS[@]}"; do
  read -r guidance_scale clip_guidance <<< "$combo"

  echo
  echo "============================================================"
  echo "Running seed=${SEED} guidance_scale=${guidance_scale} clip_guidance=${clip_guidance}"
  echo "============================================================"

  if ! python run.py \
    --function-name "${FUNCTION_NAME}" \
    --solver-name cowboys_diffusion \
    --n-dimensions 128 \
    --max-iter 300 \
    --seed "${SEED}" \
    --diffusion-checkpoint-path "${CHECKPOINT_PATH}" \
    --num-diffusion-steps 200 \
    --num-candidates 1000 \
    --distillation-n 1024 \
    --guidance-scale "${guidance_scale}" \
    --clip-guidance "${clip_guidance}" \
    --no-strict-on-hash \
    --tag diffusion-guidance-sweep \
    --sufix seed1-guidance-sweep
  then
    echo "Run failed for guidance_scale=${guidance_scale}, clip_guidance=${clip_guidance}"
    status=1
  fi
done

exit "${status}"
