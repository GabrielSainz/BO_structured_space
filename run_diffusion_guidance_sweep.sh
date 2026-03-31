#!/usr/bin/env bash

set -u

SEED=1
FUNCTION_NAME="albuterol_similarity"
CHECKPOINT_PATH="data/trained_models/training_diffusion_on_zinc_250k/latent_diffusion_latent_dim-128-seed-0.pt"

# guidance_scale clip_guidance diffusion_eta guide_every alpha_bar_lower alpha_bar_upper
# Wider sweep around stronger guidance and larger clipping, plus a few
# stochasticity / guidance-window variants.
COMBINATIONS=(
  "1 1 0.0 1 1e-4 0.999"
  "1 5 0.0 1 1e-4 0.999"
  "1 10 0.0 1 1e-4 0.999"
  "5 1 0.0 1 1e-4 0.999"
  "5 5 0.0 1 1e-4 0.999"
  "5 10 0.0 1 1e-4 0.999"
  "5 20 0.0 1 1e-4 0.999"
  "10 5 0.0 1 1e-4 0.999"
  "10 10 0.0 1 1e-4 0.999"
  "10 20 0.0 1 1e-4 0.999" 
  "10 30 0.0 1 1e-4 0.999" 
  "10 40 0.0 1 1e-4 0.999"  # **
  "10 50 0.0 1 1e-4 0.999"
  "15 5 0.0 1 1e-4 0.999"
  "15 10 0.0 1 1e-4 0.999"
  "15 20 0.0 1 1e-4 0.999"
  "15 30 0.0 1 1e-4 0.999"
  "15 40 0.0 1 1e-4 0.999"
  "15 50 0.0 1 1e-4 0.999"
  "20 5 0.0 1 1e-4 0.999"
  "20 10 0.0 1 1e-4 0.999"
  "20 20 0.0 1 1e-4 0.999"
  "20 30 0.0 1 1e-4 0.999"
  "20 40 0.0 1 1e-4 0.999" # **
  "20 50 0.0 1 1e-4 0.999"
  "25 30 0.0 1 1e-4 0.999"
  "25 50 0.0 1 1e-4 0.999"
  "30 30 0.0 1 1e-4 0.999"
  "30 50 0.0 1 1e-4 0.999"
  "15 30 0.3 1 1e-4 0.999"
  "15 30 0.6 1 1e-4 0.999"
  "15 30 0.0 2 1e-4 0.999" # **
  "15 30 0.0 1 1e-3 0.995"
)

status=0

for combo in "${COMBINATIONS[@]}"; do
  read -r guidance_scale clip_guidance diffusion_eta guide_every alpha_bar_lower alpha_bar_upper <<< "$combo"

  echo
  echo "============================================================"
  echo "Running seed=${SEED} guidance_scale=${guidance_scale} clip_guidance=${clip_guidance} eta=${diffusion_eta} guide_every=${guide_every} alpha_bar_window=(${alpha_bar_lower}, ${alpha_bar_upper})"
  echo "============================================================"

  if ! python run.py \
    --function-name "${FUNCTION_NAME}" \
    --solver-name cowboys_diffusion \
    --n-dimensions 128 \
    --max-iter 300 \
    --seed "${SEED}" \
    --diffusion-checkpoint-path "${CHECKPOINT_PATH}" \
    --num-candidates 1000 \
    --distillation-n 1024 \
    --guidance-scale "${guidance_scale}" \
    --clip-guidance "${clip_guidance}" \
    --guide-every "${guide_every}" \
    --guidance-alpha-bar-lower "${alpha_bar_lower}" \
    --guidance-alpha-bar-upper "${alpha_bar_upper}" \
    --diffusion-eta "${diffusion_eta}" \
    --no-strict-on-hash \
    --wandb-mode disabled \
    --tag diffusion-guidance-sweep \
    --sufix seed1-guidance-sweep
  then
    echo "Run failed for guidance_scale=${guidance_scale}, clip_guidance=${clip_guidance}, eta=${diffusion_eta}, guide_every=${guide_every}, alpha_bar_window=(${alpha_bar_lower}, ${alpha_bar_upper})"
    status=1
  fi
done

exit "${status}"
