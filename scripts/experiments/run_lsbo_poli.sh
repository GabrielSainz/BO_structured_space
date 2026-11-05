#!/bin/bash
# chmod +x run_lsbo_poli.sh
# ./run_lsbo_poli.sh

set -u -o pipefail

MAX_ITER="${MAX_ITER:-300}"
N_DIMENSIONS="${N_DIMENSIONS:-128}"
TAG="${TAG:-lsbo-poli}"
SUFIX="${SUFIX:-lsbo_iteration}"
WANDB_MODE="${WANDB_MODE:-disabled}"
LOG_DIR="${LOG_DIR:-logs/lsbo_poli}"

mkdir -p "${LOG_DIR}"

PMO_TASKS=(
  albuterol_similarity
  amlodipine_mpo
  #celecoxib_rediscovery
  #deco_hop
  #drd2_docking
  #fexofenadine_mpo
  #gsk3_beta
  #isomer_c7h8n2o2
  #isomer_c9h10n2o2pf2cl
  #jnk3
  #median_1
  #median_2
  #mestranol_similarity
  #osimetrinib_mpo
  #perindopril_mpo
  #ranolazine_mpo
  #rdkit_logp
  #rdkit_qed
  #sa_tdc
  #scaffold_hop
  #sitagliptin_mpo
  #thiothixene_rediscovery
  #troglitazone_rediscovery
  #valsartan_smarts
  #zaleplon_mpo
)

run_one() {
  local function_name="$1"
  local seed="$2"
  local run_name="${function_name}__seed-${seed}"
  local log_file="${LOG_DIR}/${run_name}.log"

  echo
  echo "============================================================"
  echo "Running LSBO function=${function_name} seed=${seed}"
  echo "Log file: ${log_file}"
  echo "============================================================"

  if python run.py \
    --function-name "${function_name}" \
    --solver-name lsbo \
    --n-dimensions "${N_DIMENSIONS}" \
    --max-iter "${MAX_ITER}" \
    --seed "${seed}" \
    --wandb-mode "${WANDB_MODE}" \
    --no-strict-on-hash \
    --tag "${TAG}" \
    --sufix "${SUFIX}" \
    --save-iteration-results \
    --checkpoint-every 1 \
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

for seed in 1; do
  for function_name in "${PMO_TASKS[@]}"; do
    run_one "${function_name}" "${seed}" || status=1
  done
done

exit "${status}"
