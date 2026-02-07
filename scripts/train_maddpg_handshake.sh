#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_maddpg_handshake.sh
# Optional overrides:
#   TOTAL_STEPS=20000000 NUM_ENVS=12 DEVICE=cuda WANDB_MODE=offline bash scripts/train_maddpg_handshake.sh

TOTAL_STEPS="${TOTAL_STEPS:-10000000}"
NUM_ENVS="${NUM_ENVS:-8}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-humanoid-collab}"
WANDB_GROUP="${WANDB_GROUP:-maddpg}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_NAME="${RUN_NAME:-maddpg_handshake_loco_s0}"
LOG_DIR="${LOG_DIR:-runs/maddpg_handshake_loco_s0}"
SAVE_DIR="${SAVE_DIR:-checkpoints/maddpg_handshake_loco_s0}"

python scripts/train_maddpg.py \
  --task handshake \
  --backend cpu \
  --physics-profile train_fast \
  --no-fixed-standing \
  --control-mode all \
  --observation-mode proprio \
  --stage 0 \
  --auto-curriculum \
  --num-envs "${NUM_ENVS}" \
  --vec-env-backend shared_memory \
  --total-steps "${TOTAL_STEPS}" \
  --start-steps 8000 \
  --update-after 2000 \
  --update-every 1 \
  --gradient-steps 1 \
  --policy-delay 2 \
  --batch-size 1024 \
  --buffer-size 500000 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor-lr 1e-4 \
  --critic-lr 3e-4 \
  --action-l2 1e-3 \
  --exploration-noise-start 0.25 \
  --exploration-noise-end 0.05 \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --log-dir "${LOG_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${RUN_NAME}" \
  --wandb-mode "${WANDB_MODE}" \
  --print-every-steps 2048 \
  --save-every-steps 50000
