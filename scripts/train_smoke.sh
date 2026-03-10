#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_BIN="${PY_BIN:-/usr/bin/python3}"

RUN_TAG="smoke_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${ROOT_DIR}/runs/${RUN_TAG}"
RESUME_RUN_DIR="${ROOT_DIR}/runs/${RUN_TAG}_resume"
TRAIN_CONFIG="${ROOT_DIR}/configs/train/ppo_selfplay_rllib_smoke.yaml"
BENCHMARK_CONFIG="${ROOT_DIR}/configs/eval/smoke.yaml"
EXPERIMENT_NAME="ppo_selfplay_smoke"
CHECKPOINT_DIR="${RUN_DIR}/${EXPERIMENT_NAME}"

echo "[smoke] root=${ROOT_DIR}"
echo "[smoke] python=${PY_BIN}"

echo "[smoke] unittest"
PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" -m unittest discover -s "${ROOT_DIR}/tests" -v

echo "[smoke] sim regression"
PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" -m mahjong_ai.cli.main sim --games 5 --seed 1

echo "[smoke] train 1 iter"
PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" -m mahjong_ai.cli.main train-rllib \
  --config "${TRAIN_CONFIG}" \
  --num-iterations 1 \
  --checkpoint-every 1 \
  --eval-every 1 \
  --eval-games 1 \
  --run-dir "${RUN_DIR}"

if [[ ! -f "${CHECKPOINT_DIR}/rllib_checkpoint.json" ]]; then
  echo "[smoke] ERROR: checkpoint not found at ${CHECKPOINT_DIR}" >&2
  exit 1
fi

echo "[smoke] benchmark first checkpoint"
PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" -m mahjong_ai.cli.main eval-benchmark \
  --config "${BENCHMARK_CONFIG}" \
  --checkpoint "${CHECKPOINT_DIR}"

echo "[smoke] resume 1 iter"
PYTHONPATH="${ROOT_DIR}/src" "${PY_BIN}" -m mahjong_ai.cli.main train-rllib \
  --config "${TRAIN_CONFIG}" \
  --num-iterations 1 \
  --checkpoint-every 1 \
  --eval-every 1 \
  --eval-games 1 \
  --run-dir "${RESUME_RUN_DIR}" \
  --resume-from "${CHECKPOINT_DIR}"

echo "[smoke] done"
echo "[smoke] first_run=${RUN_DIR}"
echo "[smoke] resume_run=${RESUME_RUN_DIR}"
