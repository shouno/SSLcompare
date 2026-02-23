#!/usr/bin/env bash

set -euo pipefail

# ---
# User configuration
# ---
PROJECT="${PROJECT:-sslcompare}"

# LOG_ROOT
LOCAL_ROOT="${LOCAL_ROOT:-/workspace/runs_local}"

# NFS_ARCHIVES
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/train_logs}"

# rsync 後にローカルrun を消してディスク保持するかのフラグ
CLEAN_LOCAL="${CLEAN_LOCAL:-1}"

# YAML config layout 設定は configs/* へ移行
CONFIG_BASE="${CONFIG_BASE:-configs/base.yaml}"
CONFIG_METHOD_DIR="${CONFIG_METHOD_DIR:-configs/methods}"
CONFIG_DATASET_DIR="${CONFIG_DATASET_DIR:-configs/datasets}"

# YAML を読み込む→train へ流すは，scripts/run_from_yaml.py へ
RUNNER_PY="${RUNNER_PY:-scripts/run_from_yaml.py}"   # YAML→CLI runner
TRAIN_PY="${TRAIN_PY:-train.py}"                    # your train.py path (used by runner)


# ---
# 手法
#
METHOD="${1:-}"
DATASET="${2:-}"
DATA_DIR="${3:-}"

if [[ -z "${METHOD}" || -z "${DATASET}" || -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <method> <dataset> <data_dir> [extra train.py args...]"
  echo "Example: $0 simclr cifar10 /datasets/cifar10 --max_epochs 200 --batch_size 256"
  exit 2
fi
shift 3
EXTRA_ARGS=("$@")

# ---
# Run naming
# ---
TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${METHOD}_${DATASET}_${TS}}"

CHECKPOINT_ROOT="${LOCAL_ROOT}/checkpoints"
RUN_DIR="${CHECKPOINT_ROOT}/${RUN_ID}"

# wandb の save_dir を run_dir にしているなら、wandb も RUN_DIR 配下に入る
# （もし wandb を別に逃がしたいなら、train.py 側で save_dir を変える）

echo "[run] RUN_ID=${RUN_ID}"
echo "[run] RUN_DIR=${RUN_DIR}"
echo "[run] ARCHIVE=${ARCHIVE_ROOT}/${RUN_ID}"

mkdir -p "${RUN_DIR}"
mkdir -p "${ARCHIVE_ROOT}"

# ---
# Resolve config files
# ---
METHOD_CFG="${CONFIG_METHOD_DIR}/${METHOD}.yaml"
DATASET_CFG="${CONFIG_DATASET_DIR}/${DATASET}.yaml"

[[ -f "${CONFIG_BASE}" ]]   || { echo "[ERROR] Missing ${CONFIG_BASE}"; exit 2; }
[[ -f "${METHOD_CFG}" ]]    || { echo "[ERROR] Missing ${METHOD_CFG}"; exit 2; }
[[ -f "${DATASET_CFG}" ]]   || { echo "[ERROR] Missing ${DATASET_CFG}"; exit 2; }
[[ -f "${RUNNER_PY}" ]]     || { echo "[ERROR] Missing ${RUNNER_PY}"; exit 2; }

# Provide run_id & checkpoint_root to runner via env (runner can inject if config has run_name:auto)
export RUN_ID
export CHECKPOINT_ROOT

# --------------------
# Train
# --------------------
TRAIN_PY=scripts/train.py

set +e

python3 "${RUNNER_PY}" \
  --train_py "${TRAIN_PY}" \
  --config "${CONFIG_BASE}" \  # 基本設定
  --config "${METHOD_CFG}"  \  # 学習方法
  --config "${DATASET_CFG}" \  # データセットに関して
  -- \
  --project "${PROJECT}" \
  --run_name "${RUN_ID}" \
  --data_dir "${DATA_DIR}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  "${EXTRA_ARGS[@]}"
TRAIN_RC=$?
set -e

echo "[run] train.py exit_code=${TRAIN_RC}"

# --------------------
# Sync (always try; even if training failed)
# --------------------
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
mkdir -p "${ARCHIVE_DIR}"

echo "[sync] rsync ${RUN_DIR}/ -> ${ARCHIVE_DIR}/"
rsync -a --info=progress2 --partial --inplace "${RUN_DIR}/" "${ARCHIVE_DIR}/"
RSYNC_RC=$?

if [[ ${RSYNC_RC} -ne 0 ]]; then
  echo "[ERROR] rsync failed (code=${RSYNC_RC}). Keeping local run dir: ${RUN_DIR}"
  exit ${TRAIN_RC}
fi

# 軽量チェック（サイズ比較）
LOCAL_SIZE=$(du -sb "${RUN_DIR}" | awk '{print $1}')
REMOTE_SIZE=$(du -sb "${ARCHIVE_DIR}" | awk '{print $1}')

if [[ "${LOCAL_SIZE}" != "${REMOTE_SIZE}" ]]; then
  echo "[WARNING] size mismatch local=${LOCAL_SIZE} remote=${REMOTE_SIZE}. Keeping local run dir: ${RUN_DIR}"
  exit ${TRAIN_RC}
fi

echo "[sync] verified OK (local=${LOCAL_SIZE} remote=${REMOTE_SIZE})"

if [[ "${CLEAN_LOCAL}" == "1" ]]; then
  echo "[clean] removing local run dir ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
else
  echo "[clean] CLEAN_LOCAL=0 -> keeping local run dir ${RUN_DIR}"
fi

exit ${TRAIN_RC}

echo "[run] train.py exit_code=${TRAIN_RC}"

