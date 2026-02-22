#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="$1"
REMOTE_DIR="$2"

echo "[sync] ${LOCAL_DIR} -> ${REMOTE_DIR}"

# 1. rsync（部分転送対応つき）
rsync -a --info=progress2 --partial --inplace \
  "${LOCAL_DIR}/" "${REMOTE_DIR}/"

RS=$?

if [ $RS -ne 0 ]; then
  echo "[ERROR] rsync failed. Local data kept."
  exit 1
fi

# 2. 軽量な整合チェック（サイズ比較）
LOCAL_SIZE=$(du -sb "${LOCAL_DIR}" | awk '{print $1}')
REMOTE_SIZE=$(du -sb "${REMOTE_DIR}" | awk '{print $1}')

if [ "$LOCAL_SIZE" -ne "$REMOTE_SIZE" ]; then
  echo "[WARNING] Size mismatch. Not deleting local copy."
  exit 1
fi

echo "[OK] Sync verified. Removing local copy."
rm -rf "${LOCAL_DIR}"