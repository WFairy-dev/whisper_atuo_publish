#!/usr/bin/env bash
# 在 AutoDL（GPU）上启动文字识别服务。与语音服务分开进程、分开环境。
#
# 必配建议：
#   export ASR_API_KEY='与语音服务相同的密钥'
# 可选：
#   export OCR_PORT=6008
#   export OCR_DET_BATCH=4
#   export OCR_REC_BATCH=16
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export OCR_HOST="${OCR_HOST:-0.0.0.0}"
export OCR_PORT="${OCR_PORT:-6008}"
export OCR_MODEL_DIR="${OCR_MODEL_DIR:-/root/autodl-tmp/models/ocr}"
export PADDLE_PDX_CACHE_HOME="${PADDLE_PDX_CACHE_HOME:-$OCR_MODEL_DIR}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-True}"
export OCR_DEVICE="${OCR_DEVICE:-gpu:0}"
export OCR_ENABLE_DOCS="${OCR_ENABLE_DOCS:-0}"
export OCR_LOG_RETENTION_DAYS="${OCR_LOG_RETENTION_DAYS:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${OCR_API_KEY:-${ASR_API_KEY:-}}" ]]; then
  echo "[warn] 未设置 OCR_API_KEY 或 ASR_API_KEY。内网调用时请务必配置。" >&2
fi

PY="${OCR_PYTHON:-/root/autodl-tmp/envs/ocr-pp/bin/python}"
LOG_ROOT="${OCR_LOG_DIR:-$ROOT/ocr_server/logs}"
DAY="$(date +%F)"
STAMP="$(date +%H-%M-%S)"
mkdir -p "$LOG_ROOT/$DAY"
RUN_DIR="$LOG_ROOT/$DAY/$STAMP"
suffix=2
while ! mkdir "$RUN_DIR" 2>/dev/null; do
  RUN_DIR="$LOG_ROOT/$DAY/${STAMP}-${suffix}"
  suffix=$((suffix + 1))
done
export OCR_LOG_RUN_DIR="$RUN_DIR"
exec >>"$RUN_DIR/stdout.log" 2>&1
exec "$PY" -m ocr_server
