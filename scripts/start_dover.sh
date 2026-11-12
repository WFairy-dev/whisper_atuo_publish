#!/usr/bin/env bash
# 在 AutoDL 上启动清晰度服务（DOVER-Mobile 技术支路，默认处理器）。
# 默认端口 6009（本机），不占用 AutoDL 公网 6006/6008。
#
# 必配建议：
#   export ASR_API_KEY='与语音服务相同的密钥'
# 可选：
#   export DOVER_PORT=6009
#   export DOVER_DEVICE=cpu
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DOVER_HOST="${DOVER_HOST:-0.0.0.0}"
export DOVER_PORT="${DOVER_PORT:-6009}"
export DOVER_DEVICE="${DOVER_DEVICE:-cpu}"
export DOVER_ENABLE_DOCS="${DOVER_ENABLE_DOCS:-0}"
export DOVER_LOG_RETENTION_DAYS="${DOVER_LOG_RETENTION_DAYS:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

if [[ -z "${DOVER_API_KEY:-${ASR_API_KEY:-}}" ]]; then
  echo "[warn] 未设置 DOVER_API_KEY 或 ASR_API_KEY。内网调用时请务必配置。" >&2
fi

PY="${DOVER_PYTHON:-/root/autodl-tmp/envs/dover/bin/python}"
LOG_ROOT="${DOVER_LOG_DIR:-$ROOT/dover_server/logs}"
# shellcheck source=supervise.sh
source "$ROOT/scripts/supervise.sh"
supervise_service "$LOG_ROOT" DOVER_LOG_RUN_DIR "清晰度服务" "$PY" -m dover_server
