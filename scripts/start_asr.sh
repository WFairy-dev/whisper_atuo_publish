#!/usr/bin/env bash
# 在 AutoDL（GPU）上启动语音转写服务，供公司内网 CPU 流水线调用。
#
# 必配建议：
#   export ASR_API_KEY='换成足够长的随机串'
# 可选：
#   同时转写几路：改 asr_server/service.conf 里的 workers，改完重启。
#   临时覆盖：export ASR_MAX_WORKERS=2
#   export ASR_PORT=6006
#   export ASR_ENABLE_DOCS=0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export ASR_HOST="${ASR_HOST:-0.0.0.0}"
export ASR_PORT="${ASR_PORT:-6006}"
export ASR_MAX_QUEUE_SIZE="${ASR_MAX_QUEUE_SIZE:-200}"
export ASR_MODEL_NAME="${ASR_MODEL_NAME:-turbo}"
export ASR_MODEL_DIR="${ASR_MODEL_DIR:-/root/autodl-tmp/models/whisper}"
export ASR_DEVICE="${ASR_DEVICE:-cuda}"
export ASR_RESULT_TTL_SEC="${ASR_RESULT_TTL_SEC:-86400}"   # 结果文档 24h
export ASR_LOG_RETENTION_DAYS="${ASR_LOG_RETENTION_DAYS:-3}"  # 日志保留 3 天
export ASR_ENABLE_DOCS="${ASR_ENABLE_DOCS:-0}"
export ASR_MAX_UPLOAD_BYTES="${ASR_MAX_UPLOAD_BYTES:-524288000}"  # 500MB
export PATH="/root/miniconda3/envs/whisper-turbo/bin:${PATH}"

if [[ -z "${ASR_API_KEY:-}" ]]; then
  echo "[warn] 未设置 ASR_API_KEY。内网调用 AutoDL 时请务必配置。" >&2
fi

PY="${PY:-/root/miniconda3/envs/whisper-turbo/bin/python}"
LOG_ROOT="${ASR_LOG_DIR:-$ROOT/asr_server/logs}"
DAY="$(date +%F)"
STAMP="$(date +%H-%M-%S)"
mkdir -p "$LOG_ROOT/$DAY"
# 每次启动一个目录，精确到秒。同一秒再启动则加 -2、-3。
RUN_DIR="$LOG_ROOT/$DAY/$STAMP"
suffix=2
while ! mkdir "$RUN_DIR" 2>/dev/null; do
  RUN_DIR="$LOG_ROOT/$DAY/${STAMP}-${suffix}"
  suffix=$((suffix + 1))
done
export ASR_LOG_RUN_DIR="$RUN_DIR"
# 控制台和业务日志放同一次启动的目录里。
exec >>"$RUN_DIR/stdout.log" 2>&1
exec "$PY" -m asr_server
