#!/usr/bin/env bash
# 在 6006 启动 AutoDL 实例内路径代理，把语音与清晰度对外暴露。
# 须先保证本机 7006（语音）、6009（清晰度）已在监听。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROXY_DIR="$ROOT/proxy"
cd "$PROXY_DIR"

if [[ ! -x "$PROXY_DIR/proxy_in_instance" ]]; then
  echo "缺少 proxy_in_instance，正在下载…" >&2
  wget -q -O "$PROXY_DIR/proxy_in_instance" \
    https://autodl-public.ks3-cn-beijing.ksyuncs.com/tool/api-proxy/proxy_in_instance
  chmod +x "$PROXY_DIR/proxy_in_instance"
fi

if [[ ! -f "$PROXY_DIR/config.yaml" ]]; then
  echo "缺少 $PROXY_DIR/config.yaml" >&2
  exit 1
fi

# 粗测后端是否在听，避免代理空转。
for pair in "语音:7006" "清晰度:6009"; do
  name="${pair%%:*}"
  port="${pair##*:}"
  if ! (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1; then
    echo "[warn] 本机 ${port}（${name}）尚未监听，代理启动后对应路径会失败。" >&2
  fi
done

LOG_ROOT="${PROXY_LOG_DIR:-$PROXY_DIR/logs}"
# shellcheck source=supervise.sh
source "$ROOT/scripts/supervise.sh"
supervise_service "$LOG_ROOT" PROXY_LOG_RUN_DIR "路径代理" "$PROXY_DIR/proxy_in_instance"
