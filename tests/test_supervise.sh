#!/usr/bin/env bash
# 进程自己退出会再启动；停止信号会结束整组进程且不再启动；不能同时开两个监听。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../scripts/supervise.sh
source "$ROOT/scripts/supervise.sh"

tmp="$(mktemp -d)"
cleanup() {
  local pid
  for pid in $(jobs -p); do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

export COUNT="$tmp/count"
export CHILD="$tmp/child"
cat >"$tmp/crash.sh" << 'EOF'
#!/usr/bin/env bash
echo start >> "$COUNT"
if [[ "$(wc -l < "$COUNT")" -eq 1 ]]; then
  sleep 300 &
  echo $! > "$CHILD"
  exit 7
fi
sleep 300
EOF
chmod +x "$tmp/crash.sh"

SUPERVISE_RESTART_SEC=1 SUPERVISE_MIN_UP_SEC=3600 \
  supervise_service "$tmp/logs" TEST_LOG_RUN_DIR "测试" "$tmp/crash.sh" &
sup=$!

for _ in $(seq 1 50); do
  if [[ -f "$COUNT" ]] && [[ "$(wc -l < "$COUNT")" -ge 2 ]]; then
    break
  fi
  sleep 0.2
done
[[ "$(wc -l < "$COUNT")" -ge 2 ]]
[[ -f "$CHILD" ]]
old="$(cat "$CHILD")"
for _ in $(seq 1 30); do
  if ! kill -0 "$old" 2>/dev/null; then
    break
  fi
  sleep 0.2
done
kill -0 "$old" 2>/dev/null && exit 1

kill -TERM "$sup"
wait "$sup" || true
sleep 0.5
after="$(wc -l < "$COUNT")"
sleep 1.5
[[ "$(wc -l < "$COUNT")" -eq "$after" ]]
[[ ! -f "$tmp/logs/supervise.pid" ]]
grep -q "重新启动" "$tmp/logs/supervise.log"

rm -f "$COUNT" "$CHILD"
cat >"$tmp/hold.sh" << 'EOF'
#!/usr/bin/env bash
echo start >> "$COUNT"
sleep 300
EOF
chmod +x "$tmp/hold.sh"
SUPERVISE_RESTART_SEC=1 \
  supervise_service "$tmp/logs2" TEST_LOG_RUN_DIR "测试" "$tmp/hold.sh" &
sup2=$!
for _ in $(seq 1 30); do
  [[ -f "$COUNT" ]] && break
  sleep 0.2
done
[[ -f "$COUNT" ]]
set +e
SUPERVISE_RESTART_SEC=1 \
  supervise_service "$tmp/logs2" TEST_LOG_RUN_DIR "测试" "$tmp/hold.sh"
second=$?
set -e
[[ "$second" -ne 0 ]]
kill -TERM "$sup2"
wait "$sup2" || true

echo "监听测试通过"
