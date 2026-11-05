# 由启动脚本引用。shell 守着服务进程：进程自己退出就重新拉起；
# 收到停止信号则把整组进程结束，不再启动。
supervise_service() {
  local log_root="$1"
  local env_name="$2"
  local title="$3"
  shift 3
  if [[ $# -lt 1 ]]; then
    echo "缺少启动命令" >&2
    return 1
  fi

  mkdir -p "$log_root"
  exec 9>"$log_root/supervise.lock"
  if ! flock -n 9; then
    echo "${title}的监听已在运行。要停止请执行: kill \$(cat ${log_root}/supervise.pid)" >&2
    return 1
  fi
  local pidfile="$log_root/supervise.pid"
  echo $$ >"$pidfile"
  trap "rm -f $(printf '%q' "$pidfile")" EXIT

  _svc_stop=0
  _svc_child=0
  _svc_run_dir=""
  local base_delay="${SUPERVISE_RESTART_SEC:-3}"
  local max_delay=60
  local min_up="${SUPERVISE_MIN_UP_SEC:-60}"
  local delay="$base_delay"

  _svc_note() {
    local line
    line="$(date '+%F %T') $*"
    echo "$line" >&2
    echo "$line" >>"$log_root/supervise.log"
    if [[ -n "${_svc_run_dir}" && -d "${_svc_run_dir}" ]]; then
      echo "$line" >>"${_svc_run_dir}/stdout.log"
    fi
  }

  _svc_reap() {
    local pid="$1"
    [[ "$pid" -gt 0 ]] || return 0
    if kill -0 -"$pid" 2>/dev/null; then
      kill -TERM -"$pid" 2>/dev/null || true
      sleep 1
      kill -KILL -"$pid" 2>/dev/null || true
    fi
  }

  _svc_on_stop() {
    _svc_stop=1
    if [[ "${_svc_child}" -gt 0 ]]; then
      kill -TERM -"${_svc_child}" 2>/dev/null || true
    fi
  }
  trap _svc_on_stop INT TERM
  trap 'rm -f "$log_root/supervise.pid"' EXIT

  while [[ "$_svc_stop" -eq 0 ]]; do
    local day stamp suffix started lived code remaining
    day="$(date +%F)"
    stamp="$(date +%H-%M-%S)"
    mkdir -p "$log_root/$day"
    _svc_run_dir="$log_root/$day/$stamp"
    suffix=2
    while ! mkdir "$_svc_run_dir" 2>/dev/null; do
      _svc_run_dir="$log_root/$day/${stamp}-${suffix}"
      suffix=$((suffix + 1))
    done
    export "${env_name}=${_svc_run_dir}"
    _svc_note "启动${title}，日志目录 ${_svc_run_dir}"
    setsid "$@" >>"${_svc_run_dir}/stdout.log" 2>&1 &
    _svc_child=$!
    started=$SECONDS
    set +e
    wait "$_svc_child"
    code=$?
    set -e
    lived=$((SECONDS - started))
    _svc_reap "$_svc_child"
    _svc_child=0
    if [[ "$_svc_stop" -eq 1 ]]; then
      _svc_note "收到停止信号，${title}已停止，不再启动"
      break
    fi
    if (( lived >= min_up )); then
      delay="$base_delay"
    fi
    _svc_note "${title}进程退出，退出码 ${code}，${delay} 秒后重新启动"
    remaining="$delay"
    while [[ "$_svc_stop" -eq 0 && "$remaining" -gt 0 ]]; do
      sleep 1 || true
      remaining=$((remaining - 1))
    done
    if [[ "$_svc_stop" -eq 1 ]]; then
      _svc_note "收到停止信号，${title}不再启动"
      break
    fi
    if (( lived < min_up )); then
      delay=$((delay * 2))
      if (( delay > max_delay )); then
        delay=$max_delay
      fi
    fi
  done
}
