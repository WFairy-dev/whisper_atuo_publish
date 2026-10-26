"""语音任务队列。每一路推理是独立进程，各加载一份模型，一次只领一条任务。"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Optional

from . import config
from . import result_store
from .infer_worker import worker_main

log = logging.getLogger("asr_server.engine")

_BJ = timezone(timedelta(hours=8))

# 查询接口对外的状态码。内部仍用 status_name 这几个英文。
STATUS_CODE = {
    "queued": 2,
    "running": 3,
    "completed": 0,
    "failed": 1,
    "cancelled": 4,
}
SUCCESS_CODE = 100000
TRANSCRIBE_FAIL_CODE = 100102
WORKER_EXIT_CODE = 100103
DISPATCH_FAIL_CODE = 100104


def canonical_status(status: Any) -> str:
    if status in (0, "0", "done", "completed"):
        return "completed"
    if status in (1, "1", "error", "failed"):
        return "failed"
    if status in (2, "2", "queued"):
        return "queued"
    if status in (3, "3", "running"):
        return "running"
    if status in (4, "4", "cancelled"):
        return "cancelled"
    return str(status or "")


def format_beijing(ts: Any) -> Optional[str]:
    """时刻写成北京时间，例如 2026-10-24 16:19:28.339。"""
    if ts is None or ts == "":
        return None
    if isinstance(ts, str):
        return ts
    moment = datetime.fromtimestamp(float(ts), tz=_BJ)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

_FFMPEG_CANDIDATES = [
    "/root/miniconda3/envs/whisper-turbo/bin",
    "/usr/bin",
    "/usr/local/bin",
]


class TaskAlreadyFinished(Exception):
    """任务已经成功或失败，不能再取消。"""

    def __init__(self, status: str) -> None:
        self.status = status
        super().__init__(status)


class InvalidTaskRequest(Exception):
    """调用方传入的任务号或文件路径不符合约定。"""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class DuplicateTaskId(Exception):
    """同一个任务号已经在排队、转写或已完成，不能再提交一次。"""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(task_id)


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return
    for d in _FFMPEG_CANDIDATES:
        candidate = Path(d) / "ffmpeg"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            os.environ["PATH"] = f"{d}{os.pathsep}{os.environ.get('PATH', '')}"
            log.info("已将 ffmpeg 目录加入 PATH: %s", d)
            return
    log.warning("未找到 ffmpeg，转写可能失败")


def _purge_temp_dir() -> int:
    """启动时清理临时目录残留（崩溃未删的上传文件）。"""
    if not config.TEMP_DIR.is_dir():
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    for p in config.TEMP_DIR.iterdir():
        if not p.is_file():
            continue
        try:
            p.unlink()
            removed += 1
        except OSError as e:
            log.warning("启动清理临时文件失败 path=%s err=%s", p, e)
    if removed:
        log.info("启动清理临时文件 %d 个", removed)
    return removed


def _audio_duration_s(path: Path) -> Optional[float]:
    """读取音视频时长，单位秒。读不到时返回空，不因此拒绝任务。"""
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            text=True,
            timeout=20,
        )
        value = float(out.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if value < 0:
        return None
    return round(value, 2)


def _check_source_file_path(source_file_path: str) -> str:
    path = (source_file_path or "").strip()
    if not path or len(path) > 1024:
        raise InvalidTaskRequest("source_file_path 不能为空，且最长 1024 个字符")
    if path.endswith(("/", "\\")):
        raise InvalidTaskRequest("source_file_path 必须是单个文件路径，不接受文件夹")
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    if name in ("", ".", ".."):
        raise InvalidTaskRequest("source_file_path 必须是单个文件路径，不接受文件夹")
    if "/" not in normalized:
        raise InvalidTaskRequest(
            "source_file_path 必须是带目录的完整路径，并使用正斜杠，例如 D:/视频/a.mp4"
        )
    return path


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _request_log(task: "Task") -> dict[str, Any]:
    wait = None
    if task.started_at:
        wait = round(task.started_at - task.created_at, 3)
    return {
        "task_id": task.task_id,
        "source_file_path": task.source_file_path,
        "file_size_bytes": task.file_size,
        "audio_duration_s": task.audio_duration_s,
        "created_at": format_beijing(task.created_at),
        "queue_len": task.queue_len_at_start if task.queue_len_at_start is not None else 0,
        "queue_wait_s": wait,
    }


def _log_start(task: "Task") -> None:
    log.info(
        "【开始】 task_id=%s lane=%s 请求参数=%s",
        task.task_id,
        task.lane_id,
        _compact_json(_request_log(task)),
    )


def _with_previous_status(payload: dict[str, Any], previous: str) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key, value in payload.items():
        ordered[key] = value
        if key == "status_name":
            ordered["previous_status"] = previous
    if "previous_status" not in ordered:
        ordered["previous_status"] = previous
    return ordered


def _log_end(payload: dict[str, Any], *, lane: Optional[int]) -> None:
    task_id = payload.get("task_id")
    body = _compact_json(payload)
    failed = payload.get("status_name") == "failed"
    if lane is None:
        message = "【结束】 task_id=%s 返回参数=%s"
        args: tuple[Any, ...] = (task_id, body)
    else:
        message = "【结束】 task_id=%s lane=%s 返回参数=%s"
        args = (task_id, lane, body)
    if failed:
        log.error(message, *args)
    else:
        log.info(message, *args)


def _gpu_mem() -> Optional[dict[str, float]]:
    """用 nvidia-smi 读显存，避免父进程初始化显卡。"""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
        line = out.strip().splitlines()[0]
        free_s, total_s = [part.strip() for part in line.split(",")]
        return {"free_mb": round(float(free_s), 1), "total_mb": round(float(total_s), 1)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _public_payload(data: dict[str, Any]) -> dict[str, Any]:
    """把落盘结果整理成当前查询格式。旧记录里的 done、Unix 时间也会换掉。"""
    if data.get("status_name") and isinstance(data.get("created_at"), str):
        out = {k: v for k, v in data.items() if k != "saved_at"}
        if isinstance(out.get("expire_at"), (int, float)):
            out["expire_at"] = format_beijing(out.get("expire_at"))
        return out
    name = canonical_status(data.get("status_name") or data.get("status"))
    created = data.get("created_at", data.get("created_at_s"))
    started = data.get("started_at", data.get("started_at_s"))
    finished = data.get("finished_at", data.get("finished_at_s"))
    payload: dict[str, Any] = {
        "task_id": data.get("task_id"),
        "status": STATUS_CODE.get(name),
        "status_name": name,
        "source_file_path": data.get("source_file_path") or "",
        "file_size_bytes": data.get("file_size_bytes", data.get("file_size")),
        "audio_duration_s": data.get("audio_duration_s"),
        "created_at": format_beijing(created),
        "started_at": format_beijing(started),
        "finished_at": format_beijing(finished),
        "queue_wait_s": data.get("queue_wait_s"),
        "cost_time_s": data.get("cost_time_s", data.get("cost_time")),
        "expire_at": format_beijing(data.get("expire_at", data.get("expire_at_s"))),
        "error_code": data.get("error_code"),
        "error_msg": data.get("error_msg", data.get("error")),
        "result": data.get("result"),
    }
    if name == "completed" and payload["error_code"] is None:
        payload["error_code"] = SUCCESS_CODE
        payload["error_msg"] = "success"
    if name != "completed":
        payload["result"] = data.get("result") if name == "completed" else None
    return payload


def _abandon_queue(q: Any) -> None:
    try:
        q.close()
    except Exception:
        pass
    try:
        q.cancel_join_thread()
    except Exception:
        pass


@dataclass
class Task:
    task_id: str
    file_path: Path
    source_file_path: str = ""
    original_name: str = ""
    file_size: int = 0
    audio_duration_s: Optional[float] = None
    status: str = "queued"  # queued | running | completed | failed | cancelled
    error_code: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    lane_id: Optional[int] = None
    queue_len_at_start: Optional[int] = None
    done_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class Lane:
    """一路推理：自己的进程、自己的模型、一次只做一条。"""

    lane_id: int
    cancel_kill: threading.Event = field(default_factory=threading.Event)
    ready_event: threading.Event = field(default_factory=threading.Event)
    proc_lock: threading.Lock = field(default_factory=threading.Lock)
    busy: bool = False
    loaded: bool = False
    recovering: bool = False
    inflight_id: Optional[str] = None
    proc: Any = None
    thread: Optional[threading.Thread] = None

    def current_proc(self) -> Any:
        with self.proc_lock:
            return self.proc

    def set_proc(self, proc: Any) -> None:
        with self.proc_lock:
            self.proc = proc


class AsrEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._queue: list[str] = []
        self._stop = threading.Event()
        self._accepting = True
        self._mp: Any = None
        self._lanes: list[Lane] = []

    def start(self) -> None:
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        result_store.ensure_dirs()
        _ensure_ffmpeg_on_path()
        _purge_temp_dir()
        result_store.purge_expired()

        if config.WORKERS_REQUESTED != config.MAX_WORKERS:
            log.warning(
                "配置的转写路数=%s 超过本机上限 %s，已按 %s 路启动",
                config.WORKERS_REQUESTED,
                config.MAX_WORKERS,
                config.MAX_WORKERS,
            )

        self._mp = mp.get_context("spawn")
        self._lanes = [Lane(lane_id=i) for i in range(config.MAX_WORKERS)]
        log.info(
            "启动 %d 路推理 name=%s dir=%s device=%s",
            config.MAX_WORKERS,
            config.MODEL_NAME,
            config.MODEL_DIR,
            config.DEVICE,
        )
        for lane in self._lanes:
            lane.thread = threading.Thread(
                target=self._lane_loop,
                args=(lane,),
                name=f"asr-lane-{lane.lane_id}",
                daemon=True,
            )
            lane.thread.start()
        cleaner = threading.Thread(target=self._cleanup_loop, name="asr-cleaner", daemon=True)
        cleaner.start()

        deadline = time.time() + 300
        for lane in self._lanes:
            remain = deadline - time.time()
            if remain <= 0 or not lane.ready_event.wait(timeout=remain):
                log.error("第 %d 路推理未能在时限内就绪", lane.lane_id)
                self.stop()
                raise RuntimeError("推理进程未能在 300 秒内就绪")
        log.info(
            "模型加载完成 workers=%d queue_max=%d result_ttl_h=%.1f",
            config.MAX_WORKERS,
            config.MAX_QUEUE_SIZE,
            config.RESULT_TTL_SEC / 3600,
        )

    def stop(self) -> None:
        log.info("开始退出：停止接新任务")
        self._accepting = False
        self._stop.set()
        for lane in self._lanes:
            self._kill_proc(lane.current_proc())
        for lane in self._lanes:
            thread = lane.thread
            if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=20)
        for lane in self._lanes:
            self._kill_proc(lane.current_proc())

    def _kill_proc(self, proc: Any) -> None:
        if proc is None:
            return
        try:
            alive = proc.is_alive()
        except ValueError:
            return
        if not alive:
            try:
                proc.join(timeout=1)
            except Exception:
                pass
            return
        log.info("结束推理进程 pid=%s", getattr(proc, "pid", None))
        try:
            proc.kill()
        except Exception:
            log.exception("结束推理进程失败")
        try:
            proc.join(timeout=10)
        except Exception:
            log.exception("等待推理进程结束失败")

    def stats(self) -> dict[str, Any]:
        disk = shutil.disk_usage(str(config.TEMP_DIR if config.TEMP_DIR.exists() else "/"))
        gpu_mem = _gpu_mem()
        ready = sum(1 for lane in self._lanes if lane.loaded)
        recovering = any(lane.recovering for lane in self._lanes)
        with self._lock:
            queued = sum(1 for t in self._tasks.values() if t.status == "queued")
            running = sum(1 for t in self._tasks.values() if t.status == "running")
            in_memory = len(self._tasks)
        return {
            "service_version": config.SERVICE_VERSION,
            "model_loaded": ready > 0,
            "model_name": config.MODEL_NAME,
            "device": config.DEVICE,
            "cuda_available": gpu_mem is not None,
            "accepting": self._accepting,
            "recovering": recovering,
            "max_workers": config.MAX_WORKERS,
            "ready_workers": ready,
            "max_queue_size": config.MAX_QUEUE_SIZE,
            "queue_length": queued,
            "running": running,
            "tasks_in_memory": in_memory,
            "results_on_disk": result_store.count_results(),
            "result_ttl_hours": config.RESULT_TTL_SEC / 3600,
            "max_upload_mb": round(config.MAX_UPLOAD_BYTES / (1024 * 1024), 1),
            "api_key_required": bool(config.API_KEY),
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
            },
            "gpu_mem": gpu_mem,
        }

    def submit_file(
        self,
        src_path: Path,
        suffix: str,
        *,
        task_id: str,
        source_file_path: str,
        original_name: str = "",
        file_size: int = 0,
    ) -> Task:
        if not self._accepting:
            raise RuntimeError("服务正在关闭，拒绝新任务")

        task_id = (task_id or "").strip()
        if not result_store.is_safe_task_id(task_id):
            raise InvalidTaskRequest(
                "task_id 只能包含字母、数字、下划线和中横线，长度 1～128，且不能为空"
            )
        source_file_path = _check_source_file_path(source_file_path)

        audio_duration_s = _audio_duration_s(src_path)
        dest = config.TEMP_DIR / f"{task_id}{suffix}"
        with self._lock:
            self._reject_if_duplicate(task_id)
            queued = sum(1 for t in self._tasks.values() if t.status == "queued")
            if queued >= config.MAX_QUEUE_SIZE:
                raise RuntimeError(f"队列已满（上限 {config.MAX_QUEUE_SIZE}）")
            shutil.move(str(src_path), str(dest))
            if not file_size:
                try:
                    file_size = dest.stat().st_size
                except OSError:
                    file_size = 0
            task = Task(
                task_id=task_id,
                file_path=dest,
                source_file_path=source_file_path,
                original_name=original_name,
                file_size=file_size,
                audio_duration_s=audio_duration_s,
            )
            # 移动文件期间可能有相同任务号进来，再查一次。
            try:
                self._reject_if_duplicate(task_id)
            except DuplicateTaskId:
                dest.unlink(missing_ok=True)
                raise
            self._tasks[task_id] = task
            self._queue.append(task_id)
        return task

    def _reject_if_duplicate(self, task_id: str) -> None:
        existing = self._tasks.get(task_id)
        disk = result_store.load(task_id)
        existing_name = existing.status if existing is not None else ""
        disk_name = canonical_status(disk.get("status_name") or disk.get("status")) if disk else ""
        if existing_name in ("queued", "running", "completed") or disk_name in ("queued", "running", "completed"):
            raise DuplicateTaskId(task_id)
        if existing is not None and existing_name in ("cancelled", "failed"):
            self._tasks.pop(task_id, None)
        if disk is not None and disk_name in ("cancelled", "failed"):
            result_store.delete(task_id)

    def get_task_payload(self, task_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is not None:
                return self.task_to_dict(task)

        disk = result_store.load(task_id)
        if disk is not None:
            return _public_payload(disk)
        return None

    def cancel(self, task_id: str) -> Optional[dict[str, Any]]:
        """取消任务并立刻返回。正在转写时只停它所在的那一路。"""
        need_kill = False
        proc = None
        path: Optional[Path] = None
        lane_id: Optional[int] = None
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                disk = result_store.load(task_id)
                if disk is None:
                    return None
                status = canonical_status(disk.get("status_name") or disk.get("status"))
                if status == "cancelled":
                    return _public_payload(disk)
                if status in ("completed", "failed"):
                    raise TaskAlreadyFinished(status)
                return None

            if task.status == "cancelled":
                return _with_previous_status(self.task_to_dict(task), "cancelled")
            if task.status in ("completed", "failed"):
                raise TaskAlreadyFinished(task.status)

            if task.status == "queued":
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
                self._mark_cancelled_locked(task)
                path = task.file_path
                payload = _with_previous_status(self.task_to_dict(task), "queued")
                scope = "queued"
            elif task.status == "running":
                lane_id = task.lane_id
                lane = self._lanes[lane_id] if lane_id is not None and lane_id < len(self._lanes) else None
                if lane is not None:
                    lane.cancel_kill.set()
                    lane.loaded = False
                    lane.recovering = True
                    proc = lane.current_proc()
                    need_kill = True
                self._mark_cancelled_locked(task)
                path = task.file_path
                payload = _with_previous_status(self.task_to_dict(task), "running")
                scope = "running"
            else:
                raise TaskAlreadyFinished(task.status)

        _log_end(payload, lane=None if scope == "queued" else lane_id)
        if path is not None:
            self._cleanup_file(path)
        if need_kill:
            threading.Thread(
                target=self._kill_proc,
                args=(proc,),
                name="asr-cancel-kill",
                daemon=True,
            ).start()
        return payload

    def wait_task(self, task: Task, timeout: float) -> Task:
        task.done_event.wait(timeout=timeout)
        return task

    def task_to_dict(self, task: Task) -> dict[str, Any]:
        name = canonical_status(task.status)
        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "status": STATUS_CODE.get(name),
            "status_name": name,
            "source_file_path": task.source_file_path,
            "file_size_bytes": task.file_size,
            "audio_duration_s": task.audio_duration_s,
            "created_at": format_beijing(task.created_at),
            "started_at": format_beijing(task.started_at),
            "finished_at": format_beijing(task.finished_at),
            "queue_wait_s": None,
            "cost_time_s": None,
            "expire_at": None,
            "error_code": None,
            "error_msg": None,
            "result": None,
        }
        if task.started_at:
            payload["queue_wait_s"] = round(task.started_at - task.created_at, 3)
        if name == "completed" and task.result is not None:
            payload["result"] = task.result
            if task.started_at and task.finished_at:
                payload["cost_time_s"] = round(task.finished_at - task.started_at, 3)
            payload["expire_at"] = format_beijing((task.finished_at or time.time()) + config.RESULT_TTL_SEC)
            payload["error_code"] = SUCCESS_CODE
            payload["error_msg"] = "success"
        elif name == "failed":
            if task.finished_at:
                payload["expire_at"] = format_beijing(task.finished_at + config.RESULT_TTL_SEC)
            payload["error_code"] = task.error_code or TRANSCRIBE_FAIL_CODE
            payload["error_msg"] = task.error or "转写失败"
        elif name == "cancelled" and task.finished_at:
            payload["expire_at"] = format_beijing(task.finished_at + config.RESULT_TTL_SEC)
        return payload

    def _mark_cancelled_locked(self, task: Task) -> None:
        task.status = "cancelled"
        task.finished_at = time.time()
        task.result = None
        task.error = None
        self._persist(task)
        task.done_event.set()

    def _persist(self, task: Task) -> None:
        try:
            result_store.save(task.task_id, self.task_to_dict(task))
        except Exception:
            log.exception("结果落盘失败 task_id=%s", task.task_id)

    def _mark_error(self, task: Task, message: str, error_code: int = TRANSCRIBE_FAIL_CODE) -> None:
        with self._lock:
            if task.status != "running":
                return
            task.status = "failed"
            task.error = message
            task.error_code = error_code
            task.finished_at = time.time()
        self._persist(task)
        _log_end(self.task_to_dict(task), lane=task.lane_id)
        self._cleanup_file(task.file_path)
        task.done_event.set()

    def _take_next(self, lane_id: int) -> Optional[Task]:
        with self._lock:
            while self._queue:
                task_id = self._queue.pop(0)
                task = self._tasks.get(task_id)
                if task is None or task.status != "queued":
                    continue
                task.status = "running"
                task.started_at = time.time()
                task.lane_id = lane_id
                task.queue_len_at_start = len(self._queue)
                return task
        return None

    def _try_dispatch(self, lane: Lane, job_q: Any, proc: Any) -> None:
        if self._stop.is_set() or lane.cancel_kill.is_set() or lane.busy:
            return
        try:
            alive = proc.is_alive()
        except ValueError:
            return
        if not alive:
            return
        task = self._take_next(lane.lane_id)
        if task is None:
            return
        # 取到任务后如果这一路正在被取消，放回队首，避免被一起停掉。
        if lane.cancel_kill.is_set() or self._stop.is_set():
            self._requeue_front(task)
            return
        lane.busy = True
        lane.inflight_id = task.task_id
        _log_start(task)
        try:
            job_q.put({"task_id": task.task_id, "path": str(task.file_path)}, timeout=10)
        except Exception as e:
            lane.busy = False
            lane.inflight_id = None
            self._mark_error(task, f"下发推理失败: {e}", DISPATCH_FAIL_CODE)

    def _handle_result(self, lane: Lane, msg: dict[str, Any]) -> None:
        task_id = str(msg.get("task_id", ""))
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != "running":
                log.info(
                    "忽略过期推理结果 lane=%d task_id=%s status=%s",
                    lane.lane_id,
                    task_id,
                    None if task is None else task.status,
                )
                return
            if msg.get("ok"):
                result = msg.get("result") or {}
                task.result = result
                task.status = "completed"
                task.error_code = SUCCESS_CODE
                task.error = None
                task.finished_at = time.time()
            else:
                task.error = str(msg.get("error") or "转写失败")
                task.error_code = TRANSCRIBE_FAIL_CODE
                task.status = "failed"
                task.finished_at = time.time()
        self._persist(task)
        _log_end(self.task_to_dict(task), lane=task.lane_id)
        self._cleanup_file(task.file_path)
        task.done_event.set()

    def _requeue_front(self, task: Task) -> None:
        with self._lock:
            if task.status not in ("running", "queued"):
                return
            task.status = "queued"
            task.started_at = None
            task.lane_id = None
            if task.task_id not in self._queue:
                self._queue.insert(0, task.task_id)
        log.info("任务放回队首 task_id=%s", task.task_id)

    def _on_lane_death(self, lane: Lane, *, intentional: bool) -> None:
        inflight_id = lane.inflight_id
        lane.busy = False
        lane.inflight_id = None
        lane.loaded = False
        lane.recovering = True
        with self._lock:
            task = self._tasks.get(inflight_id) if inflight_id else None
        if task is not None and task.status == "running":
            if intentional:
                self._requeue_front(task)
            else:
                self._mark_error(task, "推理进程意外退出", WORKER_EXIT_CODE)
        lane.cancel_kill.clear()

    def _spawn_child(self, lane: Lane) -> tuple[Any, Any, Any]:
        assert self._mp is not None
        job_q = self._mp.Queue()
        result_q = self._mp.Queue()
        proc = self._mp.Process(
            target=worker_main,
            args=(job_q, result_q, config.MODEL_NAME, str(config.MODEL_DIR), config.DEVICE),
            name=f"asr-infer-{lane.lane_id}",
            daemon=False,
        )
        proc.start()
        lane.set_proc(proc)
        log.info("推理进程已启动 lane=%d pid=%s", lane.lane_id, proc.pid)
        return job_q, result_q, proc

    def _lane_loop(self, lane: Lane) -> None:
        while not self._stop.is_set():
            ever_ready = lane.ready_event.is_set()
            lane.loaded = False
            lane.recovering = ever_ready
            try:
                job_q, result_q, proc = self._spawn_child(lane)
            except Exception:
                log.exception("无法启动推理进程 lane=%d", lane.lane_id)
                if self._stop.wait(3):
                    return
                continue

            got_ready = False
            try:
                while not self._stop.is_set():
                    try:
                        msg = result_q.get(timeout=0.2)
                    except Empty:
                        try:
                            alive = proc.is_alive()
                        except ValueError:
                            alive = False
                        if not alive:
                            break
                        if got_ready and not lane.busy:
                            self._try_dispatch(lane, job_q, proc)
                        continue
                    except (EOFError, OSError):
                        log.warning("推理进程结果通道已断开 lane=%d", lane.lane_id)
                        break

                    kind = msg.get("type")
                    if kind == "ready":
                        got_ready = True
                        lane.loaded = True
                        lane.recovering = False
                        lane.ready_event.set()
                        log.info("推理进程就绪 lane=%d pid=%s", lane.lane_id, proc.pid)
                        self._try_dispatch(lane, job_q, proc)
                    elif kind == "fatal":
                        log.error("推理进程启动失败 lane=%d err=%s", lane.lane_id, msg.get("error"))
                        break
                    elif kind == "result":
                        self._handle_result(lane, msg)
                        lane.busy = False
                        lane.inflight_id = None
                        try:
                            alive = proc.is_alive()
                        except ValueError:
                            alive = False
                        if alive:
                            self._try_dispatch(lane, job_q, proc)
                    else:
                        log.warning("忽略未知推理消息 lane=%d kind=%s", lane.lane_id, kind)
            finally:
                try:
                    proc.join(timeout=5)
                except Exception:
                    pass
                _abandon_queue(job_q)
                _abandon_queue(result_q)

            if self._stop.is_set():
                break

            intentional = lane.cancel_kill.is_set()
            if intentional:
                log.info("第 %d 路已因取消结束，准备重新加载模型", lane.lane_id)
            else:
                log.warning("第 %d 路推理进程退出，准备重新加载模型", lane.lane_id)
            self._on_lane_death(lane, intentional=intentional)
            pause = 1.0 if intentional else (0.5 if got_ready else 3.0)
            if self._stop.wait(pause):
                break

    @staticmethod
    def _cleanup_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("删除临时文件失败: %s", path)

    def _cleanup_loop(self) -> None:
        while not self._stop.wait(60):
            now = time.time()
            with self._lock:
                expired = [
                    tid
                    for tid, t in self._tasks.items()
                    if t.status in ("completed", "failed", "cancelled")
                    and t.finished_at
                    and now - t.finished_at > config.RESULT_TTL_SEC
                ]
                for tid in expired:
                    del self._tasks[tid]
            if expired:
                log.info("清理内存过期任务 %d 个", len(expired))
            result_store.purge_expired()


engine = AsrEngine()
