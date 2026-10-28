"""一组图片排队，由一条执行线按批计算。取消只停这一组，不关掉模型。"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from . import config
from . import result_store
from .batch import RecognitionCancelled, recognize_images

log = logging.getLogger("ocr_server.engine")

_BJ = timezone(timedelta(hours=8))

STATUS_CODE = {
    "queued": 2,
    "running": 3,
    "completed": 0,
    "failed": 1,
    "cancelled": 4,
}
SUCCESS_CODE = 100000
OCR_FAIL_CODE = 100202


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
    """同一个任务号已经在排队、识别或已完成，不能再提交一次。"""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(task_id)


class QueueFull(Exception):
    """排队组数已到上限。"""


def check_task_id(task_id: str) -> str:
    task_id = (task_id or "").strip()
    if not result_store.is_safe_task_id(task_id):
        raise InvalidTaskRequest(
            "task_id 只能包含字母、数字、下划线和中横线，长度 1～128，且不能为空"
        )
    return task_id


def check_source_file_path(source_file_path: str) -> str:
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


def check_mode(mode: str) -> str:
    chosen = (mode or "").strip().lower() or "det_rec"
    if chosen not in ("det_rec", "rec_only"):
        raise InvalidTaskRequest("mode 只能是 det_rec 或 rec_only")
    return chosen


@dataclass
class Task:
    task_id: str
    file_paths: list[Path]
    mode: str
    source_file_path: str = ""
    file_size: int = 0
    image_count: int = 0
    status: str = "queued"
    error_code: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    queue_len_at_start: Optional[int] = None
    cancel_requested: bool = False
    done_event: threading.Event = field(default_factory=threading.Event)


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _with_previous_status(payload: dict[str, Any], previous: str) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for key, value in payload.items():
        ordered[key] = value
        if key == "status_name":
            ordered["previous_status"] = previous
    if "previous_status" not in ordered:
        ordered["previous_status"] = previous
    return ordered


def _log_end(payload: dict[str, Any]) -> None:
    task_id = payload.get("task_id")
    body = _compact_json(payload)
    if payload.get("status_name") == "failed":
        log.error("【结束】 task_id=%s 返回参数=%s", task_id, body)
    else:
        log.info("【结束】 task_id=%s 返回参数=%s", task_id, body)


def _gpu_mem() -> Optional[dict[str, float]]:
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
        free_s, total_s = [part.strip() for part in out.strip().splitlines()[0].split(",")]
        return {"free_mb": round(float(free_s), 1), "total_mb": round(float(total_s), 1)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _purge_temp_dir() -> int:
    if not config.TEMP_DIR.is_dir():
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        return 0
    removed = 0
    for path in config.TEMP_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            log.warning("启动清理临时文件失败 path=%s err=%s", path, exc)
    return removed


def _public_payload(data: dict[str, Any]) -> dict[str, Any]:
    out = {key: value for key, value in data.items() if key != "saved_at"}
    if isinstance(out.get("expire_at"), (int, float)):
        out["expire_at"] = format_beijing(out.get("expire_at"))
    name = canonical_status(out.get("status_name") or out.get("status"))
    if name == "completed" and out.get("error_code") is None:
        out["error_code"] = SUCCESS_CODE
        out["error_msg"] = "success"
    if name != "completed":
        out["result"] = None
    return out


class OcrEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._queue: list[Task] = []
        self._running = 0
        self._loaded = False
        self._accepting = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._detect = None
        self._recognize = None
        self._read = read_image

    def start(self) -> None:
        from .runtime import detect_images, load_models, recognize_crops

        removed = _purge_temp_dir()
        if removed:
            log.info("启动清理临时文件 %d 个", removed)
        purged = result_store.purge_expired()
        if purged:
            log.info("启动清理过期结果 %d 个", purged)
        detector, recognizer = load_models()
        self._detect = lambda images: detect_images(detector, images)
        self._recognize = lambda specs: recognize_crops(recognizer, specs)
        self._loaded = True
        self._accepting = True
        self._thread = threading.Thread(target=self._loop, name="ocr-infer", daemon=True)
        self._thread.start()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, name="ocr-cleanup", daemon=True)
        self._cleanup_thread.start()
        log.info(
            "文字识别就绪 det_batch=%d rec_batch=%d queue_max=%d",
            config.DET_BATCH,
            config.REC_BATCH,
            config.MAX_QUEUE_SIZE,
        )

    def start_for_test(self, detect, recognize, read=None) -> None:
        """单测注入检测和识别，不加载模型。"""
        self._detect = detect
        self._recognize = recognize
        if read is not None:
            self._read = read
        self._loaded = True
        self._accepting = True
        self._thread = threading.Thread(target=self._loop, name="ocr-infer-test", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._accepting = False
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=10)
        with self._lock:
            pending = list(self._queue)
            self._queue.clear()
        for task in pending:
            if task.status == "queued":
                task.status = "failed"
                task.error = "服务正在退出"
                task.error_code = OCR_FAIL_CODE
                task.finished_at = time.time()
                task.done_event.set()

    def submit_files(
        self,
        paths: list[Path],
        *,
        task_id: str,
        source_file_path: str,
        mode: str,
        file_size: int,
    ) -> Task:
        if not self._loaded:
            raise RuntimeError("文字识别模型尚未就绪")
        if not self._accepting:
            raise RuntimeError("服务正在关闭，拒绝新任务")
        task_id = check_task_id(task_id)
        source_file_path = check_source_file_path(source_file_path)
        mode = check_mode(mode)
        with self._lock:
            self._reject_if_duplicate(task_id)
            queued = sum(1 for task in self._tasks.values() if task.status == "queued")
            if queued >= config.MAX_QUEUE_SIZE:
                raise QueueFull(f"队列已满（上限 {config.MAX_QUEUE_SIZE} 组）")
            task = Task(
                task_id=task_id,
                file_paths=list(paths),
                mode=mode,
                source_file_path=source_file_path,
                file_size=file_size,
                image_count=len(paths),
            )
            try:
                self._reject_if_duplicate(task_id)
            except DuplicateTaskId:
                raise
            self._tasks[task_id] = task
            self._queue.append(task)
        self._wake.set()
        return task

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
        """取消任务并立刻返回。正在识别的在当前这一小批算完后停下。"""
        paths: list[Path] = []
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                disk = result_store.load(task_id)
                if disk is None:
                    return None
                status = canonical_status(disk.get("status_name") or disk.get("status"))
                if status == "cancelled":
                    return _with_previous_status(_public_payload(disk), "cancelled")
                if status in ("completed", "failed"):
                    raise TaskAlreadyFinished(status)
                return None

            if task.status == "cancelled":
                return _with_previous_status(self.task_to_dict(task), "cancelled")
            if task.status in ("completed", "failed"):
                raise TaskAlreadyFinished(task.status)

            if task.status == "queued":
                try:
                    self._queue.remove(task)
                except ValueError:
                    pass
                self._mark_cancelled_locked(task)
                paths = list(task.file_paths)
                payload = _with_previous_status(self.task_to_dict(task), "queued")
            elif task.status == "running":
                task.cancel_requested = True
                self._mark_cancelled_locked(task)
                payload = _with_previous_status(self.task_to_dict(task), "running")
            else:
                raise TaskAlreadyFinished(task.status)

        _log_end(payload)
        for path in paths:
            self._cleanup_file(path)
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
            "image_count": task.image_count,
            "mode": task.mode,
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
            payload["error_code"] = task.error_code or OCR_FAIL_CODE
            payload["error_msg"] = task.error or "识别失败"
        elif name == "cancelled" and task.finished_at:
            payload["expire_at"] = format_beijing(task.finished_at + config.RESULT_TTL_SEC)
        return payload

    def stats(self) -> dict[str, Any]:
        disk = shutil.disk_usage(str(config.TEMP_DIR if config.TEMP_DIR.exists() else "/"))
        gpu_mem = _gpu_mem()
        with self._lock:
            queued = sum(1 for task in self._tasks.values() if task.status == "queued")
            running = self._running
            in_memory = len(self._tasks)
        return {
            "service_version": config.SERVICE_VERSION,
            "model_loaded": self._loaded,
            "det_model": config.DET_MODEL_NAME,
            "rec_model": config.REC_MODEL_NAME,
            "device": config.DEVICE,
            "engine": config.ENGINE_NAME,
            "cuda_available": gpu_mem is not None,
            "accepting": self._accepting,
            "det_batch": config.DET_BATCH,
            "rec_batch": config.REC_BATCH,
            "max_images": config.MAX_IMAGES,
            "max_queue_size": config.MAX_QUEUE_SIZE,
            "queue_length": queued,
            "running": running,
            "tasks_in_memory": in_memory,
            "results_on_disk": result_store.count_results(),
            "result_ttl_hours": config.RESULT_TTL_SEC / 3600,
            "max_image_mb": round(config.MAX_IMAGE_BYTES / (1024 * 1024), 1),
            "max_group_mb": round(config.MAX_GROUP_BYTES / (1024 * 1024), 1),
            "api_key_required": bool(config.API_KEY),
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
            },
            "gpu_mem": gpu_mem,
        }

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

    def _mark_cancelled_locked(self, task: Task) -> None:
        task.cancel_requested = True
        task.status = "cancelled"
        task.finished_at = time.time()
        task.result = None
        task.error = None
        task.error_code = None
        self._persist(task)
        task.done_event.set()

    def _persist(self, task: Task) -> None:
        try:
            result_store.save(task.task_id, self.task_to_dict(task))
        except Exception:
            log.exception("结果落盘失败 task_id=%s", task.task_id)

    def _take(self) -> Optional[Task]:
        with self._lock:
            while self._queue:
                task = self._queue.pop(0)
                if task.status != "queued":
                    continue
                task.status = "running"
                task.started_at = time.time()
                task.queue_len_at_start = len(self._queue)
                self._running += 1
                return task
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            task = self._take()
            if task is None:
                self._wake.wait(0.2)
                self._wake.clear()
                continue
            self._run_task(task)

    def _run_task(self, task: Task) -> None:
        log.info(
            "【开始】 task_id=%s 请求参数=%s",
            task.task_id,
            _compact_json(
                {
                    "task_id": task.task_id,
                    "source_file_path": task.source_file_path,
                    "file_size_bytes": task.file_size,
                    "image_count": task.image_count,
                    "mode": task.mode,
                    "created_at": format_beijing(task.created_at),
                    "queue_len": task.queue_len_at_start if task.queue_len_at_start is not None else 0,
                    "queue_wait_s": round(task.started_at - task.created_at, 3) if task.started_at else None,
                }
            ),
        )
        try:
            items = self._recognize_task(task)
            self._finish_completed(task, items)
        except RecognitionCancelled:
            self._finish_cancelled_by_worker(task)
        except Exception as exc:
            log.exception("识别失败 task_id=%s", task.task_id)
            self._finish_failed(task, str(exc))
        finally:
            with self._lock:
                self._running = max(0, self._running - 1)
            self._cleanup_files(task)
            task.done_event.set()

    def _cancelled(self, task: Task) -> bool:
        return task.cancel_requested or task.status == "cancelled"

    def _recognize_task(self, task: Task) -> list[dict[str, Any]]:
        if self._cancelled(task):
            raise RecognitionCancelled()
        images: list[Any] = []
        index_map: list[int] = []
        items: list[Optional[dict[str, Any]]] = [None] * len(task.file_paths)
        for index, path in enumerate(task.file_paths):
            if self._cancelled(task):
                raise RecognitionCancelled()
            try:
                images.append(self._read(path))
                index_map.append(index)
            except Exception as exc:
                if self._cancelled(task):
                    raise RecognitionCancelled()
                items[index] = {
                    "index": index,
                    "ok": False,
                    "text": "",
                    "boxes": [],
                    "error": str(exc),
                }
        if self._cancelled(task):
            raise RecognitionCancelled()
        if images:
            recognized = recognize_images(
                images,
                mode=task.mode,
                det_batch=config.DET_BATCH,
                rec_batch=config.REC_BATCH,
                detect=self._detect,
                recognize=self._recognize,
                stop=lambda: self._cancelled(task),
            )
            for item, origin in zip(recognized, index_map):
                item["index"] = origin
                items[origin] = item
        return [item if item is not None else {
            "index": index,
            "ok": False,
            "text": "",
            "boxes": [],
            "error": "识别失败",
        } for index, item in enumerate(items)]

    def _finish_completed(self, task: Task, items: list[dict[str, Any]]) -> None:
        with self._lock:
            if task.status != "running":
                return
            task.status = "completed"
            task.result = {"count": len(items), "items": items}
            task.finished_at = time.time()
            task.error = None
            task.error_code = SUCCESS_CODE
            payload = self.task_to_dict(task)
        self._persist_payload(task, payload)
        ok_count = sum(1 for item in items if item.get("ok"))
        log.info(
            "识别完成 task_id=%s mode=%s images=%d ok=%d",
            task.task_id,
            task.mode,
            task.image_count,
            ok_count,
        )
        _log_end(payload)

    def _finish_failed(self, task: Task, message: str) -> None:
        with self._lock:
            if task.status != "running":
                return
            task.status = "failed"
            task.error = message
            task.error_code = OCR_FAIL_CODE
            task.finished_at = time.time()
            task.result = None
            payload = self.task_to_dict(task)
        self._persist_payload(task, payload)
        _log_end(payload)

    def _finish_cancelled_by_worker(self, task: Task) -> None:
        with self._lock:
            if task.status == "cancelled":
                return
            self._mark_cancelled_locked(task)
            payload = self.task_to_dict(task)
        _log_end(payload)

    def _persist_payload(self, task: Task, payload: dict[str, Any]) -> None:
        try:
            result_store.save(task.task_id, payload)
        except Exception:
            log.exception("结果落盘失败 task_id=%s", task.task_id)

    def _cleanup_files(self, task: Task) -> None:
        for path in task.file_paths:
            self._cleanup_file(path)

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
                    task_id
                    for task_id, task in self._tasks.items()
                    if task.status in ("completed", "failed", "cancelled")
                    and task.finished_at
                    and now - task.finished_at > config.RESULT_TTL_SEC
                ]
                for task_id in expired:
                    del self._tasks[task_id]
            if expired:
                log.info("清理内存过期任务 %d 个", len(expired))
            result_store.purge_expired()


def read_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path.name}")
    return image


engine = OcrEngine()
