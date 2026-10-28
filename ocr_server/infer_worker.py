"""推理子进程：加载一套检测和识别模型，按任务计算，结果交回父进程。

取消只让当前这一小批算完后停下，进程不退出，模型留在显存里。
父进程不在这里导入显卡库。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

from .batch import RecognitionCancelled, recognize_images

log = logging.getLogger("ocr_server.infer")


def _setup_child_logging() -> None:
    """子进程打到标准错误。启动脚本会把它和父进程输出写进同一次日志。"""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _read_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"无法读取图片: {path.name}")
    return image


def _recognize_paths(
    paths: list[str],
    *,
    mode: str,
    det_batch: int,
    rec_batch: int,
    detect,
    recognize,
    cancelled: Callable[[], bool],
) -> list[dict[str, Any]]:
    if cancelled():
        raise RecognitionCancelled()
    images: list[Any] = []
    index_map: list[int] = []
    items: list[Optional[dict[str, Any]]] = [None] * len(paths)
    for index, raw in enumerate(paths):
        if cancelled():
            raise RecognitionCancelled()
        try:
            images.append(_read_image(Path(raw)))
            index_map.append(index)
        except Exception as exc:
            if cancelled():
                raise RecognitionCancelled()
            items[index] = {
                "index": index,
                "ok": False,
                "text": "",
                "boxes": [],
                "error": str(exc),
            }
    if cancelled():
        raise RecognitionCancelled()
    if images:
        recognized = recognize_images(
            images,
            mode=mode,
            det_batch=det_batch,
            rec_batch=rec_batch,
            detect=detect,
            recognize=recognize,
            stop=cancelled,
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


def worker_main(job_q: Any, result_q: Any, cancel_event: Any) -> None:
    """加载模型后循环领取任务。收到空任务就退出。取消不退出。"""
    _setup_child_logging()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    try:
        from .runtime import detect_images, load_models, recognize_crops

        detector, recognizer = load_models()

        def detect(images):
            return detect_images(detector, images)

        def recognize(specs):
            return recognize_crops(recognizer, specs)
    except Exception as exc:
        log.exception("推理进程加载模型失败")
        try:
            result_q.put({"type": "fatal", "error": str(exc)})
        except Exception:
            pass
        return

    try:
        result_q.put({"type": "ready"})
    except Exception:
        log.exception("无法通知父进程模型已就绪")
        return

    while True:
        try:
            job = job_q.get()
        except (EOFError, OSError, KeyboardInterrupt):
            return
        if not job:
            return
        task_id = str(job.get("task_id", ""))
        try:
            items = _recognize_paths(
                list(job.get("paths") or []),
                mode=str(job.get("mode") or "det_rec"),
                det_batch=int(job.get("det_batch") or 1),
                rec_batch=int(job.get("rec_batch") or 1),
                detect=detect,
                recognize=recognize,
                cancelled=cancel_event.is_set,
            )
            result_q.put({"type": "result", "task_id": task_id, "ok": True, "items": items})
        except Exception as exc:
            cancelled = isinstance(exc, RecognitionCancelled)
            if not cancelled:
                log.exception("识别失败 task_id=%s", task_id)
            try:
                if cancelled:
                    result_q.put({"type": "result", "task_id": task_id, "cancelled": True})
                else:
                    result_q.put({
                        "type": "result",
                        "task_id": task_id,
                        "ok": False,
                        "error": str(exc),
                    })
            except Exception:
                return
