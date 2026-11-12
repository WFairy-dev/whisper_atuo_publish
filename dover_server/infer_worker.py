"""推理子进程：加载清晰度模型，按任务打分，结果交回父进程。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import config

log = logging.getLogger("dover_server.infer")


def _setup_child_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    from .logging_setup import watch_stdio_day_change

    watch_stdio_day_change()


def worker_main(
    job_q: Any,
    result_q: Any,
    device: str,
    opt_path: str,
    weight_path: str,
) -> None:
    _setup_child_logging()
    try:
        from .clarity import ClarityEvaluator

        log.info(
            "推理进程加载模型 device=%s opt=%s weight=%s",
            device,
            opt_path,
            weight_path,
        )
        evaluator = ClarityEvaluator(
            device=device,
            opt_path=opt_path,
            weight_path=weight_path,
        )
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
        if job is None:
            return
        task_id = str(job.get("task_id", ""))
        path = str(job.get("path", ""))
        try:
            if not path or not Path(path).is_file():
                raise FileNotFoundError(f"视频文件不存在: {path}")
            result = evaluator.score_video(path)
            result_q.put({"type": "result", "task_id": task_id, "ok": True, "result": result})
        except Exception as exc:
            log.exception("清晰度评估失败 task_id=%s", task_id)
            try:
                result_q.put(
                    {"type": "result", "task_id": task_id, "ok": False, "error": str(exc)}
                )
            except Exception:
                return
