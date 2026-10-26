"""推理子进程：只负责加载模型、转写、回传结果。

父进程不导入显卡相关库。需要立刻停掉计算时，由父进程结束本进程。
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any

log = logging.getLogger("asr_server.infer")

_FFMPEG_CANDIDATES = [
    "/root/miniconda3/envs/whisper-turbo/bin",
    "/usr/bin",
    "/usr/local/bin",
]


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


def _setup_child_logging() -> None:
    """子进程只打到标准错误，避免和父进程同时写同一个日志文件。"""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def transcribe_file(model: Any, path: str) -> dict[str, Any]:
    # verbose 用默认空值，避免进度条写进日志。
    result = model.transcribe(path, verbose=None)
    segments = result.get("segments") or []
    clean_segments = []
    for s in segments:
        clean_segments.append(
            {
                "id": s.get("id"),
                "seek": s.get("seek"),
                "start_s": float(s["start"]),
                "end_s": float(s["end"]),
                "text": str(s.get("text", "")).strip(),
            }
        )
    return {
        "language": result.get("language"),
        "text": str(result.get("text", "")).strip(),
        "segments": clean_segments,
    }


def worker_main(
    job_q: Any,
    result_q: Any,
    model_name: str,
    model_dir: str,
    device: str,
) -> None:
    """加载模型后循环领取任务。父进程结束本进程即可停掉正在进行的转写。"""
    _setup_child_logging()
    _ensure_ffmpeg_on_path()
    try:
        import whisper

        log.info("推理进程加载模型 name=%s dir=%s device=%s", model_name, model_dir, device)
        model = whisper.load_model(model_name, device=device, download_root=model_dir)
    except Exception as e:
        log.exception("推理进程加载模型失败")
        try:
            result_q.put({"type": "fatal", "error": str(e)})
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
                raise FileNotFoundError(f"音频文件不存在: {path}")
            result = transcribe_file(model, path)
            result_q.put({"type": "result", "task_id": task_id, "ok": True, "result": result})
        except Exception as e:
            log.exception("转写失败 task_id=%s", task_id)
            try:
                result_q.put(
                    {"type": "result", "task_id": task_id, "ok": False, "error": str(e)}
                )
            except Exception:
                return
