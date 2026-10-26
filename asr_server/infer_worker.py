"""推理子进程：只负责加载模型、转写、回传结果。

父进程不导入显卡相关库。需要立刻停掉计算时，由父进程结束本进程。
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any

from . import config
from .quality import filter_segments, normalize_segments, validate_structure

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


def transcribe_file(model: Any, path: str, *, task_id: str = "",
                    strict_filter: bool | None = None) -> dict[str, Any]:
    # Import only in the inference process; decode once for retries and duration.
    import whisper

    if strict_filter is None:
        strict_filter = config.STRICT_FILTER
    audio = whisper.load_audio(path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Decoded audio is empty or has invalid duration")
    options = dict(
        verbose=None, fp16=str(model.device).startswith("cuda"),
        temperature=(0., .2, .4, .6, .8, 1.),
        compression_ratio_threshold=2.4, logprob_threshold=-1.,
        no_speech_threshold=.6, initial_prompt=None, word_timestamps=True,
        language=config.LANGUAGE,
    )
    log.info("转写参数 task_id=%s duration=%.3f strict_filter=%s options=%s",
             task_id, duration, strict_filter, options)
    candidates, history = [], []
    for index in range(config.MAX_ATTEMPTS):
        options["condition_on_previous_text"] = index == 0
        # Runtime errors propagate as failed tasks, never as empty speech.
        raw = model.transcribe(audio, **options)
        try:
            segments = normalize_segments(raw.get("segments"))
            if str(raw.get("text") or "").strip() and not segments:
                raise ValueError("Whisper returned text without segments")
            errors = validate_structure(segments, duration, config.QUALITY, strict_filter=strict_filter)
        except ValueError as exc:
            segments, errors = [], [str(exc)]
        history.append({"attempt": index + 1, "errors": errors})
        log.info("转写校验 task_id=%s attempt=%d segments=%d errors=%s",
                 task_id, index + 1, len(segments), errors)
        kept, rejected = filter_segments(segments, duration, config.QUALITY, strict_filter=strict_filter)
        candidates.append((raw, kept, rejected, index + 1))
        if not errors:
            selected = candidates[-1]
            break
    else:
        # Prefer usable, confident results after filtering; ties favor the first.
        selected = max(candidates, key=lambda item: (
            bool(item[1] or item[2]),
            sum(s["quality_status"] == "accepted" for s in item[1]),
            len(item[1]), -len(item[2]), -item[3]))
    raw, kept, rejected, selected_attempt = selected
    if not kept and not rejected and all(item["errors"] for item in history):
        raise ValueError(f"Whisper returned no recoverable structure: {history}")
    # Filtering must leave a structurally valid result, even after all retries fail.
    remaining_errors = validate_structure(kept, duration, config.QUALITY, strict_filter=strict_filter)
    if remaining_errors:
        raise ValueError(f"Transcript recovery failed: {remaining_errors}")
    for segment in rejected:
        log.info("转写拒绝片段 task_id=%s segment=%s", task_id, segment)
    return {
        "language": str(raw.get("language") or config.LANGUAGE or "unknown"),
        "text": " ".join(s["text"] for s in kept),
        "segments": kept,
        "rejected_segments": rejected,
        "quality_analysis": {
            "strict_filter": strict_filter,
            "audio_duration_s": duration,
            "attempts": len(history), "selected_attempt": selected_attempt,
            "structural_salvage": all(item["errors"] for item in history),
            "structural_validation": history,
            "retained_segment_count": len(kept), "rejected_segment_count": len(rejected),
            "ambiguous_segment_count": sum(s["quality_status"] == "ambiguous" for s in kept),
            "unknown_segment_count": sum(s["quality_status"] == "unknown" for s in kept),
            "result": "speech_retained" if kept else "no_valid_speech_segments",
        },
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
        log.info("推理配置 whisper=%s model=%s device=%s language=%s max_attempts=%d quality=%s",
                 version("openai-whisper"), model_name, device, config.LANGUAGE,
                 config.MAX_ATTEMPTS, config.QUALITY)
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
            result = transcribe_file(model, path, task_id=task_id, strict_filter=job.get("strict_filter"))
            result_q.put({"type": "result", "task_id": task_id, "ok": True, "result": result})
        except Exception as e:
            log.exception("转写失败 task_id=%s", task_id)
            try:
                result_q.put(
                    {"type": "result", "task_id": task_id, "ok": False, "error": str(e)}
                )
            except Exception:
                return
