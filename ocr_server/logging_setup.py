"""每次启动单独一个目录：logs/年-月-日/时-分-秒/ocr.log。"""
from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path

from . import config

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_run_dir: Path | None = None


def allocate_run_dir(root: Path, moment: time.struct_time | None = None) -> Path:
    when = moment or time.localtime()
    day = time.strftime("%Y-%m-%d", when)
    stamp = time.strftime("%H-%M-%S", when)
    parent = root / day
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / stamp
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = parent / f"{stamp}-{suffix}"


def resolve_run_log_dir() -> Path:
    global _run_dir
    if _run_dir is not None:
        return _run_dir
    raw = os.getenv("OCR_LOG_RUN_DIR", "").strip()
    if raw:
        folder = Path(raw)
        folder.mkdir(parents=True, exist_ok=True)
        _run_dir = folder
        return folder
    _run_dir = allocate_run_dir(config.LOG_DIR)
    return _run_dir


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_run_log_dir()
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    named = logging.getLogger("ocr_server")
    named.setLevel(level)
    named.propagate = False
    if not any(isinstance(handler, logging.FileHandler) for handler in named.handlers):
        file_handler = logging.FileHandler(log_dir / "ocr.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        named.addHandler(file_handler)
    if not any(type(handler) is logging.StreamHandler for handler in named.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        named.addHandler(stream)
    named.info(
        "服务启动 pid=%s workers=%s det_batch=%s rec_batch=%s log_dir=%s",
        os.getpid(),
        config.MAX_WORKERS,
        config.DET_BATCH,
        config.REC_BATCH,
        log_dir,
    )


def purge_old_logs() -> int:
    if not config.LOG_DIR.is_dir():
        return 0
    cutoff = time.time() - config.LOG_RETENTION_DAYS * 86400
    removed = 0
    for folder in config.LOG_DIR.iterdir():
        if not folder.is_dir() or not _DAY_RE.fullmatch(folder.name):
            continue
        try:
            if folder.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(folder)
            removed += 1
        except OSError:
            continue
    return removed
