"""每次启动单独一个目录：logs/年-月-日/时-分-秒/asr.log。"""
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
    """建 logs/年-月-日/时-分-秒/。同一秒再启动则加 -2、-3。"""
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
    """本进程的日志目录。启动脚本若已指定，就用那一个，避免和控制台日志拆开。"""
    global _run_dir
    if _run_dir is not None:
        return _run_dir
    raw = os.getenv("ASR_LOG_RUN_DIR", "").strip()
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

    # 只挂在本服务的日志器上。网页框架自己的访问日志走控制台，不进 asr.log。
    named = logging.getLogger("asr_server")
    named.setLevel(level)
    named.propagate = False
    log_file = log_dir / "asr.log"
    if not any(isinstance(h, logging.FileHandler) for h in named.handlers):
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        named.addHandler(fh)
    if not any(type(h) is logging.StreamHandler for h in named.handlers):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        named.addHandler(sh)

    named.info(
        "服务启动 pid=%s workers=%s log_dir=%s",
        os.getpid(),
        config.MAX_WORKERS,
        log_dir,
    )


def purge_old_logs() -> int:
    """删掉超过保留天数的日期文件夹，以及旧的单文件 asr.log。"""
    if not config.LOG_DIR.is_dir():
        return 0
    cutoff = time.time() - config.LOG_RETENTION_DAYS * 86400
    removed = 0
    for p in config.LOG_DIR.glob("asr.log*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
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
