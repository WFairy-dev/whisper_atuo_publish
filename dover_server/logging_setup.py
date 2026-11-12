"""每次启动一个时刻目录：logs/年-月-日/时-分-秒/dover.log。"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from . import config

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import run_log

_run_dir: Path | None = None
_log_root: Path | None = None
_run_stamp: str | None = None
_handler: run_log.DailyRunFileHandler | None = None


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
    raw = os.getenv("DOVER_LOG_RUN_DIR", "").strip()
    if raw:
        folder = Path(raw)
        folder.mkdir(parents=True, exist_ok=True)
        _run_dir = folder
        return folder
    _run_dir = allocate_run_dir(config.LOG_DIR)
    return _run_dir


def _protected_days() -> set[str]:
    days = {time.strftime("%Y-%m-%d")}
    if _handler is not None and _handler.day:
        days.add(_handler.day)
    days.update(run_log.stdio_day_names())
    return days


def purge_old_logs() -> int:
    root = _log_root or config.LOG_DIR
    return run_log.purge_dated_logs(
        root,
        config.LOG_RETENTION_DAYS,
        protect_days=_protected_days(),
    )


def watch_stdio_day_change() -> None:
    raw = os.getenv("DOVER_LOG_RUN_DIR", "").strip()
    if not raw:
        return
    root, stamp = run_log.split_run_location(Path(raw), config.LOG_DIR)
    run_log.start_stdio_watch(root, stamp)


def setup_logging() -> None:
    global _log_root, _run_stamp, _handler
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_run_log_dir()
    os.environ["DOVER_LOG_RUN_DIR"] = str(log_dir)
    _log_root, _run_stamp = run_log.split_run_location(log_dir, config.LOG_DIR)
    level = getattr(logging, config.LOG_LEVEL, logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    named = logging.getLogger("dover_server")
    named.setLevel(level)
    named.propagate = False
    if _handler is None:
        start_day = log_dir.parent.name if run_log._DAY_RE.fullmatch(log_dir.parent.name) else None
        _handler = run_log.DailyRunFileHandler(_log_root, _run_stamp, "dover.log", day=start_day)
        _handler.setFormatter(fmt)
        named.addHandler(_handler)
    if not any(type(handler) is logging.StreamHandler for handler in named.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(fmt)
        named.addHandler(stream)
    run_log.start_housekeeping(_log_root, _run_stamp, _handler, purge_old_logs, named)
    named.info(
        "服务启动 pid=%s workers=%s device=%s log_dir=%s retention_days=%s",
        os.getpid(),
        config.MAX_WORKERS,
        config.DEVICE,
        log_dir,
        config.LOG_RETENTION_DAYS,
    )
