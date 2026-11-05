"""按本地日期切分一次启动的日志，并在进程不退出时删除过期日期目录。"""
from __future__ import annotations

import logging
import os
import re
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CHECK_SEC = 30
_PURGE_SEC = 3600

_guard = threading.Lock()
_house_started = False
_stdio_started = False


def split_run_location(run_dir: Path, fallback_root: Path) -> tuple[Path, str]:
    """从 logs/日期/时刻 取出日志根目录和这次启动的时刻名。"""
    day_dir = run_dir.parent
    if _DAY_RE.fullmatch(day_dir.name):
        return day_dir.parent, run_dir.name
    return fallback_root, run_dir.name


def folder_expired(folder_name: str, now: float, retention_days: int) -> bool:
    """日期目录从当天零点算起，超过保留天数就过期。"""
    if not _DAY_RE.fullmatch(folder_name):
        return False
    try:
        midnight = time.mktime(time.strptime(folder_name, "%Y-%m-%d"))
    except (OverflowError, ValueError):
        return False
    return midnight < now - retention_days * 86400


def purge_dated_logs(
    root: Path,
    retention_days: int,
    *,
    now: float | None = None,
    protect_days: set[str] | None = None,
    legacy_prefix: str | None = None,
) -> int:
    """删除过期的日期目录。正在写入的日期、以及根目录上的旧单文件按修改时间判断。"""
    if not root.is_dir():
        return 0
    moment = time.time() if now is None else now
    keep = set(protect_days or ())
    keep.add(time.strftime("%Y-%m-%d", time.localtime(moment)))
    removed = 0
    if legacy_prefix:
        cutoff = moment - retention_days * 86400
        for path in root.glob(legacy_prefix + "*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError:
                continue
    for folder in root.iterdir():
        if not folder.is_dir() or folder.name in keep or not folder_expired(folder.name, moment, retention_days):
            continue
        try:
            shutil.rmtree(folder)
            removed += 1
        except OSError:
            continue
    return removed


class DailyRunFileHandler(logging.Handler):
    """写到 logs/当天/启动时刻/文件名。日期变化时换文件，不把很多天写进同一个文件。"""

    def __init__(self, root: Path, stamp: str, filename: str, day: str | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.stamp = stamp
        self.filename = filename
        self._day = ""
        self._stream = None
        self.current_path: Path | None = None
        self._open_unlocked(day or time.strftime("%Y-%m-%d"))

    @property
    def day(self) -> str:
        return self._day

    def _open_unlocked(self, day: str) -> None:
        folder = self.root / day / self.stamp
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / self.filename
        stream = open(path, "a", encoding="utf-8")
        previous = self._stream
        self._stream = stream
        self._day = day
        self.current_path = path
        if previous is not None:
            previous.flush()
            previous.close()

    def ensure_day(self, day: str) -> bool:
        if day == self._day:
            return False
        with self.lock:
            if day == self._day:
                return False
            self._open_unlocked(day)
            return True

    def emit(self, record: logging.LogRecord) -> None:
        try:
            day = time.strftime("%Y-%m-%d", time.localtime(record.created))
            if day != self._day:
                self._open_unlocked(day)
            stream = self._stream
            if stream is None:
                return
            stream.write(self.format(record) + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self.acquire()
        try:
            stream = self._stream
            self._stream = None
            if stream is not None:
                stream.flush()
                stream.close()
        finally:
            self.release()
        super().close()


def stdio_day_names(fds: tuple[int, ...] = (1, 2)) -> set[str]:
    """标准输出若正写在某个日期目录里，返回这些日期，删除时避开。"""
    days: set[str] = set()
    for fd in fds:
        path = _fd_path(fd)
        if path is None:
            continue
        day_name = path.parent.parent.name
        if _DAY_RE.fullmatch(day_name):
            days.add(day_name)
    return days


def _fd_path(fd: int) -> Path | None:
    try:
        raw = os.readlink(f"/proc/self/fd/{fd}")
    except OSError:
        return None
    if raw.endswith(" (deleted)"):
        raw = raw[: -len(" (deleted)")]
    return Path(raw)


def _same_file(current: Path, target: Path) -> bool:
    try:
        return target.exists() and os.path.samefile(current, target)
    except OSError:
        return False


def _flush_std(fd: int) -> None:
    stream = sys.stdout if fd == 1 else sys.stderr if fd == 2 else None
    if stream is None:
        return
    try:
        stream.flush()
    except Exception:
        return


def redirect_stdio(root: Path, stamp: str, fds: tuple[int, ...] = (1, 2)) -> bool:
    """把已经重定向到普通文件的标准输出、标准错误接到当天的 stdout.log。终端和管道不动。"""
    day = time.strftime("%Y-%m-%d")
    folder = root / day / stamp
    target = folder / "stdout.log"
    pending: list[int] = []
    for fd in fds:
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                continue
        except OSError:
            continue
        current = _fd_path(fd)
        if current is not None and _same_file(current, target):
            continue
        pending.append(fd)
    if not pending:
        return False
    folder.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    opened = os.open(target, flags, 0o644)
    try:
        for fd in pending:
            _flush_std(fd)
            os.dup2(opened, fd)
    finally:
        os.close(opened)
    return True


def _house_loop(
    root: Path,
    stamp: str,
    handler: DailyRunFileHandler | None,
    purge,
    logger: logging.Logger | None,
    stop: threading.Event,
) -> None:
    next_purge = time.monotonic() + _PURGE_SEC
    while not stop.wait(_CHECK_SEC):
        try:
            rotated = False
            if handler is not None and handler.ensure_day(time.strftime("%Y-%m-%d")):
                rotated = True
            if redirect_stdio(root, stamp):
                rotated = True
            now_m = time.monotonic()
            if not rotated and now_m < next_purge:
                continue
            removed = purge()
            next_purge = now_m + _PURGE_SEC
            if removed and logger is not None:
                logger.info("清理过期日志目录 %d 个", removed)
            if rotated and logger is not None:
                logger.info("日志已切到新的一天 dir=%s", root / time.strftime("%Y-%m-%d") / stamp)
        except Exception:
            if logger is not None:
                logger.exception("日志按天维护失败")


def start_housekeeping(
    root: Path,
    stamp: str,
    handler: DailyRunFileHandler | None,
    purge,
    logger: logging.Logger | None,
    stop: threading.Event | None = None,
) -> None:
    """父进程：跨天换文件，并按小时间隔删除过期日志。"""
    global _house_started
    with _guard:
        if _house_started:
            return
        _house_started = True
    threading.Thread(
        target=_house_loop,
        args=(root, stamp, handler, purge, logger, stop or threading.Event()),
        name="log-housekeeping",
        daemon=True,
    ).start()


def _stdio_loop(root: Path, stamp: str, stop: threading.Event) -> None:
    while not stop.wait(_CHECK_SEC):
        try:
            redirect_stdio(root, stamp)
        except Exception:
            continue


def start_stdio_watch(root: Path, stamp: str, stop: threading.Event | None = None) -> None:
    """子进程：只跟着把标准错误接到当天文件，不负责删除。"""
    global _stdio_started
    with _guard:
        if _stdio_started or _house_started:
            return
        _stdio_started = True
    threading.Thread(
        target=_stdio_loop,
        args=(root, stamp, stop or threading.Event()),
        name="log-stdio-watch",
        daemon=True,
    ).start()
