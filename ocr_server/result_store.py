"""识别结果落盘，默认保留 24 小时。"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from . import config

log = logging.getLogger("ocr_server.result_store")

# 任务号由调用方传入。只允许字母、数字、下划线和中横线，避免被当成路径。
_TASK_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_BJ = timezone(timedelta(hours=8))


def is_safe_task_id(task_id: str) -> bool:
    return bool(_TASK_ID_RE.fullmatch(task_id or ""))


def _path_for(task_id: str) -> Path:
    if not is_safe_task_id(task_id):
        raise ValueError("非法任务号")
    return config.RESULT_DIR / f"{task_id}.json"


def ensure_dirs() -> None:
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)


def _as_unix(value: Any) -> Optional[float]:
    """过期判断同时认 Unix 秒和北京时间字符串。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float(text)
        except ValueError:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=_BJ).timestamp()
            except ValueError:
                continue
    return None


def save(task_id: str, payload: dict[str, Any]) -> Path:
    ensure_dirs()
    path = _path_for(task_id)
    tmp = path.with_suffix(".json.tmp")
    data = dict(payload)
    data["saved_at"] = time.time()
    if _as_unix(data.get("expire_at")) is None:
        data["expire_at"] = time.time() + config.RESULT_TTL_SEC
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False)
    tmp.replace(path)
    return path


def load(task_id: str) -> Optional[dict[str, Any]]:
    if not is_safe_task_id(task_id):
        return None
    path = _path_for(task_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("读取结果失败 task_id=%s err=%s", task_id, exc)
        return None

    expire_at = _as_unix(data.get("expire_at"))
    if expire_at is not None and time.time() > expire_at:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        log.info("结果已过期并删除 task_id=%s", task_id)
        return None
    return data


def purge_expired() -> int:
    ensure_dirs()
    now = time.time()
    removed = 0
    for path in config.RESULT_DIR.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            expire_at = _as_unix(data.get("expire_at"))
            finished = _as_unix(data.get("finished_at"))
            if finished is None:
                finished = _as_unix(data.get("saved_at"))
            expired = False
            if expire_at is not None:
                expired = now > expire_at
            elif finished is not None:
                expired = now - finished > config.RESULT_TTL_SEC
            else:
                expired = now - path.stat().st_mtime > config.RESULT_TTL_SEC
            if expired:
                path.unlink(missing_ok=True)
                removed += 1
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            try:
                path.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    if removed:
        log.info("清理过期结果文档 %d 个", removed)
    return removed


def delete(task_id: str) -> None:
    if not is_safe_task_id(task_id):
        return
    path = _path_for(task_id)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("删除结果失败 task_id=%s err=%s", task_id, exc)


def count_results() -> int:
    if not config.RESULT_DIR.is_dir():
        return 0
    return sum(1 for _ in config.RESULT_DIR.glob("*.json"))
