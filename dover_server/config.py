"""清晰度服务配置（环境变量可覆盖）。"""
from __future__ import annotations

import os
from pathlib import Path

HOST = os.getenv("DOVER_HOST", "0.0.0.0")
PORT = int(os.getenv("DOVER_PORT", "6009"))
SERVICE_VERSION = "0.1.0"

_PACKAGE_DIR = Path(__file__).resolve().parent
DOVER_ROOT = Path(os.getenv("DOVER_ROOT", str(_PACKAGE_DIR / "DOVER")))
OPT_PATH = Path(os.getenv("DOVER_OPT_PATH", str(DOVER_ROOT / "clarity.yml")))
WEIGHT_PATH = Path(
    os.getenv(
        "DOVER_WEIGHT_PATH",
        str(DOVER_ROOT / "pretrained_weights" / "DOVER.pth"),
    )
)
MODEL_NAME = os.getenv("DOVER_MODEL_NAME", "DOVER")
DEVICE = os.getenv("DOVER_DEVICE", "cpu")

_CONF_PATH = Path(__file__).with_name("service.conf")
# 处理器并发实测后再定上限；当前机器约 18 核，先放开到 8。
_WORKER_LIMIT = 8


def _read_service_conf() -> dict[str, str]:
    if not _CONF_PATH.is_file():
        return {}
    data: dict[str, str] = {}
    for raw in _CONF_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def _resolve_workers() -> tuple[int, int]:
    file_value = _read_service_conf().get("workers", "1")
    raw = os.getenv("DOVER_MAX_WORKERS", "").strip() or file_value
    try:
        requested = max(1, int(raw))
    except ValueError:
        requested = 1
    return requested, min(requested, _WORKER_LIMIT)


WORKERS_REQUESTED, MAX_WORKERS = _resolve_workers()

MAX_UPLOAD_BYTES = max(1, int(os.getenv("DOVER_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024))))
MAX_QUEUE_SIZE = max(1, int(os.getenv("DOVER_MAX_QUEUE_SIZE", "64")))

TEMP_DIR = Path(
    os.getenv("DOVER_TEMP_DIR", "/root/autodl-tmp/video_cut_forigen/dover_server/tmp")
)
RESULT_DIR = Path(
    os.getenv(
        "DOVER_RESULT_DIR", "/root/autodl-tmp/video_cut_forigen/dover_server/results"
    )
)
RESULT_TTL_SEC = float(os.getenv("DOVER_RESULT_TTL_SEC", str(24 * 60 * 60)))
SYNC_TIMEOUT_SEC = float(os.getenv("DOVER_SYNC_TIMEOUT_SEC", str(60 * 60)))

API_KEY = os.getenv("DOVER_API_KEY", os.getenv("ASR_API_KEY", "")).strip()

ENABLE_DOCS = os.getenv("DOVER_ENABLE_DOCS", "0").strip() not in ("0", "false", "False", "")

LOG_DIR = Path(
    os.getenv("DOVER_LOG_DIR", "/root/autodl-tmp/video_cut_forigen/dover_server/logs")
)
LOG_LEVEL = os.getenv("DOVER_LOG_LEVEL", "INFO").upper()
LOG_RETENTION_DAYS = max(1, int(os.getenv("DOVER_LOG_RETENTION_DAYS", "3")))
