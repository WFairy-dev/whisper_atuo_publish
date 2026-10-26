"""ASR 服务配置（环境变量可覆盖）。

部署关系：
  公司内网 CPU 机器上的业务流水线
    → HTTP 调用 →
  AutoDL（本机）上的本服务（GPU 跑 Whisper）
"""
from __future__ import annotations

import os
from pathlib import Path

# 服务
HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "6006"))
SERVICE_VERSION = "1.6.0"

# 模型
MODEL_NAME = os.getenv("ASR_MODEL_NAME", "turbo")
MODEL_DIR = Path(os.getenv("ASR_MODEL_DIR", "/root/autodl-tmp/models/whisper"))
DEVICE = os.getenv("ASR_DEVICE", "cuda")

# 同时转写路数。优先读环境变量，否则读 service.conf 的 workers。
# 实测第 4 路会耗尽显存，超过上限时按上限启动。
_WORKER_LIMIT = 3
_CONF_PATH = Path(__file__).with_name("service.conf")


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
    file_value = _read_service_conf().get("workers", "2")
    raw = os.getenv("ASR_MAX_WORKERS", "").strip() or file_value
    try:
        requested = max(1, int(raw))
    except ValueError:
        requested = 2
    return requested, min(requested, _WORKER_LIMIT)


WORKERS_REQUESTED, MAX_WORKERS = _resolve_workers()
MAX_QUEUE_SIZE = max(1, int(os.getenv("ASR_MAX_QUEUE_SIZE", "200")))

# 临时上传（推理完立即删除）
TEMP_DIR = Path(os.getenv("ASR_TEMP_DIR", "/root/autodl-tmp/video_cut_forigen/asr_server/tmp"))

# 转写结果文档落盘，默认保留 24 小时
RESULT_DIR = Path(
    os.getenv("ASR_RESULT_DIR", "/root/autodl-tmp/video_cut_forigen/asr_server/results")
)
RESULT_TTL_SEC = float(os.getenv("ASR_RESULT_TTL_SEC", str(24 * 60 * 60)))

# 同步接口最长等待（秒）
SYNC_TIMEOUT_SEC = float(os.getenv("ASR_SYNC_TIMEOUT_SEC", str(60 * 60)))

# 上传限制（默认 500MB）
MAX_UPLOAD_BYTES = int(os.getenv("ASR_MAX_UPLOAD_BYTES", str(500 * 1024 * 1024)))

# 鉴权：公司内网调用 AutoDL 时建议必填；为空则启动告警且不校验（仅便于本机调试）
API_KEY = os.getenv("ASR_API_KEY", "").strip()

# 生产建议关闭 /docs /redoc（ASR_ENABLE_DOCS=0）
ENABLE_DOCS = os.getenv("ASR_ENABLE_DOCS", "0").strip() not in ("0", "false", "False", "")

# 日志
LOG_DIR = Path(os.getenv("ASR_LOG_DIR", "/root/autodl-tmp/video_cut_forigen/asr_server/logs"))
LOG_LEVEL = os.getenv("ASR_LOG_LEVEL", "INFO").upper()
LOG_RETENTION_DAYS = max(1, int(os.getenv("ASR_LOG_RETENTION_DAYS", "3")))
