"""文字识别服务配置（环境变量可覆盖）。"""
from __future__ import annotations

import os
from pathlib import Path

HOST = os.getenv("OCR_HOST", "0.0.0.0")
PORT = int(os.getenv("OCR_PORT", "6008"))
SERVICE_VERSION = "1.1.0"

MODEL_DIR = Path(os.getenv("OCR_MODEL_DIR", "/root/autodl-tmp/models/ocr"))
DET_MODEL_NAME = os.getenv("OCR_DET_MODEL_NAME", "PP-OCRv6_small_det")
REC_MODEL_NAME = os.getenv("OCR_REC_MODEL_NAME", "PP-OCRv6_small_rec")
# 官方下载落在缓存目录的 official_models 下。目录里已有推理配置时直接读本地文件。
DET_MODEL_DIR = MODEL_DIR / "official_models" / DET_MODEL_NAME
REC_MODEL_DIR = MODEL_DIR / "official_models" / REC_MODEL_NAME
DEVICE = os.getenv("OCR_DEVICE", "gpu:0")
ENGINE_NAME = "paddle_static"

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


def _positive_int(env_name: str, file_key: str, default: int) -> int:
    raw = os.getenv(env_name, "").strip() or _read_service_conf().get(file_key, str(default))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(1, value)


DET_BATCH = _positive_int("OCR_DET_BATCH", "det_batch", 8)
REC_BATCH = _positive_int("OCR_REC_BATCH", "rec_batch", 32)

MAX_IMAGES = max(1, int(os.getenv("OCR_MAX_IMAGES", "32")))
MAX_IMAGE_BYTES = max(1, int(os.getenv("OCR_MAX_IMAGE_BYTES", str(20 * 1024 * 1024))))
MAX_GROUP_BYTES = max(1, int(os.getenv("OCR_MAX_GROUP_BYTES", str(100 * 1024 * 1024))))
MAX_QUEUE_SIZE = max(1, int(os.getenv("OCR_MAX_QUEUE_SIZE", "64")))

TEMP_DIR = Path(os.getenv("OCR_TEMP_DIR", "/root/autodl-tmp/video_cut_forigen/ocr_server/tmp"))
RESULT_DIR = Path(os.getenv("OCR_RESULT_DIR", "/root/autodl-tmp/video_cut_forigen/ocr_server/results"))
RESULT_TTL_SEC = float(os.getenv("OCR_RESULT_TTL_SEC", str(24 * 60 * 60)))
SYNC_TIMEOUT_SEC = float(os.getenv("OCR_SYNC_TIMEOUT_SEC", str(60 * 60)))

# 默认同语音服务共用密钥。需要分开时再设 OCR_API_KEY。
API_KEY = os.getenv("OCR_API_KEY", os.getenv("ASR_API_KEY", "")).strip()

ENABLE_DOCS = os.getenv("OCR_ENABLE_DOCS", "0").strip() not in ("0", "false", "False", "")

LOG_DIR = Path(os.getenv("OCR_LOG_DIR", "/root/autodl-tmp/video_cut_forigen/ocr_server/logs"))
LOG_LEVEL = os.getenv("OCR_LOG_LEVEL", "INFO").upper()
LOG_RETENTION_DAYS = max(1, int(os.getenv("OCR_LOG_RETENTION_DAYS", "3")))
