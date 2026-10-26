"""FastAPI 入口：异步入队 / 查询 / 同步阻塞 / 健康检查。"""
from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from . import config
from .engine import DuplicateTaskId, InvalidTaskRequest, TaskAlreadyFinished, engine
from .logging_setup import purge_old_logs, setup_logging

log = logging.getLogger("asr_server.app")

docs_url = "/docs" if config.ENABLE_DOCS else None
redoc_url = "/redoc" if config.ENABLE_DOCS else None
openapi_url = "/openapi.json" if config.ENABLE_DOCS else None

app = FastAPI(
    title="ASR Whisper Remote",
    version=config.SERVICE_VERSION,
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

ALLOWED_SUFFIXES = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
}


def _error_body(
    *,
    http_hint: str,
    error_code: str,
    message: str,
    retryable: bool,
    task_id: Optional[str] = None,
    status: int = 1,
    status_name: str = "failed",
) -> dict:
    return {
        "task_id": task_id,
        "status": status,
        "status_name": status_name,
        "error": {
            "error_code": error_code,
            "error_message": message,
            "retryable": retryable,
        },
    }


def _http_error(
    status_code: int,
    error_code: str,
    message: str,
    *,
    retryable: bool,
    task_id: Optional[str] = None,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail=_error_body(
            http_hint="",
            error_code=error_code,
            message=message,
            retryable=retryable,
            task_id=task_id,
        ),
    )


def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> None:
    """公司内网调用 AutoDL 时用请求头鉴权。"""
    if not config.API_KEY:
        return
    token = x_api_key
    if not token and authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip()
        else:
            token = authorization.strip()
    if token != config.API_KEY:
        raise _http_error(401, "UNAUTHORIZED", "无效或缺失 API Key", retryable=False)


@app.on_event("startup")
def on_startup() -> None:
    setup_logging()
    purged_logs = purge_old_logs()
    if purged_logs:
        log.info("清理过期日志文件 %d 个", purged_logs)
    if config.API_KEY:
        log.info("API Key 鉴权已启用（请求头 X-API-Key 或 Authorization: Bearer）")
    else:
        log.warning(
            "未设置 ASR_API_KEY：任意可达客户端均可调用。"
            "公司内网 → AutoDL 场景强烈建议配置密钥。"
        )
    log.info(
        "服务启动 version=%s docs=%s max_upload_mb=%.1f result_ttl_h=%.1f",
        config.SERVICE_VERSION,
        bool(config.ENABLE_DOCS),
        config.MAX_UPLOAD_BYTES / (1024 * 1024),
        config.RESULT_TTL_SEC / 3600,
    )
    engine.start()


@app.on_event("shutdown")
def on_shutdown() -> None:
    engine.stop()
    log.info("服务已停止")


@app.exception_handler(HTTPException)
async def handle_http_error(_request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def handle_validation_error(_request, exc: RequestValidationError):
    missing = []
    for item in exc.errors():
        loc = [str(part) for part in item.get("loc", ()) if part != "body"]
        if loc:
            missing.append(".".join(loc))
    message = "请求字段缺失或类型不对"
    if missing:
        message = message + "：" + "、".join(missing)
    return JSONResponse(
        status_code=422,
        content=_error_body(
            http_hint="",
            error_code="MISSING_FIELD",
            message=message,
            retryable=False,
        ),
    )


@app.get("/health")
def health():
    """健康检查不强制鉴权，便于监控探活。"""
    return engine.stats()


def _parse_upload_name(filename: Optional[str]) -> tuple[str, str, str]:
    """从上传文件名取出要记下的路径、后缀、最后一段文件名。

    这里的路径只是文字，服务端不会按它去打开调用方磁盘。
    """
    source_path = (filename or "").strip()
    if not source_path or "\x00" in source_path:
        raise _http_error(400, "INVALID_REQUEST", "缺少文件名", retryable=False)
    original_name = source_path.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        shown = suffix or "（无后缀）"
        raise _http_error(
            400,
            "UNSUPPORTED_FORMAT",
            f"不支持的文件类型: {shown}，允许: {sorted(ALLOWED_SUFFIXES)}",
            retryable=False,
        )
    return source_path, suffix, original_name


async def _save_upload(upload: UploadFile) -> tuple[Path, str, str, str, int]:
    source_path, suffix, original_name = _parse_upload_name(upload.filename)

    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    size = 0
    with tempfile.NamedTemporaryFile(
        dir=config.TEMP_DIR, suffix=suffix, delete=False
    ) as tmp:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                tmp_path = Path(tmp.name)
                tmp.close()
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise _http_error(
                    413,
                    "FILE_TOO_LARGE",
                    (
                        f"文件过大: {size} 字节，上限 "
                        f"{config.MAX_UPLOAD_BYTES} 字节 "
                        f"({config.MAX_UPLOAD_BYTES / (1024 * 1024):.0f}MB)"
                    ),
                    retryable=False,
                )
            tmp.write(chunk)
        tmp_path = Path(tmp.name)
    return tmp_path, suffix, source_path, original_name, size


def _reject_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@app.post("/v1/asr", dependencies=[Depends(verify_api_key)])
async def submit_asr(
    source_file_path: UploadFile = File(...),
    task_id: str = Form(...),
):
    """上传音频/视频。字段名 source_file_path 同时带上视频内容和调用方路径。"""
    tmp_path, suffix, stored_path, original_name, size = await _save_upload(source_file_path)
    try:
        task = engine.submit_file(
            tmp_path,
            suffix,
            task_id=task_id,
            source_file_path=stored_path,
            original_name=original_name,
            file_size=size,
        )
    except InvalidTaskRequest as e:
        _reject_upload(tmp_path)
        raise _http_error(400, "INVALID_REQUEST", e.message, retryable=False, task_id=task_id) from e
    except DuplicateTaskId as e:
        _reject_upload(tmp_path)
        raise _http_error(
            409, "TASK_ID_EXISTS", f"任务号已存在: {e.task_id}", retryable=False, task_id=e.task_id
        ) from e
    except RuntimeError as e:
        _reject_upload(tmp_path)
        log.warning("拒绝入队 task_id=%s source_file_path=%s reason=%s", task_id, stored_path, e)
        raise _http_error(503, "QUEUE_FULL", str(e), retryable=True, task_id=task_id) from e
    return engine.task_to_dict(task)


@app.post("/v1/asr/{task_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_asr(task_id: str):
    """取消排队中或正在转写的任务，立刻返回。"""
    try:
        payload = engine.cancel(task_id)
    except TaskAlreadyFinished as e:
        log.info("拒绝取消 task_id=%s status=%s", task_id, e.status)
        raise _http_error(
            409,
            "TASK_ALREADY_FINISHED",
            f"任务已结束，无法取消（status={e.status}）",
            retryable=False,
            task_id=task_id,
        ) from e
    if payload is None:
        raise _http_error(
            404,
            "TASK_NOT_FOUND",
            "任务不存在或已过期（结果默认保留24小时）",
            retryable=False,
            task_id=task_id,
        )
    return payload


@app.get("/v1/asr/{task_id}", dependencies=[Depends(verify_api_key)])
def get_asr(task_id: str):
    payload = engine.get_task_payload(task_id)
    if payload is None:
        raise _http_error(
            404,
            "TASK_NOT_FOUND",
            "任务不存在或已过期（结果默认保留24小时）",
            retryable=False,
            task_id=task_id,
        )
    return payload


@app.post("/v1/asr/sync", dependencies=[Depends(verify_api_key)])
async def sync_asr(
    source_file_path: UploadFile = File(...),
    task_id: str = Form(...),
):
    """上传后阻塞直到转写完成（短任务/调试；高并发请用异步接口）。"""
    tmp_path, suffix, stored_path, original_name, size = await _save_upload(source_file_path)
    try:
        task = engine.submit_file(
            tmp_path,
            suffix,
            task_id=task_id,
            source_file_path=stored_path,
            original_name=original_name,
            file_size=size,
        )
    except InvalidTaskRequest as e:
        _reject_upload(tmp_path)
        raise _http_error(400, "INVALID_REQUEST", e.message, retryable=False, task_id=task_id) from e
    except DuplicateTaskId as e:
        _reject_upload(tmp_path)
        raise _http_error(
            409, "TASK_ID_EXISTS", f"任务号已存在: {e.task_id}", retryable=False, task_id=e.task_id
        ) from e
    except RuntimeError as e:
        _reject_upload(tmp_path)
        raise _http_error(503, "QUEUE_FULL", str(e), retryable=True, task_id=task_id) from e

    # 不能在异步函数里直接干等，否则查询和取消都要排到这次转写结束。
    await asyncio.to_thread(engine.wait_task, task, config.SYNC_TIMEOUT_SEC)
    payload = engine.task_to_dict(task)
    if task.status in ("queued", "running"):
        raise _http_error(
            504,
            "SYNC_TIMEOUT",
            f"同步等待超时（{config.SYNC_TIMEOUT_SEC}秒），task_id={task.task_id}",
            retryable=True,
            task_id=task.task_id,
        )
    if task.status == "failed":
        return JSONResponse(status_code=500, content=payload)
    return payload
