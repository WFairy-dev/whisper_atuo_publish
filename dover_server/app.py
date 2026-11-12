"""清晰度入口：异步提交、查询、取消、同步等待。"""
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
from .engine import (
    DuplicateTaskId,
    InvalidTaskRequest,
    TaskAlreadyFinished,
    engine,
)
from .logging_setup import purge_old_logs, setup_logging

log = logging.getLogger("dover_server.app")

docs_url = "/docs" if config.ENABLE_DOCS else None
redoc_url = "/redoc" if config.ENABLE_DOCS else None
openapi_url = "/openapi.json" if config.ENABLE_DOCS else None

app = FastAPI(
    title="DOVER Clarity Technical",
    version=config.SERVICE_VERSION,
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

ALLOWED_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".mpeg",
    ".mpg",
}


def _error_body(
    *,
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
    purged = purge_old_logs()
    if purged:
        log.info("清理过期日志目录 %d 个", purged)
    if config.API_KEY:
        log.info("API Key 鉴权已启用（请求头 X-API-Key 或 Authorization: Bearer）")
    else:
        log.warning("未设置 DOVER_API_KEY 或 ASR_API_KEY：任意可达客户端均可调用。")
    log.info(
        "服务启动 version=%s device=%s docs=%s max_upload_mb=%.1f",
        config.SERVICE_VERSION,
        config.DEVICE,
        bool(docs_url),
        config.MAX_UPLOAD_BYTES / (1024 * 1024),
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
        loc = [str(part) for part in item.get("loc", ()) if part not in ("body", "form")]
        if loc:
            missing.append(".".join(loc))
    message = "请求字段缺失或类型不对"
    if missing:
        message = message + "：" + "、".join(missing)
    return JSONResponse(
        status_code=422,
        content=_error_body(error_code="MISSING_FIELD", message=message, retryable=False),
    )


@app.get("/health")
@app.get("/v1/vqa/health")
def health():
    """本机用 /health；经 AutoDL 6006 代理时用 /v1/vqa/health。"""
    return engine.stats()


def _parse_upload_name(filename: Optional[str]) -> tuple[str, str]:
    raw_name = (filename or "").strip()
    if not raw_name or "\x00" in raw_name:
        raise _http_error(400, "INVALID_REQUEST", "缺少文件名", retryable=False)
    original_name = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        shown = suffix or "（无后缀）"
        raise _http_error(
            400,
            "UNSUPPORTED_FORMAT",
            f"不支持的文件类型: {shown}，允许: {sorted(ALLOWED_SUFFIXES)}",
            retryable=False,
        )
    return suffix, original_name


async def _save_upload(upload: UploadFile) -> tuple[Path, str, str, int]:
    suffix, original_name = _parse_upload_name(upload.filename)
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
    return tmp_path, suffix, original_name, size


def _reject_upload(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@app.post("/v1/vqa", dependencies=[Depends(verify_api_key)])
async def submit_vqa(
    file: UploadFile = File(...),
    task_id: str = Form(...),
    source_file_path: str = Form(...),
):
    """上传视频。只评估技术清晰度分。"""
    tmp_path, suffix, original_name, size = await _save_upload(file)
    try:
        task = engine.submit_file(
            tmp_path,
            suffix,
            task_id=task_id,
            source_file_path=source_file_path,
            original_name=original_name,
            file_size=size,
        )
    except InvalidTaskRequest as exc:
        _reject_upload(tmp_path)
        raise _http_error(400, "INVALID_REQUEST", exc.message, retryable=False, task_id=task_id) from exc
    except DuplicateTaskId as exc:
        _reject_upload(tmp_path)
        raise _http_error(
            409, "TASK_ID_EXISTS", f"任务号已存在: {exc.task_id}", retryable=False, task_id=exc.task_id
        ) from exc
    except RuntimeError as exc:
        _reject_upload(tmp_path)
        log.warning("拒绝入队 task_id=%s source_file_path=%s reason=%s", task_id, source_file_path, exc)
        raise _http_error(503, "QUEUE_FULL", str(exc), retryable=True, task_id=task_id) from exc
    return engine.task_to_dict(task)


@app.post("/v1/vqa/{task_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_vqa(task_id: str):
    try:
        payload = engine.cancel(task_id)
    except TaskAlreadyFinished as exc:
        log.info("拒绝取消 task_id=%s status=%s", task_id, exc.status)
        raise _http_error(
            409,
            "TASK_ALREADY_FINISHED",
            f"任务已结束，无法取消（status={exc.status}）",
            retryable=False,
            task_id=task_id,
        ) from exc
    if payload is None:
        raise _http_error(
            404,
            "TASK_NOT_FOUND",
            "任务不存在或已过期（结果默认保留24小时）",
            retryable=False,
            task_id=task_id,
        )
    return payload


@app.get("/v1/vqa/{task_id}", dependencies=[Depends(verify_api_key)])
def get_vqa(task_id: str):
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


@app.post("/v1/vqa/sync", dependencies=[Depends(verify_api_key)])
async def sync_vqa(
    file: UploadFile = File(...),
    task_id: str = Form(...),
    source_file_path: str = Form(...),
):
    """上传后阻塞直到评估结束。高并发请用异步接口。"""
    tmp_path, suffix, original_name, size = await _save_upload(file)
    try:
        task = engine.submit_file(
            tmp_path,
            suffix,
            task_id=task_id,
            source_file_path=source_file_path,
            original_name=original_name,
            file_size=size,
        )
    except InvalidTaskRequest as exc:
        _reject_upload(tmp_path)
        raise _http_error(400, "INVALID_REQUEST", exc.message, retryable=False, task_id=task_id) from exc
    except DuplicateTaskId as exc:
        _reject_upload(tmp_path)
        raise _http_error(
            409, "TASK_ID_EXISTS", f"任务号已存在: {exc.task_id}", retryable=False, task_id=exc.task_id
        ) from exc
    except RuntimeError as exc:
        _reject_upload(tmp_path)
        raise _http_error(503, "QUEUE_FULL", str(exc), retryable=True, task_id=task_id) from exc

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
