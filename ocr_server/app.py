"""文字识别入口：异步提交、查询、取消、同步等待。"""
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
    QueueFull,
    TaskAlreadyFinished,
    check_mode,
    check_source_file_path,
    check_task_id,
    engine,
)
from .logging_setup import purge_old_logs, setup_logging

log = logging.getLogger("ocr_server.app")

docs_url = "/docs" if config.ENABLE_DOCS else None
redoc_url = "/redoc" if config.ENABLE_DOCS else None
openapi_url = "/openapi.json" if config.ENABLE_DOCS else None

app = FastAPI(
    title="OCR PP-OCRv6 Small",
    version=config.SERVICE_VERSION,
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url,
)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
        log.warning("未设置 OCR_API_KEY 或 ASR_API_KEY：任意可达客户端均可调用。")
    log.info(
        "服务启动 version=%s result_ttl_h=%.1f",
        config.SERVICE_VERSION,
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
def health():
    return engine.stats()


def _suffix(filename: Optional[str]) -> str:
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
    return suffix


async def _save_one(upload: UploadFile, used: int, task_id: str) -> tuple[Path, int]:
    suffix = _suffix(upload.filename)
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    size = 0
    with tempfile.NamedTemporaryFile(dir=config.TEMP_DIR, suffix=suffix, delete=False) as tmp:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > config.MAX_IMAGE_BYTES or used + size > config.MAX_GROUP_BYTES:
                tmp_path = Path(tmp.name)
                tmp.close()
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                if size > config.MAX_IMAGE_BYTES:
                    raise _http_error(
                        413,
                        "FILE_TOO_LARGE",
                        f"单张图片超过上限 {config.MAX_IMAGE_BYTES} 字节",
                        retryable=False,
                        task_id=task_id,
                    )
                raise _http_error(
                    413,
                    "GROUP_TOO_LARGE",
                    f"一组图片超过上限 {config.MAX_GROUP_BYTES} 字节",
                    retryable=False,
                    task_id=task_id,
                )
            tmp.write(chunk)
        return Path(tmp.name), size


def _cleanup(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("删除临时文件失败: %s", path)


async def _accept_upload(
    file: list[UploadFile],
    task_id: str,
    source_file_path: str,
    mode: str,
):
    try:
        task_id = check_task_id(task_id)
        source_file_path = check_source_file_path(source_file_path)
        mode = check_mode(mode)
    except InvalidTaskRequest as exc:
        raise _http_error(400, "INVALID_REQUEST", exc.message, retryable=False, task_id=task_id) from exc
    if not file:
        raise _http_error(400, "INVALID_REQUEST", "至少上传一张图片", retryable=False, task_id=task_id)
    if len(file) > config.MAX_IMAGES:
        raise _http_error(
            400,
            "TOO_MANY_IMAGES",
            f"单次最多 {config.MAX_IMAGES} 张，收到 {len(file)} 张",
            retryable=False,
            task_id=task_id,
        )

    saved: list[Path] = []
    used = 0
    try:
        for upload in file:
            path, size = await _save_one(upload, used, task_id)
            saved.append(path)
            used += size
        task = engine.submit_files(
            saved,
            task_id=task_id,
            source_file_path=source_file_path,
            mode=mode,
            file_size=used,
        )
    except InvalidTaskRequest as exc:
        _cleanup(saved)
        raise _http_error(400, "INVALID_REQUEST", exc.message, retryable=False, task_id=task_id) from exc
    except DuplicateTaskId as exc:
        _cleanup(saved)
        raise _http_error(
            409, "TASK_ID_EXISTS", f"任务号已存在: {exc.task_id}", retryable=False, task_id=exc.task_id
        ) from exc
    except QueueFull as exc:
        _cleanup(saved)
        raise _http_error(503, "QUEUE_FULL", str(exc), retryable=True, task_id=task_id) from exc
    except RuntimeError as exc:
        _cleanup(saved)
        log.warning("拒绝入队 task_id=%s source_file_path=%s reason=%s", task_id, source_file_path, exc)
        code = "NOT_READY" if "尚未就绪" in str(exc) else "QUEUE_FULL"
        raise _http_error(503, code, str(exc), retryable=True, task_id=task_id) from exc
    except HTTPException:
        _cleanup(saved)
        raise
    except Exception:
        _cleanup(saved)
        raise
    return task


@app.post("/v1/ocr", dependencies=[Depends(verify_api_key)])
async def submit_ocr(
    file: list[UploadFile] = File(...),
    task_id: str = Form(...),
    source_file_path: str = Form(...),
    mode: str = Form("det_rec"),
):
    """上传一组图片。file 可重复。提交成功只表示已受理。"""
    task = await _accept_upload(file, task_id, source_file_path, mode)
    return engine.task_to_dict(task)


@app.post("/v1/ocr/sync", dependencies=[Depends(verify_api_key)])
async def sync_ocr(
    file: list[UploadFile] = File(...),
    task_id: str = Form(...),
    source_file_path: str = Form(...),
    mode: str = Form("det_rec"),
):
    """上传后阻塞直到识别结束。高并发请用异步接口。"""
    task = await _accept_upload(file, task_id, source_file_path, mode)
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


@app.post("/v1/ocr/{task_id}/cancel", dependencies=[Depends(verify_api_key)])
def cancel_ocr(task_id: str):
    """取消排队中或正在识别的任务，立刻返回。"""
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


@app.get("/v1/ocr/{task_id}", dependencies=[Depends(verify_api_key)])
def get_ocr(task_id: str):
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
