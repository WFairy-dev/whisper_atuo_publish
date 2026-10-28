#!/usr/bin/env python3
"""
公司内网调用文字识别服务的示例。调用方式和语音转写一样：先提交任务号，再查询或取消。

用法：
  export OCR_BASE_URL=http://<autodl可达地址>:6008
  export ASR_API_KEY=与语音服务相同的密钥
  python ocr_client_example.py a.jpg b.jpg --source D:/视频/a.mp4

取消某个任务号：
  from ocr_client_example import cancel_ocr
  cancel_ocr(base_url, task_id, api_key=...)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx


def _headers(api_key: Optional[str]) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def new_task_id() -> str:
    """仅供本机示例和自测生成任务号。正式流水线应传入自己的唯一任务号。"""
    return f"req_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def upload_filename(file_path: str | Path) -> str:
    """记到结果里的路径文字。统一成正斜杠，服务不会按它打开磁盘。"""
    return str(file_path).replace("\\", "/")


class TaskCancelled(RuntimeError):
    """任务已被取消。流水线应停止等待，不要把这次调用当成识别成功。"""


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise PermissionError("鉴权失败：请检查 OCR_API_KEY 或 ASR_API_KEY")
    if response.status_code == 404:
        raise LookupError("任务不存在或已过期")
    if response.status_code == 409:
        body = response.json() if "json" in response.headers.get("content-type", "") else {}
        err = body.get("error") if isinstance(body, dict) else None
        message = (err or {}).get("error_message") if isinstance(err, dict) else None
        raise RuntimeError(message or "任务号冲突或任务已结束")
    if response.status_code >= 400:
        body = {}
        if "json" in response.headers.get("content-type", ""):
            body = response.json()
        err = body.get("error") if isinstance(body, dict) else None
        message = (err or {}).get("error_message") if isinstance(err, dict) else response.text
        raise RuntimeError(message or f"识别失败 HTTP {response.status_code}")


def _form(paths: list[Path], task_id: str, source_file_path: str, mode: str, handles):
    data = {
        "task_id": task_id,
        "source_file_path": upload_filename(source_file_path),
        "mode": mode,
    }
    files = [
        ("file", (path.name, handle, "application/octet-stream"))
        for path, handle in zip(paths, handles)
    ]
    return data, files


def submit_ocr(
    paths: list[Path],
    *,
    base_url: str,
    task_id: str,
    source_file_path: str,
    api_key: Optional[str] = None,
    mode: str = "det_rec",
    timeout: float = 120.0,
) -> str:
    """上传一组图片。返回的任务号就是调用方传入的那个。此时还没有识别结果。"""
    url = f"{base_url.rstrip('/')}/v1/ocr"
    handles = [path.open("rb") for path in paths]
    try:
        data, files = _form(paths, task_id, source_file_path, mode, handles)
        response = httpx.post(url, data=data, files=files, headers=_headers(api_key), timeout=timeout)
    finally:
        for handle in handles:
            handle.close()
    _raise_for_status(response)
    return response.json()["task_id"]


def get_ocr(
    base_url: str,
    task_id: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/ocr/{task_id}"
    response = httpx.get(url, headers=_headers(api_key), timeout=timeout)
    _raise_for_status(response)
    return response.json()


def cancel_ocr(
    base_url: str,
    task_id: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """立刻取消该任务。排队中的马上摘掉；正在识别的在当前这一小批算完后停下。"""
    url = f"{base_url.rstrip('/')}/v1/ocr/{task_id}/cancel"
    response = httpx.post(url, headers=_headers(api_key), timeout=timeout)
    _raise_for_status(response)
    return response.json()


def recognize_wait(
    paths: list[str | Path],
    base_url: str = "http://127.0.0.1:6008",
    *,
    task_id: str,
    source_file_path: str,
    api_key: Optional[str] = None,
    mode: str = "det_rec",
    poll_interval: float = 1.0,
    timeout: float = 3600.0,
) -> dict[str, Any]:
    """提交并轮询直到结束。完成时读取 result.items。结果默认保留 24 小时。"""
    files = [Path(item) for item in paths]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("找不到图片: " + "、".join(missing))
    task_id = submit_ocr(
        files,
        base_url=base_url,
        task_id=task_id,
        source_file_path=source_file_path,
        api_key=api_key,
        mode=mode,
    )
    deadline = time.time() + timeout
    last: Optional[dict[str, Any]] = None
    while time.time() < deadline:
        last = get_ocr(base_url, task_id, api_key=api_key)
        name = last.get("status_name")
        if name == "completed":
            return last
        if name == "cancelled":
            raise TaskCancelled(f"任务已取消 task_id={task_id}")
        if name == "failed":
            raise RuntimeError(f"识别失败: {last.get('error_msg')}")
        time.sleep(poll_interval)
    raise TimeoutError(f"等待超时 task_id={task_id} last={last}")


def recognize_sync(
    paths: list[str | Path],
    base_url: str = "http://127.0.0.1:6008",
    *,
    task_id: str,
    source_file_path: str,
    api_key: Optional[str] = None,
    mode: str = "det_rec",
    timeout: float = 3600.0,
) -> dict[str, Any]:
    files = [Path(item) for item in paths]
    url = f"{base_url.rstrip('/')}/v1/ocr/sync"
    handles = [path.open("rb") for path in files]
    try:
        data, form = _form(files, task_id, source_file_path, mode, handles)
        response = httpx.post(url, data=data, files=form, headers=_headers(api_key), timeout=timeout)
    finally:
        for handle in handles:
            handle.close()
    _raise_for_status(response)
    body = response.json()
    if body.get("status_name") == "cancelled":
        raise TaskCancelled(f"任务已取消 task_id={body.get('task_id')}")
    if body.get("status_name") == "failed":
        raise RuntimeError(f"识别失败: {body.get('error_msg')}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description="调用文字识别服务。默认提交后轮询，和语音客户端一样")
    parser.add_argument("images", nargs="+", help="图片路径，最多 32 张")
    parser.add_argument("--url", default=os.getenv("OCR_BASE_URL", "http://127.0.0.1:6008"))
    parser.add_argument("--api-key", default=os.getenv("OCR_API_KEY") or os.getenv("ASR_API_KEY"))
    parser.add_argument("--task-id", default="", help="调用方自己的唯一任务号。不填时由本脚本在本地生成")
    parser.add_argument("--source", default="", help="调用方机器上的源文件路径，只用于对账。不填时用第一张图片的路径")
    parser.add_argument("--mode", choices=["det_rec", "rec_only"], default="det_rec")
    parser.add_argument("--call", choices=("wait", "sync"), default="wait", help="wait=提交后轮询；sync=阻塞到结束")
    args = parser.parse_args()
    paths = [Path(item) for item in args.images]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print("找不到图片: " + "、".join(missing), file=sys.stderr)
        sys.exit(2)
    task_id = args.task_id or new_task_id()
    source = args.source or upload_filename(paths[0].resolve())
    print(f"task_id={task_id}", flush=True)
    if args.call == "sync":
        result = recognize_sync(
            paths, base_url=args.url, task_id=task_id, source_file_path=source,
            api_key=args.api_key, mode=args.mode,
        )
    else:
        result = recognize_wait(
            paths, base_url=args.url, task_id=task_id, source_file_path=source,
            api_key=args.api_key, mode=args.mode,
        )
    for item in (result.get("result") or {}).get("items") or []:
        if item.get("ok"):
            print(f"[{item.get('index')}] {item.get('text')}")
        else:
            print(f"[{item.get('index')}] 失败: {item.get('error')}", file=sys.stderr)


if __name__ == "__main__":
    main()
