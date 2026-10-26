#!/usr/bin/env python3
"""
公司内网 CPU 流水线调用 AutoDL 语音转写服务的示例客户端。

用法：
  export ASR_BASE_URL=http://<autodl可达地址>:6006
  export ASR_API_KEY=你的密钥
  python client_example.py /path/to/audio.mp4

取消某个任务号：
  from client_example import cancel_asr
  cancel_asr(base_url, task_id, api_key=...)
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
    """仅供本机示例和自测生成任务号。正式流水线应传入自己的唯一任务号，不要依赖这个函数。"""
    return f"req_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def upload_filename(file_path: str | Path) -> str:
    """记到结果里的路径文字。统一成正斜杠，服务不会按它打开磁盘。"""
    return str(file_path).replace("\\", "/")


def _upload_form(file_path: Path, task_id: str, handle,
                 strict_filter: Optional[bool] = None) -> tuple[dict[str, str], dict]:
    recorded = upload_filename(file_path)
    data = {"task_id": task_id, "source_file_path": recorded}
    if strict_filter is not None:
        data["strict_filter"] = str(strict_filter).lower()
    files = {"file": (Path(recorded).name, handle)}
    return data, files


def submit_asr(
    base_url: str,
    file_path: Path,
    *,
    task_id: str,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
    strict_filter: Optional[bool] = None,
) -> str:
    url = f"{base_url.rstrip('/')}/v1/asr"
    with file_path.open("rb") as f:
        data, files = _upload_form(file_path, task_id, f, strict_filter)
        r = httpx.post(
            url,
            data=data,
            files=files,
            headers=_headers(api_key),
            timeout=timeout,
        )
    if r.status_code == 401:
        raise PermissionError("鉴权失败：请检查 ASR_API_KEY")
    if r.status_code == 409:
        body = r.json()
        err = body.get("error") if isinstance(body, dict) else None
        message = (err or {}).get("error_message") if isinstance(err, dict) else None
        raise RuntimeError(message or "任务号已存在")
    r.raise_for_status()
    return r.json()["task_id"]


class TaskCancelled(RuntimeError):
    """任务已被取消。流水线应停止等待，不要把这次调用当成转写成功。"""


def cancel_asr(
    base_url: str,
    task_id: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """请求服务立刻取消该任务。排队中的会马上摘掉；正在转写的会停掉计算。"""
    url = f"{base_url.rstrip('/')}/v1/asr/{task_id}/cancel"
    r = httpx.post(url, headers=_headers(api_key), timeout=timeout)
    if r.status_code == 401:
        raise PermissionError("鉴权失败：请检查 ASR_API_KEY")
    if r.status_code == 404:
        raise LookupError("任务不存在或已过期")
    if r.status_code == 409:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        err = body.get("error") if isinstance(body, dict) else None
        message = (err or {}).get("error_message") if isinstance(err, dict) else None
        raise RuntimeError(message or "任务已结束，无法取消")
    r.raise_for_status()
    return r.json()


def get_asr(
    base_url: str,
    task_id: str,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/asr/{task_id}"
    r = httpx.get(url, headers=_headers(api_key), timeout=timeout)
    if r.status_code == 401:
        raise PermissionError("鉴权失败：请检查 ASR_API_KEY")
    r.raise_for_status()
    return r.json()


def transcribe_wait(
    file_path: str | Path,
    base_url: str = "http://127.0.0.1:6006",
    *,
    task_id: str,
    api_key: Optional[str] = None,
    poll_interval: float = 1.5,
    timeout: float = 3600.0,
    strict_filter: Optional[bool] = None,
) -> dict[str, Any]:
    """
    上传并轮询直到完成。流水线可当同步使用：
      result = transcribe_wait(audio_path, base_url=..., api_key=...)
      text = result["result"]["text"]
    结果在 AutoDL 上默认保留 24 小时，可再次用 get_asr 拉取。
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    task_id = submit_asr(
        base_url,
        path,
        task_id=task_id,
        api_key=api_key,
        strict_filter=strict_filter,
    )
    deadline = time.time() + timeout
    last: Optional[dict[str, Any]] = None
    while time.time() < deadline:
        last = get_asr(base_url, task_id, api_key=api_key)
        name = last.get("status_name")
        if name == "completed":
            return last
        if name == "cancelled":
            raise TaskCancelled(f"任务已取消 task_id={task_id}")
        if name == "failed":
            raise RuntimeError(f"转写失败: {last.get('error_msg')}")
        time.sleep(poll_interval)

    raise TimeoutError(f"等待超时 task_id={task_id} last={last}")


def transcribe_sync(
    file_path: str | Path,
    base_url: str = "http://127.0.0.1:6006",
    *,
    task_id: str,
    api_key: Optional[str] = None,
    timeout: float = 3600.0,
    strict_filter: Optional[bool] = None,
) -> dict[str, Any]:
    path = Path(file_path)
    url = f"{base_url.rstrip('/')}/v1/asr/sync"
    with path.open("rb") as f:
        data, files = _upload_form(path, task_id, f, strict_filter)
        r = httpx.post(
            url,
            data=data,
            files=files,
            headers=_headers(api_key),
            timeout=timeout,
        )
    if r.status_code == 401:
        raise PermissionError("鉴权失败：请检查 ASR_API_KEY")
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    data = r.json()
    if data.get("status_name") == "cancelled":
        raise TaskCancelled(f"任务已取消 task_id={data.get('task_id')}")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ASR 远程转写客户端（内网→AutoDL）")
    parser.add_argument("file", type=Path, help="音频或视频文件路径")
    parser.add_argument(
        "--url",
        default=os.getenv("ASR_BASE_URL", "http://127.0.0.1:6006"),
        help="AutoDL 上 ASR 服务地址",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ASR_API_KEY", ""),
        help="与服务器 ASR_API_KEY 一致",
    )
    parser.add_argument("--task-id", default="", help="调用方自己的唯一任务号。不填时由本脚本在本地生成")
    parser.add_argument(
        "--mode",
        choices=("wait", "sync"),
        default="wait",
        help="wait=提交+轮询；sync=阻塞接口",
    )
    parser.add_argument("--strict-filter", action=argparse.BooleanOptionalAction, default=None,
                        help="严格过滤可疑文本；--no-strict-filter 保留存疑文本，默认跟随服务端")
    args = parser.parse_args(argv)
    key = args.api_key or None
    task_id = args.task_id or new_task_id()

    if args.mode == "sync":
        result = transcribe_sync(args.file, base_url=args.url, task_id=task_id, api_key=key,
                                 strict_filter=args.strict_filter)
    else:
        result = transcribe_wait(args.file, base_url=args.url, task_id=task_id, api_key=key,
                                 strict_filter=args.strict_filter)

    data = result.get("result") or {}
    print(f"status={result.get('status')} language={data.get('language')} task_id={result.get('task_id')}")
    print(data.get("text", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
