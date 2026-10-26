#!/usr/bin/env python3
"""
压测：并发提交 N 个转写任务，统计排队/推理/吞吐与失败率。

示例：
  python scripts/bench_asr.py --file test_video/xxx.mp4 --n 8
  python scripts/bench_asr.py --file test_video/xxx.mp4 --n 80 --url http://127.0.0.1:6006
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_example import get_asr, new_task_id, submit_asr  # noqa: E402


def wait_one(
    base_url: str,
    file_path: Path,
    poll_interval: float,
    timeout: float,
    api_key: str | None,
) -> dict:
    t0 = time.time()
    task_id = submit_asr(
        base_url,
        file_path,
        task_id=new_task_id(),
        api_key=api_key,
    )
    submit_cost = time.time() - t0
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = get_asr(base_url, task_id, api_key=api_key)
        status = last.get("status")
        if status == "completed" or last.get("status") == 0:
            return {
                "ok": True,
                "task_id": task_id,
                "submit_cost": submit_cost,
                "wall_cost": time.time() - t0,
                "infer_cost": last.get("cost_time_s"),
                "error": None,
            }
        if status == "failed" or last.get("status") == 1:
            return {
                "ok": False,
                "task_id": task_id,
                "submit_cost": submit_cost,
                "wall_cost": time.time() - t0,
                "infer_cost": None,
                "error": last.get("error_msg"),
            }
        if last.get("status_name") == "cancelled":
            return {
                "ok": False,
                "task_id": task_id,
                "submit_cost": submit_cost,
                "wall_cost": time.time() - t0,
                "infer_cost": None,
                "error": "cancelled",
            }
        time.sleep(poll_interval)
    return {
        "ok": False,
        "task_id": task_id,
        "submit_cost": submit_cost,
        "wall_cost": time.time() - t0,
        "infer_cost": None,
        "error": f"timeout last={last}",
    }


def health(base_url: str) -> dict:
    r = httpx.get(f"{base_url.rstrip('/')}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="ASR 容量压测")
    parser.add_argument("--file", type=Path, required=True, help="测试音视频路径")
    parser.add_argument("--n", type=int, default=8, help="并发提交任务数")
    parser.add_argument("--url", default="http://127.0.0.1:6006")
    parser.add_argument("--api-key", default=os.getenv("ASR_API_KEY", ""))
    parser.add_argument("--poll", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    api_key = args.api_key or None

    if not args.file.is_file():
        print(f"文件不存在: {args.file}", file=sys.stderr)
        return 1

    before = health(args.url)
    print("=== health (before) ===")
    for k, v in before.items():
        print(f"  {k}: {v}")

    print(f"\n提交 {args.n} 个任务（文件={args.file}）...")
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=min(args.n, 64)) as ex:
        futs = [
            ex.submit(wait_one, args.url, args.file, args.poll, args.timeout, api_key)
            for _ in range(args.n)
        ]
        for fut in as_completed(futs):
            results.append(fut.result())
            done = sum(1 for r in results if r["ok"])
            fail = sum(1 for r in results if not r["ok"])
            print(f"  进度 {len(results)}/{args.n} ok={done} fail={fail}", flush=True)

    wall = time.time() - t0
    ok_list = [r for r in results if r["ok"]]
    fail_list = [r for r in results if not r["ok"]]

    print("\n=== 汇总 ===")
    print(f"  总数: {args.n}")
    print(f"  成功: {len(ok_list)}")
    print(f"  失败: {len(fail_list)}")
    print(f"  总墙钟: {wall:.2f}s")
    if wall > 0:
        print(f"  吞吐: {len(ok_list) / wall:.3f} 任务/秒")

    if ok_list:
        walls = [r["wall_cost"] for r in ok_list]
        infers = [r["infer_cost"] for r in ok_list if r["infer_cost"] is not None]
        print(f"  单任务墙钟: avg={statistics.mean(walls):.2f}s p50={statistics.median(walls):.2f}s max={max(walls):.2f}s")
        if infers:
            print(
                f"  单任务推理: avg={statistics.mean(infers):.2f}s "
                f"p50={statistics.median(infers):.2f}s max={max(infers):.2f}s"
            )

    if fail_list:
        print("\n=== 失败样例 ===")
        for r in fail_list[:5]:
            print(f"  {r['task_id']}: {r['error']}")

    after = health(args.url)
    print("\n=== health (after) ===")
    for k, v in after.items():
        print(f"  {k}: {v}")

    return 0 if not fail_list else 2


if __name__ == "__main__":
    raise SystemExit(main())
