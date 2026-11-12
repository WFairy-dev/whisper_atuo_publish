#!/usr/bin/env python3
"""清晰度服务并发压测。

用法示例：
  # 只测单条时长影响（不启 HTTP）
  python scripts/bench_dover.py duration --videos bench_data/dover/v60.mp4 bench_data/dover/v90.mp4

  # 对已启动的服务压并发（先 export DOVER_MAX_WORKERS=4 再启服务）
  python scripts/bench_dover.py http --url http://127.0.0.1:6009 --file bench_data/dover/v60.mp4 --n 24

  # 一键：起服务扫 1/2/4/6/8 路，每路提交 n 条
  python scripts/bench_dover.py sweep --file bench_data/dover/v60.mp4 --n 24 --workers 1,2,4,6,8
"""
from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = Path("/root/miniconda3/envs/whisper-turbo/bin/ffmpeg")
FFPROBE = Path("/root/miniconda3/envs/whisper-turbo/bin/ffprobe")
DOVER_PY = Path("/root/autodl-tmp/envs/dover/bin/python")


def _duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        text=True,
    )
    return float(out.strip())


def prepare_videos(out_dir: Path) -> dict[str, Path]:
    """用现有素材拼出约 60s / 90s 测试片。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "test_video" / "7b000bd63eda17d6b40136d295a0e221.mp4"
    if not src.is_file():
        src = ROOT / "dover_server" / "DOVER" / "demo" / "17734.mp4"
    targets = {"v60": 60.0, "v90": 90.0}
    result: dict[str, Path] = {}
    for name, seconds in targets.items():
        dest = out_dir / f"{name}.mp4"
        if dest.is_file():
            try:
                if abs(_duration(dest) - seconds) < 2.0:
                    result[name] = dest
                    continue
            except Exception:
                pass
        # 循环拷贝截到目标时长（本机 ffmpeg 无 libx264，用 copy）。
        cmd = [
            str(FFMPEG),
            "-y",
            "-stream_loop",
            "-1",
            "-i",
            str(src),
            "-t",
            str(seconds),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(dest),
        ]
        print(f"生成 {dest.name} ≈{seconds}s …")
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        result[name] = dest
    short = ROOT / "dover_server" / "DOVER" / "demo" / "1724.mp4"
    if short.is_file():
        result["v8"] = short
    result["src"] = src
    return result


def cmd_prepare(args: argparse.Namespace) -> int:
    paths = prepare_videos(Path(args.out_dir))
    for key, path in paths.items():
        print(f"{key}: {path} duration={_duration(path):.2f}s size={path.stat().st_size}")
    return 0


def cmd_duration(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT))
    from dover_server.clarity import ClarityEvaluator

    videos = [Path(p) for p in args.videos]
    if not videos:
        prepared = prepare_videos(ROOT / "bench_data" / "dover")
        videos = [prepared[k] for k in ("v8", "src", "v60", "v90") if k in prepared]
    ev = ClarityEvaluator(device="cpu")
    print("video\tduration_s\trun\tcost_time_s\ttechnical_score")
    for path in videos:
        dur = _duration(path)
        for run in range(args.runs):
            r = ev.score_video(path)
            print(f"{path.name}\t{dur:.2f}\t{run+1}\t{r['cost_time_s']}\t{r['technical_score']:.6f}")
            sys.stdout.flush()
    return 0


def _headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"X-API-Key": api_key}


def _wait_one(
    client: httpx.Client,
    base_url: str,
    file_path: Path,
    poll: float,
    timeout: float,
) -> dict:
    t0 = time.time()
    task_id = f"bench_{uuid.uuid4().hex[:16]}"
    with file_path.open("rb") as handle:
        resp = client.post(
            f"{base_url.rstrip('/')}/v1/vqa",
            data={"task_id": task_id, "source_file_path": f"/bench/{file_path.name}"},
            files={"file": (file_path.name, handle, "video/mp4")},
        )
    submit_cost = time.time() - t0
    if resp.status_code != 200:
        return {
            "ok": False,
            "task_id": task_id,
            "submit_cost": submit_cost,
            "wall_cost": submit_cost,
            "infer_cost": None,
            "queue_wait_s": None,
            "error": f"submit HTTP {resp.status_code}: {resp.text[:200]}",
        }
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get(f"{base_url.rstrip('/')}/v1/vqa/{task_id}").json()
        name = last.get("status_name")
        if name == "completed":
            return {
                "ok": True,
                "task_id": task_id,
                "submit_cost": submit_cost,
                "wall_cost": time.time() - t0,
                "infer_cost": last.get("cost_time_s"),
                "queue_wait_s": last.get("queue_wait_s"),
                "error": None,
            }
        if name in ("failed", "cancelled"):
            return {
                "ok": False,
                "task_id": task_id,
                "submit_cost": submit_cost,
                "wall_cost": time.time() - t0,
                "infer_cost": None,
                "queue_wait_s": last.get("queue_wait_s"),
                "error": last.get("error_msg") or name,
            }
        time.sleep(poll)
    return {
        "ok": False,
        "task_id": task_id,
        "submit_cost": submit_cost,
        "wall_cost": time.time() - t0,
        "infer_cost": None,
        "queue_wait_s": None if last is None else last.get("queue_wait_s"),
        "error": "timeout",
    }


def _summarize(rows: list[dict], wall: float, n: int, workers: int | None = None) -> None:
    ok = [r for r in rows if r["ok"]]
    bad = [r for r in rows if not r["ok"]]
    infer = [float(r["infer_cost"]) for r in ok if r.get("infer_cost") is not None]
    waits = [float(r["queue_wait_s"]) for r in ok if r.get("queue_wait_s") is not None]
    walls = [float(r["wall_cost"]) for r in ok]
    prefix = f"workers={workers}\t" if workers is not None else ""
    print(
        f"{prefix}n={n}\tok={len(ok)}\tfail={len(bad)}\t"
        f"batch_wall_s={wall:.2f}\tthroughput={len(ok)/wall if wall else 0:.2f}/s"
    )
    if infer:
        print(
            f"{prefix}infer_s: mean={statistics.mean(infer):.3f} "
            f"p50={statistics.median(infer):.3f} "
            f"max={max(infer):.3f}"
        )
    if waits:
        print(
            f"{prefix}queue_wait_s: mean={statistics.mean(waits):.3f} "
            f"p50={statistics.median(waits):.3f} "
            f"max={max(waits):.3f}"
        )
    if walls:
        print(
            f"{prefix}client_wall_s: mean={statistics.mean(walls):.3f} "
            f"p50={statistics.median(walls):.3f} "
            f"max={max(walls):.3f}"
        )
    for row in bad[:5]:
        print(f"{prefix}FAIL {row['task_id']}: {row['error']}")


def cmd_http(args: argparse.Namespace) -> int:
    path = Path(args.file)
    api_key = args.api_key or os.getenv("DOVER_API_KEY") or os.getenv("ASR_API_KEY") or ""
    headers = _headers(api_key or None)
    t0 = time.time()
    rows: list[dict] = []
    with httpx.Client(timeout=args.timeout, headers=headers) as client:
        health = client.get(f"{args.url.rstrip('/')}/health")
        health.raise_for_status()
        info = health.json()
        print(
            f"health workers={info.get('max_workers')} ready={info.get('ready_workers')} "
            f"device={info.get('device')} video={path.name} duration≈{_duration(path):.1f}s"
        )
        with ThreadPoolExecutor(max_workers=args.n) as pool:
            futs = [
                pool.submit(_wait_one, client, args.url, path, args.poll, args.timeout)
                for _ in range(args.n)
            ]
            for fut in as_completed(futs):
                rows.append(fut.result())
    _summarize(rows, time.time() - t0, args.n, workers=info.get("max_workers"))
    return 0 if all(r["ok"] for r in rows) else 2


def _wait_health(url: str, timeout: float = 180.0) -> dict:
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{url.rstrip('/')}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("model_loaded"):
                        return data
                    last_err = f"model_loaded={data.get('model_loaded')}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"服务未就绪: {last_err}")


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def cmd_sweep(args: argparse.Namespace) -> int:
    path = Path(args.file)
    workers_list = [int(x) for x in args.workers.split(",") if x.strip()]
    port = args.port
    url = f"http://127.0.0.1:{port}"
    log_dir = ROOT / "bench_data" / "dover" / "sweep_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"素材 {path} duration≈{_duration(path):.1f}s；"
        f"每档并发提交 {args.n} 条；路数={workers_list}"
    )
    overall_ok = True
    for workers in workers_list:
        env = os.environ.copy()
        env["DOVER_HOST"] = "127.0.0.1"
        env["DOVER_PORT"] = str(port)
        env["DOVER_DEVICE"] = "cpu"
        env["DOVER_MAX_WORKERS"] = str(workers)
        env["DOVER_ENABLE_DOCS"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        # 避免扫档时撞上已有密钥要求。
        env.pop("DOVER_API_KEY", None)
        env.pop("ASR_API_KEY", None)
        log_path = log_dir / f"workers_{workers}.log"
        print(f"\n===== 启动 workers={workers} =====")
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                [str(DOVER_PY), "-m", "dover_server"],
                cwd=str(ROOT),
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        try:
            info = _wait_health(url, timeout=args.boot_timeout)
            print(f"就绪 max_workers={info.get('max_workers')} ready={info.get('ready_workers')}")
            ns = argparse.Namespace(
                url=url,
                file=str(path),
                n=args.n,
                poll=args.poll,
                timeout=args.timeout,
                api_key="",
            )
            code = cmd_http(ns)
            if code != 0:
                overall_ok = False
        except Exception as exc:
            overall_ok = False
            print(f"workers={workers} 失败: {exc}")
        finally:
            _stop_proc(proc)
            time.sleep(2)
    return 0 if overall_ok else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="DOVER 清晰度并发压测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("prepare", help="生成 60s/90s 测试视频")
    p0.add_argument("--out-dir", default=str(ROOT / "bench_data" / "dover"))
    p0.set_defaults(func=cmd_prepare)

    p1 = sub.add_parser("duration", help="单进程测不同时长耗时")
    p1.add_argument("--videos", nargs="*", default=[])
    p1.add_argument("--runs", type=int, default=3)
    p1.set_defaults(func=cmd_duration)

    p2 = sub.add_parser("http", help="对已启动服务做并发压测")
    p2.add_argument("--url", default="http://127.0.0.1:6009")
    p2.add_argument("--file", required=True)
    p2.add_argument("--n", type=int, default=24)
    p2.add_argument("--poll", type=float, default=0.5)
    p2.add_argument("--timeout", type=float, default=600)
    p2.add_argument("--api-key", default="")
    p2.set_defaults(func=cmd_http)

    p3 = sub.add_parser("sweep", help="自动切换路数并压测")
    p3.add_argument("--file", required=True)
    p3.add_argument("--n", type=int, default=24)
    p3.add_argument("--workers", default="1,2,4,6,8")
    p3.add_argument("--port", type=int, default=6009)
    p3.add_argument("--poll", type=float, default=0.5)
    p3.add_argument("--timeout", type=float, default=900)
    p3.add_argument("--boot-timeout", type=float, default=180)
    p3.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
