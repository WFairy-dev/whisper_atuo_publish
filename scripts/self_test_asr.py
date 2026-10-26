#!/usr/bin/env python3
"""语音服务自测：模拟调用各接口，并测量排队耗时与显卡能同时算几路。

接口测试要求服务已启动：
  python scripts/self_test_asr.py api --url http://127.0.0.1:6006

显卡并行测试要求先停掉服务：
  python scripts/self_test_asr.py parallel --max-n 4

结果写入 --out 指定的 JSON。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_example import (  # noqa: E402
    TaskCancelled,
    cancel_asr,
    get_asr,
    new_task_id,
    submit_asr,
    transcribe_sync,
    transcribe_wait,
    upload_filename,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _gpu() -> dict[str, Any] | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    name, used_s, total_s = [p.strip() for p in out.strip().splitlines()[0].split(",")]
    return {"name": name, "used_mib": float(used_s), "total_mib": float(total_s)}


def _health(base: str) -> dict[str, Any]:
    r = httpx.get(f"{base.rstrip('/')}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


class Recorder:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []

    def run(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.time()
        try:
            detail = fn()
            row = {
                "name": name,
                "ok": True,
                "elapsed_s": round(time.time() - t0, 3),
                "detail": detail if isinstance(detail, (str, int, float, dict, list)) or detail is None else str(detail),
            }
            self.cases.append(row)
            print(f"  通过 {name} {row['elapsed_s']}s", flush=True)
            return detail
        except Exception as e:
            row = {
                "name": name,
                "ok": False,
                "elapsed_s": round(time.time() - t0, 3),
                "detail": f"{type(e).__name__}: {e}",
            }
            self.cases.append(row)
            print(f"  失败 {name} {row['detail']}", flush=True)
            return None


def _make_long(src: Path, loops: int) -> Path:
    dest = Path(tempfile.gettempdir()) / "asr_self_test_long.mp4"
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    subprocess.check_call(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-stream_loop",
            str(loops),
            "-i",
            str(src),
            "-c",
            "copy",
            str(dest),
        ],
        timeout=60,
    )
    return dest


def _wait_status(base: str, key: str, task_id: str, want: set[str], timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_asr(base, task_id, api_key=key)
        if last.get("status_name") in want:
            return last
        time.sleep(0.05)
    raise TimeoutError(f"等待状态 {want} 超时，最后={last.get('status_name')}")


def _wait_recovered(base: str, timeout: float = 120) -> dict[str, Any]:
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = _health(base)
        if last.get("model_loaded") and not last.get("recovering") and last.get("running") == 0:
            return last
        time.sleep(0.5)
    raise TimeoutError(f"模型没有恢复: {last}")


def _one_task(base: str, key: str, path: Path, timeout: float) -> dict[str, Any]:
    t0 = time.time()
    task_id = submit_asr(base, path, task_id=new_task_id(), api_key=key)
    submit_s = time.time() - t0
    deadline = time.time() + timeout
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_asr(base, task_id, api_key=key)
        status = last.get("status")
        if status == "completed" or last.get("status") == 0:
            return {
                "ok": True,
                "task_id": task_id,
                "submit_s": round(submit_s, 3),
                "wall_s": round(time.time() - t0, 3),
                "infer_s": last.get("cost_time_s"),
                "language": (last.get("result") or {}).get("language"),
                "segments": len((last.get("result") or {}).get("segments") or []),
            }
        if status in ("failed", "cancelled") or last.get("status") in (1, 4):
            return {
                "ok": False,
                "task_id": task_id,
                "submit_s": round(submit_s, 3),
                "wall_s": round(time.time() - t0, 3),
                "error": last.get("error_msg") or status,
            }
        time.sleep(0.4)
    return {"ok": False, "task_id": task_id, "wall_s": round(time.time() - t0, 3), "error": "timeout"}


def _summarize_batch(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    ok = [r for r in rows if r.get("ok")]
    infers = [float(r["infer_s"]) for r in ok if r.get("infer_s") is not None]
    walls = [float(r["wall_s"]) for r in ok]
    return {
        "n": len(rows),
        "ok": len(ok),
        "fail": len(rows) - len(ok),
        "wall_s": round(wall_s, 3),
        "throughput_per_s": round(len(ok) / wall_s, 3) if wall_s else 0,
        "infer_avg_s": round(statistics.mean(infers), 3) if infers else None,
        "infer_p50_s": round(_percentile(infers, 0.5) or 0, 3) if infers else None,
        "infer_max_s": round(max(infers), 3) if infers else None,
        "queue_wall_avg_s": round(statistics.mean(walls), 3) if walls else None,
        "queue_wall_p50_s": round(_percentile(walls, 0.5) or 0, 3) if walls else None,
        "queue_wall_max_s": round(max(walls), 3) if walls else None,
        "errors": [r.get("error") for r in rows if not r.get("ok")][:5],
    }


def run_api(base: str, key: str, sample: Path, out_path: Path) -> dict[str, Any]:
    rec = Recorder()
    long_path = _make_long(sample, 5)
    print(f"加长样片 {long_path} 字节={long_path.stat().st_size}", flush=True)

    def health_ok() -> dict[str, Any]:
        body = _health(base)
        if not body.get("model_loaded"):
            raise RuntimeError(f"模型未加载 {body}")
        if int(body.get("max_workers") or 0) < 1:
            raise RuntimeError(f"转写路数异常: {body.get('max_workers')}")
        if int(body.get("ready_workers") or 0) < 1:
            raise RuntimeError(f"没有就绪的转写路: {body}")
        return {
            "version": body.get("service_version"),
            "gpu": body.get("gpu_mem"),
            "recovering": body.get("recovering"),
        }

    def reject_no_key() -> str:
        url = f"{base.rstrip('/')}/v1/asr"
        with sample.open("rb") as f:
            r = httpx.post(
                url,
                data={"task_id": new_task_id()},
                files={"source_file_path": (upload_filename(sample), f)},
                timeout=60,
            )
        if r.status_code != 401:
            raise RuntimeError(f"期望 401，实际 {r.status_code} {r.text[:200]}")
        return "401"

    def reject_bad_suffix() -> str:
        url = f"{base.rstrip('/')}/v1/asr"
        r = httpx.post(
            url,
            data={"task_id": new_task_id()},
            files={"source_file_path": ("D:/share/videos/note.txt", b"hello", "text/plain")},
            headers={"X-API-Key": key},
            timeout=30,
        )
        if r.status_code != 400:
            raise RuntimeError(f"期望 400，实际 {r.status_code} {r.text[:200]}")
        return "400"

    def full_flow() -> dict[str, Any]:
        result = transcribe_wait(
            sample,
            base_url=base,
            task_id=new_task_id(),
            api_key=key,
            poll_interval=0.3,
            timeout=180,
        )
        data = result.get("result") or {}
        if result.get("status_name") != "completed":
            raise RuntimeError(result.get("status"))
        if result.get("source_file_path") != upload_filename(sample):
            raise RuntimeError(f"路径没有原样记下: {result.get('source_file_path')}")
        if not data.get("language") or not str(data.get("text") or "").strip():
            raise RuntimeError("缺少语种或全文")
        if not data.get("segments"):
            raise RuntimeError("缺少时间轴")
        seg = data["segments"][0]
        for field in ("start_s", "end_s", "text"):
            if field not in seg:
                raise RuntimeError(f"时间轴缺少 {field}")
        return {
            "task_id": result["task_id"],
            "language": data["language"],
            "text_len": len(data["text"]),
            "segments": len(data["segments"]),
            "infer_s": round(float(result.get("cost_time_s") or 0), 3),
        }

    done_holder: dict[str, Any] = {}

    def remember_done() -> dict[str, Any]:
        detail = full_flow()
        done_holder.update(detail)
        return detail

    def cancel_finished() -> str:
        try:
            cancel_asr(base, done_holder["task_id"], api_key=key)
        except RuntimeError as e:
            if "409" in str(e) or "无法取消" in str(e):
                return "409"
            raise
        raise RuntimeError("已完成的任务被取消了")

    def missing_cancel() -> str:
        try:
            cancel_asr(base, "a" * 32, api_key=key)
        except LookupError:
            return "404"
        raise RuntimeError("不存在的任务没有返回 404")

    def missing_query() -> str:
        r = httpx.get(
            f"{base.rstrip('/')}/v1/asr/{'b' * 32}",
            headers={"X-API-Key": key},
            timeout=15,
        )
        if r.status_code != 404:
            raise RuntimeError(f"期望 404，实际 {r.status_code}")
        return "404"

    def queued_then_running_cancel() -> dict[str, Any]:
        blocker = submit_asr(base, long_path, task_id=new_task_id(), api_key=key)
        _wait_status(base, key, blocker, {"running"}, 30)
        second = submit_asr(base, sample, task_id=new_task_id(), api_key=key)
        second_state = get_asr(base, second, api_key=key)
        if second_state.get("status_name") != "queued":
            raise RuntimeError(f"第二条不是排队: {second_state.get('status_name')}")
        t0 = time.time()
        queued = cancel_asr(base, second, api_key=key)
        queued_s = time.time() - t0
        if queued.get("status_name") != "cancelled":
            raise RuntimeError(queued)
        again = cancel_asr(base, second, api_key=key)
        if again.get("status_name") != "cancelled":
            raise RuntimeError("重复取消没有保持 cancelled")
        t1 = time.time()
        running = cancel_asr(base, blocker, api_key=key)
        running_s = time.time() - t1
        if running.get("status_name") != "cancelled":
            raise RuntimeError(running)
        if queued_s >= 1 or running_s >= 1:
            raise RuntimeError(f"取消不够快 queued={queued_s:.3f}s running={running_s:.3f}s")
        recovered = _wait_recovered(base)
        return {
            "queued_cancel_s": round(queued_s, 3),
            "running_cancel_s": round(running_s, 3),
            "gpu_after_recover": recovered.get("gpu_mem"),
        }

    def after_recover() -> dict[str, Any]:
        return full_flow()

    def sync_ok() -> dict[str, Any]:
        result = transcribe_sync(
            sample,
            base_url=base,
            task_id=new_task_id(),
            api_key=key,
            timeout=180,
        )
        if result.get("status_name") != "completed":
            raise RuntimeError(result.get("status"))
        return {"infer_s": round(float(result.get("cost_time_s") or 0), 3), "language": result["result"]["language"]}

    def sync_cancel() -> dict[str, Any]:
        holder: dict[str, Any] = {}
        task_id = new_task_id()

        def _call() -> None:
            try:
                holder["result"] = transcribe_sync(
                    long_path,
                    base_url=base,
                    task_id=task_id,
                    api_key=key,
                    timeout=300,
                )
            except Exception as e:
                holder["error"] = f"{type(e).__name__}: {e}"

        th = threading.Thread(target=_call)
        th.start()
        seen = None
        deadline = time.time() + 60
        while time.time() < deadline and seen is None:
            body = _health(base)
            if body.get("running"):
                state = get_asr(base, task_id, api_key=key)
                if state.get("status_name") == "running":
                    seen = task_id
                    break
            time.sleep(0.05)
        if seen is None:
            th.join(5)
            raise RuntimeError(f"同步任务没有进入转写 {holder}")
        state = _wait_status(base, key, task_id, {"running", "completed", "cancelled"}, 20)
        if state.get("status_name") != "running":
            th.join(5)
            raise RuntimeError(f"同步任务来不及停在转写中: {state.get('status_name')}")
        t0 = time.time()
        cancelled = cancel_asr(base, task_id, api_key=key)
        elapsed = time.time() - t0
        th.join(30)
        if cancelled.get("status_name") != "cancelled" or elapsed >= 1:
            raise RuntimeError(f"取消返回 {cancelled.get('status')} {elapsed:.3f}s")
        if "TaskCancelled" not in str(holder.get("error")):
            raise RuntimeError(f"调用方没有收到取消 {holder}")
        recovered = _wait_recovered(base)
        return {"cancel_s": round(elapsed, 3), "gpu_after_recover": recovered.get("gpu_mem")}

    print("=== 接口 ===", flush=True)
    rec.run("健康检查", health_ok)
    rec.run("缺少密钥被拒绝", reject_no_key)
    rec.run("错误文件类型被拒绝", reject_bad_suffix)
    rec.run("提交并等到转写完成", remember_done)
    rec.run("已完成的任务不能取消", cancel_finished)
    rec.run("取消不存在的任务", missing_cancel)
    rec.run("查询不存在的任务", missing_query)
    rec.run("取消排队中和正在转写的任务", queued_then_running_cancel)
    rec.run("取消后恢复再转一条", after_recover)
    rec.run("同步转写短样片", sync_ok)
    rec.run("同步等待时取消正在转写的任务", sync_cancel)

    print("=== 排队 ===", flush=True)
    batches = []
    for n in (8, 24):
        print(f"提交 {n} 条", flush=True)
        t0 = time.time()
        rows: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(_one_task, base, key, sample, 1800) for _ in range(n)]
            for fut in as_completed(futs):
                rows.append(fut.result())
        summary = _summarize_batch(rows, time.time() - t0)
        summary["gpu_after"] = (_health(base).get("gpu_mem"))
        batches.append(summary)
        print(
            f"  n={n} 成功={summary['ok']} 失败={summary['fail']} "
            f"总耗时={summary['wall_s']}s 推理均值={summary['infer_avg_s']}s",
            flush=True,
        )

    single = next((c["detail"] for c in rec.cases if c["name"] == "提交并等到转写完成" and c["ok"]), None)
    report = {
        "kind": "api",
        "at": _now(),
        "gpu": _gpu(),
        "sample": str(sample),
        "cases": rec.cases,
        "passed": sum(1 for c in rec.cases if c["ok"]),
        "failed": sum(1 for c in rec.cases if not c["ok"]),
        "single": single,
        "batches": batches,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        long_path.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"接口测试 {report['passed']} 通过 / {report['failed']} 失败，写入 {out_path}", flush=True)
    return report


def _server_running() -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    return any("python -m asr_server" in line for line in out.splitlines())


def _parallel_worker(path: str, model_name: str, model_dir: str, device: str, ready, start, out) -> None:
    try:
        import whisper

        t0 = time.time()
        model = whisper.load_model(model_name, device=device, download_root=model_dir)
        ready.put({"ok": True, "load_s": time.time() - t0})
        if not start.wait(timeout=600):
            out.put({"ok": False, "error": "等待开始超时"})
            return
        t1 = time.time()
        whisper_result = model.transcribe(path, verbose=None)
        text = str(whisper_result.get("text") or "").strip()
        out.put(
            {
                "ok": bool(text),
                "load_s": time.time() - t0,
                "infer_s": time.time() - t1,
                "text_len": len(text),
                "error": None if text else "空文本",
            }
        )
    except Exception as e:
        err = f"{e}"
        try:
            ready.put({"ok": False, "error": err})
        except Exception:
            pass
        try:
            out.put({"ok": False, "error": err, "trace": traceback.format_exc()[-500:]})
        except Exception:
            pass


def _settle(baseline: float, timeout: float = 90) -> float:
    deadline = time.time() + timeout
    last = baseline
    while time.time() < deadline:
        info = _gpu()
        if info is None:
            return last
        last = info["used_mib"]
        if last <= baseline + 800:
            return last
        time.sleep(1)
    return last


def run_parallel(sample: Path, max_n: int, model_name: str, model_dir: str, device: str, out_path: Path) -> dict[str, Any]:
    if _server_running():
        raise SystemExit("语音服务还在运行，请先停掉再测并行，否则显存会被服务占住。")
    import multiprocessing as mp

    gpu = _gpu()
    if gpu is None:
        raise SystemExit("读不到显卡")
    print(f"显卡 {gpu['name']} 已用 {gpu['used_mib']:.0f} / {gpu['total_mib']:.0f} MiB", flush=True)
    ctx = mp.get_context("spawn")
    phases = []
    baseline = gpu["used_mib"]
    for n in range(1, max_n + 1):
        print(f"=== {n} 路 ===", flush=True)
        ready_q = ctx.Queue()
        out_q = ctx.Queue()
        start = ctx.Event()
        procs = [
            ctx.Process(
                target=_parallel_worker,
                args=(str(sample), model_name, model_dir, device, ready_q, start, out_q),
            )
            for _ in range(n)
        ]
        peak = baseline
        for proc in procs:
            proc.start()
        loads = []
        deadline = time.time() + 180
        while len(loads) < n and time.time() < deadline:
            info = _gpu()
            if info:
                peak = max(peak, info["used_mib"])
            try:
                loads.append(ready_q.get(timeout=0.5))
            except Exception:
                if any(not p.is_alive() for p in procs):
                    break
        bad_load = [x for x in loads if not x.get("ok")]
        if len(loads) < n or bad_load:
            start.set()
            for proc in procs:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=10)
            err = bad_load[0]["error"] if bad_load else "模型没有在 180 秒内加载完"
            phases.append({"n": n, "ok": False, "error": err, "peak_mib": round(peak, 1)})
            print(f"  失败 {err}", flush=True)
            _settle(baseline)
            break
        t0 = time.time()
        start.set()
        results = []
        deadline = time.time() + 300
        while len(results) < n and time.time() < deadline:
            info = _gpu()
            if info:
                peak = max(peak, info["used_mib"])
            try:
                results.append(out_q.get(timeout=0.5))
            except Exception:
                if all(not p.is_alive() for p in procs):
                    break
        wall = time.time() - t0
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)
        ok_rows = [r for r in results if r.get("ok")]
        bad = [r for r in results if not r.get("ok")]
        infers = [float(r["infer_s"]) for r in ok_rows]
        phase = {
            "n": n,
            "ok": len(ok_rows) == n and not bad,
            "error": (bad[0].get("error") if bad else None),
            "wall_s": round(wall, 3),
            "infer_s": [round(x, 3) for x in infers],
            "infer_max_s": round(max(infers), 3) if infers else None,
            "load_s": [round(float(x.get("load_s") or 0), 3) for x in loads if x.get("ok")],
            "peak_mib": round(peak, 1),
            "throughput_per_s": round(len(ok_rows) / wall, 3) if wall and ok_rows else 0,
        }
        phases.append(phase)
        print(
            f"  成功={phase['ok']} 墙钟={phase['wall_s']}s 峰值={phase['peak_mib']} MiB",
            flush=True,
        )
        _settle(baseline)
        if not phase["ok"] or phase["peak_mib"] >= gpu["total_mib"] * 0.92:
            break

    single = next((p for p in phases if p["n"] == 1 and p.get("ok")), None)
    best = None
    for phase in phases:
        if not phase.get("ok"):
            continue
        if best is None or phase["throughput_per_s"] > best["throughput_per_s"]:
            best = phase
    report = {
        "kind": "parallel",
        "at": _now(),
        "gpu": gpu,
        "sample": str(sample),
        "phases": phases,
        "best_n": None if best is None else best["n"],
        "single_infer_s": None if single is None else single.get("infer_max_s"),
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"并行测试写入 {out_path} 吞吐最高路数={report['best_n']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="语音服务自测")
    sub = parser.add_subparsers(dest="cmd", required=True)

    api = sub.add_parser("api")
    api.add_argument("--url", default="http://127.0.0.1:6006")
    api.add_argument("--api-key", default=os.getenv("ASR_API_KEY", ""))
    api.add_argument("--file", type=Path, required=True)
    api.add_argument("--out", type=Path, default=Path("/tmp/asr_self_test_api.json"))

    par = sub.add_parser("parallel")
    par.add_argument("--file", type=Path, required=True)
    par.add_argument("--max-n", type=int, default=4)
    par.add_argument("--model-name", default="turbo")
    par.add_argument("--model-dir", default="/root/autodl-tmp/models/whisper")
    par.add_argument("--device", default="cuda")
    par.add_argument("--out", type=Path, default=Path("/tmp/asr_self_test_parallel.json"))

    args = parser.parse_args()
    if not args.file.is_file():
        print(f"文件不存在: {args.file}", file=sys.stderr)
        return 1
    if args.cmd == "api":
        if not args.api_key:
            print("缺少密钥", file=sys.stderr)
            return 1
        report = run_api(args.url, args.api_key, args.file, args.out)
        return 0 if report["failed"] == 0 else 2
    report = run_parallel(args.file, args.max_n, args.model_name, args.model_dir, args.device, args.out)
    return 0 if any(p.get("ok") for p in report["phases"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
