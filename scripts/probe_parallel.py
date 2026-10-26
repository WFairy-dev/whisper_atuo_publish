#!/usr/bin/env python3
"""探测这张显卡同时跑 1 路和 2 路语音转写时的显存与耗时。

跑之前要先停掉语音服务，否则服务自己占着的模型会算进显存。
本脚本只测量，不改服务。生产环境仍然只开 1 路。

示例：
  python scripts/probe_parallel.py --file test_video/xxx.mp4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def _gpu_query() -> tuple[str, float, float] | None:
    """返回（显卡名，已用 MiB，总 MiB）。"""
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
    line = out.strip().splitlines()[0]
    name, used_s, total_s = [part.strip() for part in line.split(",")]
    return name, float(used_s), float(total_s)


def _asr_server_running() -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        if "python -m asr_server" in line or "asr_server.app" in line:
            return True
    return False


def _worker(
    path: str,
    model_name: str,
    model_dir: str,
    device: str,
    ready: mp.Queue,
    start: mp.synchronize.Event,
    out: mp.Queue,
) -> None:
    try:
        import whisper

        t0 = time.time()
        model = whisper.load_model(model_name, device=device, download_root=model_dir)
        load_s = time.time() - t0
        ready.put({"ok": True, "load_s": load_s})
        if not start.wait(timeout=600):
            out.put({"ok": False, "error": "等待开始超时", "load_s": load_s})
            return
        t1 = time.time()
        model.transcribe(path, verbose=None)
        out.put({"ok": True, "load_s": load_s, "infer_s": time.time() - t1})
    except Exception as e:
        try:
            ready.put({"ok": False, "error": str(e)})
        except Exception:
            pass
        try:
            out.put({"ok": False, "error": f"{e}\n{traceback.format_exc()}"})
        except Exception:
            pass


class _Sampler:
    def __init__(self) -> None:
        self.peak = 0.0
        self.peak_infer = 0.0
        self.inferring = False
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.peak = 0.0
            self.peak_infer = 0.0
            self.inferring = False

    def snapshot(self) -> tuple[float, float]:
        with self._lock:
            return self.peak, self.peak_infer

    def run(self) -> None:
        while not self._stop.is_set():
            info = _gpu_query()
            if info is not None:
                used = info[1]
                with self._lock:
                    self.peak = max(self.peak, used)
                    if self.inferring:
                        self.peak_infer = max(self.peak_infer, used)
            self._stop.wait(0.3)

    def stop(self) -> None:
        self._stop.set()


def _wait_settle(baseline_used: float, timeout: float = 90.0) -> float:
    """等显存回到接近起始占用，避免上一轮的模型还占着卡。"""
    deadline = time.time() + timeout
    last = baseline_used
    while time.time() < deadline:
        info = _gpu_query()
        if info is None:
            return last
        last = info[1]
        if last <= baseline_used + 800:
            return last
        time.sleep(1)
    return last


def _run_phase(
    ctx: mp.context.BaseContext,
    n: int,
    path: Path,
    model_name: str,
    model_dir: str,
    device: str,
    sampler: _Sampler,
) -> dict:
    sampler.reset()
    ready_q: mp.Queue = ctx.Queue()
    out_q: mp.Queue = ctx.Queue()
    start = ctx.Event()
    procs = []
    for _ in range(n):
        proc = ctx.Process(
            target=_worker,
            args=(str(path), model_name, model_dir, device, ready_q, start, out_q),
        )
        proc.start()
        procs.append(proc)

    loads = []
    deadline = time.time() + 180
    while len(loads) < n and time.time() < deadline:
        try:
            item = ready_q.get(timeout=1)
        except Exception:
            if any(not p.is_alive() for p in procs) and ready_q.empty():
                break
            continue
        loads.append(item)

    failed_load = [x for x in loads if not x.get("ok")]
    if len(loads) < n or failed_load:
        start.set()
        for proc in procs:
            if proc.is_alive():
                proc.kill()
            proc.join(timeout=10)
        err = failed_load[0]["error"] if failed_load else "模型未能在 180 秒内加载完成"
        return {"n": n, "ok": False, "error": err, "loads": loads}

    sampler.inferring = True
    t0 = time.time()
    start.set()
    results = []
    deadline = time.time() + 600
    while len(results) < n and time.time() < deadline:
        try:
            results.append(out_q.get(timeout=1))
        except Exception:
            if all(not p.is_alive() for p in procs):
                break
    wall = time.time() - t0
    sampler.inferring = False
    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10)
    peak, peak_infer = sampler.snapshot()
    ok_results = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]
    return {
        "n": n,
        "ok": len(ok_results) == n and not bad,
        "error": bad[0].get("error") if bad else (None if len(ok_results) == n else "有进程没有返回结果"),
        "wall_s": wall,
        "infer_s": [r.get("infer_s") for r in ok_results],
        "load_s": [r.get("load_s") for r in ok_results],
        "peak_mib": peak,
        "peak_infer_mib": peak_infer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="探测 1 路与 2 路语音转写的显存和耗时")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--model-name", default="turbo")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/models/whisper")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"文件不存在: {args.file}", file=sys.stderr)
        return 1
    if _asr_server_running():
        print("语音服务还在运行。请先停掉它再探测，否则显存会被服务占住。", file=sys.stderr)
        return 2

    info = _gpu_query()
    if info is None:
        print("读不到显卡信息", file=sys.stderr)
        return 1
    name, used, total = info
    print(f"显卡: {name}")
    print(f"起始显存: 已用 {used:.0f} MiB / 总共 {total:.0f} MiB")

    sampler = _Sampler()
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    ctx = mp.get_context("spawn")
    try:
        print("\n=== 1 路 ===")
        one = _run_phase(ctx, 1, args.file, args.model_name, args.model_dir, args.device, sampler)
        _print_phase(one)
        settled = _wait_settle(used)
        print(f"显存回落: {settled:.0f} MiB")
        print("\n=== 2 路 ===")
        two = _run_phase(ctx, 2, args.file, args.model_name, args.model_dir, args.device, sampler)
        _print_phase(two)
    finally:
        sampler.stop()
        thread.join(timeout=2)

    print("\n=== 结论 ===")
    conclusion = _conclude(one, two, total)
    print(conclusion)
    return 0 if one.get("ok") else 3


def _print_phase(row: dict) -> None:
    if not row.get("ok"):
        print(f"  失败: {row.get('error')}")
        return
    infers = row.get("infer_s") or []
    loads = row.get("load_s") or []
    infer_txt = ", ".join(f"{x:.2f}s" for x in infers)
    load_txt = ", ".join(f"{x:.2f}s" for x in loads)
    print(f"  加载: {load_txt}")
    print(f"  推理: {infer_txt}")
    print(f"  一起算完的墙钟: {row.get('wall_s'):.2f}s")
    print(f"  峰值显存: {row.get('peak_mib'):.0f} MiB（推理阶段 {row.get('peak_infer_mib'):.0f} MiB）")


def _conclude(one: dict, two: dict, total: float) -> str:
    if not one.get("ok"):
        return "单路都没跑通，不能判断并行能力。生产维持 1 路。"
    if not two.get("ok"):
        return (
            "双路没有跑通（多半是显存不够或进程退出）。"
            "这张卡不适合同时算 2 路。生产维持 1 路。"
        )
    infer_one = (one.get("infer_s") or [None])[0]
    wall_two = two.get("wall_s") or 0
    peak_two = max(two.get("peak_mib") or 0, two.get("peak_infer_mib") or 0)
    full = peak_two >= total * 0.9
    faster = infer_one and wall_two < infer_one * 2 * 0.85
    parts = [
        f"单路推理 {infer_one:.2f}s，峰值显存 {one.get('peak_infer_mib'):.0f} MiB。",
        f"双路一起算完 {wall_two:.2f}s，峰值显存 {peak_two:.0f} MiB / {total:.0f} MiB。",
    ]
    if full:
        parts.append("双路峰值接近显存上限。生产维持 1 路。")
    elif not faster:
        parts.append("双路总耗时接近两条单路之和，没有明显加速。生产维持 1 路。")
    else:
        parts.append("双路比排队算两条更快，但生产仍维持 1 路，把余量留给稳定性。")
    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
