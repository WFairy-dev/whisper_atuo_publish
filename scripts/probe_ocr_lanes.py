#!/usr/bin/env python3
"""比较一份文字识别模型和两份同时算时的耗时、显存。

每一份模型内部仍按检测 4、识别 16 分批。一组 32 张，由同一帧复制而来。
先让一份模型把两组排队做完，再让两份模型各做一组并且同时开始。
语音服务可以开着；最后一轮会在语音正在转写时再测一次两份同时算。

跑之前请先停掉文字识别服务，避免服务自己的那一份模型混进测量。

示例：
  /root/autodl-tmp/envs/ocr-pp/bin/python scripts/probe_ocr_lanes.py \\
    --video test_video/样片.mp4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MEMORY_LIMIT_MIB = 18 * 1024
DET_BATCH = 4
REC_BATCH = 16
GROUP = 32


def _gpu_query() -> tuple[str, float, float] | None:
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
    name, used_s, total_s = [part.strip() for part in out.strip().splitlines()[0].split(",")]
    return name, float(used_s), float(total_s)


def _process_running(snippet: str) -> bool:
    try:
        out = subprocess.check_output(["ps", "-eo", "args"], text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(snippet in line for line in out.splitlines())


def _asr_running() -> int:
    try:
        with urllib.request.urlopen("http://127.0.0.1:6006/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0
    return int(data.get("running") or 0)


class _Sampler:
    def __init__(self) -> None:
        self.peak = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.peak = 0.0

    def peak_mib(self) -> float:
        with self._lock:
            return self.peak

    def run(self) -> None:
        while not self._stop.is_set():
            info = _gpu_query()
            if info is not None:
                with self._lock:
                    self.peak = max(self.peak, info[1])
            self._stop.wait(0.2)

    def stop(self) -> None:
        self._stop.set()


def _extract_frames(video: Path, dest: Path, count: int = 5) -> list[Path]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 1:
        total = 1
    paths: list[Path] = []
    dest.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        pos = int(total * (index + 1) / (count + 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        path = dest / f"frame_{index}.png"
        if not cv2.imwrite(str(path), frame):
            continue
        paths.append(path)
    cap.release()
    if not paths:
        raise SystemExit(f"没有从视频抽出可用画面: {video}")
    return paths


def _load_copies(path: str, count: int):
    import cv2

    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"无法读取图片: {path}")
    return [image.copy() for _ in range(count)], image.shape


def _summarize(items: list[dict]) -> dict:
    boxes = 0
    chars = 0
    failed = 0
    for item in items:
        if not item.get("ok"):
            failed += 1
        for box in item.get("boxes") or []:
            boxes += 1
            chars += len(box.get("chars") or [])
    text = ""
    if items:
        text = str(items[0].get("text") or "")[:80]
    return {"failed": failed, "boxes": boxes, "chars": chars, "text": text}


def _recognize(detector, recognizer, images: list) -> list[dict]:
    from ocr_server.batch import recognize_images
    from ocr_server.runtime import detect_images, recognize_crops

    return recognize_images(
        images,
        mode="det_rec",
        det_batch=DET_BATCH,
        rec_batch=REC_BATCH,
        detect=lambda batch: detect_images(detector, batch),
        recognize=lambda specs: recognize_crops(recognizer, specs),
    )


def _picker(paths: list[str], ready: mp.Queue, out: mp.Queue) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from ocr_server.runtime import load_models

        detector, recognizer = load_models()
        best = None
        rows = []
        for path in paths:
            images, shape = _load_copies(path, 1)
            items = _recognize(detector, recognizer, images)
            summary = _summarize(items)
            summary["path"] = path
            summary["shape"] = list(shape)
            rows.append(summary)
            score = (summary["chars"], summary["boxes"])
            if best is None or score > best[0]:
                best = (score, path)
        ready.put({"ok": True})
        out.put({"ok": True, "best": best[1] if best else paths[0], "rows": rows})
    except Exception as exc:
        payload = {"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}
        try:
            ready.put(payload)
        except Exception:
            pass
        try:
            out.put(payload)
        except Exception:
            pass


def _worker(
    image_path: str,
    count: int,
    jobs: int,
    ready: mp.Queue,
    cmd: mp.Queue,
    out: mp.Queue,
) -> None:
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from ocr_server.runtime import load_models

        t0 = time.perf_counter()
        detector, recognizer = load_models()
        load_s = time.perf_counter() - t0
        images, shape = _load_copies(image_path, count)
        _recognize(detector, recognizer, images[:2])
        ready.put({"ok": True, "load_s": round(load_s, 3), "shape": list(shape)})
        while True:
            action = cmd.get()
            if action == "stop":
                break
            started = time.perf_counter()
            parts = []
            summary = {}
            for _ in range(jobs):
                one = time.perf_counter()
                items = _recognize(detector, recognizer, images)
                parts.append(round(time.perf_counter() - one, 3))
                summary = _summarize(items)
            out.put({
                "ok": True,
                "elapsed_s": parts,
                "wall_s": round(time.perf_counter() - started, 3),
                **summary,
            })
    except Exception as exc:
        payload = {"ok": False, "error": f"{exc}\n{traceback.format_exc()}"}
        try:
            ready.put(payload)
        except Exception:
            pass
        try:
            out.put(payload)
        except Exception:
            pass


def _stop_procs(procs: list) -> None:
    for proc in procs:
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=10)


def _collect(queue: mp.Queue, expect: int, procs: list, timeout: float) -> list[dict]:
    rows: list[dict] = []
    deadline = time.time() + timeout
    while len(rows) < expect and time.time() < deadline:
        try:
            rows.append(queue.get(timeout=1))
        except Exception:
            if all(not proc.is_alive() for proc in procs) and queue.empty():
                break
    return rows


def _run_phase(
    ctx: mp.context.BaseContext,
    lanes: int,
    image: Path,
    jobs: int,
    rounds: int,
    sampler: _Sampler,
) -> dict:
    sampler.reset()
    ready_q: mp.Queue = ctx.Queue()
    cmd_q: mp.Queue = ctx.Queue()
    out_q: mp.Queue = ctx.Queue()
    procs = []
    for _ in range(lanes):
        proc = ctx.Process(
            target=_worker,
            args=(str(image), GROUP, jobs, ready_q, cmd_q, out_q),
        )
        proc.start()
        procs.append(proc)
    loads = _collect(ready_q, lanes, procs, 180)
    failed = [item for item in loads if not item.get("ok")]
    if len(loads) < lanes or failed:
        _stop_procs(procs)
        err = failed[0].get("error") if failed else "模型未能在 180 秒内加载完成"
        return {"lanes": lanes, "ok": False, "error": err}
    round_rows = []
    try:
        for _ in range(rounds):
            sampler.reset()
            t0 = time.perf_counter()
            for _lane in range(lanes):
                cmd_q.put("go")
            results = _collect(out_q, lanes, procs, 180)
            wall = time.perf_counter() - t0
            bad = [item for item in results if not item.get("ok")]
            if len(results) < lanes or bad:
                err = bad[0].get("error") if bad else "有一路没有返回结果"
                return {"lanes": lanes, "ok": False, "error": err, "loads": loads}
            round_rows.append({
                "wall_s": round(wall, 3),
                "peak_mib": round(sampler.peak_mib(), 1),
                "lanes": results,
            })
    finally:
        for _lane in range(lanes):
            try:
                cmd_q.put("stop")
            except Exception:
                pass
        for proc in procs:
            proc.join(timeout=30)
        _stop_procs(procs)
    return {"lanes": lanes, "jobs": jobs, "ok": True, "load_s": [item.get("load_s") for item in loads], "rounds": round_rows}


def _keep_asr_busy(video: Path, stop: threading.Event) -> None:
    import httpx

    while not stop.is_set():
        try:
            with video.open("rb") as handle:
                httpx.post(
                    "http://127.0.0.1:6006/v1/asr/sync",
                    data={
                        "task_id": f"ocr_lane_{time.time_ns()}",
                        "source_file_path": str(video),
                    },
                    files={"file": (video.name, handle)},
                    timeout=180,
                )
        except Exception:
            time.sleep(0.5)


def _wait_settle(target: float, timeout: float = 60.0) -> float:
    """等显存回到接近起始占用，避免上一轮的模型还占着卡。"""
    deadline = time.time() + timeout
    last = target
    while time.time() < deadline:
        info = _gpu_query()
        if info is None:
            return last
        last = info[1]
        if last <= target + 800:
            return last
        time.sleep(1)
    return last


def _print_phase(title: str, row: dict) -> None:
    print(f"\n{title}")
    if not row.get("ok"):
        print(f"  失败: {row.get('error')}")
        return
    print(f"  加载耗时（秒）: {row.get('load_s')}")
    for index, item in enumerate(row["rounds"], start=1):
        parts = []
        for lane in item["lanes"]:
            parts.append(
                f"计算 {lane.get('elapsed_s')} 秒，文字框 {lane.get('boxes')}，失败 {lane.get('failed')}"
            )
        print(
            f"  第 {index} 轮 墙钟 {item['wall_s']} 秒，峰值 {item['peak_mib']} MiB；"
            + "；".join(parts)
        )


def _pick_frame(ctx: mp.context.BaseContext, frames: list[Path]) -> tuple[Path, list[dict]]:
    ready_q: mp.Queue = ctx.Queue()
    out_q: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_picker, args=([str(path) for path in frames], ready_q, out_q))
    proc.start()
    loads = _collect(ready_q, 1, [proc], 180)
    results = _collect(out_q, 1, [proc], 180)
    proc.join(timeout=30)
    _stop_procs([proc])
    if not results or not results[0].get("ok"):
        err = ""
        if results:
            err = results[0].get("error") or ""
        elif loads:
            err = loads[0].get("error") or ""
        raise SystemExit(err or "选帧失败")
    return Path(results[0]["best"]), results[0]["rows"]


def main() -> int:
    parser = argparse.ArgumentParser(description="探测一份与两份文字识别模型同时计算的耗时和显存")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--with-asr", action="store_true")
    parser.add_argument("--frame-dir", type=Path, default=Path("/tmp/ocr_lane_probe"))
    args = parser.parse_args()
    if not args.video.is_file():
        print(f"找不到视频: {args.video}", file=sys.stderr)
        return 2
    if _process_running("python -m ocr_server"):
        print("文字识别服务还在运行。请先停掉它，否则显存里会多一份模型。", file=sys.stderr)
        return 2

    info = _gpu_query()
    if info is None:
        print("读不到显卡占用。", file=sys.stderr)
        return 2
    print(f"显卡: {info[0]}  起始已用 {info[1]:.0f} MiB / {info[2]:.0f} MiB")
    print(f"批大小: 检测 {DET_BATCH}，识别 {REC_BATCH}，每组 {GROUP} 张，重复 {args.rounds} 轮")

    frames = _extract_frames(args.video, args.frame_dir)
    ctx = mp.get_context("spawn")
    sampler = _Sampler()
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    serial: dict | None = None
    parallel: dict | None = None
    busy: dict | None = None
    try:
        image, picked = _pick_frame(ctx, frames)
        settled = _wait_settle(info[1])
        print(f"选帧后显存 {settled:.0f} MiB")
        print(f"选用画面: {image.name}")
        for row in picked:
            print(
                f"  {Path(row['path']).name} 形状 {row['shape']} 文字框 {row['boxes']} "
                f"字符 {row['chars']} 文本 {row['text']!r}"
            )
        serial = _run_phase(ctx, 1, image, jobs=2, rounds=args.rounds, sampler=sampler)
        _print_phase("一份模型，两组排队", serial)
        settled = _wait_settle(info[1])
        print(f"排队测量结束后显存 {settled:.0f} MiB")
        parallel = _run_phase(ctx, 2, image, jobs=1, rounds=args.rounds, sampler=sampler)
        _print_phase("两份模型，两组同时", parallel)
        settled = _wait_settle(info[1])
        print(f"同时测量结束后显存 {settled:.0f} MiB")
        if args.with_asr:
            if not _process_running("python -m asr_server"):
                print("语音服务没有运行，跳过转写中的测量。")
            else:
                stop = threading.Event()
                asr_threads = [
                    threading.Thread(target=_keep_asr_busy, args=(args.video, stop), daemon=True)
                    for _ in range(2)
                ]
                for asr_thread in asr_threads:
                    asr_thread.start()
                deadline = time.time() + 45
                while time.time() < deadline and _asr_running() < 1:
                    time.sleep(0.3)
                print(f"语音正在转写的路数: {_asr_running()}")
                busy = _run_phase(ctx, 2, image, jobs=1, rounds=args.rounds, sampler=sampler)
                stop.set()
                _print_phase("语音转写进行中，两份模型同时", busy)
    finally:
        sampler.stop()
        thread.join(timeout=2)

    def _wall(row: dict | None) -> list[float]:
        if not row or not row.get("ok"):
            return []
        return [item["wall_s"] for item in row["rounds"]]

    serial_walls = _wall(serial)
    parallel_walls = _wall(parallel)
    if serial_walls and parallel_walls:
        base = min(serial_walls)
        fast = min(parallel_walls)
        print(
            f"\n对照: 排队最快 {base:.3f} 秒，两路同时最快 {fast:.3f} 秒，"
            f"墙钟变为原来的 {fast / base:.2f} 倍。上限 {MEMORY_LIMIT_MIB} MiB。"
        )
    if busy and busy.get("ok"):
        peaks = [item["peak_mib"] for item in busy["rounds"]]
        print(f"转写同时进行时峰值 {max(peaks):.0f} MiB，上限 {MEMORY_LIMIT_MIB} MiB。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
