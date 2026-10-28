#!/usr/bin/env python3
"""测量 PP-OCRv6_small 在不同批大小下的耗时和显存。

只用文字识别环境运行。先单独测，再在语音转写进行中复测选定批大小。

示例：
  /root/autodl-tmp/envs/ocr-pp/bin/python scripts/probe_ocr.py --phase alone --image sample.png
  /root/autodl-tmp/envs/ocr-pp/bin/python scripts/probe_ocr.py --phase with-asr --image sample.png
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DET_BATCHES = (1, 4, 8, 16)
REC_BATCHES = (1, 16, 32, 64)
MEMORY_LIMIT_MIB = 18 * 1024


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


def _load_images(path: Path, count: int):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"无法读取图片: {path}")
    return [image.copy() for _ in range(count)]


def _run_once(detector, recognizer, images, det_batch: int, rec_batch: int, sampler: _Sampler) -> dict:
    from ocr_server.batch import recognize_images
    from ocr_server.runtime import detect_images, recognize_crops

    sampler.reset()
    started = time.perf_counter()
    items = recognize_images(
        images,
        mode="det_rec",
        det_batch=det_batch,
        rec_batch=rec_batch,
        detect=lambda batch: detect_images(detector, batch),
        recognize=lambda specs: recognize_crops(recognizer, specs),
    )
    elapsed = time.perf_counter() - started
    failed = sum(1 for item in items if not item["ok"])
    return {
        "images": len(images),
        "det_batch": det_batch,
        "rec_batch": rec_batch,
        "elapsed_s": round(elapsed, 3),
        "peak_mib": round(sampler.peak_mib(), 1),
        "failed": failed,
        "texts": [item.get("text") for item in items],
        "boxes": [item.get("boxes") for item in items],
    }


def _keep_asr_busy(url: str, video: Path, api_key: str, stop: threading.Event) -> None:
    """连续提交语音，让文字识别测量时显卡上确实有转写在跑。"""
    import httpx

    while not stop.is_set():
        try:
            with video.open("rb") as handle:
                httpx.post(
                    f"{url.rstrip('/')}/v1/asr/sync",
                    data={
                        "task_id": f"ocr_probe_{time.time_ns()}",
                        "source_file_path": str(video),
                    },
                    files={"file": (video.name, handle)},
                    headers={"X-API-Key": api_key} if api_key else {},
                    timeout=120,
                )
        except Exception:
            time.sleep(0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description="探测文字识别批大小的耗时和显存")
    parser.add_argument("--phase", choices=["alone", "with-asr"], default="alone")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--counts", default="32,100", help="逗号分隔的图片张数")
    parser.add_argument("--det-batch", type=int, default=0, help="只测这一个检测批大小；0 表示扫全部")
    parser.add_argument("--rec-batch", type=int, default=0, help="只测这一个识别批大小；0 表示扫全部")
    parser.add_argument("--asr-url", default="http://127.0.0.1:6006")
    parser.add_argument("--asr-file", type=Path, default=None)
    parser.add_argument("--asr-api-key", default="")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    asr_on = _process_running("python -m asr_server")
    if args.phase == "alone" and asr_on:
        print("语音服务还在运行。单独测文字识别前请先停掉它，避免显存叠在一起。", file=sys.stderr)
        return 2
    if args.phase == "with-asr" and not asr_on:
        print("语音服务没有运行。请先按 2 路启动语音服务再复测。", file=sys.stderr)
        return 2

    info = _gpu_query()
    if info is None:
        print("读不到显卡占用。", file=sys.stderr)
        return 2
    print(f"显卡: {info[0]}  起始已用 {info[1]:.0f} MiB / {info[2]:.0f} MiB")

    from ocr_server.runtime import load_models

    detector, recognizer = load_models()
    sampler = _Sampler()
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    stop_asr = threading.Event()
    asr_thread = None
    if args.phase == "with-asr":
        if args.asr_file is None or not args.asr_file.is_file():
            print("同时测量需要 --asr-file 指向一条语音样片。", file=sys.stderr)
            return 2
        asr_thread = threading.Thread(
            target=_keep_asr_busy,
            args=(args.asr_url, args.asr_file, args.asr_api_key, stop_asr),
            daemon=True,
        )
        asr_thread.start()
        time.sleep(1.0)

    det_values = (args.det_batch,) if args.det_batch else DET_BATCHES
    rec_values = (args.rec_batch,) if args.rec_batch else REC_BATCHES
    counts = [int(part) for part in args.counts.split(",") if part.strip()]
    rows = []
    try:
        for count in counts:
            images = _load_images(args.image, count)
            for det_batch in det_values:
                for rec_batch in rec_values:
                    row = _run_once(detector, recognizer, images, det_batch, rec_batch, sampler)
                    public = {key: value for key, value in row.items() if key not in ("texts", "boxes")}
                    print(
                        f"n={public['images']} det={public['det_batch']} rec={public['rec_batch']} "
                        f"耗时 {public['elapsed_s']}s 峰值 {public['peak_mib']} MiB 失败 {public['failed']}"
                    )
                    rows.append(public)
    finally:
        stop_asr.set()
        sampler.stop()
        thread.join(timeout=2)

    baseline = next((row for row in rows if row["images"] == max(counts) and row["det_batch"] == 1 and row["rec_batch"] == 1), None)
    chosen = next((row for row in rows if row["images"] == max(counts) and row["det_batch"] == 8 and row["rec_batch"] == 32), None)
    if baseline and chosen and baseline["elapsed_s"] > 0:
        print(
            f"一百张量级：逐张约 {baseline['elapsed_s']}s，检测 8 / 识别 32 约 {chosen['elapsed_s']}s，"
            f"峰值 {chosen['peak_mib']} MiB，上限 {MEMORY_LIMIT_MIB} MiB。"
        )
    payload = {"phase": args.phase, "gpu": {"name": info[0], "total_mib": info[2]}, "rows": rows}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
