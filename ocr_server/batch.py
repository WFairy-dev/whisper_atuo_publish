"""把一组图片切成批，并在某一张失败时保留其余图片的结果。

这里不加载模型。检测和识别由调用方传入，方便无显卡单测。
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence


class RecognitionCancelled(Exception):
    """调用方要求停在下一批之前。已经送进模型的这一小批会算完。"""


def _stopped(stop: Optional[Callable[[], bool]]) -> bool:
    return bool(stop and stop())


def chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    if size < 1:
        raise ValueError("批大小至少为 1")
    return [list(items[index:index + size]) for index in range(0, len(items), size)]


def call_in_batches(
    fn: Callable[[list[Any]], list[Any]],
    items: Sequence[Any],
    batch_size: int,
    stop: Optional[Callable[[], bool]] = None,
) -> list[tuple[bool, Any]]:
    """按批调用。整批失败时改成逐条，避免一张坏图丢掉同批的其他图片。"""
    results: list[tuple[bool, Any]] = []
    for batch in chunked(items, batch_size):
        if _stopped(stop):
            raise RecognitionCancelled()
        try:
            got = fn(batch)
        except RecognitionCancelled:
            raise
        except Exception as batch_error:
            if len(batch) == 1:
                results.append((False, str(batch_error)))
                continue
            for one in batch:
                if _stopped(stop):
                    raise RecognitionCancelled()
                try:
                    single = fn([one])
                except RecognitionCancelled:
                    raise
                except Exception as one_error:
                    results.append((False, str(one_error)))
                    continue
                if len(single) != 1:
                    results.append((False, "单张结果数量不符"))
                    continue
                results.append((True, single[0]))
            continue
        if _stopped(stop):
            raise RecognitionCancelled()
        if len(got) != len(batch):
            results.extend((False, "批结果数量不符") for _ in batch)
            continue
        results.extend((True, item) for item in got)
    return results


def _points(box: Any) -> list[list[float]] | None:
    if box is None:
        return None
    raw = box.get("points") if isinstance(box, dict) else box
    if raw is None:
        return None
    return [[float(point[0]), float(point[1])] for point in raw]


def _det_score(box: Any) -> float | None:
    if isinstance(box, dict) and box.get("det_score") is not None:
        return float(box["det_score"])
    return None


def _box_width(spec: dict[str, Any]) -> float:
    box = spec.get("box")
    raw = box.get("points") if isinstance(box, dict) else box
    if not raw:
        return 0.0
    xs = [float(point[0]) for point in raw]
    return max(xs) - min(xs)


def _width_groups(specs: list[dict[str, Any]], rec_batch: int, ratio: float = 1.01) -> list[list[int]]:
    """宽度接近的文字行才进同一批，同一批最多 rec_batch 条。"""
    ordered = sorted(range(len(specs)), key=lambda index: (_box_width(specs[index]), index))
    groups: list[list[int]] = []
    current: list[int] = []
    base = 0.0
    for index in ordered:
        width = _box_width(specs[index])
        too_wide = current and width > base * ratio
        if current and (len(current) >= rec_batch or too_wide):
            groups.append(current)
            current = []
        if not current:
            base = max(width, 1.0)
        current.append(index)
    if current:
        groups.append(current)
    return groups


def _recognize_by_width(
    specs: list[dict[str, Any]],
    rec_batch: int,
    recognize: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    stop: Optional[Callable[[], bool]] = None,
) -> list[tuple[bool, Any]]:
    """按文字行宽度分批，避免长短行补齐后把文字认错。结果仍按原顺序返回。"""
    paired: list[tuple[bool, Any] | None] = [None] * len(specs)
    for indexes in _width_groups(specs, rec_batch):
        if _stopped(stop):
            raise RecognitionCancelled()
        group = [specs[index] for index in indexes]
        got = call_in_batches(recognize, group, rec_batch, stop)
        for index, item in zip(indexes, got):
            paired[index] = item
    return [item if item is not None else (False, "识别失败") for item in paired]


def recognize_images(
    images: Sequence[Any],
    *,
    mode: str,
    det_batch: int,
    rec_batch: int,
    detect: Callable[[list[Any]], list[Any]],
    recognize: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    stop: Optional[Callable[[], bool]] = None,
) -> list[dict[str, Any]]:
    """按提交顺序返回每张图的文字和文字框。mode 为 det_rec 或 rec_only。"""
    if _stopped(stop):
        raise RecognitionCancelled()
    if mode not in ("det_rec", "rec_only"):
        raise ValueError("mode 只能是 det_rec 或 rec_only")
    items = [
        {"index": index, "ok": True, "text": "", "boxes": [], "error": None}
        for index in range(len(images))
    ]
    specs: list[dict[str, Any]] = []
    if mode == "rec_only":
        specs = [
            {"image_index": index, "box": None, "image": image}
            for index, image in enumerate(images)
        ]
    else:
        detected = call_in_batches(detect, images, det_batch, stop)
        for index, (ok, value) in enumerate(detected):
            if not ok:
                items[index]["ok"] = False
                items[index]["error"] = str(value)
                continue
            for box in value:
                specs.append({"image_index": index, "box": box, "image": images[index]})

    if not specs:
        return items

    if _stopped(stop):
        raise RecognitionCancelled()
    recognized = _recognize_by_width(specs, rec_batch, recognize, stop)
    failed: set[int] = set()
    errors: dict[int, str] = {}
    for spec, (ok, value) in zip(specs, recognized):
        index = spec["image_index"]
        if not ok:
            failed.add(index)
            errors.setdefault(index, str(value))
            continue
        entry = {
            "text": str(value.get("text") or ""),
            "score": float(value.get("score") or 0.0),
            "box": _points(spec["box"]),
        }
        det_score = _det_score(spec["box"])
        if det_score is not None:
            entry["det_score"] = det_score
        items[index]["boxes"].append(entry)

    for index in failed:
        if items[index]["boxes"]:
            continue
        items[index]["ok"] = False
        items[index]["error"] = errors.get(index) or "识别失败"
    for item in items:
        if item["ok"]:
            item["text"] = "\n".join(box["text"] for box in item["boxes"] if box["text"])
            item["error"] = None
    return items
