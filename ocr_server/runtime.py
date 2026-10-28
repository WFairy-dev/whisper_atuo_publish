"""加载官方静态图推理模型，并把检测、识别结果转成普通字典。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from . import config
from .chars import char_spans, rect_boxes
from .crop import prepare_line, project_spans

log = logging.getLogger("ocr_server.runtime")


def _ready_dir(path: Path) -> Optional[str]:
    """目录里已经有推理配置时才指定路径。空目录会让加载去读一个不存在的文件。"""
    if (path / "inference.yml").is_file():
        return str(path)
    return None


def load_models() -> tuple[Any, Any]:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(config.MODEL_DIR))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    from paddleocr import TextDetection, TextRecognition

    log.info(
        "加载文字识别模型 det=%s rec=%s device=%s engine=%s cache=%s",
        config.DET_MODEL_NAME,
        config.REC_MODEL_NAME,
        config.DEVICE,
        config.ENGINE_NAME,
        os.environ["PADDLE_PDX_CACHE_HOME"],
    )
    detector = TextDetection(
        model_name=config.DET_MODEL_NAME,
        model_dir=_ready_dir(config.DET_MODEL_DIR),
        device=config.DEVICE,
        engine=config.ENGINE_NAME,
    )
    recognizer = TextRecognition(
        model_name=config.REC_MODEL_NAME,
        model_dir=_ready_dir(config.REC_MODEL_DIR),
        device=config.DEVICE,
        engine=config.ENGINE_NAME,
    )
    log.info("文字识别模型已加载 dir=%s", config.MODEL_DIR)
    return detector, recognizer


def _payload(result: Any) -> dict[str, Any]:
    data = result.json if hasattr(result, "json") else result
    if isinstance(data, dict) and isinstance(data.get("res"), dict):
        return data["res"]
    if isinstance(data, dict):
        return data
    raise TypeError("模型结果不是字典")


def _points(poly: Any) -> list[list[float]]:
    import numpy as np

    array = np.asarray(poly, dtype=float).reshape(-1, 2)
    return [[round(float(x), 2), round(float(y), 2)] for x, y in array]


def detect_images(model: Any, images: list[Any]) -> list[list[dict[str, Any]]]:
    outputs = list(model.predict(images, batch_size=len(images)))
    if len(outputs) != len(images):
        raise RuntimeError(f"检测结果数量不符: 期望 {len(images)} 实际 {len(outputs)}")
    boxes_per_image: list[list[dict[str, Any]]] = []
    for output in outputs:
        payload = _payload(output)
        polys = payload.get("dt_polys")
        if polys is None:
            polys = payload.get("polys") or []
        scores = payload.get("dt_scores") or []
        boxes = []
        for index, poly in enumerate(polys):
            box: dict[str, Any] = {"points": _points(poly)}
            if index < len(scores) and scores[index] is not None:
                box["det_score"] = round(float(scores[index]), 6)
            boxes.append(box)
        boxes_per_image.append(boxes)
    return boxes_per_image


def _aspect(image: Any) -> float:
    height, width = image.shape[:2]
    return float(width) / float(height)


def _text_and_info(raw: Any) -> tuple[str, Any]:
    if (
        isinstance(raw, (list, tuple))
        and len(raw) == 2
        and isinstance(raw[1], (list, tuple))
        and len(raw[1]) >= 4
    ):
        return str(raw[0] or ""), raw[1]
    return str(raw or ""), None


def recognize_crops(model: Any, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines: list[Any] = []
    crops = []
    for spec in specs:
        box = spec.get("box")
        if box is None:
            lines.append(None)
            crops.append(spec["image"])
            continue
        points = box.get("points") if isinstance(box, dict) else box
        line = prepare_line(spec["image"], points)
        lines.append(line)
        crops.append(line.image)
    # 模型内部按宽高比给字符位置配比例。先按同样顺序送入，返回后再排回原顺序。
    order = sorted(range(len(crops)), key=lambda index: (_aspect(crops[index]), index))
    ordered = [crops[index] for index in order]
    outputs = list(model.predict(ordered, batch_size=len(ordered), return_word_box=True))
    if len(outputs) != len(specs):
        raise RuntimeError(f"识别结果数量不符: 期望 {len(specs)} 实际 {len(outputs)}")
    restored: list[Any] = [None] * len(specs)
    for index, output in zip(order, outputs):
        restored[index] = output
    results = []
    for line, source, output in zip(lines, crops, restored):
        payload = _payload(output)
        text, info = _text_and_info(payload.get("rec_text"))
        view = line.image if line is not None else source
        height, width = view.shape[:2]
        spans = char_spans(text, info, width)
        if spans is None:
            chars = None
        elif line is not None:
            chars = project_spans(line, spans)
        else:
            chars = rect_boxes(width, height, spans)
        results.append({
            "text": text,
            "score": round(float(payload.get("rec_score") or 0.0), 6),
            "chars": chars,
        })
    return results
