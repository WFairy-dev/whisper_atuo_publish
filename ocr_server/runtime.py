"""加载官方静态图推理模型，并把检测、识别结果转成普通字典。"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from . import config
from .crop import crop_quad

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


def recognize_crops(model: Any, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    crops = []
    for spec in specs:
        box = spec.get("box")
        if box is None:
            crops.append(spec["image"])
            continue
        points = box.get("points") if isinstance(box, dict) else box
        crops.append(crop_quad(spec["image"], points))
    outputs = list(model.predict(crops, batch_size=len(crops)))
    if len(outputs) != len(specs):
        raise RuntimeError(f"识别结果数量不符: 期望 {len(specs)} 实际 {len(outputs)}")
    results = []
    for output in outputs:
        payload = _payload(output)
        results.append({
            "text": str(payload.get("rec_text") or ""),
            "score": round(float(payload.get("rec_score") or 0.0), 6),
        })
    return results
