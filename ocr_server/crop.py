"""把检测框从原图上裁成文字行，并把字符区间映回原图。"""
from __future__ import annotations

from typing import Any, Sequence

from .chars import axis_position


class LineCrop:
    """模型看到的文字行，以及把它映回原图所需的几何。"""

    def __init__(self, image: Any, warp_width: int, warp_height: int, inverse: Any, rotated: bool) -> None:
        self.image = image
        self.warp_width = warp_width
        self.warp_height = warp_height
        self.inverse = inverse
        self.rotated = rotated


def crop_quad(image: Any, points: Any) -> Any:
    return prepare_line(image, points).image


def prepare_line(image: Any, points: Any) -> LineCrop:
    import cv2
    import numpy as np

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 4:
        raise ValueError("文字框点数不足")
    if len(pts) != 4:
        rect = cv2.minAreaRect(pts)
        pts = cv2.boxPoints(rect)
    pts = _order_points(pts)
    width = int(round(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))))
    height = int(round(max(np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2]))))
    if width < 2 or height < 2:
        raise ValueError("文字框过小")
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, destination)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    rotated = bool(crop.shape[0] >= crop.shape[1] * 1.5)
    view = np.rot90(crop) if rotated else crop
    inverse = cv2.getPerspectiveTransform(destination, pts)
    return LineCrop(view, width, height, inverse, rotated)


def project_spans(
    line: LineCrop,
    spans: Sequence[tuple[str, int, float, float]],
) -> list[dict[str, Any]]:
    """字符区间先落在模型看到的文字行上，再映回上传图片的像素坐标。"""
    import cv2
    import numpy as np

    model_h, model_w = line.image.shape[:2]
    boxes = []
    for text, index, start, end in spans:
        corners = _model_corners(start, end, model_w, model_h)
        if line.rotated:
            corners = [_unrotate(x, y, line.warp_width) for x, y in corners]
        mapped = cv2.perspectiveTransform(
            np.asarray([corners], dtype=np.float32),
            line.inverse,
        )[0]
        ordered = _order_points(mapped)
        boxes.append({
            "text": text,
            "index": index,
            "box": [[round(float(x), 2), round(float(y), 2)] for x, y in ordered],
        })
    return boxes


def _model_corners(start: float, end: float, width: float, height: float) -> list[tuple[float, float]]:
    left = axis_position(start, width)
    right = axis_position(end, width)
    bottom = float(max(height - 1, 0))
    return [(left, 0.0), (right, 0.0), (right, bottom), (left, bottom)]


def _unrotate(x: float, y: float, warp_width: int) -> tuple[float, float]:
    """把逆时针转正后的坐标，还原到裁剪矩形。"""
    return (warp_width - 1) - y, x


def _order_points(pts: Any) -> Any:
    import numpy as np

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered
