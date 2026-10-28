"""把检测框从原图上裁成文字行。逻辑与常见的四点透视裁剪一致。"""
from __future__ import annotations

from typing import Any


def crop_quad(image: Any, points: Any) -> Any:
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
    if crop.shape[0] >= crop.shape[1] * 1.5:
        crop = np.rot90(crop)
    return crop


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
