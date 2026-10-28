"""把识别位置切成单个字符的区间。顺序与这一行文字一致，空格也占一个字符。"""
from __future__ import annotations

from typing import Any, Optional, Sequence


def axis_position(position: float, size: float) -> float:
    """把落在 0～size 上的位置，收成图像坐标 0～size-1。"""
    if size <= 1:
        return 0.0
    clipped = min(max(float(position), 0.0), float(size))
    return clipped / float(size) * (size - 1)


def char_spans(
    text: str,
    word_info: Any,
    width: float,
) -> Optional[list[tuple[str, int, float, float]]]:
    """返回 (字符, 在行文本中的下标, 起点, 终点)。

    行文本为空时返回空列表。有文字但对不齐每个字符时返回 None。
    """
    if not text:
        return []
    if width <= 0:
        return None
    located = _locate(text, word_info)
    if located is None:
        return None
    columns, col_num = located
    bounds = _bounds(columns, col_num, float(width))
    if bounds is None:
        return None
    return [
        (text[index], index, start, end)
        for index, (start, end) in enumerate(bounds)
    ]


def rect_boxes(
    width: float,
    height: float,
    spans: Sequence[tuple[str, int, float, float]],
) -> list[dict[str, Any]]:
    """把区间画成水平矩形。用于已经是文字行的图片。"""
    bottom = float(max(height - 1, 0))
    boxes = []
    for text, index, start, end in spans:
        left = round(axis_position(start, width), 2)
        right = round(axis_position(end, width), 2)
        boxes.append({
            "text": text,
            "index": index,
            "box": [
                [left, 0.0],
                [right, 0.0],
                [right, round(bottom, 2)],
                [left, round(bottom, 2)],
            ],
        })
    return boxes


def _locate(text: str, word_info: Any) -> Optional[tuple[list[int], float]]:
    if not isinstance(word_info, (list, tuple)) or len(word_info) < 4:
        return None
    try:
        col_num = float(word_info[0])
    except (TypeError, ValueError):
        return None
    if col_num <= 0:
        return None
    word_list, col_list = word_info[1], word_info[2]
    if not isinstance(word_list, (list, tuple)) or not isinstance(col_list, (list, tuple)):
        return None
    columns: list[int] = []
    pieces: list[str] = []
    for word, cols in zip(word_list, col_list):
        chars = _chars(word)
        got = _columns(cols)
        if chars is None or got is None or len(chars) != len(got):
            return None
        pieces.extend(chars)
        columns.extend(got)
    if "".join(pieces) != text or len(columns) != len(text):
        return None
    previous = -1
    for col in columns:
        if col < 0 or col <= previous:
            return None
        previous = col
    return columns, col_num


def _bounds(columns: list[int], col_num: float, width: float) -> Optional[list[tuple[float, float]]]:
    cell = width / col_num
    centers = [(col + 0.5) * cell for col in columns]
    if len(centers) == 1:
        half = cell / 2
        start = max(centers[0] - half, 0.0)
        end = min(centers[0] + half, width)
        if end <= start:
            return None
        return [(start, end)]
    bounds: list[tuple[float, float]] = []
    for index, center in enumerate(centers):
        if index == 0:
            start = center - (centers[1] - center) / 2
        else:
            start = (centers[index - 1] + center) / 2
        if index == len(centers) - 1:
            end = center + (center - centers[index - 1]) / 2
        else:
            end = (center + centers[index + 1]) / 2
        start = max(start, 0.0)
        end = min(end, width)
        if end <= start:
            return None
        bounds.append((start, end))
    return bounds


def _columns(cols: Any) -> Optional[list[int]]:
    if isinstance(cols, bool):
        return None
    if isinstance(cols, (int, float)):
        return [int(cols)]
    if not isinstance(cols, (list, tuple)) or not cols:
        return None
    got = []
    for col in cols:
        if isinstance(col, bool) or not isinstance(col, (int, float)):
            return None
        got.append(int(col))
    return got


def _chars(word: Any) -> Optional[list[str]]:
    if isinstance(word, str):
        chars = list(word)
    elif isinstance(word, (list, tuple)):
        chars = []
        for item in word:
            if not isinstance(item, str):
                return None
            chars.extend(list(item))
    else:
        return None
    if not chars or any(len(char) != 1 for char in chars):
        return None
    return chars
