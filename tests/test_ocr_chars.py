"""字符区间，以及映回原图后的四点。"""
from __future__ import annotations

import unittest

import numpy as np

from ocr_server.chars import char_spans, rect_boxes
from ocr_server.crop import prepare_line, project_spans


class CharSpanTests(unittest.TestCase):
    def test_each_chinese_character_has_its_own_span(self) -> None:
        spans = char_spans("你好", [10, [["你", "好"]], [[1, 5]], ["cn"]], 100)
        self.assertEqual([(item[0], item[1]) for item in spans], [("你", 0), ("好", 1)])
        self.assertAlmostEqual(spans[0][2], 0.0)
        self.assertAlmostEqual(spans[0][3], 35.0)
        self.assertAlmostEqual(spans[1][2], 35.0)
        self.assertAlmostEqual(spans[1][3], 75.0)

    def test_english_word_splits_into_letters(self) -> None:
        spans = char_spans(
            "hello",
            [20, [["h", "e", "l", "l", "o"]], [[2, 3, 4, 5, 6]], ["en&num"]],
            100,
        )
        self.assertEqual([item[0] for item in spans], ["h", "e", "l", "l", "o"])
        self.assertEqual([item[1] for item in spans], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(spans[2][2], 20.0)
        self.assertAlmostEqual(spans[3][3], 30.0)

    def test_space_is_its_own_character(self) -> None:
        spans = char_spans(
            "A B",
            [10, [["A"], [" "], ["B"]], [[1], [3], [5]], ["en&num", "symbol", "en&num"]],
            100,
        )
        self.assertEqual([(item[0], item[1]) for item in spans], [("A", 0), (" ", 1), ("B", 2)])
        self.assertLess(spans[0][3], spans[2][2])

    def test_missing_positions_are_unavailable(self) -> None:
        self.assertIsNone(char_spans("第一行", None, 90))

    def test_mismatched_text_is_unavailable(self) -> None:
        self.assertIsNone(char_spans("AB", [10, [["A"]], [[1]], ["en&num"]], 100))

    def test_empty_text_has_no_characters(self) -> None:
        self.assertEqual(char_spans("", [10, [], [], []], 90), [])

    def test_rect_boxes_keep_index_order(self) -> None:
        spans = char_spans("甲乙", [4, [["甲", "乙"]], [[0, 2]], ["cn"]], 90)
        boxes = rect_boxes(90, 20, spans)
        self.assertEqual([item["text"] for item in boxes], ["甲", "乙"])
        self.assertEqual([item["index"] for item in boxes], [0, 1])
        self.assertLessEqual(boxes[0]["box"][1][0], boxes[1]["box"][0][0])


class ProjectTests(unittest.TestCase):
    def test_horizontal_half_stays_on_the_left(self) -> None:
        image = np.zeros((40, 120, 3), dtype=np.uint8)
        points = [[10, 5], [110, 5], [110, 25], [10, 25]]
        line = prepare_line(image, points)
        width = line.image.shape[1]
        boxes = project_spans(line, [("左", 0, 0, width / 2)])
        xs = [point[0] for point in boxes[0]["box"]]
        self.assertGreater(min(xs), 8)
        self.assertLess(max(xs), 65)
        self.assertEqual(boxes[0]["index"], 0)

    def test_vertical_top_span_stays_on_the_top(self) -> None:
        image = np.zeros((130, 50, 3), dtype=np.uint8)
        points = [[10, 10], [30, 10], [30, 110], [10, 110]]
        line = prepare_line(image, points)
        self.assertTrue(line.rotated)
        width = line.image.shape[1]
        boxes = project_spans(line, [("上", 0, 0, width / 2)])
        ys = [point[1] for point in boxes[0]["box"]]
        self.assertLess(max(ys), 70)
        self.assertGreater(min(ys), 8)


if __name__ == "__main__":
    unittest.main()
