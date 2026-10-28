"""无显卡单测：组批、顺序，以及一张失败不影响其余图片。"""
from __future__ import annotations

import unittest

from ocr_server.batch import RecognitionCancelled, chunked, recognize_images


class BatchTests(unittest.TestCase):
    def test_chunk_sizes(self) -> None:
        self.assertEqual(chunked(list(range(10)), 4), [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_order_and_batch_sizes(self) -> None:
        det_sizes: list[int] = []
        rec_sizes: list[int] = []

        def detect(batch):
            det_sizes.append(len(batch))
            return [[[[0, 0], [2, 0], [2, 1], [0, 1]]] for _ in batch]

        def recognize(specs):
            rec_sizes.append(len(specs))
            return [{"text": f"字{spec['image_index']}", "score": 0.5} for spec in specs]

        items = recognize_images(
            ["a", "b", "c", "d", "e"],
            mode="det_rec",
            det_batch=2,
            rec_batch=3,
            detect=detect,
            recognize=recognize,
        )
        self.assertEqual(det_sizes, [2, 2, 1])
        self.assertEqual(rec_sizes, [3, 2])
        self.assertEqual([item["index"] for item in items], [0, 1, 2, 3, 4])
        self.assertEqual([item["text"] for item in items], ["字0", "字1", "字2", "字3", "字4"])
        self.assertEqual(items[0]["boxes"][0]["box"], [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])

    def test_one_detection_failure_keeps_the_rest(self) -> None:
        def detect(batch):
            if any(image == "bad" for image in batch):
                raise RuntimeError("检测整批失败")
            return [[[[1, 1], [3, 1], [3, 2], [1, 2]]] for _ in batch]

        def recognize(specs):
            return [{"text": spec["image"], "score": 0.8} for spec in specs]

        items = recognize_images(
            ["ok", "bad", "ok2"],
            mode="det_rec",
            det_batch=8,
            rec_batch=8,
            detect=detect,
            recognize=recognize,
        )
        self.assertTrue(items[0]["ok"])
        self.assertEqual(items[0]["text"], "ok")
        self.assertFalse(items[1]["ok"])
        self.assertIn("检测", items[1]["error"])
        self.assertTrue(items[2]["ok"])
        self.assertEqual(items[2]["text"], "ok2")

    def test_one_recognition_failure_keeps_siblings(self) -> None:
        def detect(batch):
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]]] for _ in batch]

        def recognize(specs):
            if any(spec["image"] == "rec-bad" for spec in specs):
                raise RuntimeError("识别整批失败")
            return [{"text": "好", "score": 0.7} for _ in specs]

        items = recognize_images(
            ["good", "rec-bad", "also"],
            mode="det_rec",
            det_batch=4,
            rec_batch=8,
            detect=detect,
            recognize=recognize,
        )
        self.assertTrue(items[0]["ok"])
        self.assertFalse(items[1]["ok"])
        self.assertTrue(items[2]["ok"])
        self.assertEqual(items[2]["text"], "好")

    def test_partial_line_keeps_image(self) -> None:
        def detect(batch):
            boxes = []
            for image in batch:
                boxes.append([
                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                    [[0, 2], [1, 2], [1, 3], [0, 3]],
                ])
            return boxes

        def recognize(specs):
            output = []
            for spec in specs:
                if spec["box"] == [[0, 2], [1, 2], [1, 3], [0, 3]] and len(specs) == 1:
                    raise RuntimeError("第二行失败")
                if spec["box"] == [[0, 2], [1, 2], [1, 3], [0, 3]]:
                    raise RuntimeError("第二行失败")
                output.append({"text": "第一行", "score": 0.6})
            return output

        items = recognize_images(
            ["page"],
            mode="det_rec",
            det_batch=1,
            rec_batch=8,
            detect=detect,
            recognize=recognize,
        )
        self.assertTrue(items[0]["ok"])
        self.assertEqual(items[0]["text"], "第一行")
        self.assertEqual(len(items[0]["boxes"]), 1)

    def test_rec_only_skips_detection(self) -> None:
        def detect(_batch):
            raise AssertionError("只识别模式不应检测")

        def recognize(specs):
            self.assertIsNone(specs[0]["box"])
            return [{"text": "仅识别", "score": 0.4}]

        items = recognize_images(
            ["crop"],
            mode="rec_only",
            det_batch=8,
            rec_batch=32,
            detect=detect,
            recognize=recognize,
        )
        self.assertEqual(items[0]["text"], "仅识别")
        self.assertIsNone(items[0]["boxes"][0]["box"])

    def test_similar_widths_share_a_batch(self) -> None:
        seen: list[list[float]] = []

        def detect(batch):
            boxes = []
            for image in batch:
                width = image
                boxes.append([[[0, 0], [width, 0], [width, 1], [0, 1]]])
            return boxes

        def recognize(specs):
            seen.append([spec["box"][1][0] for spec in specs])
            return [{"text": "行", "score": 1.0} for _ in specs]

        items = recognize_images(
            [100, 200, 100.4, 200.5],
            mode="det_rec",
            det_batch=4,
            rec_batch=2,
            detect=detect,
            recognize=recognize,
        )
        self.assertEqual(seen, [[100, 100.4], [200, 200.5]])
        self.assertEqual([item["text"] for item in items], ["行", "行", "行", "行"])

    def test_distant_widths_are_not_mixed(self) -> None:
        seen: list[list[float]] = []

        def detect(batch):
            return [[[[0, 0], [image, 0], [image, 1], [0, 1]]] for image in batch]

        def recognize(specs):
            seen.append([spec["box"][1][0] for spec in specs])
            return [{"text": "行", "score": 1.0} for _ in specs]

        recognize_images(
            [10, 80],
            mode="det_rec",
            det_batch=2,
            rec_batch=16,
            detect=detect,
            recognize=recognize,
        )
        self.assertEqual(seen, [[10], [80]])

    def test_stop_before_the_next_detect_batch(self) -> None:
        calls: list[int] = []

        def detect(batch):
            calls.append(len(batch))
            return [[[[0, 0], [1, 0], [1, 1], [0, 1]]] for _ in batch]

        def stop() -> bool:
            return len(calls) >= 1

        with self.assertRaises(RecognitionCancelled):
            recognize_images(
                ["a", "b"],
                mode="det_rec",
                det_batch=1,
                rec_batch=1,
                detect=detect,
                recognize=lambda specs: [{"text": "字", "score": 1.0} for _ in specs],
                stop=stop,
            )
        self.assertEqual(calls, [1])

    def test_empty_detection(self) -> None:
        items = recognize_images(
            ["blank"],
            mode="det_rec",
            det_batch=1,
            rec_batch=1,
            detect=lambda batch: [[] for _ in batch],
            recognize=lambda specs: [],
        )
        self.assertTrue(items[0]["ok"])
        self.assertEqual(items[0]["text"], "")
        self.assertEqual(items[0]["boxes"], [])


if __name__ == "__main__":
    unittest.main()
