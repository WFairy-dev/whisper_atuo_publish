"""无显卡单测：任务号、排队取消、识别中取消、重复提交。"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from ocr_server import config
from ocr_server.engine import (
    DuplicateTaskId,
    OcrEngine,
    QueueFull,
    TaskAlreadyFinished,
)


class TaskFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self._old_result = config.RESULT_DIR
        self._old_temp = config.TEMP_DIR
        self._old_det = config.DET_BATCH
        config.RESULT_DIR = Path(self.tmp.name) / "results"
        config.TEMP_DIR = Path(self.tmp.name) / "tmp"
        config.RESULT_DIR.mkdir()
        config.TEMP_DIR.mkdir()
        config.DET_BATCH = 1
        self.engine = OcrEngine()
        self.engine._loaded = True
        self.engine._accepting = True
        self.image = Path(self.tmp.name) / "a.jpg"
        self.image.write_bytes(b"img")
        self.image_b = Path(self.tmp.name) / "b.jpg"
        self.image_b.write_bytes(b"img")

    def tearDown(self) -> None:
        self.engine.stop()
        config.RESULT_DIR = self._old_result
        config.TEMP_DIR = self._old_temp
        config.DET_BATCH = self._old_det
        self.tmp.cleanup()

    def _submit(self, task_id: str, paths=None):
        return self.engine.submit_files(
            list(paths or [self.image]),
            task_id=task_id,
            source_file_path="D:/视频/a.mp4",
            mode="det_rec",
            file_size=3,
        )

    def test_cancel_queued_removes_it_before_detect(self) -> None:
        calls: list[int] = []

        def detect(batch):
            calls.append(len(batch))
            return [[] for _ in batch]

        task = self._submit("req_queued")
        payload = self.engine.cancel("req_queued")
        self.assertEqual(payload["status_name"], "cancelled")
        self.assertEqual(payload["previous_status"], "queued")
        self.assertIsNone(payload["result"])
        self.assertFalse(self.image.exists())
        self.engine.start_for_test(detect, lambda specs: [], read=lambda path: "img")
        time.sleep(0.3)
        self.assertEqual(calls, [])
        self.assertEqual(task.status, "cancelled")
        again = self.engine.cancel("req_queued")
        self.assertEqual(again["previous_status"], "cancelled")

    def test_cancel_running_returns_before_the_batch_finishes(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls: list[int] = []

        def detect(batch):
            calls.append(len(batch))
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            return [[] for _ in batch]

        self.engine.start_for_test(detect, lambda specs: [], read=lambda path: path.name)
        task = self._submit("req_running", [self.image, self.image_b])
        self.assertTrue(entered.wait(2))
        started = time.perf_counter()
        payload = self.engine.cancel("req_running")
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.5)
        self.assertEqual(payload["previous_status"], "running")
        self.assertEqual(payload["status_name"], "cancelled")
        self.assertEqual(payload["status"], 4)
        release.set()
        self.assertTrue(task.done_event.wait(2))
        self.assertEqual(calls, [1])
        self.assertEqual(self.engine.get_task_payload("req_running")["status_name"], "cancelled")

    def test_completed_cannot_be_cancelled_and_blocks_resubmit(self) -> None:
        self.engine.start_for_test(
            lambda batch: [[] for _ in batch],
            lambda specs: [],
            read=lambda path: "img",
        )
        task = self._submit("req_done")
        self.assertTrue(task.done_event.wait(2))
        payload = self.engine.get_task_payload("req_done")
        self.assertEqual(payload["status"], 0)
        self.assertEqual(payload["status_name"], "completed")
        self.assertEqual(payload["error_code"], 100000)
        self.assertEqual(payload["result"]["count"], 1)
        self.assertEqual(payload["image_count"], 1)
        with self.assertRaises(TaskAlreadyFinished):
            self.engine.cancel("req_done")
        with self.assertRaises(DuplicateTaskId):
            self._submit("req_done")

    def test_cancelled_id_can_be_submitted_again(self) -> None:
        task = self._submit("req_retry")
        self.engine.cancel("req_retry")
        self.image.write_bytes(b"img")
        self.engine.start_for_test(
            lambda batch: [[] for _ in batch],
            lambda specs: [],
            read=lambda path: "img",
        )
        again = self._submit("req_retry")
        self.assertTrue(again.done_event.wait(2))
        self.assertEqual(again.status, "completed")
        self.assertIsNot(again, task)

    def test_two_lanes_keep_each_task_on_one_lane(self) -> None:
        left_a = Path(self.tmp.name) / "left_a.jpg"
        left_b = Path(self.tmp.name) / "left_b.jpg"
        right_a = Path(self.tmp.name) / "right_a.jpg"
        right_b = Path(self.tmp.name) / "right_b.jpg"
        for path in (left_a, left_b, right_a, right_b):
            path.write_bytes(b"img")
        seen: dict[int, list[str]] = {}
        lock = threading.Lock()
        both = threading.Barrier(2)

        def detect(batch):
            ident = threading.get_ident()
            with lock:
                seen.setdefault(ident, []).extend(batch)
            both.wait(timeout=2)
            return [[] for _ in batch]

        self.engine.start_for_test(detect, lambda specs: [], read=lambda path: path.name, lanes=2)
        first = self._submit("lane_left", [left_a, left_b])
        second = self._submit("lane_right", [right_a, right_b])
        self.assertTrue(first.done_event.wait(3))
        self.assertTrue(second.done_event.wait(3))
        self.assertEqual(first.status, "completed")
        self.assertEqual(second.status, "completed")
        self.assertNotEqual(first.lane_id, second.lane_id)
        self.assertEqual({first.lane_id, second.lane_id}, {0, 1})
        groups = [sorted(names) for names in seen.values()]
        self.assertEqual(sorted(groups), [["left_a.jpg", "left_b.jpg"], ["right_a.jpg", "right_b.jpg"]])

    def test_missing_cancel_and_full_queue(self) -> None:
        self.assertIsNone(self.engine.cancel("req_missing"))
        old = config.MAX_QUEUE_SIZE
        config.MAX_QUEUE_SIZE = 1
        try:
            self._submit("req_full_a")
            with self.assertRaises(QueueFull):
                self._submit("req_full_b")
        finally:
            config.MAX_QUEUE_SIZE = old


if __name__ == "__main__":
    unittest.main()
