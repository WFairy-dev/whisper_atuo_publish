"""日志按天切分，以及不重启也会按日期删除。"""
from __future__ import annotations

import logging
import os
import tempfile
import time
import unittest
from pathlib import Path

import run_log


def _moment(text: str) -> float:
    return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S"))


class LogRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_old_day_is_removed_even_if_directory_was_touched_recently(self) -> None:
        now = _moment("2026-11-05 10:51:00")
        expired = self.root / "2026-11-01"
        kept = self.root / "2026-11-03"
        expired.mkdir()
        kept.mkdir()
        (expired / "ocr.log").write_text("old", encoding="utf-8")
        (kept / "ocr.log").write_text("new", encoding="utf-8")
        os.utime(expired, (now, now))
        os.utime(kept, (now - 10 * 86400, now - 10 * 86400))

        removed = run_log.purge_dated_logs(self.root, 3, now=now, protect_days=set())

        self.assertEqual(removed, 1)
        self.assertFalse(expired.exists())
        self.assertTrue(kept.exists())

    def test_day_being_written_is_kept_even_when_it_is_older_than_retention(self) -> None:
        now = _moment("2026-11-05 10:51:00")
        active = self.root / "2026-10-20"
        active.mkdir()
        (active / "ocr.log").write_text("still open", encoding="utf-8")

        removed = run_log.purge_dated_logs(self.root, 3, now=now, protect_days={"2026-10-20"})

        self.assertEqual(removed, 0)
        self.assertTrue((active / "ocr.log").is_file())

    def test_legacy_flat_log_uses_modification_time(self) -> None:
        now = _moment("2026-11-05 10:51:00")
        old = self.root / "asr.log"
        fresh = self.root / "asr.log.1"
        old.write_text("old", encoding="utf-8")
        fresh.write_text("fresh", encoding="utf-8")
        os.utime(old, (now - 4 * 86400, now - 4 * 86400))
        os.utime(fresh, (now - 3600, now - 3600))

        removed = run_log.purge_dated_logs(
            self.root, 3, now=now, protect_days=set(), legacy_prefix="asr.log"
        )

        self.assertEqual(removed, 1)
        self.assertFalse(old.exists())
        self.assertTrue(fresh.exists())


class DailySplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_one_process_writes_each_day_into_its_own_file(self) -> None:
        handler = run_log.DailyRunFileHandler(self.root, "10-06-53", "ocr.log", day="2026-11-04")
        handler.setFormatter(logging.Formatter("%(message)s"))
        try:
            first = logging.LogRecord("t", logging.INFO, __file__, 1, "第一天", (), None)
            first.created = _moment("2026-11-04 23:50:00")
            handler.handle(first)
            second = logging.LogRecord("t", logging.INFO, __file__, 1, "第二天", (), None)
            second.created = _moment("2026-11-05 00:01:00")
            handler.handle(second)
        finally:
            handler.close()

        self.assertEqual(
            (self.root / "2026-11-04" / "10-06-53" / "ocr.log").read_text(encoding="utf-8").strip(),
            "第一天",
        )
        self.assertEqual(
            (self.root / "2026-11-05" / "10-06-53" / "ocr.log").read_text(encoding="utf-8").strip(),
            "第二天",
        )

    def test_redirect_moves_regular_file_and_leaves_pipe_alone(self) -> None:
        old = self.root / "old.log"
        old.write_text("old\n", encoding="utf-8")
        fd = os.open(old, os.O_WRONLY | os.O_APPEND)
        read_end, write_end = os.pipe()
        try:
            self.assertTrue(run_log.redirect_stdio(self.root, "10-06-53", fds=(fd,)))
            os.write(fd, b"new\n")
            self.assertFalse(run_log.redirect_stdio(self.root, "10-06-53", fds=(fd,)))
            self.assertFalse(run_log.redirect_stdio(self.root, "10-06-53", fds=(write_end,)))
        finally:
            os.close(fd)
            os.close(read_end)
            os.close(write_end)

        day = time.strftime("%Y-%m-%d")
        target = self.root / day / "10-06-53" / "stdout.log"
        self.assertIn("new", target.read_text(encoding="utf-8"))
        self.assertNotIn("new", old.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
