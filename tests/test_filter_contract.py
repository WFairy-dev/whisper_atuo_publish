"""Exercise multipart options and queue delivery without starting GPU workers."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient

from asr_server import config
from asr_server.app import _parse_strict_filter, app
from asr_server.engine import AsrEngine, Lane
from client_example import _upload_form


class FilterContractTests(unittest.TestCase):
    def test_blank_filter_means_unspecified(self):
        self.assertIsNone(_parse_strict_filter(None))
        self.assertIsNone(_parse_strict_filter(""))
        self.assertIsNone(_parse_strict_filter("  "))
        self.assertIs(_parse_strict_filter(" TRUE "), True)
        self.assertIs(_parse_strict_filter("0"), False)

    def test_both_upload_endpoints_forward_optional_filter(self):
        # No context manager: skip the production startup/model-loading lifecycle.
        client = TestClient(app)
        for endpoint in ("/v1/asr", "/v1/asr/sync"):
            for value, expected in (
                ("false", False), ("true", True), ("0", False), ("1", True),
                (" TRUE ", True), ("", None), (None, None),
            ):
                with self.subTest(endpoint=endpoint, value=value), patch("asr_server.app.engine") as engine, \
                        patch("asr_server.app._save_upload", new_callable=AsyncMock) as save, \
                        patch.object(config, "API_KEY", ""):
                    save.return_value = (Path("unused.wav"), ".wav", "sample.wav", 1)
                    engine.task_to_dict.return_value = {"status_name": "completed", "result": {}}
                    data = {"task_id": "test", "source_file_path": "D:/sample.wav"}
                    if value is not None:
                        data["strict_filter"] = value
                    response = client.post(endpoint, data=data, files={"file": ("sample.wav", b"audio")})
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertIs(engine.submit_file.call_args.kwargs["strict_filter"], expected)

    def test_invalid_filter_rejected_before_upload_save(self):
        client = TestClient(app)
        for endpoint in ("/v1/asr", "/v1/asr/sync"):
            with self.subTest(endpoint=endpoint), patch("asr_server.app._save_upload", new_callable=AsyncMock) as save, \
                    patch.object(config, "API_KEY", ""):
                response = client.post(endpoint, data={"task_id": "test", "source_file_path": "D:/sample.wav",
                    "strict_filter": "invalid"}, files={"file": ("sample.wav", b"audio")})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(response.json()["error"]["error_code"], "INVALID_REQUEST")
                save.assert_not_called()

    def test_queue_preserves_effective_option(self):
        for default in (True, False):
            for requested in (None, True, False):
                with self.subTest(default=default, requested=requested), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    with patch.object(config, "TEMP_DIR", root), patch.object(config, "RESULT_DIR", root), \
                            patch.object(config, "STRICT_FILTER", default), patch("asr_server.engine._audio_duration_s", return_value=1.):
                        engine = AsrEngine()
                        source = root / "upload.wav"
                        source.write_bytes(b"audio")
                        task = engine.submit_file(source, ".wav", task_id="test", source_file_path="D:/sample.wav",
                                                  strict_filter=requested)
                        lane, queue, proc = Lane(0), Mock(), Mock()
                        proc.is_alive.return_value = True
                        engine._try_dispatch(lane, queue, proc)
                        expected = default if requested is None else requested
                        self.assertIs(task.strict_filter, expected)
                        self.assertIs(queue.put.call_args.args[0]["strict_filter"], expected)

    def test_client_form_omits_default_and_encodes_boolean(self):
        path = Path("D:/sample.wav")
        for value in (None, False, True):
            data, _ = _upload_form(path, "test", None, value)
            if value is None:
                self.assertNotIn("strict_filter", data)
            else:
                self.assertEqual(data["strict_filter"], str(value).lower())
