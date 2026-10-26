"""Run with python -m unittest discover -s tests -v; no Whisper/GPU required."""
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from asr_server import config
from asr_server import result_store
from asr_server.infer_worker import transcribe_file
from asr_server.quality import QualityConfig, normalize_segments, validate_structure


def segment(start=0., end=1., text="Hello.", **values):
    return dict(start=start, end=end, text=text, no_speech_prob=.1,
                avg_logprob=-.2, compression_ratio=1., **values)


def result(*segments):
    return {"language": "en", "text": " ".join(s["text"] for s in segments),
            "segments": list(segments)}


class TranscriptionTests(unittest.TestCase):
    def run_model(self, outputs, duration=5., device="cpu", strict_filter=True):
        audio = [0] * int(16000 * duration)
        backend = SimpleNamespace(load_audio=Mock(return_value=audio),
                                  audio=SimpleNamespace(SAMPLE_RATE=16000))
        model = Mock(device=device)
        model.transcribe.side_effect = outputs
        with patch.dict(sys.modules, whisper=backend), patch.object(config, "MAX_ATTEMPTS", 2):
            payload = transcribe_file(model, "sample.wav", task_id="test", strict_filter=strict_filter)
        backend.load_audio.assert_called_once_with("sample.wav")
        json.dumps(payload, allow_nan=False)
        return payload, model

    def test_short_single_sentence_kept_with_words(self):
        payload, model = self.run_model([result(segment(end=2.8, words=[
            {"start": 0., "end": 2.8, "word": "Hello."}]))], duration=3.)
        self.assertEqual(payload["text"], "Hello.")
        self.assertEqual(payload["segments"][0]["words"][0]["end"], 2.8)
        options = model.transcribe.call_args.kwargs
        self.assertTrue(options["word_timestamps"])
        self.assertTrue(options["condition_on_previous_text"])
        self.assertFalse(options["fp16"])

    def test_retry_disables_context_and_reuses_audio(self):
        payload, model = self.run_model([result(segment(end=40)), result(segment())], device="cuda:0")
        self.assertEqual(payload["quality_analysis"]["attempts"], 2)
        self.assertEqual(payload["quality_analysis"]["selected_attempt"], 2)
        self.assertFalse(model.transcribe.call_args.kwargs["condition_on_previous_text"])
        self.assertTrue(model.transcribe.call_args.kwargs["fp16"])
        self.assertIs(model.transcribe.call_args_list[0].args[0], model.transcribe.call_args_list[1].args[0])

    def test_salvage_and_rebuild_text(self):
        raw = result(segment(), segment(2, 40, "Bad guess."))
        payload, _ = self.run_model([raw, raw])
        self.assertEqual(payload["text"], "Hello.")
        self.assertTrue(payload["quality_analysis"]["structural_salvage"])
        self.assertEqual(len(payload["rejected_segments"]), 1)

    def test_confident_repeated_speech_is_retained(self):
        payload, model = self.run_model([result(*(segment(i, i + .5) for i in range(3)))])
        self.assertEqual(len(payload["segments"]), 3)
        self.assertEqual(model.transcribe.call_count, 1)

    def test_weak_repetitions_do_not_survive_recovery(self):
        segments = [segment(i, i + .5) for i in range(4)]
        segments[0]["avg_logprob"] = -.55
        raw = result(*segments)
        payload, _ = self.run_model([raw, raw])
        self.assertEqual(len(payload["segments"]), 3)
        self.assertEqual(len(payload["rejected_segments"]), 1)

    def test_quality_filter_and_unknown_metrics(self):
        low, ambiguous, unknown = segment(), segment(1, 2, "Maybe."), segment(2, 3, "Unknown.")
        low.update(no_speech_prob=.8, avg_logprob=-.55)
        ambiguous["avg_logprob"] = -.55
        unknown.pop("avg_logprob")
        payload, _ = self.run_model([result(low, ambiguous, unknown)])
        self.assertEqual(payload["text"], "Maybe. Unknown.")
        self.assertEqual([s["quality_status"] for s in payload["segments"]], ["ambiguous", "unknown"])
        self.assertIsNone(payload["segments"][1]["avg_logprob"])

    def test_nonfinite_ranges_are_rejected_and_json_safe(self):
        raw = result(segment(), segment(float("nan"), float("inf"), "Invalid."))
        payload, _ = self.run_model([raw, raw])
        self.assertEqual(payload["text"], "Hello.")
        self.assertIsNone(payload["rejected_segments"][0]["start_s"])

    def test_total_duration_is_rechecked_after_salvage(self):
        raw = result(segment(0, 4), segment(1, 5, "Overlap."))
        payload, _ = self.run_model([raw, raw])
        self.assertEqual(len(payload["segments"]), 1)
        self.assertEqual(validate_structure(payload["segments"], 5., config.QUALITY), [])

    def test_small_overrun_is_clipped_and_words_cleared(self):
        raw = result(segment(4, 5.2, words=[{"start": 4., "end": 5.2, "word": "Hello."}]))
        payload, _ = self.run_model([raw])
        self.assertEqual(payload["segments"][0]["end_s"], 5.)
        self.assertEqual(payload["segments"][0]["words"], [])

    def test_empty_speech_is_success_but_runtime_error_propagates(self):
        payload, _ = self.run_model([result()])
        self.assertEqual(payload["text"], "")
        with self.assertRaisesRegex(RuntimeError, "GPU failure"):
            self.run_model([RuntimeError("GPU failure")])

    def test_malformed_structure_retries_then_fails(self):
        with self.assertRaisesRegex(ValueError, "no recoverable structure"):
            self.run_model([{"segments": "invalid"}, {"segments": "invalid"}])

    def test_all_rejected_speech_is_successful_empty_result(self):
        raw = segment()
        raw["avg_logprob"] = -.9
        payload, _ = self.run_model([result(raw)])
        self.assertEqual(payload["segments"], [])
        self.assertEqual(payload["quality_analysis"]["rejected_segment_count"], 1)

    def test_bad_alignment_and_thresholds(self):
        normalized = normalize_segments([segment(words=[{"start": -1, "end": 1, "word": "Hello."}])])
        self.assertEqual(normalized[0]["words"], [])
        with self.assertRaises(ValueError):
            QualityConfig(duration_tolerance=float("nan"))

    def test_result_store_preserves_new_fields_and_legacy_contract(self):
        from pathlib import Path

        payload, _ = self.run_model([result(segment())])
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "RESULT_DIR", Path(directory)):
            result_store.save("test-quality", {"result": payload})
            restored = result_store.load("test-quality")["result"]
        self.assertEqual(restored, payload)
        self.assertTrue({"language", "text", "segments"} <= restored.keys())
        self.assertTrue({"id", "seek", "start_s", "end_s", "text"} <= restored["segments"][0].keys())

    def test_relaxed_filter_keeps_uncertain_text_and_repeats(self):
        segments = [segment(i, i + .5) for i in range(3)]
        for item in segments:
            item.update(avg_logprob=-.9, no_speech_prob=.9, compression_ratio=4.)
        payload, model = self.run_model([result(*segments)], strict_filter=False)
        self.assertEqual(len(payload["segments"]), 3)
        self.assertEqual(payload["rejected_segments"], [])
        self.assertFalse(payload["quality_analysis"]["strict_filter"])
        self.assertEqual(payload["quality_analysis"]["ambiguous_segment_count"], 3)
        self.assertEqual(model.transcribe.call_count, 1)

    def test_relaxed_filter_still_rejects_invalid_ranges(self):
        raw = result(segment(), segment(2, 99, "Invalid."))
        payload, _ = self.run_model([raw, raw], strict_filter=False)
        self.assertEqual(payload["text"], "Hello.")
        self.assertEqual(len(payload["rejected_segments"]), 1)

    def test_inference_uses_server_default_when_option_omitted(self):
        raw = segment()
        raw["avg_logprob"] = -.9
        with patch.object(config, "STRICT_FILTER", False):
            payload, _ = self.run_model([result(raw)], strict_filter=None)
        self.assertEqual(len(payload["segments"]), 1)
        self.assertFalse(payload["quality_analysis"]["strict_filter"])

    def test_relaxed_filter_keeps_long_valid_speech(self):
        raw = result(segment(end=35))
        payload, model = self.run_model([raw], duration=36, strict_filter=False)
        self.assertEqual(len(payload["segments"]), 1)
        self.assertEqual(payload["segments"][0]["quality_status"], "ambiguous")

    def test_out_of_order_segments_retry_then_sort(self):
        raw = result(segment(2, 3, "Second."), segment(0, 1, "First."))
        payload, model = self.run_model([raw, raw])
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertEqual(payload["text"], "First. Second.")
        self.assertEqual(payload["quality_analysis"]["structural_validation"][0]["errors"],
                         ["segment_1_out_of_order"])

    def test_overlap_rejected_even_when_total_duration_is_short(self):
        raw = result(segment(0, 2), segment(1, 3, "Overlap."))
        payload, _ = self.run_model([raw, raw], duration=20)
        self.assertEqual(payload["rejected_segments"][0]["quality_reason"], "overlap_exceeds_tolerance")

    def test_small_overlap_is_allowed(self):
        payload, model = self.run_model([result(segment(0, 2), segment(1.8, 3, "Next."))])
        self.assertEqual(len(payload["segments"]), 2)
        self.assertEqual(model.transcribe.call_count, 1)

    def test_word_alignment_must_cover_text(self):
        for words in ([{"start": 0, "end": 1, "word": "Other."}],
                      [{"start": 0, "end": 1, "word": "Hello"}]):
            normalized = normalize_segments([segment(words=words)])
            self.assertEqual(normalized[0]["words"], [])
        normalized = normalize_segments([segment(text="Hello world.", words=[
            {"start": 0, "end": .5, "word": " hello"},
            {"start": .5, "end": 1, "word": " world."}])])
        self.assertEqual(len(normalized[0]["words"]), 2)


if __name__ == "__main__":
    unittest.main()
