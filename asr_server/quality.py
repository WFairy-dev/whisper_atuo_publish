"""GPU-independent transcript validation, recovery and confidence filtering."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any


@dataclass(frozen=True)
class QualityConfig:
    segment_max_duration: float = 29.0
    duration_tolerance: float = 0.5
    total_duration_ratio: float = 1.15
    repetition_count: int = 3
    compression_limit: float = 2.8
    no_speech_threshold: float = 0.5
    low_logprob: float = -0.65
    ambiguous_logprob: float = -0.5

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in vars(self).values()):
            raise ValueError("Quality thresholds must be finite")
        if (self.segment_max_duration <= 0 or self.duration_tolerance < 0
                or self.total_duration_ratio < 1 or self.repetition_count < 2
                or self.compression_limit <= 0 or not 0 <= self.no_speech_threshold <= 1
                or self.low_logprob > self.ambiguous_logprob or self.ambiguous_logprob > 0):
            raise ValueError("Invalid quality thresholds")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def normalize_segments(raw: Any) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("Whisper segments must be a list")
    segments = []
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("Whisper segment must be an object")
        segment = {
            "id": value.get("id") if type(value.get("id")) is int else None,
            "seek": value.get("seek") if type(value.get("seek")) is int else None,
            "start_s": _number(value.get("start")),
            "end_s": _number(value.get("end")),
            "text": str(value.get("text") or "").strip(),
            **{key: _number(value.get(key)) for key in
               ("no_speech_prob", "avg_logprob", "compression_ratio")},
            "words": [],
        }
        # Discard unusable alignment as a whole; callers can use segment times.
        previous_end = segment["start_s"]
        raw_words = value.get("words")
        for word in raw_words if isinstance(raw_words, list) else []:
            start = _number(word.get("start")) if isinstance(word, dict) else None
            end = _number(word.get("end")) if isinstance(word, dict) else None
            if (start is None or end is None or previous_end is None
                    or segment["end_s"] is None or not start < end
                    or start < segment["start_s"] or end > segment["end_s"]
                    or start < previous_end - 0.1 or not isinstance(word.get("word"), str)):
                segment["words"] = []
                break
            segment["words"].append({"start": start, "end": end, "word": word["word"]})
            previous_end = end
        if (_alignment_text("".join(word["word"] for word in segment["words"]))
                != _alignment_text(segment["text"])):
            segment["words"] = []
        segments.append(segment)
    return segments


def _alignment_text(text: str) -> str:
    """Ignore whitespace/case while retaining punctuation and actual words."""
    return "".join(text.casefold().split())


def _normalized(text: str) -> str:
    return re.sub(r"[\W_]+", " ", re.sub(r"^\d+[.)、]?\s*", "", text.casefold())).strip()


def _weak(segment: dict, config: QualityConfig) -> bool:
    score = segment["avg_logprob"]
    return score is not None and score <= config.ambiguous_logprob


def _reason(segment: dict, duration: float, config: QualityConfig, strict_filter: bool) -> str | None:
    start, end = segment["start_s"], segment["end_s"]
    if start is None or end is None or start < 0 or end <= start or start >= duration:
        return "invalid_range"
    if end > duration + config.duration_tolerance:
        return "end_exceeds_audio"
    if strict_filter and end - start >= config.segment_max_duration:
        return "duration_exceeds_limit"
    compression = segment["compression_ratio"]
    if strict_filter and compression is not None and compression > config.compression_limit:
        return "compression_ratio_exceeds_limit"
    return None


def _timeline_reason(segment: dict, previous: dict | None, config: QualityConfig) -> str | None:
    if previous is not None:
        if segment["start_s"] < previous["start_s"]:
            return "out_of_order"
        if previous["end_s"] - segment["start_s"] > config.duration_tolerance:
            return "overlap_exceeds_tolerance"
    return None


def validate_structure(segments: list[dict], duration: float, config: QualityConfig,
                       *, strict_filter: bool = True) -> list[str]:
    errors = []
    for index, segment in enumerate(segments):
        reason = _reason(segment, duration, config, strict_filter)
        if reason:
            errors.append(f"segment_{index}_{reason}")
    if errors:
        return errors
    for index in range(1, len(segments)):
        reason = _timeline_reason(segments[index], segments[index - 1], config)
        if reason:
            errors.append(f"segment_{index}_{reason}")
    if sum(s["end_s"] - s["start_s"] for s in segments) > duration * config.total_duration_ratio:
        errors.append("total_segment_duration_exceeds_audio")
    counts = Counter(_normalized(s["text"]) for s in segments if s["text"])
    if strict_filter and any(counts[_normalized(s["text"])] >= config.repetition_count and _weak(s, config)
           for s in segments if s["text"]):
        errors.append("repeated_low_confidence_text")
    return errors


def filter_segments(segments: list[dict], duration: float, config: QualityConfig,
                    *, strict_filter: bool = True) -> tuple[list[dict], list[dict]]:
    kept, rejected = [], []
    counts = Counter(_normalized(s["text"]) for s in segments if s["text"])
    nonweak_texts = {_normalized(s["text"]) for s in segments
                     if s["text"] and not _weak(s, config)}
    seen = set()
    total = 0.0
    # Reorder recoverable segments without rewriting their original IDs/times.
    for original in sorted(segments, key=lambda s: s["start_s"] if s["start_s"] is not None else math.inf):
        segment = dict(original)
        reason = _reason(segment, duration, config, strict_filter)
        text = _normalized(segment["text"])
        score, no_speech = segment["avg_logprob"], segment["no_speech_prob"]
        if not reason and not segment["text"]:
            reason = "empty_text"
        if (strict_filter and not reason and counts[text] >= config.repetition_count
                and (text in seen or text in nonweak_texts) and _weak(segment, config)):
            reason = "repeated_low_confidence_text"
        if strict_filter and not reason and score is not None:
            if no_speech is not None and no_speech >= config.no_speech_threshold and _weak(segment, config):
                reason = "high_no_speech_and_low_logprob"
            elif score <= config.low_logprob:
                reason = "extremely_low_avg_logprob"
        if not reason:
            segment["end_s"] = min(segment["end_s"], duration)
            if segment["end_s"] != original["end_s"]:
                segment["words"] = []
            length = segment["end_s"] - segment["start_s"]
            reason = _timeline_reason(segment, kept[-1] if kept else None, config)
            if not reason and total + length > duration * config.total_duration_ratio:
                reason = "total_segment_duration_exceeds_audio"
        if reason:
            segment.update(quality_status="rejected", quality_reason=reason)
            rejected.append(segment)
            continue
        unknown = any(segment[key] is None for key in ("no_speech_prob", "avg_logprob", "compression_ratio"))
        ambiguous = (_weak(segment, config)
                     or (no_speech is not None and no_speech >= config.no_speech_threshold)
                     or (segment["compression_ratio"] is not None and segment["compression_ratio"] > config.compression_limit)
                     or length >= config.segment_max_duration)
        status = "unknown" if unknown else "ambiguous" if ambiguous else "accepted"
        segment.update(quality_status=status, quality_reason={
            "unknown": "quality_metrics_unavailable", "ambiguous": "uncertain_speech",
            "accepted": "quality_passed"}[status])
        kept.append(segment)
        seen.add(text)
        total += length
    return kept, rejected
