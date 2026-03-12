"""Tests for evaluation module."""

import pytest
import numpy as np

from transcription_lab.evaluation import (
    GroundTruthParser, GroundTruth, GroundTruthSegment,
    TranscriptionEvaluator, DiarizationEvaluator, Evaluator, EvaluationMetrics,
)
from transcription_lab.diarization import SpeakerSegment
from transcription_lab.transcriber import Transcript, TranscriptSegment


class TestGroundTruthParser:

    def test_parse_timestamped(self):
        content = """[00:00:05 - 00:00:12] Alice: Hello everyone, welcome to the meeting.
[00:00:13 - 00:00:18] Bob: Thanks Alice, let's get started.
[00:00:20 - 00:00:30] Alice: First item on the agenda is the project update."""

        parser = GroundTruthParser()
        gt = parser.parse(content)

        assert len(gt.segments) == 3
        assert gt.segments[0].speaker == "Alice"
        assert gt.segments[0].start == 5.0
        assert gt.segments[0].end == 12.0
        assert "Hello everyone" in gt.segments[0].text
        assert len(gt.speakers) == 2
        assert "Alice" in gt.speakers
        assert "Bob" in gt.speakers

    def test_parse_minutes_seconds(self):
        content = "[01:30 - 01:45] Speaker1: This is a test."

        parser = GroundTruthParser()
        gt = parser.parse(content)

        assert len(gt.segments) == 1
        assert gt.segments[0].start == 90.0
        assert gt.segments[0].end == 105.0

    def test_parse_decimal_seconds(self):
        content = "[00:00:05.500 - 00:00:10.250] Alice: Testing decimals."

        parser = GroundTruthParser()
        gt = parser.parse(content)

        assert gt.segments[0].start == 5.5
        assert gt.segments[0].end == 10.25

    def test_parse_rttm(self, tmp_path):
        content = """SPEAKER file 1 5.0 7.0 <NA> <NA> Alice <NA> <NA>
SPEAKER file 1 13.0 5.0 <NA> <NA> Bob <NA> <NA>
SPEAKER file 1 20.0 5.0 <NA> <NA> Alice <NA> <NA>"""

        path = tmp_path / "ref.rttm"
        path.write_text(content)

        parser = GroundTruthParser()
        gt = parser.parse_file(path)

        assert len(gt.segments) == 3
        assert gt.segments[0].speaker == "Alice"
        assert gt.segments[0].start == 5.0
        assert gt.segments[0].end == 12.0

    def test_parse_file(self, tmp_path):
        content = "[00:00:05 - 00:00:12] Alice: Hello."
        path = tmp_path / "transcript.txt"
        path.write_text(content)

        parser = GroundTruthParser()
        gt = parser.parse_file(path)

        assert len(gt.segments) == 1


class TestTranscriptionEvaluator:

    def test_perfect_match(self):
        evaluator = TranscriptionEvaluator()
        result = evaluator.evaluate("hello world how are you", "hello world how are you")
        assert result['wer'] == 0.0

    def test_all_wrong(self):
        evaluator = TranscriptionEvaluator()
        result = evaluator.evaluate("goodbye earth", "hello world")
        assert result['wer'] == 1.0

    def test_insertions(self):
        evaluator = TranscriptionEvaluator()
        result = evaluator.evaluate("hello beautiful world", "hello world")
        assert result['wer'] == 0.5  # 1 insertion / 2 ref words

    def test_deletions(self):
        evaluator = TranscriptionEvaluator()
        result = evaluator.evaluate("hello world", "hello beautiful world")
        assert abs(result['wer'] - 1 / 3) < 0.01

    def test_normalization(self):
        evaluator = TranscriptionEvaluator()
        result = evaluator.evaluate("hello world", "Hello, World!", normalize=True)
        assert result['wer'] == 0.0


class TestDiarizationEvaluator:

    def test_perfect_diarization(self):
        gt = GroundTruth(
            segments=[
                GroundTruthSegment(speaker="Alice", text="", start=0.0, end=5.0),
                GroundTruthSegment(speaker="Bob", text="", start=5.0, end=10.0),
            ],
            speakers=["Alice", "Bob"],
            duration=10.0,
        )

        hypothesis = [
            SpeakerSegment(speaker_id="Alice", start=0.0, end=5.0),
            SpeakerSegment(speaker_id="Bob", start=5.0, end=10.0),
        ]

        evaluator = DiarizationEvaluator(collar=0.0)
        result = evaluator.evaluate(hypothesis, gt)

        assert result['der'] < 0.1

    def test_speaker_confusion(self):
        gt = GroundTruth(
            segments=[GroundTruthSegment(speaker="Alice", text="", start=0.0, end=10.0)],
            speakers=["Alice"],
            duration=10.0,
        )

        hypothesis = [SpeakerSegment(speaker_id="Bob", start=0.0, end=10.0)]

        evaluator = DiarizationEvaluator(collar=0.0)
        result = evaluator.evaluate(hypothesis, gt)

        assert result['speaker_confusion'] > 0.9

    def test_uses_speaker_id_attr(self):
        """SpeakerSegment uses speaker_id, not speaker -- evaluator must handle both."""
        gt = GroundTruth(
            segments=[GroundTruthSegment(speaker="X", text="", start=0.0, end=5.0)],
            speakers=["X"],
            duration=5.0,
        )

        hypothesis = [SpeakerSegment(speaker_id="X", start=0.0, end=5.0)]

        evaluator = DiarizationEvaluator(collar=0.0)
        result = evaluator.evaluate(hypothesis, gt)

        # Should match because _get_speaker() reads speaker_id
        assert result['speaker_confusion'] == 0.0


class TestEvaluator:

    def test_full_evaluation(self):
        gt = GroundTruth(
            segments=[GroundTruthSegment(speaker="Alice", text="Hello this is a test", start=0.0, end=5.0)],
            speakers=["Alice"],
            duration=5.0,
        )

        hyp = Transcript(
            segments=[TranscriptSegment(text="Hello this is a test", start=0.0, end=5.0, speaker="Alice")],
            language="en",
            duration=5.0,
        )

        evaluator = Evaluator()
        metrics = evaluator.evaluate(hyp, gt)

        assert metrics.wer == 0.0
        assert metrics.total_words == 5


class TestEvaluationMetrics:

    def test_to_dict(self):
        metrics = EvaluationMetrics(
            wer=0.15, insertions=2, deletions=1, substitutions=3,
            der=0.10, miss=0.02, false_alarm=0.03, confusion=0.05,
            speaker_accuracy=0.90, speaker_mapping={"A": "B"},
            total_words=100, total_duration=60.0,
        )
        d = metrics.to_dict()
        assert d["wer"] == 0.15
        assert d["der"] == 0.10

    def test_str(self):
        metrics = EvaluationMetrics(
            wer=0.15, insertions=2, deletions=1, substitutions=3,
            der=0.10, miss=0.02, false_alarm=0.03, confusion=0.05,
            speaker_accuracy=0.90, speaker_mapping={},
            total_words=100, total_duration=60.0,
        )
        s = str(metrics)
        assert "WER" in s
        assert "DER" in s
