"""Tests for transcription engine module."""

import pytest
import numpy as np

from transcription_lab.transcriber import (
    TranscriptWord,
    TranscriptSegment,
    Transcript,
    TranscriptionEngine,
    RealtimeTranscriber,
    BatchTranscriber,
)


class TestTranscriptWord:

    def test_attributes(self):
        w = TranscriptWord(word="hello", start=1.0, end=1.5, probability=0.95)
        assert w.word == "hello"
        assert w.start == 1.0
        assert w.end == 1.5
        assert w.probability == 0.95


class TestTranscriptSegment:

    def test_duration(self):
        seg = TranscriptSegment(text="hello world", start=1.0, end=3.5)
        assert seg.duration == 2.5

    def test_defaults(self):
        seg = TranscriptSegment(text="test", start=0.0, end=1.0)
        assert seg.words == []
        assert seg.speaker is None


class TestTranscript:

    def test_text(self):
        t = Transcript(
            segments=[
                TranscriptSegment(text="hello", start=0.0, end=1.0),
                TranscriptSegment(text="world", start=1.0, end=2.0),
            ],
            language="en",
            duration=2.0,
        )
        assert t.text == "hello world"

    def test_text_skips_empty(self):
        t = Transcript(
            segments=[
                TranscriptSegment(text="hello", start=0.0, end=1.0),
                TranscriptSegment(text="", start=1.0, end=1.5),
                TranscriptSegment(text="world", start=1.5, end=2.0),
            ],
            language="en",
            duration=2.0,
        )
        assert t.text == "hello world"

    def test_to_text_with_timestamps(self):
        t = Transcript(
            segments=[
                TranscriptSegment(text="hello", start=0.0, end=1.0, speaker="Alice"),
                TranscriptSegment(text="world", start=1.0, end=2.0),
            ],
            language="en",
            duration=2.0,
        )
        output = t.to_text_with_timestamps()
        assert "[0.00 - 1.00] Alice: hello" in output
        assert "[1.00 - 2.00] world" in output

    def test_empty_transcript(self):
        t = Transcript(segments=[], language="en", duration=0.0)
        assert t.text == ""
        assert t.to_text_with_timestamps() == ""


class TestTranscriptionEngine:

    def test_mock_transcribe(self):
        engine = TranscriptionEngine(model_size="base")
        audio = np.random.randn(16000 * 3).astype(np.float32)
        transcript = engine.transcribe(audio, sample_rate=16000)

        assert isinstance(transcript, Transcript)
        assert transcript.duration == pytest.approx(3.0)
        assert len(transcript.segments) > 0
        assert "Mock" in transcript.text or len(transcript.text) > 0

    def test_mock_returns_correct_duration(self):
        engine = TranscriptionEngine(model_size="tiny")
        audio = np.random.randn(16000 * 5).astype(np.float32)
        transcript = engine.transcribe(audio, sample_rate=16000)
        assert transcript.duration == pytest.approx(5.0)

    def test_custom_parameters(self):
        engine = TranscriptionEngine(model_size="base", language="en")
        audio = np.random.randn(16000 * 2).astype(np.float32)
        transcript = engine.transcribe(
            audio, sample_rate=16000,
            beam_size=10, best_of=3, temperature=0.1,
        )
        assert isinstance(transcript, Transcript)

    def test_language_setting(self):
        engine = TranscriptionEngine(language="fr")
        assert engine.language == "fr"
        audio = np.random.randn(16000).astype(np.float32)
        transcript = engine._mock_transcribe(audio, 16000)
        assert transcript.language == "fr"


class TestRealtimeTranscriber:

    def test_process_single_chunk(self):
        rt = RealtimeTranscriber(model_size="base", chunk_duration_ms=1000)
        chunk = np.random.randn(16000).astype(np.float32)
        segments = rt.process_chunk(chunk, sample_rate=16000)

        assert isinstance(segments, list)

    def test_process_multiple_chunks(self):
        rt = RealtimeTranscriber(model_size="base", chunk_duration_ms=1000)
        audio = np.random.randn(16000 * 4).astype(np.float32)

        all_segments = []
        for i in range(4):
            chunk = audio[i * 16000:(i + 1) * 16000]
            segments = rt.process_chunk(chunk, sample_rate=16000)
            all_segments.extend(segments)

        transcript = rt.get_full_transcript()
        assert isinstance(transcript, Transcript)
        assert transcript.duration == pytest.approx(4.0)

    def test_reset(self):
        rt = RealtimeTranscriber(model_size="base")
        chunk = np.random.randn(16000).astype(np.float32)
        rt.process_chunk(chunk, sample_rate=16000)

        rt.reset()
        assert rt._total_duration == 0.0
        assert rt._all_segments == []
        assert rt._context_buffer is None
        assert rt._prev_text_tail == ""

    def test_get_full_transcript_empty(self):
        rt = RealtimeTranscriber(model_size="base")
        transcript = rt.get_full_transcript()
        assert isinstance(transcript, Transcript)
        assert transcript.duration == 0.0
        assert len(transcript.segments) == 0

    def test_deduplication(self):
        rt = RealtimeTranscriber(model_size="base")
        seg = TranscriptSegment(text="hello world test", start=0.0, end=1.0)

        rt._prev_text_tail = "hello world"
        deduped = rt._deduplicate(seg)
        assert "hello" not in deduped.text.lower().split()[:2] or deduped.text == "test"

    def test_deduplication_no_overlap(self):
        rt = RealtimeTranscriber(model_size="base")
        seg = TranscriptSegment(text="completely new text", start=0.0, end=1.0)

        rt._prev_text_tail = "other words"
        deduped = rt._deduplicate(seg)
        assert deduped.text == "completely new text"

    def test_deduplication_empty_tail(self):
        rt = RealtimeTranscriber(model_size="base")
        seg = TranscriptSegment(text="some text", start=0.0, end=1.0)

        rt._prev_text_tail = ""
        deduped = rt._deduplicate(seg)
        assert deduped.text == "some text"


class TestBatchTranscriber:

    def test_transcribe(self):
        bt = BatchTranscriber(model_size="large-v3")
        audio = np.random.randn(16000 * 5).astype(np.float32)
        transcript = bt.transcribe(audio, sample_rate=16000)

        assert isinstance(transcript, Transcript)
        assert transcript.duration == pytest.approx(5.0)

    def test_uses_high_accuracy_defaults(self):
        bt = BatchTranscriber(model_size="large-v3")
        assert bt.engine.model_size == "large-v3"
