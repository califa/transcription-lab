"""Tests for diarization module."""

import pytest
import numpy as np
from pathlib import Path

from transcription_lab.diarization import (
    SpeakerSegment, DiarizationResult, DiarizationEngine,
    VoicePrintDatabase, SpeakerIdentifier, _sanitize_filename,
)


class TestSpeakerSegment:

    def test_duration(self):
        seg = SpeakerSegment(speaker_id="A", start=1.0, end=3.5)
        assert seg.duration == 2.5


class TestDiarizationResult:

    def test_to_rttm(self):
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker_id="SPEAKER_00", start=1.0, end=5.0),
                SpeakerSegment(speaker_id="SPEAKER_01", start=6.0, end=10.0),
            ],
            num_speakers=2,
            duration=10.0,
        )
        rttm = result.to_rttm()

        assert "SPEAKER_00" in rttm
        assert "SPEAKER_01" in rttm
        lines = rttm.strip().split("\n")
        assert len(lines) == 2

    def test_get_speaker_at_time(self):
        result = DiarizationResult(
            segments=[
                SpeakerSegment(speaker_id="A", start=0.0, end=5.0),
                SpeakerSegment(speaker_id="B", start=5.0, end=10.0),
            ],
            num_speakers=2, duration=10.0,
        )
        assert result.get_speaker_at_time(2.5) == "A"
        assert result.get_speaker_at_time(7.0) == "B"
        assert result.get_speaker_at_time(11.0) is None


class TestDiarizationEngine:

    def test_mock_diarize_is_deterministic(self):
        engine = DiarizationEngine()
        audio = np.random.randn(16000 * 10).astype(np.float32)

        r1 = engine._mock_diarize(audio, 16000)
        r2 = engine._mock_diarize(audio, 16000)

        assert len(r1.segments) == len(r2.segments)
        for s1, s2 in zip(r1.segments, r2.segments):
            assert s1.start == s2.start
            assert s1.end == s2.end

    def test_mock_embedding_is_deterministic(self):
        engine = DiarizationEngine()
        audio = np.random.randn(16000 * 5).astype(np.float32)

        e1 = engine._mock_embedding(audio, 16000, 0.0, 5.0)
        e2 = engine._mock_embedding(audio, 16000, 0.0, 5.0)

        np.testing.assert_array_equal(e1, e2)

    def test_mock_embedding_respects_start_end(self):
        engine = DiarizationEngine()
        # Use audio with enough variance so slicing different regions produces different seeds
        rng = np.random.RandomState(123)
        audio = rng.randn(16000 * 10).astype(np.float32)

        e_start = engine._mock_embedding(audio, 16000, 0.0, 3.0)
        e_end = engine._mock_embedding(audio, 16000, 7.0, 10.0)

        # Non-overlapping segments should give different embeddings
        assert not np.allclose(e_start, e_end)


class TestSanitizeFilename:

    def test_spaces(self):
        assert _sanitize_filename("John Doe") == "John_Doe"

    def test_special_chars(self):
        safe = _sanitize_filename("O'Brien/Jr.")
        assert "/" not in safe
        assert "'" not in safe

    def test_empty(self):
        safe = _sanitize_filename("")
        assert len(safe) > 0  # Falls back to hash


class TestVoicePrintDatabase:

    def test_add_and_get(self, tmp_path):
        db = VoicePrintDatabase(tmp_path / "vp")
        embedding = np.random.randn(256).astype(np.float32)

        db.add("Alice", embedding)

        vp = db.get("Alice")
        assert vp is not None
        assert vp.speaker_name == "Alice"
        assert vp.num_samples == 1

    def test_persistence(self, tmp_path):
        """Voice prints should survive reload."""
        vp_dir = tmp_path / "vp"
        embedding = np.random.randn(256).astype(np.float32)

        db1 = VoicePrintDatabase(vp_dir)
        db1.add("Bob", embedding)

        # Create new instance (simulates restart)
        db2 = VoicePrintDatabase(vp_dir)
        vp = db2.get("Bob")

        assert vp is not None
        assert vp.speaker_name == "Bob"
        np.testing.assert_allclose(vp.embedding, embedding, atol=1e-6)

    def test_update_averages(self, tmp_path):
        db = VoicePrintDatabase(tmp_path / "vp")
        e1 = np.ones(256, dtype=np.float32)
        e2 = np.ones(256, dtype=np.float32) * 3

        db.add("Alice", e1)
        db.add("Alice", e2)

        vp = db.get("Alice")
        assert vp.num_samples == 2
        np.testing.assert_allclose(vp.embedding, np.ones(256) * 2.0, atol=1e-6)

    def test_remove(self, tmp_path):
        db = VoicePrintDatabase(tmp_path / "vp")
        db.add("Alice", np.zeros(256))
        db.remove("Alice")

        assert db.get("Alice") is None
        assert "Alice" not in db.list_speakers()

    def test_find_matching(self, tmp_path):
        db = VoicePrintDatabase(tmp_path / "vp")
        emb = np.random.randn(256).astype(np.float32)
        db.add("Alice", emb)

        match = db.find_matching_speaker(emb, threshold=0.9)
        assert match is not None
        assert match[0] == "Alice"
        assert match[1] > 0.99

    def test_find_no_match(self, tmp_path):
        db = VoicePrintDatabase(tmp_path / "vp")
        db.add("Alice", np.ones(256))

        # Very different embedding
        query = -np.ones(256)
        match = db.find_matching_speaker(query, threshold=0.5)
        assert match is None
