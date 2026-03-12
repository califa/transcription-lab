"""Tests for audio processing module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf


class TestAudioLoader:
    def test_load_mono_audio(self):
        from transcription_lab.audio import AudioLoader

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sr = 16000
            audio = np.random.randn(int(sr * 2.0)).astype(np.float32)
            sf.write(f.name, audio, sr)

            loader = AudioLoader(target_sr=16000)
            loaded, loaded_sr = loader.load(f.name)

            assert loaded_sr == 16000
            assert loaded.ndim == 1
            assert abs(len(loaded) - len(audio)) < 100

    def test_load_stereo_converts_to_mono(self):
        from transcription_lab.audio import AudioLoader

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sr = 16000
            audio = np.random.randn(int(sr * 2.0), 2).astype(np.float32)
            sf.write(f.name, audio, sr)

            loader = AudioLoader(target_sr=16000)
            loaded, _ = loader.load(f.name)

            assert loaded.ndim == 1

    def test_resamples_to_target(self):
        from transcription_lab.audio import AudioLoader

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            original_sr = 44100
            duration = 2.0
            audio = np.random.randn(int(original_sr * duration)).astype(np.float32)
            sf.write(f.name, audio, original_sr)

            loader = AudioLoader(target_sr=16000)
            loaded, sr = loader.load(f.name)

            assert sr == 16000
            expected = int(duration * 16000)
            assert abs(len(loaded) - expected) < 100

    def test_normalize_handles_silence(self):
        from transcription_lab.audio import AudioLoader

        loader = AudioLoader()
        silence = np.zeros(16000)
        normalized = loader._normalize(silence)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_file_not_found(self):
        from transcription_lab.audio import AudioLoader

        loader = AudioLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.wav")


class TestAudioMerger:
    def test_merge_equal_length(self):
        from transcription_lab.audio import AudioMerger

        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            duration = 2.0
            samples = int(sr * duration)

            mic = np.random.randn(samples).astype(np.float32)
            system = np.random.randn(samples).astype(np.float32)

            mic_path = Path(tmpdir) / "mic.wav"
            system_path = Path(tmpdir) / "system.wav"
            sf.write(mic_path, mic, sr)
            sf.write(system_path, system, sr)

            merger = AudioMerger(target_sr=sr)
            merged = merger.merge(mic_path, system_path, align=False)

            assert merged.sample_rate == sr
            assert abs(merged.duration - duration) < 0.1
            assert len(merged.data) == samples

    def test_merge_different_lengths(self):
        from transcription_lab.audio import AudioMerger

        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            mic = np.random.randn(sr * 3).astype(np.float32)
            system = np.random.randn(sr * 2).astype(np.float32)

            sf.write(Path(tmpdir) / "mic.wav", mic, sr)
            sf.write(Path(tmpdir) / "system.wav", system, sr)

            merger = AudioMerger(target_sr=sr)
            merged = merger.merge(
                Path(tmpdir) / "mic.wav", Path(tmpdir) / "system.wav", align=False,
            )

            assert len(merged.data) == sr * 3

    def test_merge_silence_no_nan(self):
        """Merging silent tracks must not produce NaN."""
        from transcription_lab.audio import AudioMerger

        with tempfile.TemporaryDirectory() as tmpdir:
            sr = 16000
            silence = np.zeros(sr * 2, dtype=np.float32)

            sf.write(Path(tmpdir) / "mic.wav", silence, sr)
            sf.write(Path(tmpdir) / "system.wav", silence, sr)

            merger = AudioMerger(target_sr=sr)
            merged = merger.merge(
                Path(tmpdir) / "mic.wav", Path(tmpdir) / "system.wav", align=False,
            )

            assert not np.any(np.isnan(merged.data))
            assert not np.any(np.isinf(merged.data))


class TestAudioChunker:
    def test_chunk_audio(self):
        from transcription_lab.audio import AudioChunker

        sr = 16000
        audio = np.random.randn(int(sr * 5.0)).astype(np.float32)

        chunker = AudioChunker(chunk_duration_ms=1000, overlap_ms=100)
        chunks = list(chunker.chunk(audio, sr))

        assert len(chunks) > 0
        assert chunks[0].start_time == 0.0

    def test_last_chunk_not_padded(self):
        """Last chunk should NOT be zero-padded (avoids hallucination)."""
        from transcription_lab.audio import AudioChunker

        sr = 16000
        # 2.5 seconds of audio with 1 second chunks = last chunk is 0.5s
        audio = np.random.randn(int(sr * 2.5)).astype(np.float32)

        chunker = AudioChunker(chunk_duration_ms=1000, overlap_ms=0)
        chunks = list(chunker.chunk(audio, sr))

        last_chunk = chunks[-1]
        assert len(last_chunk.data) < sr  # Should be < 1 second (not padded)

    def test_chunk_overlap(self):
        from transcription_lab.audio import AudioChunker

        sr = 16000
        audio = np.random.randn(sr * 3).astype(np.float32)

        chunker = AudioChunker(chunk_duration_ms=1000, overlap_ms=200)
        chunks = list(chunker.chunk(audio, sr))

        for i in range(1, len(chunks)):
            prev_end = chunks[i - 1].end_time
            curr_start = chunks[i].start_time
            assert abs((prev_end - curr_start) - 0.2) < 0.05


class TestVAD:
    def test_detects_speech(self):
        from transcription_lab.audio import VAD

        sr = 16000
        silence = np.zeros(sr)
        speech = np.random.randn(sr * 2) * 0.5
        audio = np.concatenate([silence, speech, silence])

        vad = VAD(threshold=0.3, sample_rate=sr)
        segments = vad.detect(audio)

        assert len(segments) >= 1
        start, end = segments[0]
        assert start < 2.0
        assert end > 2.0

    def test_detects_silence(self):
        from transcription_lab.audio import VAD

        sr = 16000
        audio = np.zeros(sr * 2)
        vad = VAD(threshold=0.3, sample_rate=sr)
        segments = vad.detect(audio)
        assert len(segments) == 0


class TestAudioSegment:
    def test_duration(self):
        from transcription_lab.audio import AudioSegment

        seg = AudioSegment(data=np.zeros(16000), sample_rate=16000, start_time=1.0, end_time=3.5, source="mic")
        assert seg.duration == 2.5
