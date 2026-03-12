"""Audio processing module for handling mic and system audio tracks."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
import soundfile as sf
import librosa


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    data: np.ndarray
    sample_rate: int
    start_time: float
    end_time: float
    source: str  # 'mic', 'system', or 'mixed'

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class MergedAudio:
    """Merged audio from multiple sources."""
    data: np.ndarray
    sample_rate: int
    mic_weight: float
    system_weight: float
    duration: float


class AudioLoader:
    """Loads and preprocesses audio files."""

    # Formats that soundfile supports natively
    SOUNDFILE_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, path: Path | str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in self.SOUNDFILE_EXTS:
            audio, sr = sf.read(path)
        else:
            # Try pydub for MP3, M4A, etc.
            audio, sr = self._load_with_pydub(path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        audio = self._normalize(audio)
        return audio, sr

    def _load_with_pydub(self, path: Path) -> Tuple[np.ndarray, int]:
        """Fallback loader using pydub for MP3, M4A, etc."""
        try:
            from pydub import AudioSegment as PydubSegment
            seg = PydubSegment.from_file(str(path))
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            # Normalize int16 range
            samples = samples / 32768.0
            if seg.channels > 1:
                samples = samples.reshape(-1, seg.channels)
            return samples, sr
        except ImportError:
            raise RuntimeError(
                f"Cannot load {path.suffix} files. Install pydub and ffmpeg: "
                "pip install pydub && apt-get install ffmpeg"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}")

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 1e-8:
            audio = audio / max_val
        return audio


class AudioMerger:
    """Merges mic and system audio tracks."""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.loader = AudioLoader(target_sr)

    def merge(
        self,
        mic_path: Path | str,
        system_path: Path | str,
        mic_weight: float = 0.7,
        system_weight: float = 0.3,
        align: bool = True,
    ) -> MergedAudio:
        mic_audio, _ = self.loader.load(mic_path)
        system_audio, _ = self.loader.load(system_path)

        # Ensure same length (pad shorter one)
        max_len = max(len(mic_audio), len(system_audio))
        if len(mic_audio) < max_len:
            mic_audio = np.pad(mic_audio, (0, max_len - len(mic_audio)))
        if len(system_audio) < max_len:
            system_audio = np.pad(system_audio, (0, max_len - len(system_audio)))

        if align:
            system_audio = self._align_tracks(mic_audio, system_audio)

        # Weighted mix
        merged = mic_audio * mic_weight + system_audio * system_weight

        # Safe normalization (no division by zero)
        max_val = np.max(np.abs(merged))
        if max_val > 1e-8:
            merged = merged / max_val

        duration = len(merged) / self.target_sr

        return MergedAudio(
            data=merged,
            sample_rate=self.target_sr,
            mic_weight=mic_weight,
            system_weight=system_weight,
            duration=duration,
        )

    def _align_tracks(
        self,
        reference: np.ndarray,
        target: np.ndarray,
        max_shift_ms: int = 500,
    ) -> np.ndarray:
        """Align target to reference using FFT-based cross-correlation."""
        from scipy.signal import fftconvolve

        max_shift_samples = int(max_shift_ms * self.target_sr / 1000)

        # Use a portion for alignment (first 10 seconds)
        align_len = min(len(reference), 10 * self.target_sr)
        ref_portion = reference[:align_len]
        tgt_portion = target[:align_len]

        # FFT cross-correlation (O(n log n) instead of O(n^2))
        correlation = fftconvolve(ref_portion, tgt_portion[::-1], mode='full')
        center = len(correlation) // 2

        search_start = max(0, center - max_shift_samples)
        search_end = min(len(correlation), center + max_shift_samples + 1)

        best_idx = search_start + np.argmax(correlation[search_start:search_end])
        shift = best_idx - center

        # Apply shift safely
        if shift == 0:
            return target

        aligned = np.zeros_like(target)
        if shift > 0:
            # Target needs to be shifted right (delayed)
            copy_len = len(target) - shift
            if copy_len > 0:
                aligned[shift:] = target[:copy_len]
        else:
            # Target needs to be shifted left (advanced)
            abs_shift = abs(shift)
            copy_len = len(target) - abs_shift
            if copy_len > 0:
                aligned[:copy_len] = target[abs_shift:]

        return aligned


class AudioChunker:
    """Chunks audio for real-time processing simulation."""

    def __init__(self, chunk_duration_ms: int = 2000, overlap_ms: int = 200):
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms

    def chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        source: str = "mixed",
    ) -> Iterator[AudioSegment]:
        """Yield audio chunks. Last chunk is NOT zero-padded to avoid hallucination."""
        chunk_samples = int(self.chunk_duration_ms * sample_rate / 1000)
        overlap_samples = int(self.overlap_ms * sample_rate / 1000)
        step = chunk_samples - overlap_samples

        position = 0
        while position < len(audio):
            end = min(position + chunk_samples, len(audio))
            chunk_data = audio[position:end]

            # Do NOT zero-pad the last chunk -- whisper handles variable length
            start_time = position / sample_rate
            end_time = end / sample_rate

            yield AudioSegment(
                data=chunk_data,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                source=source,
            )

            position += step


class VAD:
    """Simple Voice Activity Detection."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        min_silence_ms: int = 100,
        sample_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.sample_rate = sample_rate

    def detect(self, audio: np.ndarray) -> list[Tuple[float, float]]:
        """Detect speech segments. Returns list of (start_time, end_time)."""
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        hop_length = int(0.010 * self.sample_rate)    # 10ms

        num_frames = 1 + (len(audio) - frame_length) // hop_length
        if num_frames <= 0:
            return []

        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2))

        if np.max(energy) > 0:
            energy = energy / np.max(energy)

        is_speech = energy > self.threshold

        # Find speech segments
        segments = []
        in_speech = False
        speech_start = 0

        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                speech_start = i
                in_speech = True
            elif not speech and in_speech:
                start_time = speech_start * hop_length / self.sample_rate
                end_time = i * hop_length / self.sample_rate
                duration_ms = (end_time - start_time) * 1000

                if duration_ms >= self.min_speech_ms:
                    segments.append((start_time, end_time))
                in_speech = False

        # Handle trailing speech
        if in_speech:
            start_time = speech_start * hop_length / self.sample_rate
            end_time = len(audio) / self.sample_rate
            duration_ms = (end_time - start_time) * 1000
            if duration_ms >= self.min_speech_ms:
                segments.append((start_time, end_time))

        segments = self._merge_close_segments(segments)
        return segments

    def _merge_close_segments(
        self, segments: list[Tuple[float, float]]
    ) -> list[Tuple[float, float]]:
        if not segments:
            return []

        min_gap_s = self.min_silence_ms / 1000.0
        merged = [segments[0]]

        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end < min_gap_s:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged
