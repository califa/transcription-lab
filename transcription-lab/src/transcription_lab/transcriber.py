"""Transcription engine using faster-whisper."""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranscriptWord:
    """A single transcribed word with timing."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptSegment:
    """A segment of transcription."""
    text: str
    start: float
    end: float
    words: list[TranscriptWord] = field(default_factory=list)
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Transcript:
    """Complete transcript with segments."""
    segments: list[TranscriptSegment]
    language: str
    duration: float

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text)

    def to_text_with_timestamps(self) -> str:
        lines = []
        for seg in self.segments:
            speaker = f"{seg.speaker}: " if seg.speaker else ""
            lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {speaker}{seg.text}")
        return "\n".join(lines)


class TranscriptionEngine:
    """Transcription engine using faster-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(f"Loaded faster-whisper model: {self.model_size}")
        except ImportError:
            logger.warning("faster-whisper not installed, using mock transcriber")
            self._model = "mock"

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
        word_timestamps: bool = True,
        **kwargs,
    ) -> Transcript:
        self._load_model()

        if self._model == "mock":
            return self._mock_transcribe(audio, sample_rate)

        audio = audio.astype(np.float32)

        segments_gen, info = self._model.transcribe(
            audio,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            language=self.language,
            **kwargs,
        )

        segments = []
        for seg in segments_gen:
            words = []
            if word_timestamps and seg.words:
                for w in seg.words:
                    words.append(TranscriptWord(
                        word=w.word, start=w.start, end=w.end, probability=w.probability,
                    ))

            segments.append(TranscriptSegment(
                text=seg.text.strip(), start=seg.start, end=seg.end, words=words,
            ))

        duration = len(audio) / sample_rate
        return Transcript(segments=segments, language=info.language, duration=duration)

    def transcribe_file(self, path: Path | str, **kwargs) -> Transcript:
        from .audio import AudioLoader
        loader = AudioLoader(target_sr=16000)
        audio, sr = loader.load(path)
        return self.transcribe(audio, sr, **kwargs)

    def _mock_transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        duration = len(audio) / sample_rate
        return Transcript(
            segments=[TranscriptSegment(
                text="[Mock transcription - install faster-whisper for real results]",
                start=0.0, end=duration, words=[],
            )],
            language=self.language,
            duration=duration,
        )


class RealtimeTranscriber:
    """
    Simulates real-time transcription by processing audio chunks.
    Deduplicates text that spans context boundaries.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        chunk_duration_ms: int = 2000,
        context_duration_ms: int = 500,
    ):
        self.engine = TranscriptionEngine(
            model_size=model_size, device=device,
            compute_type=compute_type, language=language,
        )
        self.chunk_duration_ms = chunk_duration_ms
        self.context_duration_ms = context_duration_ms

        # State
        self._context_buffer: Optional[np.ndarray] = None
        self._total_duration = 0.0
        self._all_segments: list[TranscriptSegment] = []
        self._prev_text_tail: str = ""  # Last few words for dedup

    def reset(self):
        self._context_buffer = None
        self._total_duration = 0.0
        self._all_segments = []
        self._prev_text_tail = ""

    def process_chunk(
        self,
        chunk: np.ndarray,
        sample_rate: int = 16000,
        **kwargs,
    ) -> list[TranscriptSegment]:
        # Prepend context from previous chunk
        if self._context_buffer is not None:
            audio = np.concatenate([self._context_buffer, chunk])
            context_duration = len(self._context_buffer) / sample_rate
        else:
            audio = chunk
            context_duration = 0.0

        transcript = self.engine.transcribe(
            audio, sample_rate=sample_rate, word_timestamps=True, **kwargs,
        )

        # Filter to only new content (after context)
        new_segments = []
        for seg in transcript.segments:
            if seg.end <= context_duration:
                continue  # Entirely within context, skip

            # Adjust timing to global timeline
            adj_start = max(0, seg.start - context_duration) + self._total_duration
            adj_end = seg.end - context_duration + self._total_duration

            adj_words = [
                TranscriptWord(
                    word=w.word,
                    start=max(0, w.start - context_duration) + self._total_duration,
                    end=w.end - context_duration + self._total_duration,
                    probability=w.probability,
                )
                for w in seg.words
                if w.end > context_duration
            ]

            new_seg = TranscriptSegment(
                text=seg.text.strip(), start=adj_start, end=adj_end,
                words=adj_words, speaker=seg.speaker,
            )

            # Deduplicate: remove prefix overlap with previous tail
            new_seg = self._deduplicate(new_seg)
            if new_seg.text:
                new_segments.append(new_seg)

        # Update state
        context_samples = int(self.context_duration_ms * sample_rate / 1000)
        if len(chunk) > context_samples:
            self._context_buffer = chunk[-context_samples:]
        else:
            self._context_buffer = chunk.copy()
        self._total_duration += len(chunk) / sample_rate

        # Track last words for next dedup
        if new_segments:
            last_text = new_segments[-1].text
            words = last_text.split()
            self._prev_text_tail = " ".join(words[-5:]) if len(words) >= 5 else last_text

        self._all_segments.extend(new_segments)
        return new_segments

    def _deduplicate(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Remove duplicated prefix from context overlap."""
        if not self._prev_text_tail or not segment.text:
            return segment

        prev_words = self._prev_text_tail.lower().split()
        new_words = segment.text.split()
        new_words_lower = [w.lower() for w in new_words]

        # Find longest common prefix between new segment and tail of previous
        best_overlap = 0
        for overlap_len in range(1, min(len(prev_words), len(new_words)) + 1):
            if prev_words[-overlap_len:] == new_words_lower[:overlap_len]:
                best_overlap = overlap_len

        if best_overlap > 0:
            deduped_text = " ".join(new_words[best_overlap:])
            deduped_words = segment.words[best_overlap:] if len(segment.words) >= best_overlap else segment.words
            return TranscriptSegment(
                text=deduped_text,
                start=segment.start,
                end=segment.end,
                words=deduped_words,
                speaker=segment.speaker,
            )

        return segment

    def get_full_transcript(self) -> Transcript:
        return Transcript(
            segments=self._all_segments,
            language=self.engine.language,
            duration=self._total_duration,
        )


class BatchTranscriber:
    """High-accuracy batch transcription for post-processing."""

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
    ):
        self.engine = TranscriptionEngine(
            model_size=model_size, device=device,
            compute_type=compute_type, language=language,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 10,
        best_of: int = 10,
    ) -> Transcript:
        return self.engine.transcribe(
            audio, sample_rate=sample_rate,
            beam_size=beam_size, best_of=best_of,
            temperature=0.0, vad_filter=True, word_timestamps=True,
        )

    def transcribe_file(self, path: Path | str, **kwargs) -> Transcript:
        return self.engine.transcribe_file(path, **kwargs)
