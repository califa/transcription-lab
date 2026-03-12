"""Speaker diarization and voice print management."""

import numpy as np
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
import json
import logging
import hashlib

logger = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Convert speaker name to a safe filename."""
    safe = re.sub(r'[^\w\-.]', '_', name)
    if not safe:
        safe = hashlib.md5(name.encode()).hexdigest()[:12]
    return safe


@dataclass
class SpeakerSegment:
    """A segment attributed to a speaker."""
    speaker_id: str
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class DiarizationResult:
    """Complete diarization result."""
    segments: list[SpeakerSegment]
    num_speakers: int
    duration: float
    embeddings: Optional[dict[str, np.ndarray]] = None

    def get_speaker_timeline(self, speaker_id: str) -> list[Tuple[float, float]]:
        return [(s.start, s.end) for s in self.segments if s.speaker_id == speaker_id]

    def get_speaker_at_time(self, time: float) -> Optional[str]:
        for seg in self.segments:
            if seg.start <= time <= seg.end:
                return seg.speaker_id
        return None

    def to_rttm(self, file_id: str = "file") -> str:
        """Export to RTTM format."""
        lines = []
        for seg in self.segments:
            dur = seg.end - seg.start
            lines.append(
                f"SPEAKER {file_id} 1 {seg.start:.3f} {dur:.3f} "
                f"<NA> <NA> {seg.speaker_id} <NA> <NA>"
            )
        return "\n".join(lines)


@dataclass
class VoicePrint:
    """A speaker's voice print (embedding)."""
    speaker_name: str
    embedding: np.ndarray
    num_samples: int = 1
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "speaker_name": self.speaker_name,
            "embedding": self.embedding.tolist(),
            "num_samples": self.num_samples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'VoicePrint':
        return cls(
            speaker_name=data["speaker_name"],
            embedding=np.array(data["embedding"]),
            num_samples=data.get("num_samples", 1),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


class DiarizationEngine:
    """Speaker diarization using pyannote-audio."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        embedding_model: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        device: str = "auto",
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.device = device
        self.hf_token = hf_token or self._get_hf_token()
        self._pipeline = None
        self._embedding_model = None

    @staticmethod
    def _get_hf_token() -> Optional[str]:
        """Try to read HuggingFace token from environment."""
        import os
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from pyannote.audio import Pipeline
            import torch

            device = self._resolve_device()
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
            self._pipeline.to(torch.device(device))
            logger.info(f"Loaded diarization pipeline: {self.model_name} on {device}")
        except Exception as e:
            logger.warning(f"Could not load pyannote pipeline: {e}")
            self._pipeline = "mock"

    def _load_embedding_model(self):
        if self._embedding_model is not None:
            return
        try:
            from pyannote.audio import Inference
            import torch

            device = self._resolve_device()
            self._embedding_model = Inference(
                self.embedding_model,
                use_auth_token=self.hf_token,
            )
            self._embedding_model.to(torch.device(device))
            logger.info(f"Loaded embedding model: {self.embedding_model} on {device}")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self._embedding_model = "mock"

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        min_speakers: int = 1,
        max_speakers: int = 10,
        **kwargs,
    ) -> DiarizationResult:
        self._load_pipeline()

        if self._pipeline == "mock":
            return self._mock_diarize(audio, sample_rate)

        import torch

        audio_flat = audio.flatten() if audio.ndim > 1 else audio
        audio_tensor = torch.from_numpy(audio_flat).float().unsqueeze(0)
        audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}

        diarization = self._pipeline(
            audio_dict,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            **kwargs,
        )

        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(speaker_id=speaker, start=turn.start, end=turn.end))

        speakers = list(set(s.speaker_id for s in segments))
        duration = len(audio_flat) / sample_rate

        return DiarizationResult(segments=segments, num_speakers=len(speakers), duration=duration)

    def diarize_file(self, path: Path | str, **kwargs) -> DiarizationResult:
        from .audio import AudioLoader
        loader = AudioLoader(target_sr=16000)
        audio, sr = loader.load(path)
        return self.diarize(audio, sr, **kwargs)

    def extract_embedding(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        start: float = 0.0,
        end: Optional[float] = None,
    ) -> np.ndarray:
        """Extract speaker embedding from an audio segment."""
        self._load_embedding_model()

        if self._embedding_model == "mock":
            return self._mock_embedding(audio, sample_rate, start, end)

        import torch
        from pyannote.core import Segment

        audio_flat = audio.flatten() if audio.ndim > 1 else audio
        if end is None:
            end = len(audio_flat) / sample_rate

        audio_tensor = torch.from_numpy(audio_flat).float().unsqueeze(0)
        audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}

        embedding = self._embedding_model.crop(audio_dict, Segment(start, end))
        return embedding.numpy()

    def _mock_diarize(self, audio: np.ndarray, sample_rate: int) -> DiarizationResult:
        """Deterministic mock diarization for testing."""
        audio_flat = audio.flatten() if audio.ndim > 1 else audio
        duration = len(audio_flat) / sample_rate

        rng = np.random.RandomState(42)  # Deterministic
        segments = []
        current = 0.0
        idx = 0
        while current < duration:
            seg_dur = rng.uniform(2.0, 8.0)
            end_time = min(current + seg_dur, duration)
            segments.append(SpeakerSegment(
                speaker_id=f"SPEAKER_{idx:02d}",
                start=current,
                end=end_time,
            ))
            current = end_time + rng.uniform(0.1, 0.5)
            idx = (idx + 1) % 3

        return DiarizationResult(segments=segments, num_speakers=min(3, len(set(s.speaker_id for s in segments))), duration=duration)

    def _mock_embedding(
        self, audio: np.ndarray, sample_rate: int, start: float, end: Optional[float]
    ) -> np.ndarray:
        """Deterministic mock embedding based on audio content."""
        audio_flat = audio.flatten() if audio.ndim > 1 else audio
        s = int(start * sample_rate)
        e = int((end or len(audio_flat) / sample_rate) * sample_rate)
        segment = audio_flat[s:e] if e <= len(audio_flat) else audio_flat[s:]

        # Create a deterministic seed from audio content AND segment length
        if len(segment) > 0:
            content_hash = int(np.abs(segment[:min(1000, len(segment))]).sum() * 10000)
            seed = (content_hash + len(segment) * 7919) % (2**31)
        else:
            seed = 0
        rng = np.random.RandomState(seed)
        return rng.randn(256).astype(np.float32)


class VoicePrintDatabase:
    """Persistent storage for speaker voice prints."""

    def __init__(self, storage_path: Path | str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._voiceprints: dict[str, VoicePrint] = {}
        self._load()

    def _load(self):
        index_path = self.storage_path / "index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
            for name, filename in index.items():
                vp_path = self.storage_path / filename
                if vp_path.exists():
                    with open(vp_path, "r") as f:
                        data = json.load(f)
                    self._voiceprints[name] = VoicePrint.from_dict(data)
            logger.info(f"Loaded {len(self._voiceprints)} voice prints")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load voice print index: {e}")

    def _save_index(self):
        index = {name: f"{_sanitize_filename(name)}.json" for name in self._voiceprints}
        with open(self.storage_path / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    def add(self, speaker_name: str, embedding: np.ndarray, update_if_exists: bool = True):
        from datetime import datetime
        now = datetime.now().isoformat()

        if speaker_name in self._voiceprints and update_if_exists:
            existing = self._voiceprints[speaker_name]
            total_samples = existing.num_samples + 1
            new_embedding = (existing.embedding * existing.num_samples + embedding) / total_samples
            vp = VoicePrint(
                speaker_name=speaker_name,
                embedding=new_embedding,
                num_samples=total_samples,
                created_at=existing.created_at,
                updated_at=now,
            )
        else:
            vp = VoicePrint(
                speaker_name=speaker_name,
                embedding=embedding,
                num_samples=1,
                created_at=now,
                updated_at=now,
            )

        self._voiceprints[speaker_name] = vp

        # Write voice print file first, then index (safer order)
        filename = f"{_sanitize_filename(speaker_name)}.json"
        with open(self.storage_path / filename, "w") as f:
            json.dump(vp.to_dict(), f, indent=2)
        self._save_index()
        logger.info(f"Saved voice print for: {speaker_name}")

    def get(self, speaker_name: str) -> Optional[VoicePrint]:
        return self._voiceprints.get(speaker_name)

    def remove(self, speaker_name: str):
        if speaker_name in self._voiceprints:
            del self._voiceprints[speaker_name]
            filename = f"{_sanitize_filename(speaker_name)}.json"
            vp_path = self.storage_path / filename
            if vp_path.exists():
                vp_path.unlink()
            self._save_index()

    def list_speakers(self) -> list[str]:
        return list(self._voiceprints.keys())

    def find_matching_speaker(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> Optional[Tuple[str, float]]:
        if not self._voiceprints:
            return None

        best_match = None
        best_score = -1.0

        embedding_flat = embedding.flatten()
        for name, vp in self._voiceprints.items():
            score = self._cosine_similarity(embedding_flat, vp.embedding.flatten())
            if score > best_score:
                best_score = score
                best_match = name

        if best_score >= threshold:
            return (best_match, best_score)
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class SpeakerIdentifier:
    """Identifies speakers using diarization + voice print matching."""

    def __init__(
        self,
        diarization_engine: DiarizationEngine,
        voiceprint_db: VoicePrintDatabase,
        similarity_threshold: float = 0.75,
    ):
        self.diarization = diarization_engine
        self.voiceprints = voiceprint_db
        self.similarity_threshold = similarity_threshold

    def identify_speakers(
        self,
        audio: np.ndarray,
        sample_rate: int,
        diarization_result: Optional[DiarizationResult] = None,
    ) -> dict[str, str]:
        """Map anonymous speaker IDs to known names."""
        if diarization_result is None:
            diarization_result = self.diarization.diarize(audio, sample_rate)

        audio_flat = audio.flatten() if audio.ndim > 1 else audio
        speaker_ids = list(set(s.speaker_id for s in diarization_result.segments))

        mapping = {}
        used_names = set()

        for spk_id in speaker_ids:
            segments = [s for s in diarization_result.segments if s.speaker_id == spk_id]
            longest = max(segments, key=lambda s: s.duration)

            # Slice the audio for embedding extraction -- pass full audio with start/end
            embedding = self.diarization.extract_embedding(
                audio_flat, sample_rate, start=longest.start, end=longest.end
            )

            match = self.voiceprints.find_matching_speaker(embedding, self.similarity_threshold)

            if match and match[0] not in used_names:
                mapping[spk_id] = match[0]
                used_names.add(match[0])
            else:
                mapping[spk_id] = spk_id

        return mapping

    def enroll_speaker(
        self,
        speaker_name: str,
        audio: np.ndarray,
        sample_rate: int,
        start: float = 0.0,
        end: Optional[float] = None,
    ):
        embedding = self.diarization.extract_embedding(audio, sample_rate, start, end)
        self.voiceprints.add(speaker_name, embedding)
