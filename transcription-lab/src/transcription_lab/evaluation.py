"""Evaluation metrics and ground truth parsing."""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _get_speaker(seg) -> Optional[str]:
    """Extract speaker from a segment regardless of attribute name."""
    if hasattr(seg, 'speaker') and seg.speaker:
        return seg.speaker
    if hasattr(seg, 'speaker_id') and seg.speaker_id:
        return seg.speaker_id
    return None


@dataclass
class GroundTruthSegment:
    """A segment from ground truth transcript."""
    speaker: str
    text: str
    start: float
    end: float
    words: list[Tuple[str, float, float]] = field(default_factory=list)


@dataclass
class GroundTruth:
    """Complete ground truth data for evaluation."""
    segments: list[GroundTruthSegment]
    speakers: list[str]
    duration: float

    @property
    def text(self) -> str:
        return " ".join(seg.text for seg in self.segments if seg.text)

    def get_speaker_timeline(self, speaker: str) -> list[Tuple[float, float]]:
        return [(s.start, s.end) for s in self.segments if s.speaker == speaker]


class GroundTruthParser:
    """
    Parses ground truth transcript files.

    Supported formats:
    1. Timestamped: [00:00:05 - 00:00:12] Alice: Text
    2. Simple: Alice: Text
    3. RTTM: SPEAKER file 1 5.0 7.0 <NA> <NA> Alice <NA> <NA>
    """

    TIMESTAMP_PATTERN = re.compile(
        r'\[(\d+(?::\d+)*(?:\.\d+)?)\s*-\s*(\d+(?::\d+)*(?:\.\d+)?)\]\s*'
        r'([^:]+):\s*(.+)',
        re.MULTILINE
    )

    SIMPLE_PATTERN = re.compile(r'^([^:\[\]]+):\s*(.+)$', re.MULTILINE)

    RTTM_PATTERN = re.compile(
        r'SPEAKER\s+\S+\s+\d+\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+'
        r'<NA>\s+<NA>\s+(\S+)\s+<NA>\s+<NA>'
    )

    def parse_file(self, path: Path | str) -> GroundTruth:
        """Parse a ground truth file."""
        path = Path(path)
        content = path.read_text(encoding='utf-8')
        return self.parse(content)

    def parse(self, content: str) -> GroundTruth:
        """Parse ground truth from a string."""
        if self._is_rttm(content):
            return self._parse_rttm(content)
        elif self._has_timestamps(content):
            return self._parse_timestamped(content)
        else:
            return self._parse_simple(content)

    def _is_rttm(self, content: str) -> bool:
        return content.strip().startswith("SPEAKER")

    def _has_timestamps(self, content: str) -> bool:
        return bool(self.TIMESTAMP_PATTERN.search(content))

    def _parse_time(self, time_str: str) -> float:
        parts = time_str.split(':')
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_str}")

    def _parse_timestamped(self, content: str) -> GroundTruth:
        segments = []
        speakers = set()

        for match in self.TIMESTAMP_PATTERN.finditer(content):
            start_str, end_str, speaker, text = match.groups()
            start = self._parse_time(start_str)
            end = self._parse_time(end_str)
            speaker = speaker.strip()
            text = text.strip()
            segments.append(GroundTruthSegment(speaker=speaker, text=text, start=start, end=end))
            speakers.add(speaker)

        if not segments:
            raise ValueError("No segments found in timestamped transcript")

        segments.sort(key=lambda s: s.start)
        duration = max(s.end for s in segments)
        return GroundTruth(segments=segments, speakers=sorted(speakers), duration=duration)

    def _parse_simple(self, content: str) -> GroundTruth:
        segments = []
        speakers = set()

        for match in self.SIMPLE_PATTERN.finditer(content):
            speaker, text = match.groups()
            speaker = speaker.strip()
            text = text.strip()
            if speaker and text:
                segments.append(GroundTruthSegment(speaker=speaker, text=text, start=0.0, end=0.0))
                speakers.add(speaker)

        return GroundTruth(segments=segments, speakers=sorted(speakers), duration=0.0)

    def _parse_rttm(self, content: str) -> GroundTruth:
        segments = []
        speakers = set()

        for match in self.RTTM_PATTERN.finditer(content):
            start = float(match.group(1))
            dur = float(match.group(2))
            speaker = match.group(3)
            segments.append(GroundTruthSegment(speaker=speaker, text="", start=start, end=start + dur))
            speakers.add(speaker)

        segments.sort(key=lambda s: s.start)
        duration = max(s.end for s in segments) if segments else 0.0
        return GroundTruth(segments=segments, speakers=sorted(speakers), duration=duration)


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics."""
    wer: float
    insertions: int
    deletions: int
    substitutions: int
    der: float
    miss: float
    false_alarm: float
    confusion: float
    speaker_accuracy: float
    speaker_mapping: dict[str, str]
    total_words: int
    total_duration: float

    def to_dict(self) -> dict:
        return {
            "wer": self.wer,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "substitutions": self.substitutions,
            "der": self.der,
            "miss": self.miss,
            "false_alarm": self.false_alarm,
            "confusion": self.confusion,
            "speaker_accuracy": self.speaker_accuracy,
            "speaker_mapping": self.speaker_mapping,
            "total_words": self.total_words,
            "total_duration": self.total_duration,
        }

    def __str__(self) -> str:
        return (
            f"WER: {self.wer:.2%} (I:{self.insertions} D:{self.deletions} S:{self.substitutions})\n"
            f"DER: {self.der:.2%} (Miss:{self.miss:.2%} FA:{self.false_alarm:.2%} Conf:{self.confusion:.2%})\n"
            f"Speaker Accuracy: {self.speaker_accuracy:.2%}"
        )


class TranscriptionEvaluator:
    """Evaluates transcription accuracy using WER and related metrics."""

    def evaluate(
        self,
        hypothesis: str,
        reference: str,
        normalize: bool = True,
    ) -> dict:
        if normalize:
            hypothesis = self._normalize(hypothesis)
            reference = self._normalize(reference)
        try:
            import jiwer
            output = jiwer.process_words(reference, hypothesis)
            return {
                "wer": output.wer,
                "insertions": output.insertions,
                "deletions": output.deletions,
                "substitutions": output.substitutions,
                "hits": getattr(output, 'hits', 0),
                "wip": 1.0 - output.wer,
                "total_words": len(reference.split()),
            }
        except ImportError:
            return self._manual_wer(hypothesis, reference)

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _manual_wer(self, hypothesis: str, reference: str) -> dict:
        hyp_words = hypothesis.split()
        ref_words = reference.split()

        n, m = len(ref_words), len(hyp_words)
        d = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(n + 1):
            d[i, 0] = i
        for j in range(m + 1):
            d[0, j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i, j] = d[i - 1, j - 1]
                else:
                    d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + 1)

        # Backtrace
        i, j = n, m
        insertions = deletions = substitutions = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
                substitutions += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i, j] == d[i, j - 1] + 1:
                insertions += 1
                j -= 1
            else:
                deletions += 1
                i -= 1

        total = max(n, 1)
        wer = (insertions + deletions + substitutions) / total
        return {
            "wer": wer,
            "insertions": insertions,
            "deletions": deletions,
            "substitutions": substitutions,
            "hits": n - substitutions - deletions,
            "wip": 1.0 - wer,
            "total_words": n,
        }


class DiarizationEvaluator:
    """Evaluates diarization accuracy using DER."""

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False):
        self.collar = collar
        self.skip_overlap = skip_overlap

    def evaluate(
        self,
        hypothesis_segments: list,
        reference: GroundTruth,
        speaker_mapping: Optional[dict[str, str]] = None,
    ) -> dict:
        try:
            from pyannote.metrics.diarization import DiarizationErrorRate
            from pyannote.core import Annotation, Segment

            ref_annotation = Annotation()
            for seg in reference.segments:
                ref_annotation[Segment(seg.start, seg.end)] = seg.speaker

            hyp_annotation = Annotation()
            for seg in hypothesis_segments:
                spk = _get_speaker(seg) or "UNKNOWN"
                if speaker_mapping:
                    spk = speaker_mapping.get(spk, spk)
                hyp_annotation[Segment(seg.start, seg.end)] = spk

            metric = DiarizationErrorRate(collar=self.collar, skip_overlap=self.skip_overlap)
            der = metric(ref_annotation, hyp_annotation)
            components = metric.compute_components(ref_annotation, hyp_annotation)
            total = max(components.get("total", 1), 1e-8)

            return {
                "der": der,
                "false_alarm": components.get("false alarm", 0) / total,
                "missed_speech": components.get("miss", 0) / total,
                "speaker_confusion": components.get("confusion", 0) / total,
                "total_speech": total,
            }
        except ImportError:
            return self._manual_der(hypothesis_segments, reference, speaker_mapping)

    def _manual_der(
        self,
        hypothesis_segments: list,
        reference: GroundTruth,
        speaker_mapping: Optional[dict[str, str]] = None,
    ) -> dict:
        duration = reference.duration
        if duration == 0:
            return {"der": 0.0, "false_alarm": 0.0, "missed_speech": 0.0, "speaker_confusion": 0.0, "total_speech": 0.0}

        step = 0.05  # 50ms resolution (improved from 100ms)
        n_samples = int(duration / step)

        miss = fa = conf = scored = 0

        for i in range(n_samples):
            t = i * step

            # Find ALL reference speakers at time t (handle overlap)
            ref_speakers_at_t = set()
            for seg in reference.segments:
                if seg.start <= t < seg.end:
                    ref_speakers_at_t.add(seg.speaker)

            # Find ALL hypothesis speakers at time t
            hyp_speakers_at_t = set()
            for seg in hypothesis_segments:
                spk = _get_speaker(seg) or "UNKNOWN"
                if speaker_mapping:
                    spk = speaker_mapping.get(spk, spk)
                if seg.start <= t < seg.end:
                    hyp_speakers_at_t.add(spk)

            n_ref = len(ref_speakers_at_t)
            n_hyp = len(hyp_speakers_at_t)

            if n_ref > 0:
                scored += n_ref
                matched = len(ref_speakers_at_t & hyp_speakers_at_t)
                miss += n_ref - matched  # Ref speakers not in hypothesis
                conf += len(hyp_speakers_at_t - ref_speakers_at_t)  # Hyp speakers not in ref but ref exists
            else:
                fa += n_hyp  # No reference speech but hypothesis says there is

        scored = max(scored, 1)
        return {
            "der": (miss + fa + conf) / scored,
            "false_alarm": fa / scored,
            "missed_speech": miss / scored,
            "speaker_confusion": conf / scored,
            "total_speech": scored * step,
        }


class Evaluator:
    """Combined transcription and diarization evaluator."""

    def __init__(self, collar: float = 0.25):
        self.collar = collar
        self._trans_eval = TranscriptionEvaluator()
        self._diar_eval = DiarizationEvaluator(collar=collar)

    def evaluate(
        self,
        hypothesis: "Transcript",
        reference: GroundTruth,
        speaker_mapping: Optional[dict[str, str]] = None,
    ) -> EvaluationMetrics:
        # WER
        wer_result = self._trans_eval.evaluate(hypothesis.text, reference.text)

        # DER
        der_result = self._diar_eval.evaluate(hypothesis.segments, reference, speaker_mapping)

        # Speaker mapping
        if speaker_mapping is None:
            speaker_mapping = self._infer_speaker_mapping(hypothesis.segments, reference.segments)

        speaker_accuracy = self._compute_speaker_accuracy(
            hypothesis.segments, reference.segments, speaker_mapping
        )

        return EvaluationMetrics(
            wer=wer_result["wer"],
            insertions=wer_result["insertions"],
            deletions=wer_result["deletions"],
            substitutions=wer_result["substitutions"],
            der=der_result["der"],
            miss=der_result["missed_speech"],
            false_alarm=der_result["false_alarm"],
            confusion=der_result["speaker_confusion"],
            speaker_accuracy=speaker_accuracy,
            speaker_mapping=speaker_mapping,
            total_words=wer_result["total_words"],
            total_duration=reference.duration,
        )

    def _infer_speaker_mapping(
        self,
        hypothesis_segments: list,
        reference_segments: list,
    ) -> dict[str, str]:
        """Infer optimal mapping using Hungarian algorithm when available."""
        hyp_speakers = set()
        ref_speakers = set()

        for seg in hypothesis_segments:
            spk = _get_speaker(seg)
            if spk:
                hyp_speakers.add(spk)
        for seg in reference_segments:
            ref_speakers.add(seg.speaker)

        if not hyp_speakers or not ref_speakers:
            return {}

        hyp_list = sorted(hyp_speakers)
        ref_list = sorted(ref_speakers)

        # Build overlap (cost) matrix
        overlap = np.zeros((len(hyp_list), len(ref_list)))

        for hseg in hypothesis_segments:
            hs = _get_speaker(hseg)
            if hs is None or hs not in hyp_speakers:
                continue
            hi = hyp_list.index(hs)
            for rseg in reference_segments:
                ri = ref_list.index(rseg.speaker)
                seg_start = max(hseg.start, rseg.start)
                seg_end = min(hseg.end, rseg.end)
                if seg_end > seg_start:
                    overlap[hi, ri] += seg_end - seg_start

        # Use Hungarian algorithm for optimal assignment
        try:
            from scipy.optimize import linear_sum_assignment
            # We want to maximize overlap, so negate for minimization
            cost = -overlap
            row_ind, col_ind = linear_sum_assignment(cost)

            mapping = {}
            for r, c in zip(row_ind, col_ind):
                if overlap[r, c] > 0:
                    mapping[hyp_list[r]] = ref_list[c]
                else:
                    mapping[hyp_list[r]] = hyp_list[r]

            # Map any unmatched hyp speakers to themselves
            for hs in hyp_list:
                if hs not in mapping:
                    mapping[hs] = hs
            return mapping

        except ImportError:
            # Fallback to greedy
            return self._greedy_speaker_mapping(hyp_list, ref_list, overlap)

    def _greedy_speaker_mapping(
        self, hyp_list: list, ref_list: list, overlap: np.ndarray
    ) -> dict[str, str]:
        mapping = {}
        used_ref = set()

        # Sort hyp speakers by total overlap descending
        order = sorted(range(len(hyp_list)), key=lambda i: -overlap[i].sum())
        for hi in order:
            best_ri = None
            best_val = 0
            for ri, rs in enumerate(ref_list):
                if ri not in used_ref and overlap[hi, ri] > best_val:
                    best_ri = ri
                    best_val = overlap[hi, ri]
            if best_ri is not None and best_val > 0:
                mapping[hyp_list[hi]] = ref_list[best_ri]
                used_ref.add(best_ri)
            else:
                mapping[hyp_list[hi]] = hyp_list[hi]

        return mapping

    def _compute_speaker_accuracy(
        self,
        hypothesis_segments: list,
        reference_segments: list,
        mapping: dict[str, str],
    ) -> float:
        correct = 0
        total = 0

        for hseg in hypothesis_segments:
            hs = _get_speaker(hseg)
            if hs is None:
                continue
            mapped_speaker = mapping.get(hs, hs)
            mid_time = (hseg.start + hseg.end) / 2

            for rseg in reference_segments:
                if rseg.start <= mid_time <= rseg.end:
                    total += 1
                    if mapped_speaker == rseg.speaker:
                        correct += 1
                    break

        return correct / max(total, 1)
