"""Parameter optimization for transcription and diarization."""

import csv
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class OptimizationTarget:
    target_wer: float = 0.05
    target_der: float = 0.10
    target_speaker_accuracy: float = 0.95
    wer_weight: float = 0.4
    der_weight: float = 0.4
    speaker_weight: float = 0.2

    def compute_score(self, wer: float, der: float, speaker_acc: float) -> float:
        wer_score = max(0, wer - self.target_wer) / max(self.target_wer, 1e-8)
        der_score = max(0, der - self.target_der) / max(self.target_der, 1e-8)
        spk_score = max(0, self.target_speaker_accuracy - speaker_acc) / max(self.target_speaker_accuracy, 1e-8)
        return wer_score * self.wer_weight + der_score * self.der_weight + spk_score * self.speaker_weight

    def is_satisfied(self, wer: float, der: float, speaker_acc: float) -> bool:
        return wer <= self.target_wer and der <= self.target_der and speaker_acc >= self.target_speaker_accuracy


@dataclass
class ParameterRange:
    name: str
    min_value: float
    max_value: float
    step: Optional[float] = None
    is_int: bool = False

    def sample(self) -> float:
        value = np.random.uniform(self.min_value, self.max_value)
        if self.step:
            value = round(value / self.step) * self.step
        if self.is_int:
            value = int(round(value))
        return value

    def grid(self, num_points: int = 5) -> list:
        values = np.linspace(self.min_value, self.max_value, num_points)
        if self.is_int:
            values = [int(round(v)) for v in values]
        return list(values)


@dataclass
class OptimizationResult:
    parameters: dict
    wer: float
    der: float
    speaker_accuracy: float
    score: float
    iteration: int
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "parameters": self.parameters,
            "wer": self.wer,
            "der": self.der,
            "speaker_accuracy": self.speaker_accuracy,
            "score": self.score,
            "iteration": self.iteration,
            "timestamp": self.timestamp,
        }


@dataclass
class OptimizationHistory:
    results: list[OptimizationResult] = field(default_factory=list)
    best_result: Optional[OptimizationResult] = None

    def add(self, result: OptimizationResult):
        self.results.append(result)
        if self.best_result is None or result.score < self.best_result.score:
            self.best_result = result
            logger.info(f"New best score: {result.score:.4f} (WER={result.wer:.4f}, DER={result.der:.4f})")

    def save(self, path: Path | str):
        data = {
            "results": [r.to_dict() for r in self.results],
            "best_result": self.best_result.to_dict() if self.best_result else None,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> 'OptimizationHistory':
        with open(path, 'r') as f:
            data = json.load(f)
        history = cls()
        for r in data.get("results", []):
            result = OptimizationResult(**r)
            history.results.append(result)
        if data.get("best_result"):
            history.best_result = OptimizationResult(**data["best_result"])
        return history


class MetricsLogger:
    """Appends iteration metrics to both a CSV and a Markdown log file."""

    CSV_FIELDS = [
        "timestamp", "iteration", "strategy",
        "wer", "der", "speaker_accuracy", "score",
        "beam_size", "best_of", "temperature", "vad_threshold",
        "clustering_threshold", "min_speakers", "max_speakers",
        "similarity_threshold", "notes",
    ]

    def __init__(self, results_dir: Path | str):
        self.results_dir = Path(results_dir)
        self.csv_path = self.results_dir / "metrics_log.csv"
        self.md_path = self.results_dir / "metrics_log.md"

    def append(self, result: "OptimizationResult", strategy: str = "", notes: str = ""):
        self._append_csv(result, strategy, notes)
        self._rewrite_md(result, strategy)

    def _append_csv(self, result: "OptimizationResult", strategy: str, notes: str):
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            if write_header:
                writer.writeheader()
            p = result.parameters
            writer.writerow({
                "timestamp": result.timestamp,
                "iteration": result.iteration,
                "strategy": strategy,
                "wer": f"{result.wer:.4f}",
                "der": f"{result.der:.4f}",
                "speaker_accuracy": f"{result.speaker_accuracy:.4f}",
                "score": f"{result.score:.4f}",
                "beam_size": p.get("beam_size", ""),
                "best_of": p.get("best_of", ""),
                "temperature": p.get("temperature", ""),
                "vad_threshold": p.get("vad_threshold", ""),
                "clustering_threshold": p.get("clustering_threshold", ""),
                "min_speakers": p.get("min_speakers", ""),
                "max_speakers": p.get("max_speakers", ""),
                "similarity_threshold": p.get("similarity_threshold", ""),
                "notes": notes,
            })

    def _rewrite_md(self, latest: "OptimizationResult", strategy: str):
        # Read all existing CSV rows for history table
        rows = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        best = min(rows, key=lambda r: float(r["score"])) if rows else None

        def status(val, target, lower_is_better=True):
            try:
                v = float(val)
                t = float(target)
                ok = v <= t if lower_is_better else v >= t
                return "✅" if ok else "❌"
            except (ValueError, TypeError):
                return "⏳"

        best_wer = best["wer"] if best else "—"
        best_der = best["der"] if best else "—"
        best_spk = best["speaker_accuracy"] if best else "—"

        lines = [
            "# Transcription Lab — Metrics Log",
            "",
            "Targets: **WER < 5%** | **DER < 10%** | **Speaker Accuracy > 95%**",
            "",
            "Updated automatically after every optimization iteration.",
            "",
            "---",
            "",
            "## Best Result So Far",
            "",
            "| Metric | Value | Target | Status |",
            "|---|---|---|---|",
            f"| WER | {f'{float(best_wer):.2%}' if best else '—'} | < 5% | {status(best_wer, 0.05)} |",
            f"| DER | {f'{float(best_der):.2%}' if best else '—'} | < 10% | {status(best_der, 0.10)} |",
            f"| Speaker Accuracy | {f'{float(best_spk):.2%}' if best else '—'} | > 95% | {status(best_spk, 0.95, lower_is_better=False)} |",
            "",
            "---",
            "",
            "## Iteration History",
            "",
            "| # | Time | WER | DER | Spk Acc | Score | Strategy | Key Params |",
            "|---|---|---|---|---|---|---|---|",
        ]

        for r in rows[-100:]:  # Cap at last 100 rows
            p_summary = (
                f"beam={r.get('beam_size','')} temp={r.get('temperature','')} "
                f"vad={r.get('vad_threshold','')}"
            ).strip()
            ts = r["timestamp"][:16].replace("T", " ") if r.get("timestamp") else "—"
            lines.append(
                f"| {r['iteration']} | {ts} "
                f"| {float(r['wer']):.2%} | {float(r['der']):.2%} "
                f"| {float(r['speaker_accuracy']):.2%} | {float(r['score']):.4f} "
                f"| {r.get('strategy','')} | {p_summary} |"
            )

        lines += [
            "",
            "---",
            "",
            "## Notes",
            "",
            "- Log updated after every completed test iteration",
            "- Score = weighted miss from targets (0.0 = all targets met)",
            "- Params logged: beam_size, best_of, temperature, vad_threshold, clustering_threshold, similarity_threshold",
        ]

        self.md_path.write_text("\n".join(lines) + "\n")


class ParameterOptimizer:
    DEFAULT_RANGES = {
        "beam_size": ParameterRange("beam_size", 1, 15, is_int=True),
        "best_of": ParameterRange("best_of", 1, 15, is_int=True),
        "temperature": ParameterRange("temperature", 0.0, 0.5, step=0.1),
        "vad_threshold": ParameterRange("vad_threshold", 0.3, 0.8, step=0.05),
        "min_speech_duration_ms": ParameterRange("min_speech_duration_ms", 100, 500, is_int=True),
        "clustering_threshold": ParameterRange("clustering_threshold", 0.3, 0.8, step=0.05),
        "min_speakers": ParameterRange("min_speakers", 1, 5, is_int=True),
        "max_speakers": ParameterRange("max_speakers", 5, 15, is_int=True),
        "similarity_threshold": ParameterRange("similarity_threshold", 0.5, 0.9, step=0.05),
    }

    def __init__(
        self,
        evaluate_fn: Callable[[dict], tuple[float, float, float]],
        parameter_ranges: Optional[dict[str, ParameterRange]] = None,
        target: Optional[OptimizationTarget] = None,
        results_dir: Path | str = "results",
        resume_from: Optional[Path | str] = None,
    ):
        self.evaluate_fn = evaluate_fn
        self.parameter_ranges = parameter_ranges or self.DEFAULT_RANGES
        self.target = target or OptimizationTarget()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = MetricsLogger(self.results_dir)
        self._current_strategy = "unknown"

        if resume_from and Path(resume_from).exists():
            self.history = OptimizationHistory.load(resume_from)
            logger.info(f"Resumed from {resume_from} with {len(self.history.results)} prior results")
        else:
            self.history = OptimizationHistory()

    def optimize(
        self,
        max_iterations: int = 100,
        parameters_to_tune: Optional[list[str]] = None,
        base_params: Optional[dict] = None,
        strategy: str = "bayesian",
    ) -> OptimizationResult:
        if parameters_to_tune is None:
            parameters_to_tune = list(self.parameter_ranges.keys())
        base_params = base_params or {}

        logger.info(f"Strategy: {strategy}, max_iter: {max_iterations}, params: {parameters_to_tune}")

        self._current_strategy = strategy
        if strategy == "grid":
            return self._grid_search(max_iterations, parameters_to_tune, base_params)
        elif strategy == "random":
            return self._random_search(max_iterations, parameters_to_tune, base_params)
        else:
            return self._bayesian_search(max_iterations, parameters_to_tune, base_params)

    def _evaluate_and_record(self, params: dict, iteration: int) -> OptimizationResult:
        try:
            wer, der, speaker_acc = self.evaluate_fn(params)
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            wer, der, speaker_acc = 1.0, 1.0, 0.0

        score = self.target.compute_score(wer, der, speaker_acc)
        result = OptimizationResult(
            parameters=params.copy(), wer=wer, der=der, speaker_accuracy=speaker_acc,
            score=score, iteration=iteration, timestamp=datetime.now().isoformat(),
        )
        self.history.add(result)
        self.metrics_logger.append(result, strategy=self._current_strategy)
        logger.info(f"Iter {iteration}: WER={wer:.4f}, DER={der:.4f}, SpkAcc={speaker_acc:.4f}, Score={score:.4f}")
        return result

    def _grid_search(self, max_iterations, parameters_to_tune, base_params):
        from itertools import product

        param_grids = {}
        for name in parameters_to_tune:
            if name in self.parameter_ranges:
                grid_size = min(5, max(2, int(max_iterations ** (1 / max(len(parameters_to_tune), 1)))))
                param_grids[name] = self.parameter_ranges[name].grid(grid_size)

        keys = list(param_grids.keys())
        combinations = list(product(*[param_grids[k] for k in keys]))[:max_iterations]

        for i, values in enumerate(combinations):
            params = base_params.copy()
            for k, v in zip(keys, values):
                params[k] = v
            result = self._evaluate_and_record(params, i)
            if self.target.is_satisfied(result.wer, result.der, result.speaker_accuracy):
                break

        self._save_results("grid_search")
        return self.history.best_result

    def _random_search(self, max_iterations, parameters_to_tune, base_params):
        for i in range(max_iterations):
            params = base_params.copy()
            for name in parameters_to_tune:
                if name in self.parameter_ranges:
                    params[name] = self.parameter_ranges[name].sample()
            result = self._evaluate_and_record(params, i)
            if self.target.is_satisfied(result.wer, result.der, result.speaker_accuracy):
                break

        self._save_results("random_search")
        return self.history.best_result

    def _bayesian_search(self, max_iterations, parameters_to_tune, base_params):
        exploration_fraction = 0.3
        exploration_count = int(max_iterations * exploration_fraction)

        for i in range(exploration_count):
            params = base_params.copy()
            for name in parameters_to_tune:
                if name in self.parameter_ranges:
                    params[name] = self.parameter_ranges[name].sample()
            result = self._evaluate_and_record(params, i)
            if self.target.is_satisfied(result.wer, result.der, result.speaker_accuracy):
                self._save_results("bayesian_search")
                return self.history.best_result

        for i in range(exploration_count, max_iterations):
            sorted_results = sorted(self.history.results, key=lambda r: r.score)
            top_results = sorted_results[:min(5, len(sorted_results))]
            base_result = top_results[np.random.randint(len(top_results))]
            params = base_result.parameters.copy()

            for name in parameters_to_tune:
                if name in self.parameter_ranges and name in params:
                    r = self.parameter_ranges[name]
                    spread = (r.max_value - r.min_value) * 0.1
                    new_val = params[name] + np.random.normal(0, spread)
                    new_val = np.clip(new_val, r.min_value, r.max_value)
                    if r.is_int:
                        new_val = int(round(new_val))
                    params[name] = new_val

            result = self._evaluate_and_record(params, i)
            if self.target.is_satisfied(result.wer, result.der, result.speaker_accuracy):
                break

        self._save_results("bayesian_search")
        return self.history.best_result

    def _save_results(self, name: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.results_dir / f"{name}_{ts}.json"
        self.history.save(path)
        logger.info(f"Results saved to: {path}")


class ExperimentRunner:
    """Runs optimization experiments with cached models."""

    def __init__(
        self,
        audio_dir: Path | str,
        transcripts_dir: Path | str,
        voiceprints_dir: Path | str,
        results_dir: Path | str,
    ):
        self.audio_dir = Path(audio_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.voiceprints_dir = Path(voiceprints_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cached instances (avoid reloading models per iteration)
        self._transcriber_cache: dict[str, "TranscriptionEngine"] = {}
        self._diarizer: Optional["DiarizationEngine"] = None
        self._voiceprints: Optional["VoicePrintDatabase"] = None

    def _get_transcriber(self, model_size: str, device: str, language: str):
        """Cache transcription engines by model_size to avoid reloading."""
        from .transcriber import TranscriptionEngine

        key = f"{model_size}_{device}_{language}"
        if key not in self._transcriber_cache:
            self._transcriber_cache[key] = TranscriptionEngine(
                model_size=model_size, device=device, language=language,
            )
        return self._transcriber_cache[key]

    def _get_diarizer(self, device: str):
        """Cache the diarization engine."""
        from .diarization import DiarizationEngine

        if self._diarizer is None:
            self._diarizer = DiarizationEngine(device=device)
        return self._diarizer

    def _get_voiceprints(self):
        """Cache the voice print database."""
        from .diarization import VoicePrintDatabase

        if self._voiceprints is None:
            self._voiceprints = VoicePrintDatabase(self.voiceprints_dir)
        return self._voiceprints

    def load_test_case(self, name: str) -> dict:
        mic_path = self.audio_dir / f"{name}_mic.wav"
        system_path = self.audio_dir / f"{name}_system.wav"
        transcript_path = self.transcripts_dir / f"{name}_transcript.txt"

        found = {
            "mic": mic_path.exists(),
            "system": system_path.exists(),
            "transcript": transcript_path.exists(),
        }

        if not found["transcript"]:
            raise FileNotFoundError(f"Transcript not found for test case: {name}")
        if not found["mic"] and not found["system"]:
            raise FileNotFoundError(f"No audio files found for test case: {name}")

        return {
            "name": name,
            "mic_path": mic_path if found["mic"] else None,
            "system_path": system_path if found["system"] else None,
            "transcript_path": transcript_path,
        }

    def list_test_cases(self) -> list[str]:
        if not self.transcripts_dir.exists():
            return []
        transcript_files = list(self.transcripts_dir.glob("*_transcript.txt"))
        names = []
        for f in transcript_files:
            name = f.stem.replace("_transcript", "")
            # Validate that at least one audio file exists
            mic = self.audio_dir / f"{name}_mic.wav"
            sys = self.audio_dir / f"{name}_system.wav"
            if mic.exists() or sys.exists():
                names.append(name)
        return names

    def run_evaluation(self, test_case: dict, params: dict) -> tuple[float, float, float]:
        """Run full evaluation pipeline. Returns (wer, der, speaker_accuracy)."""
        from .audio import AudioMerger, AudioLoader
        from .evaluation import TranscriptionEvaluator, DiarizationEvaluator, GroundTruthParser
        from .diarization import SpeakerIdentifier

        # Load audio
        mic_path = test_case["mic_path"]
        sys_path = test_case["system_path"]

        if mic_path and sys_path:
            merger = AudioMerger()
            merged = merger.merge(
                mic_path, sys_path,
                mic_weight=params.get("mic_weight", 0.7),
                system_weight=params.get("system_weight", 0.3),
            )
            audio = merged.data
            sr = merged.sample_rate
        else:
            # Only one track available
            loader = AudioLoader()
            path = mic_path or sys_path
            audio, sr = loader.load(path)

        # Ground truth
        parser = GroundTruthParser()
        ground_truth = parser.parse_file(test_case["transcript_path"])

        # Transcribe (cached engine)
        model_size = params.get("model_size", "base")
        device = params.get("device", "auto")
        language = params.get("language", "en")
        transcriber = self._get_transcriber(model_size, device, language)

        transcript = transcriber.transcribe(
            audio, sr,
            beam_size=params.get("beam_size", 5),
            best_of=params.get("best_of", 5),
            temperature=params.get("temperature", 0.0),
            vad_filter=True,
            word_timestamps=True,
        )

        # Diarize (cached engine)
        diarizer = self._get_diarizer(device)
        diarization = diarizer.diarize(
            audio, sr,
            min_speakers=params.get("min_speakers", 1),
            max_speakers=params.get("max_speakers", 10),
        )

        # Speaker identification
        voiceprints = self._get_voiceprints()
        identifier = SpeakerIdentifier(
            diarizer, voiceprints,
            similarity_threshold=params.get("similarity_threshold", 0.75),
        )
        speaker_mapping = identifier.identify_speakers(audio, sr, diarization)

        # Evaluate
        trans_eval = TranscriptionEvaluator()
        trans_metrics = trans_eval.evaluate(transcript.text, ground_truth.text)
        wer = trans_metrics["wer"]

        diar_eval = DiarizationEvaluator(collar=params.get("collar", 0.25))
        diar_metrics = diar_eval.evaluate(diarization.segments, ground_truth, speaker_mapping)
        der = diar_metrics["der"]

        # Speaker accuracy
        correct = sum(1 for name in speaker_mapping.values() if name in ground_truth.speakers)
        speaker_accuracy = correct / max(len(speaker_mapping), 1)

        return wer, der, speaker_accuracy

    def run_optimization(
        self,
        test_cases: Optional[list[str]] = None,
        max_iterations: int = 100,
        strategy: str = "bayesian",
        resume_from: Optional[Path | str] = None,
    ) -> OptimizationResult:
        if test_cases is None:
            test_cases = self.list_test_cases()
        if not test_cases:
            raise ValueError("No test cases available")

        loaded_cases = [self.load_test_case(name) for name in test_cases]

        def evaluate_fn(params: dict) -> tuple[float, float, float]:
            wers, ders, spk_accs = [], [], []
            for case in loaded_cases:
                try:
                    wer, der, spk_acc = self.run_evaluation(case, params)
                    wers.append(wer)
                    ders.append(der)
                    spk_accs.append(spk_acc)
                except Exception as e:
                    logger.warning(f"Failed to evaluate {case['name']}: {e}")
                    wers.append(1.0)
                    ders.append(1.0)
                    spk_accs.append(0.0)

            return float(np.mean(wers)), float(np.mean(ders)), float(np.mean(spk_accs))

        optimizer = ParameterOptimizer(
            evaluate_fn=evaluate_fn,
            results_dir=self.results_dir,
            resume_from=resume_from,
        )

        return optimizer.optimize(
            max_iterations=max_iterations,
            strategy=strategy,
        )
