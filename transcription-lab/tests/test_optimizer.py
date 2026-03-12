"""Tests for optimizer module."""

import pytest
import numpy as np
import tempfile

from transcription_lab.optimizer import (
    OptimizationTarget, ParameterRange, OptimizationResult,
    OptimizationHistory, ParameterOptimizer, MetricsLogger,
)


class TestOptimizationTarget:

    def test_score_all_passing(self):
        target = OptimizationTarget(target_wer=0.10, target_der=0.15, target_speaker_accuracy=0.90)
        score = target.compute_score(wer=0.05, der=0.10, speaker_acc=0.95)
        assert score == 0.0

    def test_score_failing(self):
        target = OptimizationTarget(target_wer=0.05, target_der=0.10, target_speaker_accuracy=0.95)
        score = target.compute_score(wer=0.20, der=0.30, speaker_acc=0.70)
        assert score > 0.0

    def test_is_satisfied(self):
        target = OptimizationTarget(target_wer=0.10, target_der=0.15, target_speaker_accuracy=0.90)
        assert target.is_satisfied(0.05, 0.10, 0.95) is True
        assert target.is_satisfied(0.15, 0.10, 0.95) is False
        assert target.is_satisfied(0.05, 0.10, 0.85) is False


class TestParameterRange:

    def test_sample_in_range(self):
        pr = ParameterRange("test", min_value=0.0, max_value=1.0)
        for _ in range(100):
            assert 0.0 <= pr.sample() <= 1.0

    def test_sample_int(self):
        pr = ParameterRange("test", min_value=1, max_value=10, is_int=True)
        for _ in range(100):
            v = pr.sample()
            assert isinstance(v, int)
            assert 1 <= v <= 10

    def test_grid(self):
        pr = ParameterRange("test", min_value=0.0, max_value=1.0)
        grid = pr.grid(num_points=5)
        assert len(grid) == 5
        assert grid[0] == 0.0
        assert grid[-1] == 1.0

    def test_grid_int(self):
        pr = ParameterRange("test", min_value=1, max_value=10, is_int=True)
        grid = pr.grid(num_points=5)
        for v in grid:
            assert isinstance(v, int)


class TestOptimizationHistory:

    def test_add_and_track_best(self):
        history = OptimizationHistory()

        r1 = OptimizationResult(parameters={"a": 1}, wer=0.20, der=0.25, speaker_accuracy=0.80, score=0.5, iteration=0)
        r2 = OptimizationResult(parameters={"a": 2}, wer=0.10, der=0.15, speaker_accuracy=0.90, score=0.2, iteration=1)

        history.add(r1)
        assert history.best_result == r1

        history.add(r2)
        assert history.best_result == r2

    def test_save_and_load(self, tmp_path):
        history = OptimizationHistory()
        history.add(OptimizationResult(
            parameters={"beam_size": 5}, wer=0.15, der=0.20, speaker_accuracy=0.85,
            score=0.3, iteration=0, timestamp="2024-01-01T00:00:00",
        ))

        path = tmp_path / "history.json"
        history.save(path)

        loaded = OptimizationHistory.load(path)
        assert len(loaded.results) == 1
        assert loaded.results[0].parameters["beam_size"] == 5
        assert loaded.best_result is not None


class TestParameterOptimizer:

    def test_random_search(self, tmp_path):
        call_count = [0]

        def mock_evaluate(params):
            call_count[0] += 1
            wer = 0.20 - params.get("beam_size", 5) * 0.01
            der = 0.25 - params.get("beam_size", 5) * 0.01
            return (max(wer, 0.05), max(der, 0.08), 0.90)

        optimizer = ParameterOptimizer(
            evaluate_fn=mock_evaluate,
            parameter_ranges={"beam_size": ParameterRange("beam_size", 1, 15, is_int=True)},
            results_dir=tmp_path,
        )

        result = optimizer.optimize(max_iterations=10, strategy="random", parameters_to_tune=["beam_size"])

        assert result is not None
        assert call_count[0] == 10
        assert "beam_size" in result.parameters

    def test_grid_search(self, tmp_path):
        def mock_evaluate(params):
            beam = params.get("beam_size", 5)
            # Higher beam = lower WER/DER = better
            return (0.50 / beam, 0.50 / beam, 0.80 + beam * 0.01)

        optimizer = ParameterOptimizer(
            evaluate_fn=mock_evaluate,
            parameter_ranges={"beam_size": ParameterRange("beam_size", 1, 5, is_int=True)},
            results_dir=tmp_path,
        )

        result = optimizer.optimize(max_iterations=10, strategy="grid", parameters_to_tune=["beam_size"])
        # Best beam_size should be the highest in the grid
        assert result.parameters["beam_size"] >= 4

    def test_early_stopping(self, tmp_path):
        def perfect_evaluate(params):
            return (0.01, 0.05, 0.99)

        optimizer = ParameterOptimizer(
            evaluate_fn=perfect_evaluate,
            parameter_ranges={"beam_size": ParameterRange("beam_size", 1, 10, is_int=True)},
            target=OptimizationTarget(target_wer=0.05, target_der=0.10, target_speaker_accuracy=0.95),
            results_dir=tmp_path,
        )

        result = optimizer.optimize(max_iterations=100, strategy="random")
        assert len(optimizer.history.results) < 100


class TestMetricsLogger:

    def _make_result(self, iteration=0, wer=0.15, der=0.20, spk_acc=0.85):
        return OptimizationResult(
            parameters={
                "beam_size": 5, "best_of": 3, "temperature": 0.1,
                "vad_threshold": 0.5, "clustering_threshold": 0.6,
                "min_speakers": 2, "max_speakers": 8,
                "similarity_threshold": 0.75,
            },
            wer=wer, der=der, speaker_accuracy=spk_acc,
            score=0.3, iteration=iteration,
            timestamp="2024-06-15T12:00:00",
        )

    def test_creates_csv(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(), strategy="random")

        assert logger.csv_path.exists()
        content = logger.csv_path.read_text()
        assert "timestamp" in content
        assert "0.1500" in content

    def test_creates_markdown(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(), strategy="bayesian")

        assert logger.md_path.exists()
        content = logger.md_path.read_text()
        assert "Metrics Log" in content
        assert "Best Result" in content
        assert "bayesian" in content

    def test_csv_appends_rows(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(iteration=0, wer=0.20), strategy="random")
        logger.append(self._make_result(iteration=1, wer=0.10), strategy="random")

        lines = logger.csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_markdown_shows_best(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(iteration=0, wer=0.20, der=0.25), strategy="grid")
        logger.append(self._make_result(iteration=1, wer=0.03, der=0.08, spk_acc=0.96), strategy="grid")

        content = logger.md_path.read_text()
        assert "3.00%" in content  # Best WER should be 3%

    def test_csv_header_written_once(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(iteration=0), strategy="random")
        logger.append(self._make_result(iteration=1), strategy="random")

        content = logger.csv_path.read_text()
        assert content.count("timestamp") == 1

    def test_notes_included(self, tmp_path):
        logger = MetricsLogger(tmp_path)
        logger.append(self._make_result(), strategy="random", notes="test run")

        content = logger.csv_path.read_text()
        assert "test run" in content


class TestOptimizationResult:

    def test_to_dict(self):
        result = OptimizationResult(
            parameters={"beam_size": 5, "temperature": 0.1},
            wer=0.12, der=0.18, speaker_accuracy=0.88,
            score=0.25, iteration=42, timestamp="2024-01-15T10:30:00",
        )
        d = result.to_dict()
        assert d["parameters"]["beam_size"] == 5
        assert d["wer"] == 0.12
        assert d["iteration"] == 42
