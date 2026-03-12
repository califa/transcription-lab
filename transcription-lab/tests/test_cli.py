"""Tests for the CLI module."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from click.testing import CliRunner

from transcription_lab.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def audio_files(tmp_path):
    sr = 16000
    duration = 2.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)

    mic_path = tmp_path / "mic.wav"
    sys_path = tmp_path / "system.wav"
    sf.write(mic_path, audio, sr)
    sf.write(sys_path, audio, sr)

    return mic_path, sys_path


@pytest.fixture
def transcript_file(tmp_path):
    content = "[00:00:00 - 00:00:02] Alice: Hello this is a test.\n"
    path = tmp_path / "ground_truth.txt"
    path.write_text(content)
    return path


class TestStatusCommand:

    def test_status_runs(self, runner):
        result = runner.invoke(main, ["status"])
        assert result.exit_code == 0
        assert "Status" in result.output or "Dependencies" in result.output


class TestTranscribeCommand:

    def test_transcribe_runs(self, runner, audio_files):
        mic, sys = audio_files
        result = runner.invoke(main, ["transcribe", str(mic), str(sys)])
        assert result.exit_code == 0
        assert "Transcript" in result.output or "Mock" in result.output

    def test_transcribe_with_output(self, runner, audio_files, tmp_path):
        mic, sys = audio_files
        output = tmp_path / "out.txt"
        result = runner.invoke(main, ["transcribe", str(mic), str(sys), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()


class TestDiarizeCommand:

    def test_diarize_runs(self, runner, audio_files):
        mic, sys = audio_files
        result = runner.invoke(main, ["diarize", str(mic), str(sys)])
        assert result.exit_code == 0
        assert "Speaker" in result.output or "speaker" in result.output.lower()

    def test_diarize_with_output(self, runner, audio_files, tmp_path):
        mic, sys = audio_files
        output = tmp_path / "out.rttm"
        result = runner.invoke(main, ["diarize", str(mic), str(sys), "-o", str(output)])
        assert result.exit_code == 0
        assert output.exists()


class TestEvaluateCommand:

    def test_evaluate_runs(self, runner, audio_files, transcript_file, tmp_path):
        mic, sys = audio_files
        # First produce a hypothesis transcript
        hyp_path = tmp_path / "hypothesis.txt"
        hyp_path.write_text("Hello this is a test.")

        result = runner.invoke(main, ["evaluate", str(hyp_path), str(transcript_file)])
        assert result.exit_code == 0
        assert "WER" in result.output


class TestOptimizeCommand:

    def test_optimize_no_data(self, runner, tmp_path):
        result = runner.invoke(main, [
            "--config", str(tmp_path / "nonexistent.yaml"),
            "optimize", "--max-iterations", "1",
        ])
        assert result.exit_code == 0
        assert "No test cases" in result.output or "no" in result.output.lower()


class TestSpeakersCommand:

    def test_speakers_empty(self, runner):
        result = runner.invoke(main, ["speakers"])
        assert result.exit_code == 0
        assert "No speakers" in result.output or "enrolled" in result.output.lower()
