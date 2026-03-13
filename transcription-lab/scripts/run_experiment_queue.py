#!/usr/bin/env python3
"""Run the next experiment from the queue.

Reads experiment_log.csv to see what's been done, picks the next
untried configuration, runs it, and logs the result.

Never repeats a configuration that's already been tried.
"""

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_CSV = PROJECT_ROOT / "results" / "experiment_log.csv"

# Experiment queue: ordered list of configurations to try.
# Each entry is a dict of CLI args for evaluate_wer.py.
# The queue is ordered by expected impact (most promising first).
EXPERIMENT_QUEUE = [
    # --- Model scaling (best lever) ---
    {"model": "large-v2", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "large-v2 baseline"},

    {"model": "large-v3", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "large-v3 baseline"},

    # --- Beam size tuning on best model so far (medium.en as fallback) ---
    {"model": "medium.en", "beam_size": 10, "best_of": 10, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "medium.en beam=10"},

    {"model": "medium.en", "beam_size": 1, "best_of": 1, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "medium.en beam=1 greedy"},

    # --- VAD threshold tuning ---
    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.35,
     "notes": "medium.en vad=0.35 more sensitive"},

    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.65,
     "notes": "medium.en vad=0.65 less sensitive"},

    # --- No-speech threshold tuning ---
    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.4, "vad_threshold": 0.5,
     "notes": "medium.en no_speech=0.4"},

    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.8, "vad_threshold": 0.5,
     "notes": "medium.en no_speech=0.8"},

    # --- VAD off ---
    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "no_vad_filter": True,
     "notes": "medium.en no VAD"},

    # --- Temperature sampling ---
    {"model": "medium.en", "beam_size": 5, "best_of": 5, "temperature": 0.2,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "medium.en temp=0.2"},

    # --- Large model with tuned params (applied after finding best params) ---
    {"model": "large-v2", "beam_size": 10, "best_of": 10, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "large-v2 beam=10"},

    {"model": "large-v3", "beam_size": 10, "best_of": 10, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "large-v3 beam=10"},

    # --- Distil-large-v3 (faster, sometimes competitive) ---
    {"model": "distil-large-v3", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "distil-large-v3"},

    # --- large-v3-turbo ---
    {"model": "large-v3-turbo", "beam_size": 5, "best_of": 5, "temperature": 0.0,
     "condition_on_previous_text": 1, "no_speech_threshold": 0.6, "vad_threshold": 0.5,
     "notes": "large-v3-turbo"},
]


def config_key(cfg: dict) -> str:
    """Create a unique hash for a config to detect duplicates."""
    relevant = {k: v for k, v in sorted(cfg.items()) if k != "notes"}
    return hashlib.md5(str(relevant).encode()).hexdigest()


def get_completed_configs() -> set:
    """Read experiment log and return set of config hashes already run."""
    if not EXPERIMENT_CSV.exists():
        return set()
    completed = set()
    with open(EXPERIMENT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key_parts = {
                "model": row.get("model", ""),
                "beam_size": int(row.get("beam_size", 0)),
                "best_of": int(row.get("best_of", 0)),
                "temperature": float(row.get("temperature", 0)),
                "condition_on_previous_text": row.get("condition_on_previous_text", ""),
                "no_speech_threshold": float(row.get("no_speech_threshold", 0)),
                "vad_threshold": float(row.get("vad_threshold", 0)),
                "vad_filter": row.get("vad_filter", "True"),
            }
            completed.add(hashlib.md5(str(sorted(key_parts.items())).encode()).hexdigest())
    return completed


def config_to_completed_key(cfg: dict) -> str:
    """Match the format used in get_completed_configs."""
    key_parts = {
        "model": cfg.get("model", ""),
        "beam_size": int(cfg.get("beam_size", 0)),
        "best_of": int(cfg.get("best_of", 0)),
        "temperature": float(cfg.get("temperature", 0)),
        "condition_on_previous_text": str(bool(cfg.get("condition_on_previous_text", 0))),
        "no_speech_threshold": float(cfg.get("no_speech_threshold", 0)),
        "vad_threshold": float(cfg.get("vad_threshold", 0)),
        "vad_filter": str(not cfg.get("no_vad_filter", False)),
    }
    return hashlib.md5(str(sorted(key_parts.items())).encode()).hexdigest()


def build_cmd(cfg: dict) -> list:
    """Build the evaluate_wer.py command from a config dict."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_wer.py"),
        "--model", cfg["model"],
        "--beam-size", str(cfg["beam_size"]),
        "--best-of", str(cfg["best_of"]),
        "--temperature", str(cfg["temperature"]),
        "--condition-on-previous-text", str(cfg.get("condition_on_previous_text", 0)),
        "--no-speech-threshold", str(cfg.get("no_speech_threshold", 0.6)),
        "--vad-threshold", str(cfg.get("vad_threshold", 0.5)),
        "--compute-type", "int8",
        "--sys-only",
        "--notes", cfg.get("notes", ""),
    ]
    if cfg.get("no_vad_filter"):
        cmd.append("--no-vad-filter")
    if cfg.get("initial_prompt"):
        cmd.extend(["--initial-prompt", cfg["initial_prompt"]])
    return cmd


def main():
    completed = get_completed_configs()
    print(f"Completed experiments: {len(completed)}")

    for i, cfg in enumerate(EXPERIMENT_QUEUE):
        key = config_to_completed_key(cfg)
        if key in completed:
            continue

        print(f"\n=== Running experiment {i+1}/{len(EXPERIMENT_QUEUE)}: {cfg.get('notes', '')} ===")
        cmd = build_cmd(cfg)
        print(f"Command: {' '.join(cmd)}")

        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if result.returncode != 0:
            print(f"Experiment failed with exit code {result.returncode}")
        return  # Run one experiment at a time (cron will trigger next)

    print("All queued experiments complete!")


if __name__ == "__main__":
    main()
