#!/usr/bin/env python3
"""Evaluate transcription WER against ground truth (per-track).

Compares:
  - system audio transcription vs non-Joel speaker reference
  - mic audio transcription vs Joel reference (optional)

Reports per-track WER and weighted average.

Logs every run to results/experiment_log.csv
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

EXPERIMENT_CSV = PROJECT_ROOT / "results" / "experiment_log.csv"
EXPERIMENT_MD = PROJECT_ROOT / "results" / "experiment_log.md"

CSV_FIELDS = [
    "run_id", "timestamp", "model", "beam_size", "best_of", "temperature",
    "vad_filter", "vad_threshold", "compute_type",
    "condition_on_previous_text", "no_speech_threshold",
    "initial_prompt",
    "weighted_wer", "mic_wer", "sys_wer",
    "mic_ref_words", "mic_hyp_words", "sys_ref_words", "sys_hyp_words",
    "mic_ins", "mic_del", "mic_sub",
    "sys_ins", "sys_del", "sys_sub",
    "elapsed_seconds", "notes",
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # remove [Laughter] etc
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer(hypothesis: str, reference: str) -> dict:
    import jiwer
    out = jiwer.process_words(reference, hypothesis)
    return {
        "wer": out.wer,
        "insertions": out.insertions,
        "deletions": out.deletions,
        "substitutions": out.substitutions,
        "ref_words": len(reference.split()),
        "hyp_words": len(hypothesis.split()),
    }


def load_audio(path: Path) -> np.ndarray:
    from transcription_lab.audio import AudioLoader
    loader = AudioLoader(target_sr=16000)
    audio, sr = loader.load(path)
    return audio.astype(np.float32)


def transcribe(model, audio, args) -> str:
    vad_params = {"threshold": args.vad_threshold} if not args.no_vad_filter else None
    kwargs = dict(
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        vad_filter=not args.no_vad_filter,
        vad_parameters=vad_params,
        language="en",
        word_timestamps=False,
        condition_on_previous_text=bool(args.condition_on_previous_text),
        no_speech_threshold=args.no_speech_threshold,
    )
    if args.initial_prompt:
        kwargs["initial_prompt"] = args.initial_prompt
    segments, _ = model.transcribe(audio, **kwargs)
    return " ".join(seg.text.strip() for seg in segments)


def get_next_run_id() -> int:
    if not EXPERIMENT_CSV.exists():
        return 1
    with open(EXPERIMENT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        ids = [int(r["run_id"]) for r in reader if r.get("run_id", "").isdigit()]
    return max(ids, default=0) + 1


def log_result(run_id, args, mic_wer, sys_wer, weighted_wer, elapsed, notes):
    write_header = not EXPERIMENT_CSV.exists() or EXPERIMENT_CSV.stat().st_size == 0
    with open(EXPERIMENT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": args.model,
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "temperature": args.temperature,
            "vad_filter": not args.no_vad_filter,
            "vad_threshold": args.vad_threshold,
            "compute_type": args.compute_type,
            "condition_on_previous_text": bool(args.condition_on_previous_text),
            "no_speech_threshold": args.no_speech_threshold,
            "initial_prompt": args.initial_prompt or "",
            "weighted_wer": f"{weighted_wer:.4f}",
            "mic_wer": f"{mic_wer['wer']:.4f}" if mic_wer else "",
            "sys_wer": f"{sys_wer['wer']:.4f}",
            "mic_ref_words": mic_wer["ref_words"] if mic_wer else "",
            "mic_hyp_words": mic_wer["hyp_words"] if mic_wer else "",
            "sys_ref_words": sys_wer["ref_words"],
            "sys_hyp_words": sys_wer["hyp_words"],
            "mic_ins": mic_wer["insertions"] if mic_wer else "",
            "mic_del": mic_wer["deletions"] if mic_wer else "",
            "mic_sub": mic_wer["substitutions"] if mic_wer else "",
            "sys_ins": sys_wer["insertions"],
            "sys_del": sys_wer["deletions"],
            "sys_sub": sys_wer["substitutions"],
            "elapsed_seconds": f"{elapsed:.1f}",
            "notes": notes,
        })
    rewrite_md()


def rewrite_md():
    rows = []
    if EXPERIMENT_CSV.exists():
        with open(EXPERIMENT_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
    if not rows:
        return

    best = min(rows, key=lambda r: float(r.get("weighted_wer") or r.get("wer", "999")))
    best_wer = float(best.get("weighted_wer") or best.get("wer", "999"))

    lines = [
        "# Transcription Lab -- Experiment Log",
        "",
        "**Target: WER < 5%**",
        "",
        f"**Best weighted WER: {best_wer:.2%}** (run #{best['run_id']}, model={best['model']})",
        "",
        "---",
        "",
        "| Run | Model | Beam | Temp | VAD | CoPrev | NoSp | Prompt | SysWER | MicWER | WeightedWER | Time | Notes |",
        "|-----|-------|------|------|-----|--------|------|--------|--------|--------|-------------|------|-------|",
    ]

    for r in rows:
        w_wer = r.get("weighted_wer") or r.get("wer", "?")
        try:
            w_pct = f"{float(w_wer):.2%}"
        except (ValueError, TypeError):
            w_pct = w_wer
        sys_w = f"{float(r.get('sys_wer', 0)):.2%}" if r.get("sys_wer") else "N/A"
        mic_w = f"{float(r.get('mic_wer', 0)):.2%}" if r.get("mic_wer") else "N/A"
        prompt = (r.get("initial_prompt") or "")[:20]
        marker = " **" if r["run_id"] == best["run_id"] else ""
        lines.append(
            f"| {r['run_id']} | {r['model']} | {r['beam_size']} | {r['temperature']} "
            f"| {r.get('vad_filter','')}/{r.get('vad_threshold','')} "
            f"| {r.get('condition_on_previous_text','')} "
            f"| {r.get('no_speech_threshold','')} "
            f"| {prompt} "
            f"| {sys_w} | {mic_w} | {w_pct}{marker} "
            f"| {r.get('elapsed_seconds','')}s | {r.get('notes', '')} |"
        )

    EXPERIMENT_MD.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription WER (per-track)")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--best-of", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-vad-filter", action="store_true", default=False)
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--condition-on-previous-text", type=int, default=0)
    parser.add_argument("--no-speech-threshold", type=float, default=0.6)
    parser.add_argument("--initial-prompt", default="", help="Initial prompt for Whisper")
    parser.add_argument("--sys-only", action="store_true", help="Only evaluate system track")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    mic_path = PROJECT_ROOT / "data" / "audio" / "meeting1_mic.wav"
    sys_path = PROJECT_ROOT / "data" / "audio" / "meeting1_system.wav"
    mic_ref_path = PROJECT_ROOT / "data" / "transcripts" / "meeting1_mic_ref.txt"
    sys_ref_path = PROJECT_ROOT / "data" / "transcripts" / "meeting1_system_ref.txt"

    sys_ref = normalize_text(sys_ref_path.read_text())
    mic_ref = normalize_text(mic_ref_path.read_text()) if not args.sys_only else None

    vad_on = not args.no_vad_filter
    print(f"=== Config ===")
    print(f"  Model: {args.model}, Beam: {args.beam_size}, BestOf: {args.best_of}")
    print(f"  Temp: {args.temperature}, VAD: {vad_on}/{args.vad_threshold}")
    print(f"  cond_prev: {bool(args.condition_on_previous_text)}, no_speech: {args.no_speech_threshold}")
    if args.initial_prompt:
        print(f"  prompt: {args.initial_prompt[:80]}")

    from faster_whisper import WhisperModel
    print(f"\n  Loading model: {args.model} ({args.compute_type})...")
    model = WhisperModel(args.model, device="cpu", compute_type=args.compute_type)

    start = time.time()

    # System track
    print(f"  Transcribing system audio...")
    sys_audio = load_audio(sys_path)
    sys_hyp = transcribe(model, sys_audio, args)
    sys_hyp_n = normalize_text(sys_hyp)
    sys_wer_result = compute_wer(sys_hyp_n, sys_ref)
    print(f"    Sys: {sys_wer_result['hyp_words']} words, WER={sys_wer_result['wer']:.2%}")

    # Mic track
    mic_wer_result = None
    if not args.sys_only:
        print(f"  Transcribing mic audio...")
        mic_audio = load_audio(mic_path)
        mic_hyp = transcribe(model, mic_audio, args)
        mic_hyp_n = normalize_text(mic_hyp)
        mic_wer_result = compute_wer(mic_hyp_n, mic_ref)
        print(f"    Mic: {mic_wer_result['hyp_words']} words, WER={mic_wer_result['wer']:.2%}")

    elapsed = time.time() - start

    # Weighted average
    if mic_wer_result:
        total_ref = sys_wer_result["ref_words"] + mic_wer_result["ref_words"]
        total_errors = (sys_wer_result["insertions"] + sys_wer_result["deletions"] +
                        sys_wer_result["substitutions"] + mic_wer_result["insertions"] +
                        mic_wer_result["deletions"] + mic_wer_result["substitutions"])
        weighted_wer = total_errors / max(total_ref, 1)
    else:
        weighted_wer = sys_wer_result["wer"]

    run_id = get_next_run_id()
    log_result(run_id, args, mic_wer_result, sys_wer_result, weighted_wer, elapsed, args.notes)

    # Save hypotheses
    hyp_dir = PROJECT_ROOT / "results"
    (hyp_dir / f"sys_hyp_run{run_id}.txt").write_text(sys_hyp)
    if not args.sys_only:
        (hyp_dir / f"mic_hyp_run{run_id}.txt").write_text(mic_hyp if mic_wer_result else "")

    print(f"\n=== Results (Run #{run_id}) ===")
    print(f"  System WER:    {sys_wer_result['wer']:.2%}  (I={sys_wer_result['insertions']} D={sys_wer_result['deletions']} S={sys_wer_result['substitutions']})")
    if mic_wer_result:
        print(f"  Mic WER:       {mic_wer_result['wer']:.2%}  (I={mic_wer_result['insertions']} D={mic_wer_result['deletions']} S={mic_wer_result['substitutions']})")
    print(f"  Weighted WER:  {weighted_wer:.2%}")
    print(f"  Elapsed:       {elapsed:.1f}s")

    if weighted_wer < 0.05:
        print(f"\n  *** TARGET MET: WER < 5% ***")
    else:
        print(f"\n  Gap to target: {weighted_wer - 0.05:+.2%}")


if __name__ == "__main__":
    main()
