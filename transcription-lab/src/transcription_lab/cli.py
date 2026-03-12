"""Command-line interface for the transcription lab."""

import click
import yaml
import logging
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("transcription_lab")


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _get_paths(config: dict) -> dict:
    """Extract paths from config with defaults."""
    paths = config.get('paths', {})
    return {
        'audio_dir': paths.get('audio_dir', 'data/audio'),
        'transcripts_dir': paths.get('transcripts_dir', 'data/transcripts'),
        'voiceprints_dir': paths.get('voiceprints_dir', 'data/voiceprints'),
        'results_dir': paths.get('results_dir', 'results'),
    }


@click.group()
@click.option('--config', '-c', default='configs/default.yaml', help='Config file path')
@click.pass_context
def main(ctx, config):
    """Transcription Lab - High-accuracy transcription optimization."""
    ctx.ensure_object(dict)
    config_path = Path(config)
    if config_path.exists():
        ctx.obj['config'] = load_config(config_path)
    else:
        ctx.obj['config'] = {}


@main.command()
@click.argument('mic_audio', type=click.Path(exists=True))
@click.argument('system_audio', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output transcript file')
@click.option('--model', '-m', default=None, help='Whisper model size')
@click.option('--realtime/--no-realtime', default=False, help='Simulate real-time processing')
@click.pass_context
def transcribe(ctx, mic_audio, system_audio, output, model, realtime):
    """Transcribe audio files."""
    from .audio import AudioMerger
    from .transcriber import TranscriptionEngine, RealtimeTranscriber

    config = ctx.obj['config']
    rt_cfg = config.get('realtime', {})
    model_size = model or rt_cfg.get('model_size', 'base')

    console.print(f"[bold]Transcribing...[/bold]")
    console.print(f"  Model: {model_size}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading audio...", total=None)
        merger = AudioMerger()
        merged = merger.merge(mic_audio, system_audio)
        progress.update(task, description=f"Loaded {merged.duration:.1f}s of audio")

        if realtime:
            task = progress.add_task("Transcribing (real-time)...", total=None)
            transcriber = RealtimeTranscriber(
                model_size=model_size,
                chunk_duration_ms=rt_cfg.get('chunk_duration_ms', 2000),
            )

            from .audio import AudioChunker
            chunker = AudioChunker(chunk_duration_ms=rt_cfg.get('chunk_duration_ms', 2000))

            for chunk in chunker.chunk(merged.data, merged.sample_rate):
                segments = transcriber.process_chunk(chunk.data, chunk.sample_rate)
                for seg in segments:
                    console.print(f"[dim][{seg.start:.1f}s][/dim] {seg.text}")

            transcript = transcriber.get_full_transcript()
        else:
            task = progress.add_task("Transcribing...", total=None)
            engine = TranscriptionEngine(model_size=model_size)
            transcript = engine.transcribe(
                merged.data, merged.sample_rate,
                beam_size=rt_cfg.get('beam_size', 5),
            )

        progress.update(task, description="Transcription complete")

    console.print("\n[bold green]Transcript:[/bold green]")
    console.print(transcript.to_text_with_timestamps())

    if output:
        with open(output, 'w') as f:
            f.write(transcript.to_text_with_timestamps())
        console.print(f"\n[dim]Saved to: {output}[/dim]")


@main.command()
@click.argument('mic_audio', type=click.Path(exists=True))
@click.argument('system_audio', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output RTTM file')
@click.option('--min-speakers', default=None, type=int, help='Minimum speakers')
@click.option('--max-speakers', default=None, type=int, help='Maximum speakers')
@click.pass_context
def diarize(ctx, mic_audio, system_audio, output, min_speakers, max_speakers):
    """Perform speaker diarization."""
    from .audio import AudioMerger
    from .diarization import DiarizationEngine

    config = ctx.obj['config']
    diar_cfg = config.get('diarization', {})

    min_spk = min_speakers or diar_cfg.get('min_speakers', 1)
    max_spk = max_speakers or diar_cfg.get('max_speakers', 10)

    console.print("[bold]Running speaker diarization...[/bold]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Loading audio...", total=None)
        merger = AudioMerger()
        merged = merger.merge(mic_audio, system_audio)

        task = progress.add_task("Diarizing...", total=None)
        engine = DiarizationEngine()
        result = engine.diarize(merged.data, merged.sample_rate, min_speakers=min_spk, max_speakers=max_spk)

    console.print(f"\n[bold green]Found {result.num_speakers} speakers[/bold green]")

    table = Table(title="Speaker Segments")
    table.add_column("Speaker")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Duration")

    for seg in result.segments[:20]:
        table.add_row(seg.speaker_id, f"{seg.start:.2f}s", f"{seg.end:.2f}s", f"{seg.duration:.2f}s")

    console.print(table)
    if len(result.segments) > 20:
        console.print(f"[dim]... and {len(result.segments) - 20} more segments[/dim]")

    if output:
        with open(output, 'w') as f:
            f.write(result.to_rttm())
        console.print(f"\n[dim]Saved RTTM to: {output}[/dim]")


@main.command()
@click.argument('transcript', type=click.Path(exists=True))
@click.argument('ground_truth', type=click.Path(exists=True))
@click.option('--collar', default=0.25, help='DER collar tolerance (seconds)')
@click.pass_context
def evaluate(ctx, transcript, ground_truth, collar):
    """Evaluate transcript against ground truth."""
    from .evaluation import GroundTruthParser, Evaluator
    from .transcriber import Transcript, TranscriptSegment

    console.print("[bold]Evaluating...[/bold]")

    parser = GroundTruthParser()
    gt = parser.parse_file(ground_truth)

    with open(transcript, 'r') as f:
        hyp_text = f.read()

    hyp = Transcript(
        segments=[TranscriptSegment(text=hyp_text, start=0, end=gt.duration)],
        language="en",
        duration=gt.duration,
    )

    evaluator = Evaluator(collar=collar)
    metrics = evaluator.evaluate(hyp, gt)

    table = Table(title="Evaluation Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("WER", f"{metrics.wer:.2%}")
    table.add_row("  Insertions", str(metrics.insertions))
    table.add_row("  Deletions", str(metrics.deletions))
    table.add_row("  Substitutions", str(metrics.substitutions))
    table.add_row("DER", f"{metrics.der:.2%}")
    table.add_row("  Miss", f"{metrics.miss:.2%}")
    table.add_row("  False Alarm", f"{metrics.false_alarm:.2%}")
    table.add_row("  Confusion", f"{metrics.confusion:.2%}")
    table.add_row("Speaker Accuracy", f"{metrics.speaker_accuracy:.2%}")

    console.print(table)


@main.command()
@click.option('--max-iterations', '-n', default=100, help='Max iterations')
@click.option('--strategy', '-s', default='bayesian', type=click.Choice(['grid', 'random', 'bayesian']))
@click.option('--target-wer', default=0.05, help='Target WER')
@click.option('--target-der', default=0.10, help='Target DER')
@click.option('--resume', '-r', default=None, type=click.Path(exists=True), help='Resume from results file')
@click.pass_context
def optimize(ctx, max_iterations, strategy, target_wer, target_der, resume):
    """Run parameter optimization."""
    from .optimizer import ExperimentRunner

    config = ctx.obj['config']
    paths = _get_paths(config)

    console.print("[bold]Starting optimization...[/bold]")
    console.print(f"  Strategy: {strategy}, Max: {max_iterations}")
    console.print(f"  Targets: WER<={target_wer:.1%}, DER<={target_der:.1%}")

    runner = ExperimentRunner(
        audio_dir=paths['audio_dir'],
        transcripts_dir=paths['transcripts_dir'],
        voiceprints_dir=paths['voiceprints_dir'],
        results_dir=paths['results_dir'],
    )

    test_cases = runner.list_test_cases()
    if not test_cases:
        console.print("[red]No test cases found![/red]")
        console.print(f"Add audio to: {paths['audio_dir']}")
        console.print(f"Add transcripts to: {paths['transcripts_dir']}")
        return

    console.print(f"Found {len(test_cases)} test case(s): {', '.join(test_cases)}")

    result = runner.run_optimization(
        test_cases=test_cases,
        max_iterations=max_iterations,
        strategy=strategy,
        resume_from=resume,
    )

    console.print("\n[bold green]Optimization Complete![/bold green]")

    table = Table(title="Best Parameters")
    table.add_column("Parameter")
    table.add_column("Value")
    for param, value in result.parameters.items():
        table.add_row(param, str(value))

    console.print(table)
    console.print(f"\nWER: {result.wer:.2%}, DER: {result.der:.2%}, SpkAcc: {result.speaker_accuracy:.2%}")


@main.command()
@click.argument('speaker_name')
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--start', default=0.0, help='Start time (seconds)')
@click.option('--end', default=None, type=float, help='End time (seconds)')
@click.pass_context
def enroll(ctx, speaker_name, audio_file, start, end):
    """Enroll a speaker voice print."""
    from .audio import AudioLoader
    from .diarization import DiarizationEngine, VoicePrintDatabase

    config = ctx.obj['config']
    voiceprints_dir = _get_paths(config)['voiceprints_dir']

    console.print(f"[bold]Enrolling speaker: {speaker_name}[/bold]")

    loader = AudioLoader()
    audio, sr = loader.load(audio_file)

    if end is None:
        end = len(audio) / sr

    diarizer = DiarizationEngine()
    embedding = diarizer.extract_embedding(audio, sr, start, end)

    db = VoicePrintDatabase(voiceprints_dir)
    db.add(speaker_name, embedding)

    console.print(f"[green]Enrolled: {speaker_name} ({start:.1f}s - {end:.1f}s)[/green]")


@main.command()
@click.pass_context
def speakers(ctx):
    """List enrolled speakers."""
    from .diarization import VoicePrintDatabase

    config = ctx.obj['config']
    voiceprints_dir = _get_paths(config)['voiceprints_dir']

    db = VoicePrintDatabase(voiceprints_dir)
    speaker_list = db.list_speakers()

    if not speaker_list:
        console.print("[yellow]No speakers enrolled.[/yellow]")
        return

    table = Table(title="Enrolled Speakers")
    table.add_column("Name")
    table.add_column("Samples")
    table.add_column("Created")

    for name in speaker_list:
        vp = db.get(name)
        table.add_row(name, str(vp.num_samples), vp.created_at[:10] if vp.created_at else "N/A")

    console.print(table)


@main.command()
@click.pass_context
def status(ctx):
    """Show lab status."""
    config = ctx.obj['config']
    paths = _get_paths(config)

    console.print("[bold]Transcription Lab Status[/bold]\n")

    table = Table(title="Directories")
    table.add_column("Name")
    table.add_column("Path")
    table.add_column("Status")

    for key, path in paths.items():
        p = Path(path)
        st = "[green]OK[/green]" if p.exists() else "[red]Missing[/red]"
        table.add_row(key, str(p), st)
    console.print(table)

    # Dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    for name, mod in [("faster-whisper", "faster_whisper"), ("pyannote.audio", "pyannote.audio"),
                      ("torch", "torch"), ("jiwer", "jiwer"), ("librosa", "librosa"),
                      ("soundfile", "soundfile"), ("scipy", "scipy")]:
        try:
            __import__(mod)
            console.print(f"  [green]+[/green] {name}")
        except ImportError:
            console.print(f"  [red]-[/red] {name}")


if __name__ == '__main__':
    main()
