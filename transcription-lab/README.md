# Transcription Lab

High-accuracy real-time meeting transcription and diarization optimization lab.

## Overview

This lab provides tools to:
- **Transcribe** audio with real-time and batch processing modes
- **Diarize** audio to identify who spoke when
- **Match voices** across meetings using persistent voice prints
- **Evaluate** against ground truth (WER, DER, speaker accuracy)
- **Optimize** parameters to achieve target accuracy thresholds

## Installation

```bash
pip install -e .
```

### Dependencies

Required models will download automatically on first use:
- **faster-whisper**: Transcription (various model sizes)
- **pyannote-audio**: Diarization and speaker embeddings

For pyannote models, you need a HuggingFace token (models require acceptance of terms):
```bash
export HF_TOKEN=your_token_here
```

## Quick Start

### 1. Add Test Data

Place your test files in the data directories:

```
data/
├── audio/
│   ├── meeting1_mic.wav      # Your microphone recording
│   └── meeting1_system.wav   # System audio (other participants)
├── transcripts/
│   └── meeting1_transcript.txt   # Ground truth transcript
└── voiceprints/              # Auto-populated
```

### 2. Ground Truth Format

```
[00:00:05 - 00:00:12] Alice: Let's discuss the Q3 roadmap.
[00:00:13 - 00:00:18] Bob: I think we should prioritize the API work.
[00:00:20 - 00:00:30] Alice: Great point. Let me pull up the timeline.
```

### 3. Run Evaluation

```bash
# Single evaluation
tlab evaluate --mic data/audio/meeting1_mic.wav \
              --system data/audio/meeting1_system.wav \
              --transcript data/transcripts/meeting1_transcript.txt

# Run optimization
tlab optimize --max-iterations 100 --strategy bayesian
```

## CLI Commands

```bash
# Transcribe audio
tlab transcribe mic.wav system.wav --output transcript.txt

# Diarize audio (identify speakers)
tlab diarize mic.wav system.wav --output speakers.rttm

# Evaluate against ground truth
tlab evaluate --mic mic.wav --system sys.wav --transcript ground_truth.txt

# Run parameter optimization
tlab optimize --max-iterations 100 --target-wer 0.05 --target-der 0.10

# Enroll a speaker's voice
tlab enroll "Alice" alice_sample.wav --start 0.0 --end 30.0

# List enrolled speakers
tlab speakers

# Check lab status
tlab status
```

## Parameter Optimization

The optimizer tunes these parameters:

### Transcription
- `beam_size`: Beam search width (1-15)
- `best_of`: Candidates to consider
- `temperature`: Sampling temperature
- `vad_threshold`: Voice activity detection sensitivity

### Diarization
- `clustering_threshold`: Speaker clustering threshold
- `min/max_speakers`: Expected speaker count range

### Voice Prints
- `similarity_threshold`: Match confidence threshold

### Strategies

- **grid**: Systematic search over parameter grid
- **random**: Random sampling
- **bayesian**: Smart search focusing on promising regions

## Target Metrics

Default optimization targets:
- **WER (Word Error Rate)**: < 5%
- **DER (Diarization Error Rate)**: < 10%
- **Speaker Accuracy**: > 95%

## Architecture

```
src/transcription_lab/
├── audio.py        # Audio loading, merging, chunking
├── transcriber.py  # Whisper-based transcription
├── diarization.py  # Speaker diarization + voice prints
├── evaluation.py   # WER, DER metrics
├── optimizer.py    # Parameter optimization
└── cli.py          # Command-line interface
```

## Voice Print System

Voice prints allow speaker recognition across meetings:

1. **Enroll**: Extract embedding from known speaker audio
2. **Store**: Embeddings saved persistently
3. **Match**: New speakers matched against database
4. **Update**: Embeddings refined with more samples

```bash
# Enroll from a meeting where Alice is speaking from 0:30 to 1:00
tlab enroll "Alice" meeting1_mic.wav --start 30 --end 60

# Future meetings will identify Alice automatically
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
realtime:
  model_size: "base"
  chunk_duration_ms: 2000

postprocess:
  model_size: "large-v3"

diarization:
  min_speakers: 1
  max_speakers: 10

optimization:
  target_wer: 0.05
  target_der: 0.10
```

## Testing

```bash
pytest tests/ -v
```

## Performance Tips

1. **GPU**: Enable CUDA for 5-10x faster processing
2. **Model size**: Use "base" for real-time, "large-v3" for post-processing
3. **Chunk size**: 2-3 seconds balances latency and accuracy
4. **Voice prints**: More enrollment samples = better matching
