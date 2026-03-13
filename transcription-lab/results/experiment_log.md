# Transcription Lab -- Experiment Log

**Target: WER < 5%**

**Best weighted WER: 19.37%** (run #9, model=medium.en)

---

| Run | Model | Beam | Temp | VAD | CoPrev | NoSp | Prompt | SysWER | MicWER | WeightedWER | Time | Notes |
|-----|-------|------|------|-----|--------|------|--------|--------|--------|-------------|------|-------|
| 1 | small.en | 5 | 0.0 | True/0.5 | False | 0.6 |  | 22.85% | N/A | 22.85% | 522.9s | baseline small.en sys-only new GT |
| 2 | small.en | 5 | 0.0 | True/0.5 | True | 0.6 |  | 20.79% | N/A | 20.79% | 644.9s | small.en cond_prev=True |
| 3 | medium.en | 5 | 0.0 | True/0.5 | True | 0.6 |  | 19.62% | N/A | 19.62% | 1445.3s | medium.en cond_prev=True |
| 4 | medium.en | 5 | 0.0 | True/0.5 | True | 0.6 | Product and strategy | 21.54% | N/A | 21.54% | 1529.0s | medium.en with context prompt |
| 5 | large-v2 | 5 | 0.0 | True/0.5 | True | 0.6 |  | 95.25% | N/A | 95.25% | 2301.3s | large-v2 baseline |
| 6 | medium.en | 10 | 0.0 | True/0.5 | True | 0.6 |  | 21.21% | N/A | 21.21% | 2187.7s | medium.en beam=10 |
| 7 | medium.en | 10 | 0.0 | True/0.5 | True | 0.6 |  | 21.21% | N/A | 21.21% | 2188.1s | medium.en beam=10 |
| 8 | medium.en | 1 | 0.0 | True/0.5 | True | 0.6 |  | 21.40% | N/A | 21.40% | 872.1s | medium.en beam=1 greedy |
| 9 | medium.en | 5 | 0.0 | True/0.35 | True | 0.6 |  | 19.37% | N/A | 19.37% ** | 1624.0s | medium.en vad=0.35 more sensitive |
| 10 | medium.en | 5 | 0.0 | True/0.65 | True | 0.6 |  | 19.87% | N/A | 19.87% | 1604.8s | medium.en vad=0.65 less sensitive |
| 11 | medium.en | 5 | 0.0 | True/0.65 | True | 0.6 |  | 19.87% | N/A | 19.87% | 1607.0s | medium.en vad=0.65 less sensitive |
| 12 | medium.en | 5 | 0.0 | True/0.2 | True | 0.6 |  | 23.15% | N/A | 23.15% | 1590.9s | medium.en vad=0.2 very sensitive |
| 13 | medium.en | 5 | 0.0 | True/0.5 | True | 0.4 |  | 19.62% | N/A | 19.62% | 1688.0s | medium.en no_speech=0.4 |
| 14 | medium.en | 5 | 0.0 | True/0.35 | True | 0.4 |  | 19.37% | N/A | 19.37% | 1693.5s | medium.en vad=0.35 no_speech=0.4 |
| 15 | medium.en | 5 | 0.0 | True/0.5 | True | 0.8 |  | 19.62% | N/A | 19.62% | 1715.5s | medium.en no_speech=0.8 |
