# DeepVoiceDive

**Deep, mathematical voice analysis.** DeepVoiceDive goes far beyond the simple
pitch (frequency) measurement of typical online apps. It decomposes a human voice
recording into its physical and acoustic building blocks to make the unique
**acoustic fingerprint** of a voice digitally measurable and comparable.

## Features

1. **Feature extraction & analysis**
   - **Scientific report** — reads an uncompressed voice recording (WAV) and
     generates a table of all **88 eGeMAPSv02 parameters** (formants, jitter,
     shimmer, loudness, spectral descriptors …) via openSMILE.
   - **Vector voice fingerprint** — compresses high-dimensional frequency data
     into a single, **40-dimensional voice embedding** (a numeric voice ID).
   - **Visual reporting** — renders a figure showing the frequency spectrum
     (formants) and the MFCC feature matrix.
2. **Biometric voice comparison**
   - **Similarity check** — compares two audio files mathematically.
   - **Cosine distance** — using vector geometry (`scipy`) to measure how close
     two voice fingerprints lie.
   - **Percentage match** — a "Voice Match" score showing how stable your own
     voice is across different days, or whether two recordings stem from the
     same person.

## Technology

- **[librosa](https://librosa.org/)** — high-resolution signal processing,
  frequency visualisation and MFCC extraction.
- **[openSMILE](https://audeering.github.io/opensmile-python/) (eGeMAPS)** — a
  scientific gold-standard extractor isolating 88 clinically relevant voice
  parameters.
- **scipy / numpy / pandas / matplotlib** — vector geometry, numerics, tables
  and plotting.

## Installation

```bash
pip install -e .          # runtime
pip install -e ".[dev]"   # + test tooling (pytest)
```

Requires Python ≥ 3.9.

## Usage

### Analyse a single recording

```bash
deepvoicedive analyze my_voice.wav --output-dir results/
```

Produces in `results/`:

| File                  | Contents                                        |
| --------------------- | ----------------------------------------------- |
| `egemaps_report.csv`  | All 88 eGeMAPS parameters (`parameter`, `value`)|
| `embedding.json`      | 40-dim voice fingerprint (human-readable)       |
| `embedding.npy`       | 40-dim voice fingerprint (NumPy binary)         |
| `report.png`          | Spectrogram (formants) + MFCC matrix            |

Skip the openSMILE step with `--no-egemaps`.

### Compare two recordings

```bash
deepvoicedive compare day1.wav day2.wav
```

```
Kosinus-Distanz : 0.0123
Ähnlichkeit     : 0.9877
Voice Match     : 98.77 %
```

## Use as a library

```python
from deepvoicedive import embedding_from_file, compare_files

emb = embedding_from_file("my_voice.wav")        # 40-dim numpy array
dist, similarity, match = compare_files("a.wav", "b.wav")
```

## How the metrics work

- The **voice embedding** is the L2-normalised, time-averaged MFCC vector
  (40 dimensions). Normalisation makes the comparison invariant to loudness and
  recording length.
- The **cosine distance** between two embeddings lies in `[0, 2]`; the **match
  percentage** is `max(0, 1 - distance) * 100`.

> Note: the embedding is a reproducible signal-processing fingerprint that runs
> entirely offline (no model download or GPU). A trained neural speaker
> embedding would be even more discriminative and can be added later as an
> option.

## Development

```bash
pytest -q
```

Tests synthesise their own audio (sine tones), so no binary fixtures are stored
in the repository. The eGeMAPS test is skipped automatically if openSMILE is not
installed.

## License

See [LICENSE](LICENSE).
