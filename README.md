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

### Compare many recordings at once

```bash
deepvoicedive matrix rec1.wav rec2.wav rec3.wav --output-dir results/
```

Computes the **all-pairs Voice Match** between every recording and writes:

| File                     | Contents                                            |
| ------------------------ | --------------------------------------------------- |
| `similarity_matrix.csv`  | N×N table of match percentages                      |
| `similarity_matrix.png`  | Annotated heatmap of the same matrix                |

The matrix is symmetric and its diagonal is 100 % (every recording matches
itself). High values off the diagonal mark recordings that likely come from the
same voice — handy for grouping several takes or checking voice stability across
days.

## Use as a library

```python
from deepvoicedive import embedding_from_file, compare_files, similarity_matrix

emb = embedding_from_file("my_voice.wav")        # 40-dim numpy array
dist, similarity, match = compare_files("a.wav", "b.wav")
labels, matrix = similarity_matrix(["a.wav", "b.wav", "c.wav"])  # all-pairs %
```

## Voice-swap screening workflow

If you produce AI covers — e.g. swapping the vocal of a generated song for your
own cloned voice — the swap works best when the source vocal already resembles
your voice. DeepVoiceDive can **screen many candidate vocal stems** and tell you
which is the best fit.

**1. Build a reusable profile of your own voice** (once), from a few clean,
ideally *sung* clips:

```bash
deepvoicedive profile me1.wav me2.wav me3.wav -o my_voice.json
```

**2. Screen candidate stems against it:**

```bash
deepvoicedive screen --profile my_voice.json stemA.wav stemB.wav stemC.wav
```

```
Stem                       Match  Tonlage  Eignung
stemA.wav                   100%     100%     100%  ✅
stemC.wav                   100%      96%      99%  ✅
stemB.wav                    90%      28%      71%  ❌
```

You also get `screening.csv` and a green/red bar chart. Each stem gets a
**Match %** (timbre similarity to you), a **Tonlage %** (pitch-register
compatibility) and a combined **Eignung %** (suitability, default `0.7·Match +
0.3·Tonlage`). The ✅/❌ verdict uses `--threshold` (default 75). You can also
skip the saved profile and pass `--reference-clips a.wav b.wav` directly.

> **Honest expectations:** speaker models are trained on *speech*, so comparing
> spoken vs sung audio is rougher and the absolute percentages are indicative,
> not authoritative. The **ranking** (which stem is closest) is the robust part —
> and that's exactly what this workflow needs. The suitability score is a
> well-founded estimate; the real test is running the actual swap. For best
> results, use clean, lightly-processed (dry, de-reverbed) stems and a *sung*
> reference of your own voice.

### Higher accuracy: the neural model (optional)

For much more accurate speaker matching, install the optional neural backend,
which uses the SpeechBrain **ECAPA-TDNN** model:

```bash
pip install -e ".[neural]"
deepvoicedive screen --profile my_voice.json --method neural stemA.wav stemB.wav
```

It downloads a small (~80 MB) model on first use and runs comfortably on a
laptop CPU (e.g. Apple Silicon) — no GPU or training needed. If the neural
backend or its model isn't available, the tools **automatically fall back to the
offline MFCC method** with a note, so they always run.

## How the metrics work

- The **voice embedding** is the L2-normalised, time-averaged MFCC vector
  (40 dimensions) by default, or a 192-dim ECAPA speaker embedding with
  `--method neural`. Normalisation makes the comparison invariant to loudness and
  recording length.
- The **cosine distance** between two embeddings lies in `[0, 2]`; the **match
  percentage** is `max(0, 1 - distance) * 100`.
- **Pitch compatibility** compares the median F0 of two voices: identical pitch
  is 100 %, decaying to 0 % at one octave apart.

> Note: the default MFCC embedding is a reproducible signal-processing fingerprint
> that runs entirely offline (no model download or GPU). The optional neural
> ECAPA embedding is more discriminative but needs the `neural` extra.

## Development

```bash
pytest -q
```

Tests synthesise their own audio (sine tones), so no binary fixtures are stored
in the repository. The eGeMAPS and neural tests are skipped automatically if
openSMILE / the neural backend are not installed.

## License

See [LICENSE](LICENSE).
