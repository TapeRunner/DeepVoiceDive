"""Visual reporting: spectrogram (formants) + MFCC feature matrix."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / CI-safe backend; must be set before pyplot.

import librosa.display  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from .features import compute_spectrogram, extract_mfcc  # noqa: E402

# Formants of human speech mostly live below this frequency.
FORMANT_MAX_HZ = 5000


def render_report(y, sr, out_path):
    """Render a two-panel PNG: spectrogram (left) and MFCC matrix (right)."""
    s_db, _ = compute_spectrogram(y, sr)
    mfcc = extract_mfcc(y, sr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    img0 = librosa.display.specshow(
        s_db, sr=sr, x_axis="time", y_axis="hz", ax=axes[0]
    )
    axes[0].set_title("Frequenzspektrum (Formanten)")
    axes[0].set_ylim(0, FORMANT_MAX_HZ)
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")

    img1 = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=axes[1])
    axes[1].set_title("MFCC Feature-Matrix")
    axes[1].set_ylabel("MFCC-Koeffizient")
    fig.colorbar(img1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return out_path


def render_similarity_matrix(labels, matrix, out_path):
    """Render an all-pairs Voice Match matrix as an annotated heatmap."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * n, 1.5 + 1.1 * n))

    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)

    # Annotate each cell with its percentage; pick a readable text colour.
    for i in range(n):
        for j in range(n):
            value = matrix[i, j]
            colour = "white" if value < 50 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=colour)

    ax.set_title("Voice Match (%)")
    fig.colorbar(im, ax=ax, label="Match %")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120)
    plt.close(fig)
    return out_path
