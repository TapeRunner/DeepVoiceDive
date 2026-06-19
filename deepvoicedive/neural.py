"""Neural speaker embedding via SpeechBrain ECAPA-TDNN (optional).

This module provides a 192-dimensional speaker embedding from the pretrained
``speechbrain/spkrec-ecapa-voxceleb`` model. It is the most accurate way to tell
whether two recordings come from the same voice, but it requires the heavy
``neural`` extra (PyTorch + SpeechBrain) and downloads the model on first use::

    pip install -e ".[neural]"

If those dependencies (or the model download) are unavailable, callers should
fall back to the offline MFCC embedding.
"""
from __future__ import annotations

import numpy as np

from .audio_io import DEFAULT_SAMPLE_RATE, load_wav

_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
_encoder = None


class NeuralBackendUnavailable(RuntimeError):
    """Raised when the neural embedding backend cannot be used."""


def _get_encoder():
    """Load and cache the ECAPA speaker encoder (lazy, imported on demand)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        import torch  # noqa: F401
        from speechbrain.inference.speaker import EncoderClassifier
    except Exception as exc:  # noqa: BLE001
        raise NeuralBackendUnavailable(
            "The neural backend requires PyTorch and SpeechBrain. "
            'Install them with: pip install -e ".[neural]"'
        ) from exc
    try:
        _encoder = EncoderClassifier.from_hparams(
            source=_MODEL_SOURCE,
            savedir=f"pretrained_models/{_MODEL_SOURCE.split('/')[-1]}",
        )
    except Exception as exc:  # noqa: BLE001
        raise NeuralBackendUnavailable(
            f"Could not load the ECAPA model '{_MODEL_SOURCE}'. The first run "
            "needs internet access to download it. Original error: " + str(exc)
        ) from exc
    return _encoder


def neural_embedding(path):
    """Return an L2-normalised 192-dim ECAPA speaker embedding for an audio file."""
    import torch

    encoder = _get_encoder()
    y, _sr = load_wav(path, sr=DEFAULT_SAMPLE_RATE)
    signal = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(signal).squeeze().cpu().numpy()
    emb = emb.astype(np.float64)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb
