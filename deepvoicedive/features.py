"""Spectral feature extraction (MFCCs and spectrogram) via librosa."""
from __future__ import annotations

import librosa
import numpy as np

# Dimensionality of the spectral "texture" captured per frame. The time-averaged
# MFCC vector forms the 40-dimensional voice embedding (see embedding.py).
N_MFCC = 40


def extract_mfcc(y, sr, n_mfcc: int = N_MFCC):
    """Return the MFCC matrix of shape (n_mfcc, n_frames)."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def compute_spectrogram(y, sr):
    """Return a dB-scaled magnitude spectrogram and its frequency axis.

    Returns
    -------
    (S_db, freqs): the spectrogram (n_freq_bins, n_frames) in decibels and the
    corresponding frequency axis (n_freq_bins,).
    """
    magnitude = np.abs(librosa.stft(y))
    s_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)
    return s_db, freqs
