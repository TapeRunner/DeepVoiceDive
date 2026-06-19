from deepvoicedive.audio_io import DEFAULT_SAMPLE_RATE, load_wav
from deepvoicedive.features import N_MFCC, compute_spectrogram, extract_mfcc


def test_load_wav_returns_mono(tone_a):
    y, sr = load_wav(tone_a)
    assert sr == DEFAULT_SAMPLE_RATE
    assert y.ndim == 1


def test_mfcc_shape(tone_a):
    y, sr = load_wav(tone_a)
    mfcc = extract_mfcc(y, sr)
    assert mfcc.shape[0] == N_MFCC


def test_spectrogram_axes_align(tone_a):
    y, sr = load_wav(tone_a)
    s_db, freqs = compute_spectrogram(y, sr)
    assert s_db.shape[0] == freqs.shape[0]
