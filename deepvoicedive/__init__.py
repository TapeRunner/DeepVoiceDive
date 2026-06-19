"""DeepVoiceDive — deep, mathematical voice analysis.

Public API re-exports for convenient access::

    from deepvoicedive import embedding_from_file, compare_files
"""
from .audio_io import load_wav, DEFAULT_SAMPLE_RATE
from .features import extract_mfcc, compute_spectrogram, N_MFCC
from .embedding import compute_embedding, embedding_from_file, EMBEDDING_DIM
from .compare import cosine_distance, voice_match, compare_files
from .batch import similarity_matrix

__version__ = "0.1.0"

__all__ = [
    "load_wav",
    "DEFAULT_SAMPLE_RATE",
    "extract_mfcc",
    "compute_spectrogram",
    "N_MFCC",
    "compute_embedding",
    "embedding_from_file",
    "EMBEDDING_DIM",
    "cosine_distance",
    "voice_match",
    "compare_files",
    "similarity_matrix",
    "__version__",
]
