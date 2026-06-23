"""Voz Expresiva Phideus — descriptor extraction module for paralinguistic / affective speech.

See `Documents/01_FRENTES_ACTIVOS/Voz_Expresiva_Phideus/ROADMAP_VOZ_EXPRESIVA_PHIDEUS.md`.
"""

from src.voz_expresiva.esd_loader import ESDLoader
from src.voz_expresiva.voice_quality import compute_voice_quality
from src.voz_expresiva.compound_descriptor import compute_all_descriptors

__all__ = [
    "ESDLoader",
    "compute_voice_quality",
    "compute_all_descriptors",
]
