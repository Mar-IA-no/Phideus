"""ESD (Emotional Speech Dataset) loader.

ESD structure (per Zhou et al. 2021, github.com/HLTSingapore/Emotional-Speech-Data):
    Emotional Speech Dataset/
    ├── 0001/                    # Mandarin speakers 0001-0010
    │   ├── Angry/
    │   │   ├── 0001_000351.wav
    │   │   ├── ...
    │   ├── Happy/
    │   ├── Neutral/
    │   ├── Sad/
    │   └── Surprise/
    ├── 0002/
    ├── ...
    ├── 0010/
    ├── 0011/                    # English speakers 0011-0020
    ├── ...
    └── 0020/

Each speaker has ~350 sentences per emotion, recorded as WAV (typically 16kHz).

This loader provides the file inventory + metadata (speaker, emotion, sentence id, language).
It does NOT load audio — that's done lazily by the extraction script per utterance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

EMOTIONS = ("Angry", "Happy", "Neutral", "Sad", "Surprise")

# Speaker IDs by language per ESD documentation
MANDARIN_SPEAKERS = tuple(f"{i:04d}" for i in range(1, 11))   # 0001..0010
ENGLISH_SPEAKERS = tuple(f"{i:04d}" for i in range(11, 21))   # 0011..0020


@dataclass(frozen=True)
class ESDUtterance:
    """A single utterance entry in ESD."""
    wav_path: Path
    speaker_id: str       # "0001" .. "0020"
    emotion: str          # one of EMOTIONS
    sentence_id: str      # e.g. "000351"
    language: str         # "EN" or "ZH"


class ESDLoader:
    """Loads ESD metadata. Audio is loaded lazily by consumers."""

    def __init__(
        self,
        root_dir: str | Path,
        language: str = "EN",
        speakers: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
    ) -> None:
        """Args:
            root_dir: path to the ESD root (containing 0001/, 0002/, ... subdirs).
            language: "EN", "ZH", or "ALL".
            speakers: optional explicit list of speaker IDs to include (overrides language).
            emotions: optional explicit list of emotions (default: all 5).
        """
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"ESD root not found: {self.root_dir}")

        if speakers is not None:
            self.speakers = tuple(speakers)
        else:
            if language == "EN":
                self.speakers = ENGLISH_SPEAKERS
            elif language == "ZH":
                self.speakers = MANDARIN_SPEAKERS
            elif language == "ALL":
                self.speakers = MANDARIN_SPEAKERS + ENGLISH_SPEAKERS
            else:
                raise ValueError(f"Unknown language: {language!r}")

        self.emotions = tuple(emotions) if emotions is not None else EMOTIONS
        self._cache: Optional[List[ESDUtterance]] = None

    @staticmethod
    def _language_of(speaker_id: str) -> str:
        if speaker_id in ENGLISH_SPEAKERS:
            return "EN"
        if speaker_id in MANDARIN_SPEAKERS:
            return "ZH"
        return "UNKNOWN"

    def iter_utterances(self) -> Iterator[ESDUtterance]:
        """Yield all utterances matching the loader's filters."""
        for spk in self.speakers:
            spk_dir = self.root_dir / spk
            if not spk_dir.is_dir():
                logger.warning("Speaker dir missing: %s", spk_dir)
                continue
            lang = self._language_of(spk)
            for emo in self.emotions:
                emo_dir = spk_dir / emo
                if not emo_dir.is_dir():
                    logger.warning("Emotion dir missing: %s", emo_dir)
                    continue
                for wav in sorted(emo_dir.glob("*.wav")):
                    # filename convention: {speaker}_{sentence}.wav
                    stem = wav.stem  # "0001_000351"
                    parts = stem.split("_")
                    sentence_id = parts[1] if len(parts) >= 2 else stem
                    yield ESDUtterance(
                        wav_path=wav,
                        speaker_id=spk,
                        emotion=emo,
                        sentence_id=sentence_id,
                        language=lang,
                    )

    def list_all(self) -> List[ESDUtterance]:
        """Cached list of all utterances matching the loader's filters."""
        if self._cache is None:
            self._cache = list(self.iter_utterances())
        return self._cache

    def summary(self) -> dict:
        """Quick stats about the inventory."""
        utts = self.list_all()
        by_spk = {}
        by_emo = {}
        for u in utts:
            by_spk[u.speaker_id] = by_spk.get(u.speaker_id, 0) + 1
            by_emo[u.emotion] = by_emo.get(u.emotion, 0) + 1
        return {
            "n_utterances": len(utts),
            "n_speakers": len(set(u.speaker_id for u in utts)),
            "n_emotions": len(set(u.emotion for u in utts)),
            "by_speaker": dict(sorted(by_spk.items())),
            "by_emotion": dict(sorted(by_emo.items())),
        }
