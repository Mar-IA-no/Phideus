#!/usr/bin/env python3
"""ESD dataset acquisition helper.

ESD (Emotional Speech Dataset, Zhou et al. 2021) is NOT directly wget-able —
the dataset is distributed via Google Drive after completing a registration form.

Project page: https://github.com/HLTSingapore/Emotional-Speech-Data

This script:
    1. Prints the manual steps required to obtain ESD.
    2. Verifies a candidate extracted directory looks like ESD.

It does NOT download anything automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

INSTRUCTIONS = """
═══════════════════════════════════════════════════════════════════════════
ESD (Emotional Speech Dataset) — acquisition steps (manual)
═══════════════════════════════════════════════════════════════════════════

1. Visit:  https://github.com/HLTSingapore/Emotional-Speech-Data
2. Follow the README to fill the dataset request form (Google Form).
3. Download the provided Google Drive archive (~10 GB).
4. Extract to:

     {target}

   The resulting layout must be:

     {target}/Emotional Speech Dataset/0001/Angry/0001_000351.wav
     {target}/Emotional Speech Dataset/0001/Happy/...
     ...
     {target}/Emotional Speech Dataset/0020/Surprise/...

   (Speakers 0001-0010 Mandarin · 0011-0020 English · 5 emotions each)

5. After extraction, verify the layout:

     python experiments/voz_expresiva/download_esd.py --verify {target}/'Emotional Speech Dataset'

═══════════════════════════════════════════════════════════════════════════
"""


def verify(root: Path, languages: tuple[str, ...] = ("EN",)) -> bool:
    """Verify ESD root contains the expected layout."""
    from src.voz_expresiva.esd_loader import ESDLoader
    ok = True
    for lang in languages:
        try:
            loader = ESDLoader(root, language=lang)
            summary = loader.summary()
            print(f"  language={lang}: {summary['n_utterances']} utterances, "
                  f"{summary['n_speakers']} speakers, {summary['n_emotions']} emotions")
            if summary["n_utterances"] < 100:
                print(f"    WARN: very few utterances — verify extraction.")
                ok = False
        except Exception as exc:
            print(f"  language={lang}: FAIL ({exc})")
            ok = False
    return ok


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    default_target = repo_root / "data" / "esd" / "raw"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", type=Path, default=None,
                   help="Verify an extracted ESD root directory.")
    p.add_argument("--target", type=Path, default=default_target,
                   help="Where to extract ESD after manual download.")
    args = p.parse_args()

    if args.verify is None:
        print(INSTRUCTIONS.format(target=args.target))
        return

    print(f"Verifying ESD at: {args.verify}")
    sys.path.insert(0, str(repo_root))  # so src.voz_expresiva can be imported
    ok = verify(args.verify)
    if ok:
        print("\nESD layout looks good. Ready to run 0A_extract.py.")
        sys.exit(0)
    else:
        print("\nESD layout has issues. See above warnings.")
        sys.exit(1)


if __name__ == "__main__":
    main()
