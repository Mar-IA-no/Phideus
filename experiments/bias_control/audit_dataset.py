#!/usr/bin/env python3
"""Audit: Test MaestroSegmentDataset with real data."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_maestro_json_loading(maestro_dir: Path):
    """Test that MAESTRO JSON is loaded correctly."""
    import json

    json_path = maestro_dir / "maestro-v3.0.0.json"
    if not json_path.exists():
        print(f"  ❌ JSON not found: {json_path}")
        return False

    with open(json_path) as f:
        data = json.load(f)

    print(f"  JSON type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        first_key = next(iter(data.keys()))
        n_entries = len(data[first_key])
        print(f"  Entries: {n_entries}")
    elif isinstance(data, list):
        print(f"  Entries: {len(data)}")
        n_entries = len(data)
    else:
        print(f"  ❌ Unexpected type: {type(data)}")
        return False

    if n_entries < 1000:
        print(f"  ⚠️ Expected 1276 pieces, got {n_entries}")
        return False

    print(f"  ✅ JSON structure OK ({n_entries} pieces)")
    return True


def test_file_existence(maestro_dir: Path):
    """Test that audio/MIDI files exist."""
    import json

    json_path = maestro_dir / "maestro-v3.0.0.json"
    with open(json_path) as f:
        raw_data = json.load(f)

    # Convert to list format
    if isinstance(raw_data, dict):
        first_key = next(iter(raw_data.keys()))
        n_entries = len(raw_data[first_key])
        audio_files = raw_data.get('audio_filename', {})
        midi_files = raw_data.get('midi_filename', {})

        # Check first 10 entries
        missing_audio = 0
        missing_midi = 0
        for i in range(min(10, n_entries)):
            idx = str(i)
            audio_path = maestro_dir / (audio_files.get(idx) or audio_files.get(i, ''))
            midi_path = maestro_dir / (midi_files.get(idx) or midi_files.get(i, ''))

            if not audio_path.exists():
                missing_audio += 1
            if not midi_path.exists():
                missing_midi += 1
    else:
        missing_audio = 0
        missing_midi = 0
        for i, piece in enumerate(raw_data[:10]):
            if not (maestro_dir / piece['audio_filename']).exists():
                missing_audio += 1
            if not (maestro_dir / piece['midi_filename']).exists():
                missing_midi += 1

    print(f"  Audio missing (of 10): {missing_audio}")
    print(f"  MIDI missing (of 10): {missing_midi}")

    if missing_audio > 0 or missing_midi > 0:
        print(f"  ❌ Some files missing")
        return False

    print(f"  ✅ Files exist")
    return True


def test_dataset_loading(maestro_dir: Path):
    """Test MaestroSegmentDataset loading."""
    from src.bias_control.datasets.maestro_segments import (
        MaestroSegmentDataset,
        collate_segments,
    )

    # Test with small sample
    print("  Creating dataset (validation split, max 2 segments/piece)...")
    dataset = MaestroSegmentDataset(
        maestro_dir=maestro_dir,
        segment_len=8.0,
        hop=2.0,
        split='validation',
        max_segments_per_piece=2,
    )

    print(f"  Dataset size: {len(dataset)}")
    if len(dataset) == 0:
        print("  ❌ Dataset is empty")
        return False

    # Test single sample
    print("  Loading sample 0...")
    try:
        sample = dataset[0]
        print(f"    Audio shape: {sample['audio'].shape}")
        print(f"    MIDI events: {sample['midi_n_events']}")
        print(f"    Piece idx: {sample['piece_idx']}")

        # Validate shapes
        expected_audio_len = int(8.0 * 24000)
        if sample['audio'].shape[0] != expected_audio_len:
            print(f"    ⚠️ Audio length mismatch: expected {expected_audio_len}")

    except Exception as e:
        print(f"  ❌ Failed to load sample: {e}")
        return False

    # Test collate
    print("  Testing collate function...")
    try:
        batch = [dataset[i] for i in range(min(4, len(dataset)))]
        collated = collate_segments(batch)
        print(f"    Batch audio: {collated['audio'].shape}")
        print(f"    Batch MIDI pitch: {collated['midi_pitch'].shape}")
        print(f"    Batch MIDI mask: {collated['midi_mask'].shape}")
    except Exception as e:
        print(f"  ❌ Collate failed: {e}")
        return False

    print("  ✅ Dataset loading OK")
    return True


def test_all_splits(maestro_dir: Path):
    """Test that all splits have segments."""
    from src.bias_control.datasets.maestro_segments import MaestroSegmentDataset

    split_counts = {}
    for split in ['train', 'validation', 'test']:
        try:
            dataset = MaestroSegmentDataset(
                maestro_dir=maestro_dir,
                segment_len=8.0,
                hop=2.0,
                split=split,
                load_audio=False,
                load_midi=False,
            )
            split_counts[split] = len(dataset.segments)
        except Exception as e:
            print(f"  ❌ {split}: {e}")
            split_counts[split] = 0

    print(f"  Train segments: {split_counts['train']}")
    print(f"  Validation segments: {split_counts['validation']}")
    print(f"  Test segments: {split_counts['test']}")

    total = sum(split_counts.values())
    if total < 10000:
        print(f"  ⚠️ Total {total} < 10000 threshold")
        return False

    print(f"  ✅ Total segments: {total}")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--maestro-dir',
        type=str,
        default='data/maestro_v3/maestro-v3.0.0',
        help='Path to MAESTRO dataset'
    )
    args = parser.parse_args()

    maestro_dir = Path(args.maestro_dir)

    print("=" * 60)
    print("BIAS_CONTROL DATASET AUDIT")
    print("=" * 60)
    print(f"\nMAESTRO directory: {maestro_dir}")

    if not maestro_dir.exists():
        print(f"\n❌ MAESTRO directory not found: {maestro_dir}")
        return 1

    tests = [
        ("JSON Loading", lambda: test_maestro_json_loading(maestro_dir)),
        ("File Existence", lambda: test_file_existence(maestro_dir)),
        ("Dataset Loading", lambda: test_dataset_loading(maestro_dir)),
        ("All Splits", lambda: test_all_splits(maestro_dir)),
    ]

    all_passed = True
    for name, test_fn in tests:
        print(f"\n{name}:")
        try:
            if not test_fn():
                all_passed = False
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("STATUS: ✅ ALL DATASET TESTS PASSED")
        return 0
    else:
        print("STATUS: ❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
