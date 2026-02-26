#!/usr/bin/env python3
"""Human evaluation helper for Test11 perceptual outputs."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


RECOGNIZABLE_THRESHOLD = 0.80
RELATED_THRESHOLD = 0.60


@dataclass
class EvalItem:
    item_id: str
    arm: str
    task: str
    mode: str
    pred_wav: str
    pred_mid: str
    truth_wav: str
    truth_mid: str


def _detect_samples(samples_dir: Path, arm: str) -> List[EvalItem]:
    items: List[EvalItem] = []
    for pred_mid in sorted(samples_dir.glob('*_pred_*.mid')):
        stem = pred_mid.stem  # e.g. midi2events_pred_00
        if '_pred_' not in stem:
            continue

        task, idx = stem.split('_pred_')
        pred_wav = samples_dir / f'{task}_pred_{idx}.wav'
        truth_mid = samples_dir / f'{task}_truth_{idx}.mid'
        truth_wav = samples_dir / f'{task}_truth_{idx}.wav'

        mode = 'intra' if task.startswith('midi2') else 'cross'
        item_id = f'{arm}_{task}_{idx}'

        items.append(
            EvalItem(
                item_id=item_id,
                arm=arm,
                task=task,
                mode=mode,
                pred_wav=str(pred_wav),
                pred_mid=str(pred_mid),
                truth_wav=str(truth_wav),
                truth_mid=str(truth_mid),
            )
        )
    return items


def write_manifest(items: List[EvalItem], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'item_id', 'arm', 'task', 'mode',
                'pred_wav', 'pred_mid', 'truth_wav', 'truth_mid',
                'q1_recognizable', 'q2_related',
                'reviewer', 'notes',
            ],
        )
        writer.writeheader()
        for it in items:
            writer.writerow(
                {
                    'item_id': it.item_id,
                    'arm': it.arm,
                    'task': it.task,
                    'mode': it.mode,
                    'pred_wav': it.pred_wav,
                    'pred_mid': it.pred_mid,
                    'truth_wav': it.truth_wav,
                    'truth_mid': it.truth_mid,
                    'q1_recognizable': '',
                    'q2_related': '',
                    'reviewer': '',
                    'notes': '',
                }
            )


def _read_manifest(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _ask_binary(prompt: str) -> str:
    while True:
        ans = input(f'{prompt} [y/n]: ').strip().lower()
        if ans in {'y', 'n'}:
            return '1' if ans == 'y' else '0'


def run_interactive(
    csv_path: Path,
    n_intra: int = 15,
    n_cross: int = 15,
    seed: int = 42,
    reviewer: str = 'human',
) -> None:
    rows = _read_manifest(csv_path)
    intra = [r for r in rows if r.get('mode') == 'intra']
    cross = [r for r in rows if r.get('mode') == 'cross']

    rng = random.Random(seed)
    rng.shuffle(intra)
    rng.shuffle(cross)

    selected = intra[:n_intra] + cross[:n_cross]
    rng.shuffle(selected)

    print(f'Interactive eval on {len(selected)} clips ({len(intra[:n_intra])} intra, {len(cross[:n_cross])} cross).')

    selected_ids = [x['item_id'] for x in selected]
    selected_map = {x['item_id']: x for x in selected}

    for item_id in selected_ids:
        row = selected_map[item_id]
        print('')
        print(f"Item: {row['item_id']} | task={row['task']} | mode={row['mode']}")
        print(f"pred_wav:  {row['pred_wav']}")
        print(f"truth_wav: {row['truth_wav']}")
        print(f"pred_mid:  {row['pred_mid']}")
        print(f"truth_mid: {row['truth_mid']}")

        row['q1_recognizable'] = _ask_binary('Q1: Es reconocible como ejecucion de piano?')
        row['q2_related'] = _ask_binary('Q2: Se relaciona musicalmente con ground truth?')
        row['reviewer'] = reviewer

    row_by_id = {r['item_id']: r for r in rows}
    row_by_id.update(selected_map)
    merged = [row_by_id[r['item_id']] for r in rows]

    _write_rows(csv_path, merged)


def summarize(csv_path: Path) -> Dict[str, float | bool | int]:
    rows = _read_manifest(csv_path)
    labeled = [r for r in rows if r.get('q1_recognizable') in {'0', '1'} and r.get('q2_related') in {'0', '1'}]

    n = len(labeled)
    if n == 0:
        return {
            'n_labeled': 0,
            'recognizable_rate': float('nan'),
            'related_to_truth_rate': float('nan'),
            'passes_gate': False,
        }

    recog = sum(int(r['q1_recognizable']) for r in labeled) / n
    related = sum(int(r['q2_related']) for r in labeled) / n
    passes = recog >= RECOGNIZABLE_THRESHOLD and related >= RELATED_THRESHOLD

    return {
        'n_labeled': n,
        'recognizable_rate': recog,
        'related_to_truth_rate': related,
        'recognizable_threshold': RECOGNIZABLE_THRESHOLD,
        'related_threshold': RELATED_THRESHOLD,
        'passes_gate': passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Human evaluation for Test11 perceptual outputs.')
    parser.add_argument('--samples-dir', type=str, required=True)
    parser.add_argument('--arm', type=str, required=True)
    parser.add_argument('--csv', type=str, default=None)
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--n-intra', type=int, default=15)
    parser.add_argument('--n-cross', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reviewer', type=str, default='human')
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    out_csv = Path(args.csv) if args.csv else samples_dir / 'human_eval_manifest.csv'

    items = _detect_samples(samples_dir=samples_dir, arm=args.arm)
    if not items:
        raise RuntimeError(f'No *_pred_*.mid files found in {samples_dir}')

    write_manifest(items, out_csv)
    print(f'Manifest written: {out_csv} ({len(items)} items)')

    if args.interactive and not args.build_only:
        run_interactive(
            csv_path=out_csv,
            n_intra=args.n_intra,
            n_cross=args.n_cross,
            seed=args.seed,
            reviewer=args.reviewer,
        )

    summary = summarize(out_csv)
    print('Summary:')
    for k, v in summary.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
