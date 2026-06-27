#!/usr/bin/env python3
"""E3-P0: Canonical Lissajous scene generator.

Generates deterministic scenes for Escalón 3. Each scene consists of:
- xy_audio.wav: Stereo 44.1kHz, 2s (x(t), y(t))
- xy_trace.npy: Raw coordinate array [N_samples, 2]
- figure_clean.png: 128x128 grayscale Lissajous figure
- figure_style_noisy.png: Same with additive Gaussian noise (SNR=20dB)
- figure_style_thick.png: Same with line_width=3
- meta.json: Full metadata (identity, equivalence, geometric, dynamic, parameters, renders)

Training data (P0-P2) uses ONLY reduced ratios (pure Tier 0).
Non-reduced ratios are generated for equivalence-OOD evaluation only.

Usage:
    python experiments/escalon3/generate_lissajous_dataset.py \
        --output data/escalon3/scenes \
        --workers 14
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from math import gcd
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import scipy.io.wavfile as wavfile


# ============================================================
# Ratio definitions
# ============================================================

# Group A: 16 reduced ratios (TRAINING = pure Tier 0)
REDUCED_RATIOS = [
    (1, 1), (2, 1), (3, 2), (4, 3), (5, 4), (5, 3),
    (3, 1), (4, 1), (5, 2), (7, 4), (6, 5), (7, 5),
    (8, 5), (7, 3), (8, 3), (9, 8),
]

# Group B: 6 non-reduced equivalents (equivalence-OOD ONLY)
NON_REDUCED_RATIOS = [
    (6, 4),   # equiv 3:2
    (4, 2),   # equiv 2:1
    (6, 3),   # equiv 2:1
    (8, 6),   # equiv 4:3
    (9, 6),   # equiv 3:2
    (10, 8),  # equiv 5:4
]

# Group D: 4 novel OOD ratios
OOD_RATIOS = [
    (7, 6), (9, 7), (11, 8), (11, 9),
]

# Parameter grids
BASE_FREQS = [110, 220, 330, 440, 550, 660]
SCALE_OOD_FREQS = [165, 275]
PHASES = [i * math.pi / 8 for i in range(8)]
AMP_RATIOS = [0.5, 0.75, 1.0, 1.5]

# Audio params
SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Image params
IMG_SIZE = 128


# ============================================================
# Reduction and equivalence
# ============================================================

def reduce_ratio(p: int, q: int) -> Tuple[int, int]:
    """Reduce p:q to coprime form with p >= q."""
    g = gcd(p, q)
    pr, qr = p // g, q // g
    if pr < qr:
        pr, qr = qr, pr
    return pr, qr


def build_ratio_id_map():
    """Map each reduced ratio to a unique ID (0-15).
    Also map OOD ratios to IDs 16-19."""
    ratio_map = {}
    for i, (p, q) in enumerate(REDUCED_RATIOS):
        ratio_map[(p, q)] = i
    for i, (p, q) in enumerate(OOD_RATIOS):
        ratio_map[(p, q)] = len(REDUCED_RATIOS) + i
    return ratio_map


RATIO_ID_MAP = build_ratio_id_map()


def get_equiv_id(p_raw: int, q_raw: int) -> int:
    """Get the equivalence class ID for any ratio (reduced or not).
    Returns the ratio_id of the reduced form. -1 if unknown."""
    pr, qr = reduce_ratio(p_raw, q_raw)
    return RATIO_ID_MAP.get((pr, qr), -1)


def get_ratio_id(p_raw: int, q_raw: int) -> int:
    """Get the ratio_id for the RAW (unreduced) ratio.
    For reduced ratios: same as equiv_id.
    For non-reduced: same as their reduced equivalent's ID.
    For OOD: their own unique ID (16-19)."""
    pr, qr = reduce_ratio(p_raw, q_raw)
    return RATIO_ID_MAP.get((pr, qr), -1)


# ============================================================
# Scene generation
# ============================================================

@dataclass
class SceneSpec:
    """Full specification of a Lissajous scene."""
    p_raw: int
    q_raw: int
    base_frequency: float
    phase_rad: float
    phase_idx: int
    amp_ratio: float
    split: str
    is_reduced: bool

    @property
    def p(self) -> int:
        return reduce_ratio(self.p_raw, self.q_raw)[0]

    @property
    def q(self) -> int:
        return reduce_ratio(self.p_raw, self.q_raw)[1]

    @property
    def fx(self) -> float:
        return self.base_frequency * self.p_raw

    @property
    def fy(self) -> float:
        return self.base_frequency * self.q_raw

    @property
    def ratio_float(self) -> float:
        return self.p / self.q

    @property
    def equiv_id(self) -> int:
        return get_equiv_id(self.p_raw, self.q_raw)

    @property
    def scene_id(self) -> str:
        prefix = "r" if self.is_reduced else "nr"
        return f"{prefix}{self.p_raw}_{self.q_raw}_f{int(self.base_frequency)}_p{self.phase_idx}_a{int(self.amp_ratio * 100):03d}"


def generate_xy_signal(spec: SceneSpec) -> np.ndarray:
    """Generate stereo XY audio signal using RAW frequencies. Returns [N_samples, 2] float32."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False, dtype=np.float32)
    x = np.sin(2 * np.pi * spec.fx * t + spec.phase_rad).astype(np.float32)
    y = (spec.amp_ratio * np.sin(2 * np.pi * spec.fy * t)).astype(np.float32)
    return np.stack([x, y], axis=-1)  # [N, 2]


def generate_canonical_trace(spec: SceneSpec, n_points: int = 10000) -> np.ndarray:
    """Generate canonical figure trace using REDUCED ratio.

    This ensures equivalent ratios (6:4 and 3:2) produce identical figures.
    The trace covers exactly one closure period of the reduced ratio with
    high point density, so rendering is frequency-independent.

    Returns [n_points, 2] float32.
    """
    p, q = spec.p, spec.q
    # One closure period for reduced ratio: t ∈ [0, q] in units of 1/fy_canonical
    # Use normalized parametric form: x = sin(2π·p·τ + δ), y = amp·sin(2π·q·τ)
    # where τ ∈ [0, 1] covers exactly one full Lissajous period
    tau = np.linspace(0, 1, n_points, endpoint=False, dtype=np.float32)
    x = np.sin(2 * np.pi * p * tau + spec.phase_rad).astype(np.float32)
    y = (spec.amp_ratio * np.sin(2 * np.pi * q * tau)).astype(np.float32)
    return np.stack([x, y], axis=-1)  # [n_points, 2]


def render_figure(trace: np.ndarray, line_width: int = 1,
                  noise_snr_db: Optional[float] = None,
                  noise_seed: Optional[int] = None) -> Image.Image:
    """Render Lissajous figure from trace as 128x128 grayscale image."""
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), 0)
    draw = ImageDraw.Draw(img)

    # Map trace [-1, max_amp] to pixel coords with margin
    margin = 6
    x_data = trace[:, 0]
    y_data = trace[:, 1]
    x_range = max(abs(x_data.max()), abs(x_data.min()))
    y_range = max(abs(y_data.max()), abs(y_data.min()))
    if x_range < 1e-8:
        x_range = 1.0
    if y_range < 1e-8:
        y_range = 1.0

    # Normalize to pixel space
    px = ((x_data / x_range + 1) / 2 * (IMG_SIZE - 2 * margin) + margin).astype(np.float32)
    py = ((1 - (y_data / y_range + 1) / 2) * (IMG_SIZE - 2 * margin) + margin).astype(np.float32)

    # Use all points (canonical trace already has optimal density)
    points = list(zip(px.tolist(), py.tolist()))

    if len(points) > 1:
        draw.line(points, fill=255, width=line_width)

    # Add noise if requested
    if noise_snr_db is not None:
        rng = np.random.RandomState(noise_seed)
        img_arr = np.array(img, dtype=np.float32)
        signal_power = np.mean(img_arr ** 2) + 1e-8
        noise_power = signal_power / (10 ** (noise_snr_db / 10))
        noise = rng.randn(*img_arr.shape) * np.sqrt(noise_power)
        img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode='L')

    return img


def compute_geometric_metadata(spec: SceneSpec) -> dict:
    """Compute analytical geometric metadata from reduced ratio and phase."""
    p, q = spec.p, spec.q
    phase = spec.phase_rad

    # Lobes
    lobes_x = p
    lobes_y = q

    # Self-intersections (analytical formula for coprime p:q)
    # General: (p-1)(q-1) for generic phase, fewer for special phases
    if abs(phase) < 1e-10 or abs(phase - math.pi) < 1e-10:
        # Degenerate: curve may collapse to a line segment for 1:1
        if p == q:
            n_intersections = 0
        else:
            n_intersections = max(0, (p - 1) * (q - 1) // 2)
    elif abs(phase - math.pi / 2) < 1e-10:
        n_intersections = max(0, (p - 1) * (q - 1))
    else:
        n_intersections = (p - 1) * (q - 1)

    # Eccentricity (from amplitude ratio)
    eccentricity = min(1.0, spec.amp_ratio) / max(1.0, spec.amp_ratio)

    # Symmetry (analytical)
    symmetry_x = abs(phase) < 1e-10 or abs(phase - math.pi) < 1e-10
    symmetry_y = (q % 2 == 0) or abs(phase - math.pi / 2) < 1e-10

    return {
        "lobes_x": lobes_x,
        "lobes_y": lobes_y,
        "self_intersections": n_intersections,
        "eccentricity": round(eccentricity, 4),
        "symmetry_x": symmetry_x,
        "symmetry_y": symmetry_y,
    }


def compute_dynamic_metadata(spec: SceneSpec) -> dict:
    """Compute dynamic metadata from raw frequencies."""
    fx, fy = spec.fx, spec.fy
    g = gcd(int(round(fx)), int(round(fy)))
    closure_period_s = 1.0 / g if g > 0 else float('inf')

    return {
        "base_frequency": spec.base_frequency,
        "fx": fx,
        "fy": fy,
        "closure_period_s": round(closure_period_s, 8),
        "phase_velocity": fy,
    }


def build_meta(spec: SceneSpec) -> dict:
    """Build complete meta.json for a scene."""
    geom = compute_geometric_metadata(spec)
    dyn = compute_dynamic_metadata(spec)

    tier = "0" if spec.is_reduced else "equiv_ood"
    if spec.split == "ratio_ood":
        tier = "ratio_ood"
    elif spec.split == "scale_ood":
        tier = "scale_ood"

    meta = {
        # Identity
        "scene_id": spec.scene_id,
        "p_raw": spec.p_raw,
        "q_raw": spec.q_raw,
        "is_reduced": spec.is_reduced,
        "split": spec.split,
        "tier": tier,
        # Equivalence (from reduced form)
        "p": spec.p,
        "q": spec.q,
        "ratio_float": round(spec.ratio_float, 6),
        "ratio_label": f"{spec.p}:{spec.q}",
        "ratio_id": get_ratio_id(spec.p_raw, spec.q_raw),
        "equiv_id": spec.equiv_id,
        "harmonic_order": spec.p + spec.q,
        "max_component": max(spec.p, spec.q),
        "coprime": gcd(spec.p, spec.q) == 1,
        # Geometric (from reduced — invariant under equivalence)
        **geom,
        # Dynamic (from raw fx, fy — varies with equivalence)
        **dyn,
        # Parameters
        "phase_rad": round(spec.phase_rad, 8),
        "phase_idx": spec.phase_idx,
        "amp_x": 1.0,
        "amp_y": spec.amp_ratio,
        "amp_ratio": spec.amp_ratio,
        "duration": DURATION,
        "sample_rate": SAMPLE_RATE,
        "n_samples": N_SAMPLES,
        "image_size": IMG_SIZE,
        "closed_curve": True,
        # Renders
        "renders": {
            "clean": {"line_width": 1, "noise_snr_db": None, "noise_seed": None},
            "noisy": {"line_width": 1, "noise_snr_db": 20,
                      "noise_seed": hash(spec.scene_id) % (2**31)},
            "thick": {"line_width": 3, "noise_snr_db": None, "noise_seed": None},
        },
        "activation_regime": "locked",
    }
    return meta


def generate_scene(spec: SceneSpec, output_dir: Path) -> dict:
    """Generate a complete scene and write to disk."""
    scene_dir = output_dir / spec.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    # Generate XY audio signal (uses RAW frequencies)
    audio_trace = generate_xy_signal(spec)

    # Save audio (stereo WAV)
    audio_int16 = (audio_trace * 32767).astype(np.int16)
    wavfile.write(str(scene_dir / "xy_audio.wav"), SAMPLE_RATE, audio_int16)

    # Save audio trace (raw frequencies)
    np.save(str(scene_dir / "xy_trace.npy"), audio_trace)

    # Generate CANONICAL figure trace (uses REDUCED ratio)
    # This ensures equivalent ratios produce identical figures
    fig_trace = generate_canonical_trace(spec)

    # Render figures from canonical trace
    meta = build_meta(spec)

    for style_name, style_params in meta["renders"].items():
        img = render_figure(
            fig_trace,
            line_width=style_params["line_width"],
            noise_snr_db=style_params["noise_snr_db"],
            noise_seed=style_params["noise_seed"],
        )
        if style_name == "clean":
            img.save(str(scene_dir / "figure_clean.png"))
        else:
            img.save(str(scene_dir / f"figure_style_{style_name}.png"))

    # Save metadata
    with open(scene_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    return meta


# ============================================================
# Dataset assembly
# ============================================================

def build_all_specs(output_dir: Path, seed: int = 42) -> List[SceneSpec]:
    """Build all scene specifications with split assignment."""
    rng = np.random.RandomState(seed)
    specs = []

    # Group A: Reduced ratios on standard frequencies
    group_a_specs = []
    for p, q in REDUCED_RATIOS:
        for freq in BASE_FREQS:
            for pi, phase in enumerate(PHASES):
                for amp in AMP_RATIOS:
                    group_a_specs.append(SceneSpec(
                        p_raw=p, q_raw=q,
                        base_frequency=freq,
                        phase_rad=phase, phase_idx=pi,
                        amp_ratio=amp,
                        split="",  # assigned below
                        is_reduced=True,
                    ))

    # Random 70/15/15 IID split, stratified by ratio
    ratio_groups = {}
    for spec in group_a_specs:
        key = (spec.p, spec.q)
        ratio_groups.setdefault(key, []).append(spec)

    for key, group in ratio_groups.items():
        indices = rng.permutation(len(group))
        n_train = int(0.70 * len(group))
        n_val = int(0.15 * len(group))
        for i, idx in enumerate(indices):
            if i < n_train:
                group[idx].split = "train"
            elif i < n_train + n_val:
                group[idx].split = "val"
            else:
                group[idx].split = "test"

    specs.extend(group_a_specs)

    # Group B: Non-reduced equivalents (equivalence-OOD)
    for p, q in NON_REDUCED_RATIOS:
        for freq in BASE_FREQS:
            for pi, phase in enumerate(PHASES):
                for amp in AMP_RATIOS:
                    specs.append(SceneSpec(
                        p_raw=p, q_raw=q,
                        base_frequency=freq,
                        phase_rad=phase, phase_idx=pi,
                        amp_ratio=amp,
                        split="equiv_ood",
                        is_reduced=False,
                    ))

    # Group D: Novel OOD ratios
    for p, q in OOD_RATIOS:
        for freq in BASE_FREQS:
            for pi, phase in enumerate(PHASES):
                for amp in AMP_RATIOS:
                    specs.append(SceneSpec(
                        p_raw=p, q_raw=q,
                        base_frequency=freq,
                        phase_rad=phase, phase_idx=pi,
                        amp_ratio=amp,
                        split="ratio_ood",
                        is_reduced=True,
                    ))

    # Scale-OOD: Reduced ratios on unseen frequencies
    for p, q in REDUCED_RATIOS:
        for freq in SCALE_OOD_FREQS:
            for pi, phase in enumerate(PHASES):
                for amp in AMP_RATIOS:
                    specs.append(SceneSpec(
                        p_raw=p, q_raw=q,
                        base_frequency=freq,
                        phase_rad=phase, phase_idx=pi,
                        amp_ratio=amp,
                        split="scale_ood",
                        is_reduced=True,
                    ))

    return specs


def generate_worker(args):
    """Worker for parallel generation."""
    spec, output_dir = args
    try:
        meta = generate_scene(spec, output_dir)
        return spec.scene_id, True
    except Exception as e:
        return spec.scene_id, str(e)


def main():
    parser = argparse.ArgumentParser(description="E3-P0: Lissajous scene generator")
    parser.add_argument("--output", type=str, default="data/escalon3/scenes",
                        help="Output directory for scenes")
    parser.add_argument("--workers", type=int, default=14,
                        help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split assignment")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building scene specifications...")
    specs = build_all_specs(output_dir, seed=args.seed)

    # Print summary
    split_counts = {}
    for s in specs:
        split_counts[s.split] = split_counts.get(s.split, 0) + 1
    print(f"\nTotal scenes: {len(specs)}")
    for split, count in sorted(split_counts.items()):
        print(f"  {split}: {count}")

    # Print ratio distribution
    ratio_counts = {}
    for s in specs:
        key = f"{s.p_raw}:{s.q_raw}"
        ratio_counts[key] = ratio_counts.get(key, 0) + 1
    print(f"\nRatios: {len(ratio_counts)} unique")
    for ratio, count in sorted(ratio_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {ratio}: {count}")

    # Generate scenes
    print(f"\nGenerating {len(specs)} scenes with {args.workers} workers...")
    work_items = [(spec, output_dir) for spec in specs]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = pool.map(generate_worker, work_items)
    else:
        results = [generate_worker(item) for item in work_items]

    # Report
    n_ok = sum(1 for _, status in results if status is True)
    n_fail = sum(1 for _, status in results if status is not True)
    print(f"\nDone: {n_ok} OK, {n_fail} failed")

    if n_fail > 0:
        for scene_id, status in results:
            if status is not True:
                print(f"  FAILED: {scene_id}: {status}")

    # Sanity checks
    print("\nRunning sanity checks...")
    sample_specs = [s for s in specs if s.split == "train"][:10]
    for spec in sample_specs:
        scene_dir = output_dir / spec.scene_id
        # Check files exist
        assert (scene_dir / "xy_audio.wav").exists(), f"Missing WAV: {spec.scene_id}"
        assert (scene_dir / "xy_trace.npy").exists(), f"Missing trace: {spec.scene_id}"
        assert (scene_dir / "figure_clean.png").exists(), f"Missing clean PNG: {spec.scene_id}"
        assert (scene_dir / "figure_style_noisy.png").exists(), f"Missing noisy PNG: {spec.scene_id}"
        assert (scene_dir / "figure_style_thick.png").exists(), f"Missing thick PNG: {spec.scene_id}"
        assert (scene_dir / "meta.json").exists(), f"Missing meta: {spec.scene_id}"

        # Check WAV is stereo
        sr, audio = wavfile.read(str(scene_dir / "xy_audio.wav"))
        assert sr == SAMPLE_RATE, f"Wrong SR: {sr}"
        assert audio.shape == (N_SAMPLES, 2), f"Wrong audio shape: {audio.shape}"

        # Check PNG size
        img = Image.open(scene_dir / "figure_clean.png")
        assert img.size == (IMG_SIZE, IMG_SIZE), f"Wrong image size: {img.size}"

        # Check trace
        trace = np.load(scene_dir / "xy_trace.npy")
        assert trace.shape == (N_SAMPLES, 2), f"Wrong trace shape: {trace.shape}"

        # Check meta
        with open(scene_dir / "meta.json") as f:
            meta = json.load(f)
        assert meta["p"] == spec.p, f"Wrong p: {meta['p']} vs {spec.p}"
        assert meta["q"] == spec.q, f"Wrong q: {meta['q']} vs {spec.q}"
        assert meta["split"] == spec.split, f"Wrong split: {meta['split']}"

        # Check closure
        assert meta["closed_curve"] is True
        fx, fy = meta["fx"], meta["fy"]
        expected_closure = 1.0 / gcd(int(round(fx)), int(round(fy)))
        assert abs(meta["closure_period_s"] - expected_closure) < 1e-6, \
            f"Wrong closure: {meta['closure_period_s']} vs {expected_closure}"

    print("All sanity checks passed!")

    # Check no duplicate scene_ids
    all_ids = [s.scene_id for s in specs]
    assert len(all_ids) == len(set(all_ids)), \
        f"Duplicate scene_ids! {len(all_ids)} total, {len(set(all_ids))} unique"
    print("No duplicate scene_ids.")

    # Equivalence check
    print("\nEquivalence check:")
    for p_nr, q_nr in NON_REDUCED_RATIOS[:2]:
        pr, qr = reduce_ratio(p_nr, q_nr)
        nr_dir = output_dir / f"nr{p_nr}_{q_nr}_f440_p0_a100"
        r_dir = output_dir / f"r{pr}_{qr}_f440_p0_a100"
        if nr_dir.exists() and r_dir.exists():
            img_nr = np.array(Image.open(nr_dir / "figure_clean.png"))
            img_r = np.array(Image.open(r_dir / "figure_clean.png"))
            diff = np.abs(img_nr.astype(float) - img_r.astype(float)).mean()
            print(f"  {p_nr}:{q_nr} vs {pr}:{qr}: figure diff = {diff:.2f} "
                  f"(should be ~0 for same figure)")

    print("\nGeneration complete.")


if __name__ == "__main__":
    main()
