"""Producer-side separation of physical emission, observable audio and truth.

No scene drawing, file IO, checkpoint loading or campaign admission. The caller
may use these transformations only on a draw admitted by the campaign gate.
Only this producer-side adapter reads sidecar fields; render/detector do not.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from .measurement_contract import validate_identity
from .measurement_sensor import CONDITIONS, render_audio

SIDECAR_KEYS = {"scene_id", "split_seed", "sources", "sigma_cents", "mean_log_f_observed",
                "log_f_ideal", "sensor_log_noise", "source_ids", "partial_indices", "permutation"}


def emission_from_draw(unit, observation, sidecar):
    """Build two disjoint ports; never reinterpret centered q32 as absolute Hz.

Checks the alignment/reconstruction needed for rendering. The authenticated
sampler/source manifest, not these checks, establishes the exact source law.
"""
    unit = validate_identity(unit)
    if unit["condition"] != "canonical":
        raise ValueError("one emission belongs to the canonical scene identity")
    if (not isinstance(observation, dict) or set(observation) != {"scene_id", "split_seed", "log_f"}
            or not isinstance(sidecar, dict) or set(sidecar) != SIDECAR_KEYS
            or any(type(record[k]) is not int or record[k] != unit[k]
                   for record in (observation, sidecar) for k in ("scene_id", "split_seed"))
            or sidecar["sigma_cents"] != 2.):
        raise ValueError("draw identity, schema or original detuning differs")
    ideal, detuning, original = [np.asarray(x, dtype=np.float64) for x in
                                (sidecar["log_f_ideal"], sidecar["sensor_log_noise"], observation["log_f"])]
    center = sidecar["mean_log_f_observed"]
    if (original.ndim != 1 or not 8 <= len(original) <= 32
            or ideal.shape != original.shape or detuning.shape != original.shape
            or not all(np.isfinite(a).all() for a in (ideal, detuning, original))
            or type(center) not in (int, float) or not np.isfinite(center)):
        raise ValueError("invalid finite emission coordinates")
    n = len(original)
    for key in ("source_ids", "partial_indices", "permutation"):
        if (not isinstance(sidecar[key], list) or len(sidecar[key]) != n
                or any(type(x) is not int for x in sidecar[key])):
            raise ValueError("misaligned sidecar event IDs")
    if sorted(sidecar["permutation"]) != list(range(n)):
        raise ValueError("sidecar permutation differs from the event roster")
    q32 = (ideal+detuning-center).astype(np.float32)
    if not np.array_equal(original, q32.astype(np.float64)):
        raise ValueError("canonical input is not the original delivered q32")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        frequencies = np.exp(ideal+detuning)
    if not np.isfinite(frequencies).all() or np.any(frequencies <= 0):
        raise ValueError("absolute emitted frequencies must be positive and finite")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [2026091540, unit["split_seed"], unit["scene_id"]])))
    amplitude_db = rng.uniform(-12., 0., n)
    phases = rng.uniform(0., 2*np.pi, n)
    noise = rng.normal(size=24000)
    scene_identity = {k: v for k, v in unit.items() if k != "condition"}
    observable = {"metadata": {"schema": "measurement-emission-v1", "scene": scene_identity,
        "observation": deepcopy(observation), "event_ids": list(range(n)),
        "detuning_semantics": "original-2cents-emission-no-second-extraction-noise"},
        "arrays": {"frequencies": frequencies, "amplitude_db": amplitude_db,
                   "phases": phases, "noise": noise, "canonical_q32": q32}}
    truth = {"schema": "measurement-truth-v1", "scene": scene_identity,
             "sidecar": deepcopy(sidecar), "event_ids": list(range(n))}
    return observable, truth


def audio_from_emission(emission):
    """Observable port only; no source IDs or sidecar enters the renderer."""
    if (not isinstance(emission, dict) or set(emission) != {"metadata", "arrays"}
            or set(emission["metadata"]) != {"schema", "scene", "observation", "event_ids", "detuning_semantics"}
            or emission["metadata"]["schema"] != "measurement-emission-v1"
            or set(emission["arrays"]) != {"frequencies", "amplitude_db", "phases", "noise", "canonical_q32"}):
        raise ValueError("unexpected fields at renderer observable port")
    a = emission["arrays"]
    rendered = render_audio(a["frequencies"], a["amplitude_db"], a["phases"], a["noise"])
    arrays = {k: rendered[k] for k in ("clean", "amplitude", "aliased_emission", "outside_detection_band")}
    diagnostics = {"common_gain": rendered["common_gain"], "conditions": {}}
    for name, _, _ in CONDITIONS:
        row = rendered["conditions"][name]
        arrays[name] = row["waveform"]
        diagnostics["conditions"][name] = {k: v for k, v in row.items() if k != "waveform"}
    return diagnostics, arrays
