"""Full observable batch and probes under external freeze/resource authority.

No sampler, sidecar reader or GPU acquisition. The same path is intended for
bounded OPEN profiling and admitted fresh observations. A batch receipt is
not the four-test global seal and never authorizes truth access.
"""
from __future__ import annotations

from copy import deepcopy

from .geometric_decision_archive_admission import verify_heads
from .geometric_decision_observables import roundtrip_observation
from .geometric_decision_pipeline import prepare_sources, prepare_inputs
from .geometric_decision_predictions import input_batch, preserve_prediction, preserve_transport
from .partial_compatibility_cache import encoded


def observed_batch(store, folder, split, observations, *, expected_seed, observation_origin,
                   checkpoints, runtime, forward, normalizers, normalization_ref, scale,
                   fit_origin, fit_candidates, check):
    """One shared geometric fit per scene, then an authenticated cached batch."""
    source = prepare_sources(store, folder, split, observations, expected_seed=expected_seed,
        observation_origin=observation_origin, checkpoints=checkpoints, runtime=runtime,
        forward=forward, check=check)
    inputs = prepare_inputs(store, folder, source, normalizers=normalizers,
        normalization_ref=normalization_ref, scale=scale, fit_origin=fit_origin,
        fit_candidates=fit_candidates, check=check)
    cache = input_batch(store, inputs, normalization_ref=normalization_ref, scale=scale, check=check)
    return source, inputs, cache


def readout_roster(store, folder, cache, head_store, heads, *, device, runtime, transport, check):
    """The complete ordered 144-member roster, never a caller-selected subset."""
    outputs = []
    for head_ref in heads:
        check()
        prediction = preserve_prediction(store, folder+"/predictions", cache, head_store, head_ref,
            device=device, runtime=runtime, check=check)
        probe = (preserve_transport(store, folder+"/transport", cache, prediction, head_store,
                                   check=check) if transport else None)
        outputs.append({"head": head_ref, "prediction": prediction, "transport": probe})
    return outputs


def run_observed(store, folder, split, observations, *, expected_seed, observation_origin,
                 checkpoints, runtime, forward, normalizers, normalization_ref, scale,
                 fit_origin, fit_candidates, head_store, archive_ref, selection_store,
                 selection_ref, device, check):
    """Original batch + all readouts + first-four full roundtrip pipelines.

    Caller supplies observations already admitted by OPEN provenance or the
    once-only draw index, and the COMPLETE archive operator's authenticated
    archive_ref. Membership in that archive is checked here before any forward.
    Interruption preserves lower-stage receipts; recovery traverses/revalidates
    those stages without repeating completed forwards or geometric fits.
    """
    verify_heads(head_store, archive_ref, selection_store, selection_ref, check=check)
    heads = deepcopy(head_store.json(archive_ref)["records"])
    observations = deepcopy(observations)
    args = dict(expected_seed=expected_seed, checkpoints=deepcopy(checkpoints), runtime=deepcopy(runtime),
        forward=forward, normalizers=deepcopy(normalizers), normalization_ref=deepcopy(normalization_ref),
        scale=scale, fit_origin=deepcopy(fit_origin), fit_candidates=fit_candidates, check=check)
    original_source, original_inputs, cache = observed_batch(store, folder+"/original", split,
        observations, observation_origin=deepcopy(observation_origin), **args)
    # Save this list before callbacks can access the cached inputs.
    indices = tuple(cache["probe_indices"])
    source_records = store.json(original_source)["sources"]
    parents = [source_records[i] for i in indices]
    derived = [roundtrip_observation(observations[i], expected_seed=expected_seed)["observation"] for i in indices]
    original = readout_roster(store, folder+"/original", cache, head_store, heads,
        device=device, runtime=runtime, transport=True, check=check)
    roundtrip = None
    if derived:
        source, inputs, transformed = observed_batch(store, folder+"/roundtrip", split, derived,
            observation_origin={"kind": "roundtrip", "parents": parents}, **args)
        outputs = readout_roster(store, folder+"/roundtrip", transformed, head_store, heads,
            device=device, runtime=runtime, transport=False, check=check)
        roundtrip = {"sources": source, "inputs": inputs, "records": outputs}
    check()
    # No head subset, late substitution or metadata mutation can be committed.
    if encoded(head_store.json(archive_ref)["records"]) != encoded(heads):
        raise ValueError("head roster changed during observed execution")
    return store.publish_json(folder+"/observable-complete.json", {
        "schema": "geometric-decision-observed-run-v1", "binding": store.binding,
        "split": split, "split_seed": expected_seed, "scene_ids": [o["scene_id"] for o in observations],
        "archive": archive_ref, "archive_binding": head_store.binding, "archive_root": str(head_store.root),
        "sources": original_source, "inputs": original_inputs, "records": original,
        "roundtrip_scene_ids": [observations[i]["scene_id"] for i in indices], "roundtrip": roundtrip,
        "truth_access": False, "global_seal": False})
