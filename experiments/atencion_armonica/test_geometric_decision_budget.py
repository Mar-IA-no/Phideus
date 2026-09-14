"""Budget arithmetic, immutable receipts and crash accounting with a fake clock."""
from copy import deepcopy

import pytest

from src.atencion_armonica.geometric_decision_budget import LIMITS, StageBudget, BudgetExceeded, admit_projection, owned_bytes
from src.atencion_armonica.geometric_decision_store import ArtifactStore


def fixture(tmp_path, prior=None):
    prior = prior or []
    root = tmp_path/"owned"
    store = ArtifactStore(root, binding={"limits": deepcopy(LIMITS), "prior_charges": prior,
                                        "output_roots": [str(root.resolve())]})
    now = [0.]
    options = {"prior_charges": prior, "output_roots": [root], "clock": lambda: now[0],
               "manifest_ref": store.publish_json("fixture-manifest.json", {"fixture": "arithmetic-stage"}),
               "rss": lambda: 1, "vram": lambda: 0, "disk_free": lambda: 100*1024**3,
               "bytes_used": lambda: 100}
    return store, now, options


def test_stages_keep_prior_smoke_and_partial_attempt_costs(tmp_path):
    prior = [{"stage": "open", "seconds": 3.794000603025779,
              "source": {"fixture": "the-bound-historical-smoke-receipt"}}]
    store, now, options = fixture(tmp_path, prior)
    first = StageBudget(store, "open", reservation_seconds=100., **options)
    now[0] = 12.
    first.finish("PAUSED")
    second = StageBudget(store, "open", reservation_seconds=100., **options)
    assert second.charged["open"] == prior[0]["seconds"]+12.
    now[0] = 20.
    second.finish("COMPLETE", completion={"fixture": "open-complete"})
    train = StageBudget(store, "training", reservation_seconds=120., **options)
    assert train.charged["open"] == prior[0]["seconds"]+20.
    assert train.charged["training"] == 0.
    train.finish("PAUSED")


def test_unobserved_crash_charges_full_reservation(tmp_path):
    store, now, options = fixture(tmp_path)
    StageBudget(store, "profile", reservation_seconds=120., **options)
    # Only an exclusive lock holder may start this after the prior process died.
    now[0] = 200.
    recovered = StageBudget(store, "profile", reservation_seconds=120., **options)
    assert recovered.charged["profile"] == 120.
    recovered.finish("COMPLETE")


def test_limits_exhaustion_and_finish_are_not_silent_success(tmp_path):
    store, now, options = fixture(tmp_path)
    first = StageBudget(store, "profile", reservation_seconds=600., **options)
    now[0] = 600.
    with pytest.raises(BudgetExceeded):
        first.check()
    with pytest.raises(BudgetExceeded):
        first.finish("LIMIT_REACHED")
    with pytest.raises(ValueError):
        first.finish("COMPLETE")
    with pytest.raises(BudgetExceeded):
        StageBudget(store, "profile", reservation_seconds=1., **options)


@pytest.mark.parametrize("guard", ["rss", "vram", "disk_free", "bytes_used"])
def test_sampled_resource_guards_retain_failed_attempt(tmp_path, guard):
    store, now, options = fixture(tmp_path)
    options[guard] = (lambda: 0) if guard == "disk_free" else (lambda: 200*1024**3)
    with pytest.raises(BudgetExceeded):
        StageBudget(store, "training", reservation_seconds=60., **options)
    assert store.path("attempts/0000/finish.json").exists()


def test_projection_requires_margin_without_changing_recipe():
    result = admit_projection("training", measured_seconds=10., units_measured=100,
                               remaining_units=1000, charged_seconds=10.)
    assert result["projected_remaining_seconds"] == 125.
    with pytest.raises(BudgetExceeded):
        admit_projection("training", measured_seconds=10., units_measured=1,
                         remaining_units=2000, charged_seconds=0.)


def test_late_finish_cannot_publish_complete_and_keeps_actual_cost(tmp_path):
    store, now, options = fixture(tmp_path)
    budget = StageBudget(store, "open", reservation_seconds=10., **options)
    now[0] = 10.5
    with pytest.raises(BudgetExceeded):
        budget.finish("COMPLETE", completion={"fixture": "prepared-but-over-budget"})
    record = store.json(store.reference(store.path("attempts/0000/finish.json")))
    assert record["status"] == "LIMIT_REACHED" and record["completion"] is None
    assert record["seconds"] == 10.5 and record["charged_after"]["open"] == 10.5
    resumed = StageBudget(store, "open", reservation_seconds=10., **options)
    assert resumed.charged["open"] == 10.5
    resumed.finish("PAUSED")


def test_accounting_counts_links_without_following_external_targets_or_cycles(tmp_path):
    root, outside = tmp_path/"owned", tmp_path/"external-fixture"
    root.mkdir()
    outside.mkdir()
    (root/"local.bin").write_bytes(b"fixture")
    (outside/"large.bin").write_bytes(b"x"*16384)
    links = [root/"current", root/"file-link", root/"cycle"]
    links[0].symlink_to(outside, target_is_directory=True)
    links[1].symlink_to(outside/"large.bin")
    links[2].symlink_to(root, target_is_directory=True)
    assert owned_bytes([root]) == len(b"fixture")+sum(p.lstat().st_size for p in links)
