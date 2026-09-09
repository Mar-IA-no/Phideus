"""Check the recovered fixture catalog against the original fixture functions."""
import numpy as np
import pytest

from experiments.atencion_armonica import build_generative_exclusions as b
from experiments.atencion_armonica.test_generative_evidence import fixture as generative_fixture
from experiments.atencion_armonica.test_observable_source_rivals import fixture as rival_fixture
from experiments.atencion_armonica.test_observable_rival_campaign import truth_fixture
from src.atencion_armonica.generative_evidence_exclusions import fingerprint


@pytest.fixture(scope="module")
def catalog():
    return b.mechanical_catalog()


def test_original_fixtures_match_recovered_q32(catalog):
    groups = {r["name"]: r for r in catalog["records"]}
    arithmetic = groups["arithmetic8_16_shift_ties"]["q32"]
    for i, n in enumerate((8, 16)):
        np.testing.assert_array_equal(arithmetic[i], generative_fixture(n)[0])
    np.testing.assert_array_equal(groups["rival_geometry_and_quantization"]["q32"][0], rival_fixture()[0])
    scene, _ = truth_fixture()
    assert fingerprint(groups["rival_truth_and_port_fixtures"]["q32"][0]) == fingerprint(scene["q32"])


def test_profile_observations_are_explicit_aliases(catalog):
    profiles = [r for r in catalog["records"] if r["name"].startswith("profile_")]
    assert len(profiles) == 66  # two authorities + 64 stored observations
    assert all(r["alias"] is True for r in profiles)
    assert sum(len(r["q32"]) for r in profiles) == 64
    assert len({fingerprint(q) for r in profiles for q in r["q32"]}) == 32
    assert catalog["scope"]["no_campaign_draws"] is True
    assert catalog["scope"]["status"] == "REVIEW_MATERIAL_NOT_TEST_FREEZE"


def test_output_cannot_escape_owned_review_area():
    with pytest.raises(ValueError, match="owned"):
        b.run(b.ROOT/".agent-work/phideus-exclusions-outside-negative-fixture")
