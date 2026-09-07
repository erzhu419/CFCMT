import numpy as np
import pytest

import cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting as v140
import cf_h2o.eval.traffic_signal_total_budget_policy_aligned_selection as v142


SELECTOR_SEEDS = (
    22208,
    27178,
    31657,
    34310,
    46535,
    47023,
    50079,
    50744,
    50945,
    51400,
    52367,
    61481,
    61994,
    63775,
    73412,
    74374,
    74744,
    77524,
    80314,
    84579,
    84751,
    88625,
)


def test_v142_seed_partition_is_exact_alternating_and_disjoint() -> None:
    adaptation, evaluation = v142.split_policy_aligned_seeds(
        tuple(reversed(SELECTOR_SEEDS))
    )
    assert adaptation == SELECTOR_SEEDS[::2]
    assert evaluation == SELECTOR_SEEDS[1::2]
    assert not set(adaptation) & set(evaluation)
    with pytest.raises(ValueError, match="exactly 22 unique"):
        v142.split_policy_aligned_seeds(SELECTOR_SEEDS[:-1])


def test_policy_aligned_candidate_family_is_frozen_and_inside_simplex() -> None:
    sources = tuple(f"source_{index}" for index in range(7))
    candidates = v140.policy_aligned_source_candidates(sources)
    assert len(candidates) == 33
    assert np.array_equal(candidates["source_null"], np.zeros(7))
    assert np.sum(candidates["uniform:mass=1"]) == pytest.approx(1.0)
    assert all(np.all(weights >= 0.0) for weights in candidates.values())
    assert all(float(np.sum(weights)) <= 1.0 for weights in candidates.values())


def _gate_arrays(*, harmful_source: bool = False):
    seeds = tuple(range(11))
    actual = []
    target = []
    source = []
    placebo = []
    references = []
    for _seed in seeds:
        actual.extend((1.0 if harmful_source else -1.0, 0.0))
        target.extend((1.0, 0.0))
        source.extend(((-2.0,), (0.0,)))
        placebo.extend(((0.0,), (0.0,)))
        references.extend((False, True))
    return {
        "actual": np.asarray(actual),
        "target_score": np.asarray(target),
        "source_residuals": np.asarray(source),
        "placebo_residuals": np.asarray(placebo),
        "policy_rows": np.arange(22, dtype=int).reshape(11, 2),
        "references": np.asarray(references, dtype=bool),
        "group_seeds": np.asarray(seeds, dtype=int),
        "adaptation_seeds": seeds,
        "source_order": ("source",),
    }


def test_policy_aligned_gate_uses_heldout_policy_value_and_beats_placebo() -> None:
    gate = v140.nested_policy_aligned_source_gate(**_gate_arrays())
    assert gate["aligned_authorized"] is True
    assert gate["selected_full_candidate_raw"] != "source_null"
    assert gate["aligned_minus_target"]["mean"] == pytest.approx(-1.0)
    assert gate["aligned_minus_placebo"]["mean"] == pytest.approx(-1.0)
    assert all(
        row["heldout_seed"] not in row["fit_seeds"] for row in gate["folds"]
    )


def test_policy_aligned_gate_returns_exact_null_for_harmful_source() -> None:
    gate = v140.nested_policy_aligned_source_gate(
        **_gate_arrays(harmful_source=True)
    )
    assert gate["aligned_authorized"] is False
    assert gate["selected_full_candidate_raw"] == "source_null"
    assert gate["aligned_full_weights_effective"] == [0.0]
