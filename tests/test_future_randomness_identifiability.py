from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_future_randomness_identifiability import (
    _snapshot_contains_rng_state,
    summarize_fixed_state_records,
)


def _record(checkpoint: int, seed: int, utility: float) -> dict:
    reference = [10.0 + seed, float(checkpoint)]
    alternative = [10.0 + seed - utility, float(checkpoint)]
    return {
        "checkpoint_index": checkpoint,
        "future_seed": seed,
        "valid": True,
        "reference_cumulative_cost": reference[0],
        "alternative_cumulative_cost": alternative[0],
        "reference_minus_alternative_utility": utility,
        "reference_outcome_vector": reference,
        "alternative_outcome_vector": alternative,
        **(
            {"replay_max_abs_difference": 0.0}
            if seed == 1
            else {}
        ),
    }


def test_fixed_state_summary_separates_future_and_state_variance():
    records = [
        _record(0, 1, 1.0),
        _record(0, 2, 3.0),
        _record(1, 1, 5.0),
        _record(1, 2, 7.0),
    ]

    result = summarize_fixed_state_records(records)

    assert result["checkpoint_count"] == 2
    assert result["valid_future_pair_count"] == 4
    assert result["future_randomness_effective"] is True
    assert result["future_randomness_cost_label_effective"] is True
    assert result["same_seed_replay_passed"] is True
    decomposition = result["utility_variance_decomposition"]
    assert np.isclose(decomposition["mean_within_state_future_variance"], 2.0)
    assert np.isclose(decomposition["between_state_mean_variance"], 8.0)
    assert np.isclose(decomposition["within_state_fraction"], 0.2)


def test_rng_state_tag_audit(tmp_path: Path):
    without_rng = tmp_path / "without.xml"
    without_rng.write_text("<snapshot><vehicle id='v0'/></snapshot>", encoding="utf-8")
    with_rng = tmp_path / "with.xml"
    with_rng.write_text("<snapshot><rngState default='1'/></snapshot>", encoding="utf-8")

    assert _snapshot_contains_rng_state(without_rng) is False
    assert _snapshot_contains_rng_state(with_rng) is True
