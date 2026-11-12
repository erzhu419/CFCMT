from copy import deepcopy

import pytest

from scripts.cluster import tsc_v93_selected_action_calibration as calibration


def row(seed, scenario, suffix, *, error=0.0, score=-0.2, delta=None,
        reference=False, pressure=True, old_uncertainty=1.0):
    if delta is None:
        delta = score + error
    return dict(group_id=f"{scenario}:seed{seed}:{suffix}", seed=seed, scenario=scenario,
                score=score, original_uncertainty=old_uncertainty, is_reference=reference,
                pressure_eligible=pressure, gp_cost=10.0, selected_cost=10.0 + delta,
                normalization_range=1.0)


def oof_rows(error=0.2):
    return [row(seed, scenario, "base", error=error) for seed in calibration.OOF_SEEDS
            for scenario in calibration.SCENARIOS]


def validation_rows():
    return [row(8171, scenario, "validation", score=score, delta=delta)
            for scenario, score, delta in zip(calibration.SCENARIOS, [-0.3, -0.4, -0.2], [0.4, -0.6, 0.0])]


def test_quantile_conditions_on_final_action_and_equal_nonempty_strata():
    first = calibration.SCENARIOS[0]
    records = [row(seed, scenario, "agreement", reference=True, error=100)
               for seed in calibration.OOF_SEEDS for scenario in calibration.SCENARIOS[1:]]
    records += [row(5057, first, str(i), error=0.0) for i in range(9)]
    records += [row(5057, first, "high", error=1.0), row(6067, first, "single", error=3.0)]
    records += [row(5057, first, "pressure_rejected", pressure=False, error=100)]
    result = calibration.calibrate(records)
    assert result["q"] == pytest.approx(3.0)
    assert result["eligible_group_count"] == 11
    assert result["eligible_stratum_count"] == 2
    assert len(result["empty_strata"]) == 4


def test_raw_signs_false_releases_and_fixed_scalar_application():
    records = oof_rows()
    frozen = calibration.calibrate(records)
    result = calibration.analyze_with_calibration(frozen, records, validation_rows())
    new = result["historical_validation"]["policies"]["new_guard"]
    assert [new[k] for k in ("released_action_count", "benefit_count", "harm_count", "tie_count")] == [3, 1, 1, 1]
    assert new["false_release_rate"] == pytest.approx(1 / 3)
    assert new["raw_mean_delta"] == pytest.approx(-0.2 / 3)
    assert new["cumulative_harmful_cost"] == pytest.approx(0.4)
    assert new["cumulative_beneficial_cost_reduction"] == pytest.approx(0.6)
    assert result["decision"] == "HOLD_REQUIRES_FRESH_MULTI_SEED_VALIDATION"
    assert result["closed_loop_expansion_authorized"] is False


def test_historical_validation_never_fits_or_changes_q():
    records = oof_rows()
    frozen = calibration.calibrate(records)
    before = deepcopy(frozen)
    validation = validation_rows()
    for r in validation:
        r["selected_cost"] = 1000
    result = calibration.analyze_with_calibration(frozen, records, validation)
    assert result["calibration"] == before == frozen
    assert result["decision"] == "HOLD_NO_NET_BENEFIT"
    with pytest.raises(ValueError, match="Expected only simulator seeds"):
        calibration.calibrate(validation)


def test_no_release_is_no_effect_and_not_zero_risk_evidence():
    records = oof_rows()
    frozen = calibration.calibrate(records)
    validation = validation_rows()
    for r in validation:
        r["pressure_eligible"] = False
    result = calibration.analyze_with_calibration(frozen, records, validation)
    assert result["decision"] == "HOLD_NO_EFFECT"
    assert result["historical_validation"]["policies"]["new_guard"]["false_release_rate"] is None
