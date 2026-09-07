import json
from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    TARGET_GROUP_BUDGET,
    _combine_anchored_ensemble_scores,
    _load_protocol,
    _local_group_alpha,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_local_alpha_shrinks_large_pairwise_correction() -> None:
    alpha, diagnostics = _local_group_alpha(
        anchor_score=np.asarray([0.0, 1.0]),
        correction_score=np.asarray([0.0, 5.0]),
        anchor_uncertainty=np.asarray([0.0, 0.2]),
        correction_uncertainty=np.asarray([0.0, 0.2]),
        group_rows=np.asarray([0, 1]),
        disagreement_cap=1.0,
        confidence_multiplier=0.0,
    )

    assert alpha == 0.25
    assert diagnostics["correction_disagreement_ratio"] == 4.0


def test_local_alpha_rejects_uncertain_pairwise_switch() -> None:
    alpha, diagnostics = _local_group_alpha(
        anchor_score=np.asarray([0.0, 0.2]),
        correction_score=np.asarray([0.1, 0.0]),
        anchor_uncertainty=np.asarray([0.1, 0.1]),
        correction_uncertainty=np.asarray([1.0, 1.0]),
        group_rows=np.asarray([0, 1]),
        disagreement_cap=4.0,
        confidence_multiplier=0.5,
    )

    assert alpha == 0.0
    assert diagnostics["confidence_passed"] is False


def test_ensemble_combines_within_and_between_model_uncertainty() -> None:
    score, uncertainty, trust = _combine_anchored_ensemble_scores(
        [np.asarray([1.0, 2.0]), np.asarray([3.0, 2.0])],
        [np.asarray([2.0, 1.0]), np.asarray([2.0, 1.0])],
        [np.asarray([0.4, 0.8]), np.asarray([0.8, 0.6])],
    )

    assert np.allclose(score, [2.0, 2.0])
    assert np.allclose(uncertainty, [np.sqrt(5.0), 1.0])
    assert np.allclose(trust, [0.6, 0.7])


def test_ensemble_rejects_misaligned_members() -> None:
    try:
        _combine_anchored_ensemble_scores(
            [np.asarray([1.0]), np.asarray([2.0])],
            [np.asarray([1.0])],
            [np.asarray([0.5]), np.asarray([0.5])],
        )
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("misaligned ensemble members must be rejected")


def test_capacity_admitted_protocol_keeps_exact_b60_partition() -> None:
    payload = _load_protocol(
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v22_anchored_pairwise_capacity_admitted.json"
    )
    development = payload["development"]

    assert development["target_group_budget"] == TARGET_GROUP_BUDGET
    assert len(development["evaluation_targets"]) == 14
    assert development["source_only_capacity_exclusions"] == {
        "hangzhou_bc_tyc": 25,
        "hangzhou_kn_hz": 42,
        "hangzhou_qc_yn": 33,
        "hangzhou_sb_sx": 36,
    }


def test_capacity_protocol_rejects_target_exclusion_overlap(tmp_path: Path) -> None:
    source = PROJECT_ROOT / (
        "cf_h2o/config/traffic_signal_tsc_v22_anchored_pairwise_capacity_admitted.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["development"]["evaluation_targets"].append("hangzhou_bc_tyc")
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        _load_protocol(path)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("capacity overlap must be rejected")
