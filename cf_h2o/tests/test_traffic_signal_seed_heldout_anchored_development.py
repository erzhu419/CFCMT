import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_seed_heldout_anchored_development import (
    ADAPTATION_SEEDS,
    EVALUATION_SEEDS,
    FOLD_COUNT,
    GROUPS_PER_FOLD,
    TARGET_GROUP_BUDGET,
    TRAINING_GROUPS_PER_FOLD,
    _load_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v41_protocol_freezes_seed_heldout_b100_contract() -> None:
    payload = _load_protocol(
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v24_seed_heldout_b100_ensemble.json"
    )
    development = payload["development"]

    assert TARGET_GROUP_BUDGET == 100
    assert FOLD_COUNT == 5
    assert GROUPS_PER_FOLD == 20
    assert TRAINING_GROUPS_PER_FOLD == 80
    assert tuple(development["adaptation_seeds"]) == ADAPTATION_SEEDS
    assert tuple(development["evaluation_seeds"]) == EVALUATION_SEEDS
    assert set(ADAPTATION_SEEDS).isdisjoint(EVALUATION_SEEDS)
    assert len(development["evaluation_targets"]) == 12
    assert set(development["source_only_capacity_exclusions"]) == {
        "cologne1",
        "ingolstadt1",
        "hangzhou_bc_tyc",
        "hangzhou_kn_hz",
        "hangzhou_qc_yn",
        "hangzhou_sb_sx",
    }


def test_v41_protocol_rejects_stale_family_name(tmp_path: Path) -> None:
    source = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v24_seed_heldout_b100_ensemble.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["development"]["base_families"][1] = (
        "causal_pairwise_preference_advantage"
    )
    path = tmp_path / "stale-family.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="base family identities"):
        _load_protocol(path)
