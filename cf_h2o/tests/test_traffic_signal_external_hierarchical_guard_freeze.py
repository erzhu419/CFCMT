from pathlib import Path

from cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze import (
    _future_cache_file_count,
    _guard_payload,
    _regularizer_payload,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
)


def test_future_cache_count_requires_real_files(tmp_path: Path) -> None:
    root = tmp_path / "future"
    assert _future_cache_file_count(root) == 0
    (root / "nested").mkdir(parents=True)
    assert _future_cache_file_count(root) == 0
    (root / "nested" / "cache.npz").write_bytes(b"evidence")
    assert _future_cache_file_count(root) == 1


def test_gate_payloads_are_exact_numeric_contracts() -> None:
    guard = ContrastGuardConfig(
        enabled=True,
        risk_multiplier=0.5,
        min_context_trust=0.25,
        margin=0.1,
        max_relative_rule_gap=0.2,
    )
    regularizer = PriorRegularizationConfig(
        enabled=True,
        blend_weight=2.0,
        risk_multiplier=0.5,
        min_context_trust=0.25,
    )

    assert _guard_payload(guard) == {
        "enabled": True,
        "risk_multiplier": 0.5,
        "min_context_trust": 0.25,
        "margin": 0.1,
        "max_relative_rule_gap": 0.2,
    }
    assert _regularizer_payload(regularizer) == {
        "enabled": True,
        "blend_weight": 2.0,
        "risk_multiplier": 0.5,
        "min_context_trust": 0.25,
    }
