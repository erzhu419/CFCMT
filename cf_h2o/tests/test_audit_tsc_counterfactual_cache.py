from scripts.cluster.audit_tsc_counterfactual_cache import (
    _aggregate_cache_sha256,
)


def test_aggregate_cache_sha256_is_root_and_order_independent() -> None:
    first = _aggregate_cache_sha256({"b.npz": "2" * 64, "a.npz": "1" * 64})
    second = _aggregate_cache_sha256({"a.npz": "1" * 64, "b.npz": "2" * 64})

    assert first == second
    assert len(first) == 64


def test_aggregate_cache_sha256_binds_file_names() -> None:
    original = _aggregate_cache_sha256({"a.npz": "1" * 64})
    renamed = _aggregate_cache_sha256({"b.npz": "1" * 64})

    assert original != renamed
