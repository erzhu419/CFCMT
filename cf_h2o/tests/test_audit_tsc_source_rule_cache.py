from scripts.cluster.audit_tsc_source_rule_cache import (
    _aggregate_cache_sha256,
)


def test_source_rule_aggregate_is_order_independent_and_name_bound() -> None:
    first = _aggregate_cache_sha256({"b.json": "2" * 64, "a.json": "1" * 64})
    second = _aggregate_cache_sha256({"a.json": "1" * 64, "b.json": "2" * 64})
    renamed = _aggregate_cache_sha256({"c.json": "1" * 64, "b.json": "2" * 64})

    assert first == second
    assert first != renamed
    assert len(first) == 64
