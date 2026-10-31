import pytest

from cf_h2o.eval.traffic_signal_external_estimand_aligned_freeze import (
    _group_seeds,
)


def test_group_seeds_extracts_unique_adaptation_seed_roles() -> None:
    assert _group_seeds(
        (
            "la_1x4:seed5057:0001:tls_0",
            "la_1x4:seed6067:0002:tls_1",
            "la_1x4:seed5057:0003:tls_2",
        )
    ) == {5057, 6067}


def test_group_seeds_rejects_ambiguous_identifiers() -> None:
    with pytest.raises(ValueError, match="unique simulator seed"):
        _group_seeds(("la_1x4:seed5057:seed6067:tls_0",))
