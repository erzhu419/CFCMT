from scripts.cluster.freeze_tsc_external_v9_hierarchical_protocol import (
    derive_fresh_seeds,
)


def test_fresh_seed_derivation_is_stable_and_unique() -> None:
    digest = "f5365e81a7059622663a27a561b413b4e34ac3ef23891671fdb01117407fc91c"

    seeds = derive_fresh_seeds(digest)

    assert seeds == (80314, 88625, 27178, 77524, 63775, 50945)
    assert len(seeds) == len(set(seeds)) == 6


def test_fresh_seed_derivation_binds_diagnostic_hash() -> None:
    first = derive_fresh_seeds("0" * 64)
    second = derive_fresh_seeds("1" * 64)

    assert first != second
