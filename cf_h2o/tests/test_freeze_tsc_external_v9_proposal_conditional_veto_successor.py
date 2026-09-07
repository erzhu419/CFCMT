from scripts.cluster.freeze_tsc_external_v9_proposal_conditional_veto_successor import (
    QUANTILES,
)


def test_successor_quantile_grid_is_small_and_predeclared() -> None:
    assert QUANTILES == (0.05, 0.075, 0.10, 0.125, 0.15)
    assert len(set(QUANTILES)) == 5
