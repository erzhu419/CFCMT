"""Keep equal-seed calibration means distinct from pooled descriptive bins."""
import numpy as np
import pytest

from experiments.tsc_corridor_probability_calibration_20261109.run_diagnostic import (
    ProbabilityCalibration, calibration_summary,
)


def test_seed_means_are_not_weighted_by_number_of_repeated_flight_snapshots():
    first, second = ProbabilityCalibration(), ProbabilityCalibration()
    for _ in range(9):
        first.update(np.ones((3, 14)), np.zeros(14))
    second.update(np.zeros((3, 14)), np.zeros(14))
    summary = calibration_summary([first.result(), second.result()])
    endpoint = summary['S']['detector_10']
    assert endpoint['n'] == 10
    assert endpoint['bias'] == pytest.approx(.9)
    assert endpoint['equal_seed']['bias'] == pytest.approx(.5)
    assert endpoint['equal_seed']['brier'] == pytest.approx(.5)
    assert endpoint['nonempty_seed_count'] == 2


def test_empty_signed_queries_are_missing_instead_of_zero_calibration_error():
    first, empty = ProbabilityCalibration(), ProbabilityCalibration()
    mask = np.zeros(14, dtype=bool); mask[12] = True
    first.update(np.full((3, 14), .2), np.zeros(14), mask=mask)
    summary = calibration_summary([first.result(), empty.result()])
    service = summary['S']['service_13']
    assert service['nonempty_seed_count'] == 1
    assert service['equal_seed']['bias'] == pytest.approx(.2)
    endpoint = summary['S']['detector_10']
    assert endpoint['n'] == 0 and endpoint['nonempty_seed_count'] == 0
    assert endpoint['equal_seed']['bias'] is None
