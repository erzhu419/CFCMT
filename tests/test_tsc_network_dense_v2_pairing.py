"""The corrected collector must match the preserved state before acting."""
import sys
from types import SimpleNamespace

import pytest

from experiments.tsc_network_dense_measurement_v2_20261109.collect_dense import verify_prior_snapshot


def test_original_pairing_snapshot_is_exact_and_mismatch_stops(monkeypatch):
    calls = []

    def compare(prior, current, tolerance):
        calls.append(tolerance)
        return {'passed': prior['vehicles'] == current['vehicles'] and prior['time_sec'] == current['time_sec']}

    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks', SimpleNamespace(compare_physical=compare))
    prior = dict(time_sec=610, vehicles={'car': dict(position=12.)}, pending_ids=['waiting'])
    assert all(verify_prior_snapshot(dict(prior), prior).values())
    changed = dict(prior, vehicles={'car': dict(position=12.00001)})
    with pytest.raises(ValueError, match='preserved original'):
        verify_prior_snapshot(changed, prior)
    with pytest.raises(ValueError, match='preserved original'):
        verify_prior_snapshot(dict(prior, pending_ids=[]), prior)
    assert calls == [0, 0, 0]
