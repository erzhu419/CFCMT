"""The speed join can only enrich an observation at the identical t0."""
from copy import deepcopy

import pytest

from experiments.tsc_response_cost_readout_20261109.check_interface import enrich_t0_speed


def test_exact_t0_speed_join_preserves_observed_speed_and_rejects_future_or_position_changes():
    frame = dict(time_sec=610, offset_sec=0, vehicles={'v': {'lane_id': 'lane', 'position_m': 4.}})
    start = dict(time_sec=610, vehicles={'v': {'lane_id': 'lane', 'lane_position': 4., 'speed': 12.}})
    enriched = enrich_t0_speed(frame, start)
    assert enriched['vehicles']['v']['speed_mps'] == 12.
    assert 'speed_mps' not in frame['vehicles']['v']
    bad = deepcopy(start); bad['time_sec'] = 620
    with pytest.raises(ValueError, match='same t0'):
        enrich_t0_speed(frame, bad)
    bad = deepcopy(start); bad['vehicles']['v']['lane_position'] = 4.1
    with pytest.raises(ValueError, match='position'):
        enrich_t0_speed(frame, bad)
