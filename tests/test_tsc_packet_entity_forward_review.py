"""Independent full-step checks using the new entity envelope table only."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110.geometry import pair_geometry
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_spatial_forward_review import crossing_kernel


def entity_kernel(packets, *, starts=False):
    kernel = crossing_kernel(packets, starts=starts)
    state, signals = deepcopy(kernel.state), deepcopy(kernel.signals)
    table = build_entity_geometry(kernel.topology, 2.)
    configure_geometry(kernel, table)
    assert kernel.state == state and kernel.signals == signals
    assert {lane['width_m'] for lane in kernel.topology.lanes.values()} == {3.2}
    first = kernel.topology.connections_from('a')[0]
    second = kernel.topology.connections_from('b')[0]
    geometry = pair_geometry(table, first, second)
    # Incoming20m plus an internal crossing at15m, entity radius2m.
    assert geometry['zones'][0]['ego_interval_m'] == pytest.approx([33., 37.])
    assert geometry['first_receiving_prefix_m'] == pytest.approx(3.2)
    assert geometry['second_receiving_prefix_m'] == pytest.approx(3.2)
    assert geometry['contact_pieces']
    return kernel


def test_entity_table_moves_internal_stop_boundary_but_keeps_tail_occupancy():
    kernel = entity_kernel([
        packet('primary', 'Ia', 18, 0),
        packet('secondary', 'b', 19, 5),
    ])
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['secondary']['lane_id'] == 'Ib'
    assert kernel.step()['ledger_passed']
    # Head18 is beyond region13..17, but its5m body still occupies it.
    assert kernel.state['packets']['secondary']['position_m'] == pytest.approx(13., abs=1e-7)
    assert kernel.state['packets']['primary']['position_m'] == 18

    clear = entity_kernel([
        packet('primary', 'Ia', 23, 0),
        packet('secondary', 'Ib', 10, 5),
    ])
    assert clear.step()['ledger_passed']
    assert clear.state['packets']['secondary']['position_m'] > 13


def test_entity_table_still_reserves_a_realized_crossing_within_the_current_step():
    kernel = entity_kernel([
        packet('primary', 'Ia', 10, 15),
        packet('secondary', 'Ib', 10, 15),
    ], starts=True)
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['primary']['position_m'] > 22
    assert kernel.state['packets']['secondary']['position_m'] == pytest.approx(13., abs=1e-7)
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['secondary']['position_m'] > 13
