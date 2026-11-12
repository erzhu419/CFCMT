"""Processed reservations must describe a physically taken or current movement."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_mean_alignment_20261110.conflict import _movements
from experiments.tsc_packet_spatial_conflict_20261110.conflict import _movements as frozen_movements
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_spatial_forward_review import crossing_kernel


def configured(packets):
    kernel = crossing_kernel(packets)
    kernel._connections = {key: kernel._connection(row) for key, row in kernel.state['packets'].items()}
    return kernel, kernel.state['packets'], kernel._connections['secondary'][0]


def test_incompatible_unaccepted_processed_candidate_cannot_create_a_borrowed_reservation():
    kernel, packets, own = configured([packet('primary', 'a', 19, 0), packet('secondary', 'b', 19, 5)])
    borrowed = kernel._connections['primary'][0]
    # A target-road candidate can borrow a sibling's connection while its
    # current ordinary lane is incompatible and its body has never entered.
    kernel.topology.lanes['a_sibling'] = dict(kernel.topology.lanes['a'], index=1)
    packets['primary'].update(lane_id='a_sibling', body_trail=['a_sibling'])
    kernel._connections['primary'] = borrowed, False
    processed = {'primary': dict(packet=packets['primary'], connection=borrowed,
        speed=0., committed=False, accepted=False)}
    speeds = {'primary': 4., 'secondary': 8.}
    old = frozen_movements(kernel, own, packets, speeds, processed)
    assert len(old) == 1 and old[0]['connection'] == borrowed
    assert old[0]['packet']['lane_id'] not in borrowed['path']
    assert not old[0]['committed'] and not old[0]['accepted']
    assert _movements(kernel, own, packets, speeds, processed) == []


def test_legal_green_approach_keeps_its_actual_processed_speed_without_entering():
    kernel, packets, own = configured([packet('primary', 'a', 19, 0), packet('secondary', 'b', 19, 5)])
    connection = kernel._connections['primary'][0]
    assert kernel._signal(connection) == 'G'
    processed = {'primary': dict(packet=packets['primary'], connection=connection,
        speed=0., committed=False, accepted=False)}
    speeds = {'primary': 4., 'secondary': 8.}
    result = _movements(kernel, own, packets, speeds, processed)
    assert result == frozen_movements(kernel, own, packets, speeds, processed)
    assert len(result) == 1 and result[0]['connection'] == connection
    assert result[0]['speed'] == 0. and result[0]['processed']


def test_real_internal_body_tail_keeps_its_old_movement_and_actual_processed_speed():
    kernel, packets, own = configured([packet('primary', 'out_a', 3, 0, trail=['Ia', 'out_a']),
        packet('secondary', 'b', 19, 5)])
    assert kernel._connections['primary'] == (None, True)
    processed = {'primary': dict(packet=packets['primary'], connection=None,
        speed=0., committed=False, accepted=False)}
    speeds = {'primary': 4., 'secondary': 8.}
    result = _movements(kernel, own, packets, speeds, processed)
    assert result == frozen_movements(kernel, own, packets, speeds, processed)
    assert result and all(row['committed'] and row['speed'] == 0. for row in result)
    assert any('Ia' in row['connection']['path'] for row in result)


def test_accepted_in_step_path_still_reserves_after_current_body_has_cleared():
    kernel, packets, own = configured([packet('primary', 'out_a', 20, 10), packet('secondary', 'b', 19, 5)])
    connection = next(c for c in kernel.topology.connections if c['from_lane'] == 'a')
    old_start = deepcopy(packets['primary'])
    old_start.update(lane_id='a', position_m=19., body_trail=['a'])
    processed = {'primary': dict(packet=old_start, connection=connection,
        speed=20., committed=False, accepted=True)}
    speeds = {'primary': 4., 'secondary': 8.}
    result = _movements(kernel, own, packets, speeds, processed)
    assert result == frozen_movements(kernel, own, packets, speeds, processed)
    assert len(result) == 1 and result[0]['accepted'] and result[0]['packet'] == old_start
    assert result[0]['speed'] == 20.


def test_whole_step_uses_corrected_binding_for_an_incompatible_processed_first_packet():
    from experiments.tsc_packet_mean_alignment_20261110.kernel import PacketKernel
    from experiments.tsc_packet_same_step_receiving_20261110.kernel import PacketKernel as FrozenKernel
    from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
    from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
    from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
        coefficients, competing_approaches, executor_seed, initial_observation,
    )

    static, laws = competing_approaches()
    for lane, shape in {'a': '-35,0 -15,0', 'Ia': '-15,0 15,0', 'out_a': '15,0 115,0',
            'b': '0,-35 0,-15', 'Ib': '0,-15 0,15', 'out_b': '0,15 0,115'}.items():
        static['lanes'][lane]['shape'] = shape
    for lane in ('Ia', 'Ib'):
        static['lanes'][lane]['length_m'] = 30.
    static['lanes']['a_sibling'] = dict(static['lanes']['a'], index=1,
        has_lane_changer=True, shape='-35,3.2 -15,3.2')
    static['lanes']['a']['has_lane_changer'] = True
    static['successors']['a_sibling'] = []
    static['predecessors']['a_sibling'] = []
    static['lane_to_tls']['a_sibling'] = []
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    initial = initial_observation(static, [packet('primary', 'a_sibling', 20, 0),
        packet('secondary', 'b', 19, 5)], executors=seeds, time_s=1)
    table = build_entity_geometry(RoadTopology(static), 2.)

    def make(kernel_class):
        return kernel_class(static, initial, seeds, laws, coefficients(),
            random_seed=0, conflict_geometry=table)

    # The frozen comparator exposes the same invalid off-path reservation as
    # the real failing cases, rather than hiding it behind a synthetic stub.
    with pytest.raises(KeyError, match='a_sibling'):
        make(FrozenKernel).step()
    corrected = make(PacketKernel)
    assert corrected.step()['ledger_passed']
    resolution = corrected._last_receiving_resolution
    assert resolution['accepted_order'].index('primary') < resolution['accepted_order'].index('secondary')
    assert not resolution['processed']['primary']['accepted']
    assert not resolution['processed']['primary']['committed']
    assert not resolution['results']['secondary']['spatial_summary']['active_conflict']
    assert corrected.state['packets']['primary']['lane_id'] == 'a_sibling'
    assert corrected.state['packets']['secondary']['lane_id'] == 'Ib'
    assert corrected.state['packets']['secondary']['position_m'] > 0.
