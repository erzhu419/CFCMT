"""Local occupancy and unresolved fallback checks; synthetic only."""
from experiments.tsc_packet_spatial_conflict_20261110.conflict import _movements, _occupancy_interval, _overlap
from test_tsc_packet_spatial_forward_review import crossing_kernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet


def test_stationary_occupied_region_has_no_fabricated_finite_clear_time():
    stopped = _occupancy_interval(20., 5., 0., (11.8, 18.2))
    assert stopped == dict(start_sec=0., end_sec=None, occupied_now=True)
    assert _overlap(dict(start_sec=100., end_sec=101.), stopped)
    assert _occupancy_interval(24., 5., 0., (11.8, 18.2)) is None
    assert _occupancy_interval(10., 5., 0., (11.8, 18.2)) is None
    # Even a retained point-contact zone remains occupied until the tail clears.
    point = _occupancy_interval(15., 5., 0., (10., 10.))
    assert point['end_sec'] is None and point['occupied_now']


def test_unresolved_active_pair_keeps_external_stop_and_internal_no_advance():
    for lane, position in (('b', 19), ('Ib', 10)):
        kernel = crossing_kernel([packet('primary', 'Ia', 15, 0),
                                  packet('secondary', lane, position, 5)])
        for pair in kernel._spatial_geometry['pairs'].values():
            pair.update(status='unresolved', zones=[], reason='synthetic_missing_shape')
        assert kernel.step()['ledger_passed']
        ego = kernel.state['packets']['secondary']
        assert ego['lane_id'] == lane
        if lane == 'Ib':
            assert ego['position_m'] == position
        else:
            assert ego['position_m'] < 20


def test_existing_lower_priority_tail_is_not_erased_by_priority_of_an_internal_front():
    kernel = crossing_kernel([packet('primary', 'Ia', 12, 5), packet('secondary', 'Ib', 20, 0)])
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['primary']['position_m'] == 12
    assert kernel.state['packets']['secondary']['position_m'] == 20


def test_processed_actual_speed_updates_all_previous_body_movement_fragments():
    kernel = crossing_kernel([packet('primary', 'out_a', 3, 0, trail=['Ia', 'out_a']),
                              packet('secondary', 'b', 19, 5)])
    packets = kernel.state['packets']
    kernel._connections = {key: kernel._connection(p) for key, p in packets.items()}
    own = kernel._connections['secondary'][0]
    processed = {'primary': dict(packet=packets['primary'], connection=kernel._connections['primary'][0],
                                speed=0., committed=False, accepted=False)}
    movements = _movements(kernel, own, packets, {'primary': 4., 'secondary': 8.}, processed)
    fragments = [row for row in movements if row['packet']['packet_id'] == 'primary']
    assert fragments and all(row['speed'] == 0 for row in fragments)


def test_internal_waiting_vehicle_still_obeys_an_approaching_priority_foe():
    kernel = crossing_kernel([packet('primary', 'a', 19, 10), packet('secondary', 'Ib', 10, 5)])
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['secondary']['position_m'] < 11.8
    assert kernel.state['packets']['primary']['lane_id'] == 'Ia'
    assert kernel.state['packets']['primary']['position_m'] > 11.8
    # A motion already resolved as stopping before the region does not invent
    # a later crossing beyond its known t=1 endpoint.
    assert _occupancy_interval(10., 5., 1.8-1e-9, (11.8, 18.2), processed=True) is None
