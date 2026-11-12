"""Independent full-step checks for a local crossing zone; synthetic only."""
from copy import deepcopy

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    competing_approaches, coefficients, executor_seed, initial_observation, packet,
)
from experiments.tsc_packet_spatial_conflict_20261110.kernel import PacketKernel


def crossing_kernel(packets, *, starts=False):
    static, target_laws = competing_approaches()
    shapes = {
        'a': '-35,0 -15,0', 'Ia': '-15,0 15,0', 'out_a': '15,0 115,0',
        'b': '0,-35 0,-15', 'Ib': '0,-15 0,15', 'out_b': '0,15 0,115',
    }
    for lane, shape in shapes.items():
        static['lanes'][lane].update(shape=shape, width_m=3.2)
    for lane in ('Ia', 'Ib'):
        static['lanes'][lane]['length_m'] = 30.
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    prepared = deepcopy(packets)
    for item in prepared:
        item['parameters']['min_gap_m'] = 2.5
    observed = initial_observation(static, prepared, executors=seeds)
    return PacketKernel(static, observed, seeds, target_laws,
                        coefficients(start=starts), random_seed=0)


def test_entry_can_use_internal_prefix_then_internal_progression_waits_before_crossing():
    # The declared sum-half-width capsule zone spans local arc11.8..18.2.
    # A stopped primary vehicle is in that zone, but b has an internal prefix.
    kernel = crossing_kernel([
        packet('primary', 'Ia', 15, 0),
        packet('secondary', 'b', 19, 5),
    ])
    assert kernel.step()['ledger_passed']
    ego = kernel.state['packets']['secondary']
    assert ego['lane_id'] == 'Ib' and 0 < ego['position_m'] < 11.8
    first_position = ego['position_m']
    assert kernel.step()['ledger_passed']
    ego = kernel.state['packets']['secondary']
    assert ego['lane_id'] == 'Ib' and first_position <= ego['position_m'] < 11.8
    assert kernel.state['packets']['primary']['position_m'] == 15


def test_head_past_zone_does_not_release_until_tail_is_clear():
    blocked = crossing_kernel([
        packet('primary', 'Ia', 20, 0),
        packet('secondary', 'Ib', 10, 5),
    ])
    assert blocked.step()['ledger_passed']
    assert blocked.state['packets']['secondary']['position_m'] < 11.8

    clear = crossing_kernel([
        packet('primary', 'Ia', 29, 0),
        packet('secondary', 'Ib', 10, 5),
    ])
    assert clear.step()['ledger_passed']
    assert clear.state['packets']['secondary']['position_m'] > 11.8


def test_processed_vehicle_retains_this_steps_zone_reservation_after_its_tail_exits():
    kernel = crossing_kernel([
        packet('primary', 'Ia', 10, 15),
        packet('secondary', 'Ib', 10, 15),
    ], starts=True)
    assert kernel.step()['ledger_passed']
    # The primary moves first under the unchanged directed right of way and
    # clears the zone in this step. Its old->new occupancy still blocks b's
    # overlapping in-step crossing; poststep-empty geometry alone is insufficient.
    assert kernel.state['packets']['primary']['position_m'] > 23.2
    assert kernel.state['packets']['secondary']['position_m'] < 11.8
    assert kernel.step()['ledger_passed']
    assert kernel.state['packets']['secondary']['position_m'] > 11.8


def test_internal_waiter_yields_before_zone_without_reserving_it_against_priority_approach():
    kernel = crossing_kernel([
        packet('primary', 'a', 19, 15),
        packet('secondary', 'Ib', 10, 5),
    ], starts=True)
    assert kernel.step()['ledger_passed']
    secondary = kernel.state['packets']['secondary']
    primary = kernel.state['packets']['primary']
    assert secondary['lane_id'] == 'Ib' and secondary['position_m'] < 11.8
    assert primary['lane_id'] == 'Ia' and primary['position_m'] > 11.8
