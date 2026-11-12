"""Pure static counts for current gates, future goals and domain obstruction."""
from types import SimpleNamespace

from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import connection, lane, laws, packet, road_graph
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


class StaticView(SimpleNamespace):
    def __getattr__(self, name):
        raise AssertionError('A symbolic budget tried to access a response/model field: '+name)


def view(rows, *, signal='g'):
    static = road_graph([lane('in', 20), lane('I0', 2, internal=True),
        lane('out0', 100, edge='out', index=0), lane('out1', 100, edge='out', index=1),
        lane('next', 100)], [connection('in', 'out0', 'I0', tls='J', link=0),
                            connection('I0', 'out0'), connection('out1', 'next')])
    topology = RoadTopology(static)
    history = laws(static, intents={'in': {'out': .7, 'END': .3, 'unused': 0.},
        'out': {'next': .25, 'END': .75, 'unused': 0.}, 'next': {'END': 1.}})
    return StaticView(topology=topology, laws=history, state={'packets': {r['packet_id']: r for r in rows}},
        signals={'J': signal}, _allowed_speed=lambda p: min(p['parameters']['max_speed_mps'],
            topology.lanes[p['lane_id']]['speed_limit_mps']*p['parameters']['speed_factor']))


def test_moving_compatible_g_and_end_count_differ_and_new_receiver_prior_is_integrated_bound():
    current = view([packet('focal', 'in', 19, 10)])
    result = symbolic_budget(current, select_stochastic(current, 'focal'))
    assert result['status'] == 'REGISTERED_SYMBOLIC_UPPER_BOUND'
    factors = {r['goal']: r for r in result['per_actor']['focal']['current_goal_factors']}
    assert factors['out']['physical_branches'] == 4  # 2 residuals * 2 yield permissions.
    assert factors['END']['physical_branches'] == 2
    assert result['physical_resolver_calls_upper_bound'] == 6
    assert result['receiving_goal_product_upper_bound'] == 2
    assert result['lateral_leaf_branches_upper_bound'] == 36


def test_start_gate_doubles_current_stopped_count_but_protected_G_does_not_add_yield_gate():
    current = view([packet('focal', 'in', 19, 0)])
    result = symbolic_budget(current, select_stochastic(current, 'focal'))
    assert result['physical_resolver_calls_upper_bound'] == 12
    current.signals['J'] = 'G'
    assert symbolic_budget(current, select_stochastic(current, 'focal'))['physical_resolver_calls_upper_bound'] == 8


def test_internal_actor_keeps_one_committed_goal_without_yield_permission_factor():
    row = packet('focal', 'I0', 1, 5)
    current = view([row])
    row['_movement_key'] = current.topology.connections[0]['key']
    result = symbolic_budget(current, select_stochastic(current, 'focal'))
    assert result['per_actor']['focal']['current_positive_goals'] == 1
    assert result['physical_resolver_calls_upper_bound'] == 2
    assert result['receiving_goal_product_upper_bound'] == 2
    assert result['lateral_leaf_branches_upper_bound'] == 12


def test_overflow_is_undefined_with_full_required_set_and_no_fake_zero_cost_bound():
    static = road_graph([lane('a0', 100, edge='a', index=0), lane('a1', 100, edge='a', index=1),
                         lane('a2', 100, edge='a', index=2)], [])
    topology = RoadTopology(static)
    rows = [packet('focal', 'a0', 20, 5), packet('near', 'a1', 24, 5),
            packet('two_lanes', 'a2', 26, 5), packet('ahead', 'a0', 35, 5)]
    current = StaticView(topology=topology, laws=laws(static), state={'packets': {r['packet_id']: r for r in rows}},
        signals={}, _allowed_speed=lambda p: 20.)
    selection = select_stochastic(current, 'focal')
    result = symbolic_budget(current, selection)
    assert result['status'] == 'UNDEFINED_SELECTOR_DOMAIN'
    assert result['required_participants'] == ['focal', 'near', 'two_lanes', 'ahead']
    assert result['participants'] == [] and result['physical_resolver_calls_upper_bound'] is None
    assert result['lateral_leaf_branches_upper_bound'] is None
