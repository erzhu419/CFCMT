"""Fixed4-8 actor frames for the conditional-law stage; invented data only."""
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import make_kernel, vehicle, cases as tiny_cases
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    competing_approaches, lane, laws, road_graph,
)


def three_lanes():
    return road_graph([lane('a0', 300, edge='a', index=0), lane('a1', 300, edge='a', index=1),
                       lane('a2', 300, edge='a', index=2)], [])


def cases():
    far = [vehicle('far'+str(i), 'a'+str(i % 3), 130+20*(i // 3), 1) for i in range(6)]
    out = dict(distant_background=(make_kernel(three_lanes(),
        [vehicle('focal', 'a0', 40, 5), vehicle('near', 'a1', 44, 5)]+far), 'focal'))
    out['binding_background'] = make_kernel(three_lanes(), [vehicle('focal', 'a0', 40, 5),
        vehicle('near', 'a1', 52, 10), vehicle('blocker', 'a1', 64, 0),
        vehicle('far0', 'a2', 140, 1), vehicle('far1', 'a0', 180, 1)]), 'focal'
    graph, unused = competing_approaches()
    shapes = {'a': '-35,0 -15,0', 'Ia': '-15,0 15,0', 'out_a': '15,0 115,0',
              'b': '0,-35 0,-15', 'Ib': '0,-15 0,15', 'out_b': '0,15 0,115'}
    for key, shape in shapes.items():
        graph['lanes'][key].update(shape=shape, width_m=3.2)
    for key in ('Ia', 'Ib'):
        graph['lanes'][key]['length_m'] = 30.
    graph = road_graph(list(graph['lanes'].items())+[lane('distant', 300)], graph['connections'],
                       foes=((0, 1), (1, 0)), yields=((1, 0),))
    history = laws(graph, intents={'a': {'out_a': 1.}, 'b': {'out_b': 1.},
        'out_a': {'END': 1.}, 'out_b': {'END': 1.}, 'distant': {'END': 1.}})
    out['responsive_background'] = make_kernel(graph, [vehicle('focal', 'a', 19, 15),
        vehicle('crossing_background', 'b', 19, 15)]+
        [vehicle('far'+str(i), 'distant', 50+40*i, 1) for i in range(4)], history, 'Gg'), 'focal'
    single = road_graph([lane('a', 200)], [])
    out['start_with_background'] = make_kernel(single, [vehicle('focal', 'a', 20, 0)]+
        [vehicle('far'+str(i), 'a', 80+40*i, 1) for i in range(3)]), 'focal'
    old_cases = tiny_cases()
    old, unused = old_cases['receiving_intent']
    out['receiving_with_background'] = make_kernel(old.static, [vehicle('ego', 'I1', 7, 10),
        vehicle('far0', 'out0', 60, 1), vehicle('far1', 'out1', 80, 1),
        vehicle('far2', 'next', 50, 1)], old.laws, 'GG'), 'ego'
    old, unused = old_cases['intent_and_survival']
    out['survival_with_background'] = make_kernel(old.static, [vehicle('ego', 'in', 9, 1),
        vehicle('far0', 'out', 10, 1), vehicle('far1', 'out', 30, 1),
        vehicle('far2', 'out', 45, 0)], old.laws, 'G'), 'ego'
    graph = road_graph([lane('a'+str(i), 200, edge='a', index=i) for i in range(5)], [])
    out['overflow'] = make_kernel(graph, [vehicle('focal', 'a2', 40, 5)]+
        [vehicle('near'+str(i), 'a'+str(i), 42+i, 5) for i in (0, 1, 3, 4)]), 'focal'
    return out
