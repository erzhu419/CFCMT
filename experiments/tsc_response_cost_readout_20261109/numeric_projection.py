"""Frozen numeric dynamics and PP-anchored reader representation.

Every projection input is observed at t0 or computed by a nominal executor
clone. Future lane labels, realized signals and cost labels are never read.
"""
from math import fsum, sqrt
from statistics import mean

import numpy as np

SIGNALS = ('G', 'g', 's', 'O', 'Y', 'y', 'r')
MODES = ('green', 'yellow', 'all_red')
STATES = ('rhoN', 'rhoH', 'rhoM', 'occupancy')
RELATIONS = ('foes', 'self_yields', 'other_yields', 'shared_receiving')
LOCAL_NAMES = (
    *STATES, 'empty', 'freeflow_time_10', 'lane_limit_11_11', 'shared_turn',
    'head0_present', 'head0_distance_ratio', 'head0_speed_ratio',
    'head1_present', 'head1_distance_ratio', 'head1_speed_ratio', 'near_count',
    'length_7_5', 'min_gap_7_5', 'accel_10_limit', 'decel_10_limit', 'tau_10',
    'imperfection', 'max_speed_limit', 'speed_factor', 'allowed_speed_limit', 'parameters_observed',
    'receiving_rhoN', 'receiving_rhoH', 'receiving_rhoM', 'receiving_occupancy',
    'internal_vehicles_per_lane', 'executor_green', 'executor_yellow', 'executor_all_red',
    'executor_green_age_min_green', 'executor_remaining_10', 'executor_pending_all_red_10',
)
OWN_NAMES = tuple(f'own_t{step:02d}_{color}_{factor}' for step in range(10) for color in SIGNALS
                  for factor in ('fraction', 'rhoN', 'rhoH', 'rhoM'))
JOINT_NAMES = tuple(f'{relation}_t{step:02d}_{color}_{density}' for relation in RELATIONS
                    for step in range(10) for color in ('G', 'g') for density in ('rhoN', 'rhoH'))
DYNAMICS_FEATURE_NAMES = LOCAL_NAMES+OWN_NAMES+JOINT_NAMES
U_NAMES = ('nominal10_green', 'nominal10_yellow', 'nominal10_all_red',
           'nominal10_green_age_min_green', 'nominal10_remaining_10', 'nominal10_pending_all_red_10',
           'nominal10_feasible_eighths', 'request_differs',
           *(f'halted_weighted_exposure_{color}' for color in SIGNALS))
G0_NAMES = tuple(f'{region}_{state}' for region in ('global', 'incoming', 'receiving') for state in STATES)
PHI_NAMES = tuple(f'predicted_{state}_{statistic}' for state in STATES
                  for statistic in ('mean', 'std', 'min', 'max'))
READER_FEATURE_NAMES = U_NAMES+tuple(f'{context}_times_{action}' for context in G0_NAMES for action in U_NAMES)+PHI_NAMES


def _state0(node):
    return np.asarray([node['N']/node['storage'], node['H']/node['storage'],
                       node['M']/node['storage'], node['occupancy']], dtype=float)


def _means(nodes):
    rows = [_state0(node) for node in nodes]
    if not rows:
        raise ValueError('A declared lane region must contain observed t0 lane states')
    return np.asarray([mean(float(row[column]) for row in rows) for column in range(4)])


def _executor(executor):
    snapshot = executor['snapshot']
    if snapshot['mode'] not in MODES or executor['minimum_green_sec'] <= 0:
        raise ValueError('Executor mode and positive minimum green must be observed at input time')
    return [*(float(snapshot['mode'] == mode) for mode in MODES),
            snapshot['green_elapsed_sec']/executor['minimum_green_sec'],
            snapshot['remaining_sec']/10., snapshot['pending_all_red_sec']/10.]


def _local(inputs, node):
    state = _state0(node)
    limit = node['speed_limit_mps']
    outgoing_edges = {movement['outgoing_edge'] for movement in inputs['movements'].values()
                      if node['lane_id'] in movement['incoming_lanes']}
    values = [*state, float(node['N'] == 0), node['length_m']/(10*limit), limit/11.11,
              float(len(outgoing_edges) > 1)]
    for head in node['heads']:
        values += [float(head['present']), head['distance_ratio'] if head['present'] else 0.,
                   head['speed_ratio'] if head['present'] else 0.]
    values.append(node['near_count'])
    parameters = node['parameters']['mean']
    if node['N']:
        if parameters is None:
            raise ValueError('Present vehicles require actual observed parameters')
        values += [parameters['length_m']/7.5, parameters['min_gap_m']/7.5,
                   parameters['accel_mps2']*10/limit, parameters['decel_mps2']*10/limit,
                   parameters['tau_s']/10., parameters['imperfection'], parameters['max_speed_mps']/limit,
                   parameters['speed_factor'], parameters['allowed_speed_mps']/limit, 1.]
    else:
        values += [0.]*10
    values.extend(_means(inputs['receiving_nodes'][lane] for lane in node['to_lanes']))
    count = len(inputs['internal_nodes'])
    values.append(inputs['internal_vehicle_count']/count if count else 0.)
    values.extend(_executor(inputs['executor0']))
    return np.asarray(values, dtype=float)


def _joint(inputs, node, nodes):
    """Average relation-and-signal indicators over exact physical connection pairs.

    The denominator includes all own/other pairs. A partially permissive own
    lane therefore retains its fraction; it never becomes an any-G boolean.
    Other-lane contributions sum once per lane, with the fixed divisor eleven.
    """
    connections = inputs['connections']
    own = [row for row in connections if row['from_lane'] == node['lane_id']]
    foes = {tuple(pair) for pair in inputs['relations']['foes']}
    yields = {tuple(pair) for pair in inputs['relations']['yields_to']}
    sequences = inputs['nominal_link_sequences']
    contributions = []
    for other in nodes:
        if other['lane_id'] == node['lane_id']:
            continue
        neighbor = [row for row in connections if row['from_lane'] == other['lane_id']]
        counts = np.zeros((4, 10, 2), dtype=int)
        for a in own:
            for b in neighbor:
                i, j = a['link_index'], b['link_index']
                relation = np.asarray([(i, j) in foes, (i, j) in yields, (j, i) in yields,
                                       a['to_lane'] == b['to_lane']], dtype=int)
                for step in range(10):
                    own_color, other_color = sequences[str(i)][step], sequences[str(j)][step]
                    if own_color in 'GgsO' and other_color in ('G', 'g'):
                        counts[:, step, int(other_color == 'g')] += relation
        density = _state0(other)[:2]
        contributions.append((counts[..., None]*density/(len(own)*len(neighbor))).ravel())
    return np.asarray([fsum(float(values[index]) for values in contributions)/11.
                       for index in range(len(JOINT_NAMES))])


def project_dynamics(inputs):
    """Return (local lane join keys, float64 lane-by-476 predecision design)."""
    if not inputs['can_full_fit']:
        raise ValueError('Numeric dynamics require the complete frozen predecision interface')
    nodes = inputs['lane_nodes']
    result = []
    for node in nodes:
        state = _state0(node)
        fractions = np.asarray(node['nominal_signal_fractions'], dtype=float)
        own = (fractions[..., None]*np.r_[1., state[:3]]).ravel()
        result.append(np.r_[_local(inputs, node), own, _joint(inputs, node, nodes)])
    matrix = np.asarray(result, dtype=float)
    if matrix.shape != (len(nodes), len(DYNAMICS_FEATURE_NAMES)) or not np.all(np.isfinite(matrix)):
        raise ValueError('Frozen dynamics projection has missing fields or a changed numeric roster')
    return [node['lane_id'] for node in nodes], matrix


def project_predicted_state(inputs, delta):
    """Project predicted normalized states identically for all dynamics arms."""
    nodes = inputs['lane_nodes']
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (len(nodes), 4) or not np.all(np.isfinite(delta)):
        raise ValueError('Predicted response must align with every input lane and four target columns')
    result = {}
    for node, difference in zip(nodes, delta):
        state = _state0(node)+difference
        state[0] = max(state[0], 0.)
        state[1] = np.clip(state[1], 0., state[0])
        state[2:] = np.maximum(state[2:], 0.)
        result[node['lane_id']] = state
    return result


def _reader_u(inputs):
    executor = inputs['nominal_executor10']
    nodes = inputs['lane_nodes']
    exposures = []
    for color in range(len(SIGNALS)):
        exposures.append(mean(float(_state0(node)[1])*mean(row[color] for row in node['nominal_signal_fractions'])
                              for node in nodes))
    return np.asarray([*_executor(executor), len(executor['feasible_states'])/8.,
                       float(inputs['request_differs']), *exposures])


def _reader_g0(inputs):
    return np.r_[_means(inputs['global_nodes'].values()), _means(inputs['lane_nodes']),
                 _means(inputs['receiving_nodes'].values())]


def project_reader(inputs, predicted_state):
    """Return H=[u,g0 tensor u,Phi]; the runner computes F(a)=H(a)-H(PP).

    g0 is the shared t0 context within one native group. Subtracting H therefore
    yields precisely [delta_u,g0 tensor delta_u,delta_Phi], with exact PP zero.
    No real endpoint10 or realized executor state is accepted by this API.
    """
    lane_ids = [node['lane_id'] for node in inputs['lane_nodes']]
    if set(predicted_state) != set(lane_ids):
        raise ValueError('Reader predictions must use the same local lane join keys')
    values = np.asarray([predicted_state[lane] for lane in lane_ids], dtype=float)
    if values.shape != (len(lane_ids), 4) or not np.all(np.isfinite(values)):
        raise ValueError('Reader requires four finite predicted normalized state components')
    phi = []
    for column in values.T:
        average = mean(map(float, column))
        deviation = sqrt(mean((float(value)-average)**2 for value in column))
        phi.extend((average, deviation, float(min(column)), float(max(column))))
    u, g0 = _reader_u(inputs), _reader_g0(inputs)
    result = np.r_[u, np.outer(g0, u).ravel(), phi]
    if result.shape != (len(READER_FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError('Frozen reader projection has missing fields or a changed numeric roster')
    return result
