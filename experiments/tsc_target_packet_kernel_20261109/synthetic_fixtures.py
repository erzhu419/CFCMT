"""Small invented road graphs for packet-kernel acceptance; no native data."""
from copy import deepcopy

from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver


PARAMETERS = dict(length_m=5., min_gap_m=2., accel_mps2=3., decel_mps2=4.,
                  tau_s=1., imperfection=0., max_speed_mps=20., speed_factor=1.)


def packet(key, lane, position, speed=0., *, length=5., trail=None):
    parameters = dict(PARAMETERS, length_m=float(length))
    return dict(packet_id=key, lane_id=lane, position_m=float(position), speed_mps=float(speed),
        acceleration_mps2=0., lateral_position_m=0., lateral_speed_mps=0.,
        allowed_speed_mps=20., is_stopped=False, type_id='synthetic', parameters=parameters,
        body_trail=list(trail or [lane]), body_path_complete=True)


def lane(key, length, *, edge=None, index=0, internal=False):
    return key, dict(edge_id=edge or key, length_m=float(length), speed_limit_mps=20.,
        internal=internal, has_lane_changer=False, index=index, width_m=3.2,
        shape='0,0 100,0')


def connection(origin, destination, via=None, *, tls=None, link=None):
    return dict(from_lane=origin, to_lane=destination, via=via, direction='s',
                tls_id=tls, link_index=link)


def road_graph(lanes, connections, *, foes=(), yields=(), cost_lanes=None):
    geometries = dict(lanes)
    for geometry in geometries.values():
        geometry['has_lane_changer'] = sum(other['edge_id'] == geometry['edge_id']
                                           for other in geometries.values()) > 1
    successors = {key: [] for key in geometries}
    predecessors = {key: [] for key in geometries}
    lane_to_tls = {key: [] for key in geometries}
    controlled = []
    for row in connections:
        following = row['via'] or row['to_lane']
        if following not in successors[row['from_lane']]:
            successors[row['from_lane']].append(following)
            predecessors[following].append(row['from_lane'])
        if row['tls_id'] is not None:
            if row['from_lane'] not in controlled:
                controlled.append(row['from_lane'])
            pending = [row['from_lane'], following]
            while pending:
                key = pending.pop()
                if row['tls_id'] not in lane_to_tls[key]:
                    lane_to_tls[key].append(row['tls_id'])
                if geometries[key]['internal']:
                    pending.extend(successors[key])
    # Connections are declared in XML order; finish internal TLS membership
    # after all internal continuation edges have been added.
    for row in connections:
        if row['tls_id'] is None:
            continue
        pending = [row['via']] if row['via'] else []
        seen = set()
        while pending:
            key = pending.pop()
            if key in seen or not geometries[key]['internal']:
                continue
            seen.add(key)
            if row['tls_id'] not in lane_to_tls[key]:
                lane_to_tls[key].append(row['tls_id'])
            pending.extend(successors[key])
    tls_ids = {row['tls_id'] for row in connections if row['tls_id'] is not None}
    return dict(netfile='synthetic-only', lanes=geometries, connections=deepcopy(connections),
        successors=successors, predecessors=predecessors,
        internal_lane_ids=[key for key, value in geometries.items() if value['internal']],
        controlled_lane_ids=list(controlled if cost_lanes is None else cost_lanes),
        lane_to_tls=lane_to_tls,
        conflicts_by_tls={tls: dict(foes=[list(pair) for pair in foes], response=[list(pair) for pair in yields],
                                    full_context_complete=True) for tls in tls_ids})


def executor_seed(states=('G',), *, current=None, mode='green', remaining=0., min_green=5.,
                  green_age=30., yellow=2., all_red=1., clearance=()):
    green = states[0]
    return dict(snapshot=dict(tls_id='J', mode=mode, current_state=current or green,
        current_green_state=green, target_green_state=green, green_elapsed_sec=green_age,
        remaining_sec=remaining, pending_all_red_sec=0., audit={}),
        timings=[dict(state=state, min_green_sec=min_green, yellow_sec=yellow, all_red_sec=all_red)
                 for state in states], clearance_lane_ids=list(clearance), clearance_poll_sec=1.)


def initial_observation(static, packets=(), pending=(), executors=None, time_s=0):
    """Go through the real causal initializer rather than inventing extra state."""
    packets = {row['packet_id']: deepcopy(row) for row in packets}
    sensors = PacketObserver(static)._reconstruct(packets)
    seeds = deepcopy(executors or {})
    frame = dict(time_s=time_s, packets=packets, native_lanes=sensors, native_pending_ids=list(pending),
                 executors={tls: dict(snapshot=row['snapshot']) for tls, row in seeds.items()},
                 unresolved=[], checks=dict(synthetic_state_complete=True))
    return initialize_observation(frame, time_s)


def laws(static, *, intents=None, rate=0., entries=None):
    if intents is None:
        intents = {}
        for key, geometry in static['lanes'].items():
            if not geometry['internal']:
                intents.setdefault(geometry['edge_id'], {'END': 1.})
    return dict(intent_probs=deepcopy(intents), due_rate_per_sec=float(rate),
        entry_probs=dict(entries or {next(iter(static['lanes'])): 1.}),
        profiles=[dict(parameters=dict(PARAMETERS), type_id='synthetic', allowed_speed_mps=20.)])


def coefficients(*, start=1, yielding=1, lateral='stay', motion=0.):
    result = {name: [0.]*38 for name in ('motion', 'start', 'yield_entry', 'lane_left', 'lane_right')}
    result['motion'][0] = float(motion)
    result['start'][0] = 1000. if start else -1000.
    result['yield_entry'][0] = 1000. if yielding else -1000.
    result['lane_left'][0] = 1000. if lateral == 'left' else -1000.
    result['lane_right'][0] = 1000. if lateral == 'right' else -1000.
    return result


def corridor(*, incoming_length=10., internal_lengths=(2., 3.), outgoing_length=100.):
    lanes = [lane('in', incoming_length)]
    lanes += [lane(f'I{i}', length, internal=True) for i, length in enumerate(internal_lengths)]
    lanes += [lane('out', outgoing_length)]
    chain = ['in']+[f'I{i}' for i in range(len(internal_lengths))]+['out']
    connections = [connection('in', 'out', chain[1] if len(chain) > 2 else None, tls='J', link=0)]
    connections += [connection(key, 'out', chain[index+1] if index+1 < len(chain)-1 else None)
                    for index, key in enumerate(chain[1:-1], 1)]
    static = road_graph(lanes, connections)
    target_laws = laws(static, intents={'in': {'out': 1.}, 'out': {'END': 1.}})
    return static, target_laws


def competing_approaches(*, with_foes=True):
    static = road_graph(
        [lane('a', 20), lane('b', 20), lane('Ia', 10, internal=True), lane('Ib', 10, internal=True),
         lane('out_a', 100), lane('out_b', 100)],
        [connection('a', 'out_a', 'Ia', tls='J', link=0),
         connection('b', 'out_b', 'Ib', tls='J', link=1),
         connection('Ia', 'out_a'), connection('Ib', 'out_b')],
        foes=((0, 1), (1, 0)) if with_foes else (), yields=((1, 0),) if with_foes else ())
    target_laws = laws(static, intents={'a': {'out_a': 1.}, 'b': {'out_b': 1.},
                                       'out_a': {'END': 1.}, 'out_b': {'END': 1.}})
    return static, target_laws


def parallel_receivers():
    static = road_graph(
        [lane('a0', 20, edge='a', index=0), lane('a1', 20, edge='a', index=1),
         lane('I0', 8, edge=':I', index=0, internal=True), lane('I1', 8, edge=':I', index=1, internal=True),
         lane('out0', 100, edge='out', index=0), lane('out1', 100, edge='out', index=1)],
        [connection('a0', 'out0', 'I0', tls='J', link=0), connection('a1', 'out1', 'I1', tls='J', link=1),
         connection('I0', 'out0'), connection('I1', 'out1')])
    return static, laws(static, intents={'a': {'out': 1.}, 'out': {'END': 1.}})


def turn_lane_two_changes():
    """The sampled road-level turn is available only from the third lane."""
    static = road_graph(
        [lane('a0', 20, edge='a', index=0), lane('a1', 20, edge='a', index=1),
         lane('a2', 20, edge='a', index=2), lane('I0', 2, internal=True), lane('out', 100)],
        [connection('a2', 'out', 'I0', tls='J', link=0), connection('I0', 'out')],
        cost_lanes=['a0', 'a1', 'a2'])
    return static, laws(static, intents={'a': {'out': 1.}, 'out': {'END': 1.}})
