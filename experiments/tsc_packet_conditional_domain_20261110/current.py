"""Current-state geometry only; no model response, feature or kernel creation."""
from copy import deepcopy

from experiments.tsc_target_packet_kernel_20261109.scenario_noise import ScenarioNoise
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation


class GeometryView:
    """Reproduce the frozen initializer fields required by the selector.

    initialize_observation deliberately omits raw _movement_key and extra
    intent/route fields. As in the old kernel constructor, committed internal
    paths are reconstructed from CURRENT body trails and static candidates,
    using the registered nuisance seed only when those remain ambiguous.
    Ordinary-road goals are not drawn: the selector uses their prior support.
    """
    def __init__(self, static, frame, laws, applied_states, random_seed):
        self.topology = RoadTopology(static)
        self.state = initialize_observation(frame, frame['time_s'])
        self.laws = deepcopy(laws)
        self.signals = dict(applied_states)
        self.random_seed = random_seed
        active = set(self.state['packets'])
        pending = {row['packet_id'] for row in self.state['pending']}
        self.noise = ScenarioNoise(random_seed, active | pending, self.state['time_s'])
        self.internal_bindings = {}
        for key, packet in self.state['packets'].items():
            if not self.topology.lanes[packet['lane_id']]['internal']:
                continue
            candidates = list(self.topology.internal_connections[packet['lane_id']])
            compatible = [c for c in candidates if all(lane in c['path'] for lane in packet['body_trail'])]
            selected = compatible if compatible else candidates
            if not selected:
                raise ValueError('An internal packet needs a legal committed movement')
            index = int(self.noise.intent_uniform(key)*len(selected)) if len(selected) > 1 else 0
            connection = selected[index]
            packet['_movement_key'] = connection['key']
            self.internal_bindings[key] = dict(
                source='frozen_current_body_trail_and_seed', movement_key=connection['key'],
                static_candidate_count=len(candidates), compatible_body_trail_count=len(compatible),
                binding_candidate_count=len(selected), ambiguous=len(selected) > 1,
                body_trail_fallback=not bool(compatible),
                raw_movement_key_present_but_not_used='_movement_key' in frame['packets'][key])

    def _allowed_speed(self, packet):
        parameters = packet['parameters']
        return min(parameters['max_speed_mps'],
                   self.topology.lanes[packet['lane_id']]['speed_limit_mps']*parameters['speed_factor'])
