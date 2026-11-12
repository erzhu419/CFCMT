"""Read-only replay of the two exact failed queries; no model step or fitting."""
import inspect
import json
from pathlib import Path

from experiments.tsc_packet_same_step_receiving_20261110 import conflict
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, _candidates, RESERVOIR_SEED
from .conditional_features import prepare_feature_kernel, candidate_features


def run(root):
    cache, base = root/'target_packet_training', root/'target_packet_mean_alignment'
    static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
    laws = json.loads((cache/'target_laws_server_only.json').read_text())
    width, unused = verify_entity_preflight(root/'target_packet_entity_footprint/entity_input_preflight.json')
    table = build_entity_geometry(RoadTopology(static), width)
    queries = json.loads((base/'failure_queries_attempt1.json').read_text())
    records = []
    original = conflict._progress
    try:
        for query in queries:
            seed, t, pid = query['seed'], query['time_sec'], query['packet_id']
            directory = cache/'training'/f'seed{seed}'
            for frame, unused_next, action in iter_transitions(directory):
                if frame['time_s'] == t:
                    break
            executors = json.loads((directory/'executor_seed_server_only.json').read_text())
            kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'],
                RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
            record = dict(query=query, off_path_progress_calls=[])
            def observed_progress(active_kernel, packet, connection):
                if packet['lane_id'] not in connection['path']:
                    local = inspect.currentframe().f_back.f_locals
                    movement = local.get('movement', {})
                    current, compatible = active_kernel._connections[packet['packet_id']]
                    record['off_path_progress_calls'].append(dict(
                        packet={key: packet[key] for key in ('packet_id', 'lane_id', 'position_m', 'body_trail', 'body_path_complete')},
                        reserved_connection=connection, current_connection=current,
                        current_compatible=bool(compatible),
                        movement={key: value for key, value in movement.items() if key not in ('packet', 'connection')},
                        queried_packet_id=local.get('packet', {}).get('packet_id')))
                return original(active_kernel, packet, connection)
            conflict._progress = observed_progress
            for intent in _candidates(kernel, pid):
                try:
                    candidate_features(kernel, pid, intent)
                except Exception as error:
                    record.update(failed_intent=intent, error=f'{type(error).__name__}: {error}')
                    break
            records.append(record)
    finally:
        conflict._progress = original
    (base/'failure_movement_context_attempt1.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    run(Path('/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109'))
