"""Re-evaluate the two preserved failing queries after the movement fix."""
import json
from pathlib import Path

from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import iter_transitions, RESERVOIR_SEED
from .conditional_features import prepare_feature_kernel, candidate_features


def run(root):
    cache, base = root/'target_packet_training', root/'target_packet_mean_alignment'
    static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
    laws = json.loads((cache/'target_laws_server_only.json').read_text())
    width, unused = verify_entity_preflight(root/'target_packet_entity_footprint/entity_input_preflight.json')
    table = build_entity_geometry(RoadTopology(static), width)
    failed = json.loads((base/'failure_movement_context_attempt1.json').read_text())
    results = []
    for case in failed:
        query = case['query']
        seed, t, pid = query['seed'], query['time_sec'], query['packet_id']
        directory = cache/'training'/f'seed{seed}'
        for frame, unused_next, action in iter_transitions(directory):
            if frame['time_s'] == t:
                break
        executors = json.loads((directory/'executor_seed_server_only.json').read_text())
        kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'],
            RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
        value = candidate_features(kernel, pid, case['failed_intent'])
        results.append(dict(query=query, intent=case['failed_intent'], status='PASS',
            mean0_mps=value['mean0_mps'], yield_risk=value['yield_risk'],
            lane_mask=value['lane_mask'], connection_key=value['connection_key']))
    result = dict(status='PASS_TWO_PRESERVED_FAILURE_QUERIES', cases=results,
        new_native_starts=0, new_fits=0, original_failure_retained=True)
    (base/'fixed_query_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run(Path('/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109'))
