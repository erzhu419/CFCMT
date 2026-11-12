"""Zero-fit semantic checks on the two excluded measurement-pilot states."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lane_interface import build_inputs, extract_targets, read_lane_static


def read(path):
    return json.loads(path.read_text())


def enrich_t0_speed(frame, physical_start):
    """Join a separately saved observation at exactly the same time; no future."""
    if frame['offset_sec'] != 0 or frame['time_sec'] != physical_start['time_sec']:
        raise ValueError('Physical start and observer frame are not the same t0')
    enriched = deepcopy(frame)
    for vid, vehicle in enriched['vehicles'].items():
        original = physical_start['vehicles'][vid]
        if original['lane_id'] != vehicle['lane_id'] or original['lane_position'] != vehicle['position_m']:
            raise ValueError('Current vehicle position does not match its physical-start record')
        vehicle['speed_mps'] = float(original['speed'])
    return enriched


def run(pilot_base, output):
    receipt = read(pilot_base/'result.json')
    if receipt['status'] != 'PASS_MEASUREMENT':
        raise ValueError('Use only the completed two-state measurement pilot for schema checks')
    records = []
    for group in receipt['groups']:
        case = group['case']
        directory = pilot_base/'measurement'/case['case_id']
        data = read(directory/'group_data_server_only.json')
        physical = read(directory/'physical_start_server_only.json')
        static = read_lane_static(case['sumocfg'])
        for branch in data['branches']:
            raw = directory/branch['raw_directory']
            observation = read(raw/'observations_server_only.json')
            future = read(raw/'future_execution_server_only.json')
            frame = enrich_t0_speed(observation['frames'][0], physical)
            inputs = build_inputs(future['endpoint_states']['0'], frame, observation['catalog'], static,
                                  branch['preaction']['nominal_states'], case['tls_id'])
            lane_ids = [node['lane_id'] for node in inputs['lane_nodes']]
            labels = extract_targets(future['endpoint_states']['0'], future['endpoint_states']['10'], lane_ids, static)
            fractions = np.asarray([node['nominal_signal_fractions'] for node in inputs['lane_nodes']])
            valid = (len(lane_ids) == 12 and inputs['relation_context_complete'] and
                     all(node['spatial_complete'] for node in inputs['lane_nodes']) and
                     fractions.shape == (12, 10, 7) and np.allclose(fractions.sum(axis=2), 1., rtol=0, atol=1e-12)
                     and labels['y'].shape == (12, 4) and np.isfinite(labels['y']).all())
            if not valid:
                raise ValueError('Pilot lane roles, spatial inputs, nominal signals or labels are incomplete')
            graph_path = output/'inputs_server_only'/case['case_id']/f'action_{branch["candidate_index"]}.json'
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            graph_path.write_text(json.dumps(inputs, allow_nan=False)+'\n')
            speeds = [node['mean_speed_mps']/node['speed_limit_mps']
                      for node in inputs['lane_nodes'] if node['N']]
            records.append(dict(city=case['city'], case_id=case['case_id'],
                candidate_index=branch['candidate_index'], lane_count=len(lane_ids),
                link_count=len(inputs['nominal_link_sequences']), receiving_lane_count=len(inputs['receiving_nodes']),
                lane_rows_retained=12, target_columns=labels['columns'], relation_context_complete=True,
                spatial_inputs_complete=True, speed_join='exact_same_time_vehicle_lane_position',
                max_nonempty_lane_speed_ratio=max(speeds) if speeds else None,
                parameter_observation_status=inputs['param_observation_status'],
                dynamics_inputs_fully_observed=inputs['can_full_fit'],
                nominal_executor10_observed='nominal_executor10' in branch['preaction']))
    result = dict(status='CORE_INTERFACE_PASS_NEW_OBSERVATIONS_REQUIRED',
        new_native_starts=0, real_data_model_fits=0, checked_pilot_branches=len(records),
        pilot_states_excluded_from_training=True, checked_lane_action_rows=12*len(records),
        semantic_checks_passed=True, current_pilot_records_eligible_for_new_training=False,
        required_new_collection_fields=['actual current vehicle parameters', 'all t0 executor timings',
                                        'isolated nominal executor10 state'], records=records)
    output.mkdir(parents=True, exist_ok=True)
    (output/'interface_result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], len(records), 'pilot branches; zero fits/starts', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.pilot_base, args.output)
