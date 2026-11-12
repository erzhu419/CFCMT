"""Recover singleton t0 lane measurements in saved graphs, without simulation."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tsc_joint_response_interface import PROTOCOL, build_joint_response_interface


def run(root, previous, output):
    output.mkdir(parents=True, exist_ok=False)
    graphs = output / 'interfaces_server_only'
    graphs.mkdir()
    cities = defaultdict(Counter)
    cases = []
    for case in json.loads((root / 'protocol.json').read_text())['cases']:
        directory = root / 'collection' / f"seed_{case['seed']}" / case['case_id']
        group = json.loads((directory / 'group_data_server_only.json').read_text())
        obs = json.loads((directory / 'first_pp/observations_server_only.json').read_text())
        old = json.loads((previous / 'interfaces_server_only' / f"{case['case_id']}.json").read_text())
        clean = {k: group[k] for k in ['case_id', 'movement_ids', 'reference_index', 'feature_names', 'X']}
        clean['branches'] = [{'candidate_index': b['candidate_index'], 'preaction': {
            k: b['preaction'][k] for k in ['nominal_states', 'nominal_request_accepted']}}
            for b in group['branches']]
        graph = build_joint_response_interface(clean, obs['frames'][0], obs['catalog'],
            obs['lane_geometry'], {n['lane_id']: n['speed_limit_mps'] for n in old['lane_nodes']},
            conflict_relations=old['conflicts'])
        # The only intended changes are measurement availability and its semantics.
        for key in old:
            if key not in ('protocol', 'lane_nodes', 'semantics'):
                assert graph[key] == old[key], (case['case_id'], key)
        counts = Counter(groups=1, candidate_views=len(graph['candidate_views']))
        for before, after in zip(old['lane_nodes'], graph['lane_nodes']):
            for key in before:
                if key != 'lane_measurements':
                    assert before[key] == after[key], (case['case_id'], key)
            if 'incoming' not in after['roles']:
                continue
            counts['incoming_lane_nodes'] += 1
            counts['previous_missing'] += before['lane_measurements'] is None
            counts[after['lane_measurements_source'] or 'still_unavailable'] += 1
            measurement = after['lane_measurements']
            if measurement is not None:
                assert measurement['vehicles'] == after['vehicle_count'], case['case_id']
                if after['lane_measurements_source'] == 'preaction_single_lane_row':
                    counts['recovered_empty_speed_null'] += measurement['mean_speed_mps'] is None
        cities[case['city']].update(counts)
        cases.append({'case_id': case['case_id'], **dict(counts)})
        (graphs / f"{case['case_id']}.json").write_text(json.dumps(graph, separators=(',', ':'), allow_nan=False) + '\n')
    result = dict(status='EXACT_LANE_RECOVERY_AUDITED', interface_protocol=PROTOCOL,
        new_fits=0, new_simulations=0, groups=len(cases), cities=dict(cities), case_results=cases,
        joint_signals_topology_supply_executor_unchanged=True,
        singleton_counts_equal_saved_vehicle_positions=True,
        previous_audit=str(previous / 'result.json'),
        graph_artifacts_bytes=sum(p.stat().st_size for p in graphs.glob('*.json')),
        empty_speed_semantics='Null for recovered empty lanes: stored zero is an aggregate convention.',
        old_response_gate='HOLD_RESPONSE')
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['status', 'interface_protocol', 'groups', 'cities']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'previous', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.previous, args.output)
