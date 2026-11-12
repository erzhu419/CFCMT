"""Read archived action labels in server memory; fit and simulate nothing."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import shlex
from statistics import mean
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_stats import audit_group, summarize


def remote_json(host, path):
    proc = subprocess.run(['ssh', '-o', 'BatchMode=yes', host,
                           shlex.join(['cat', '--', str(path)])],
                          check=True, capture_output=True, text=True)
    return json.loads(proc.stdout)


def prepare_group(case, receipt, data):
    """Retain canonical action rows and recover the original integer cost grid."""
    for key in ('case_id', 'city', 'seed'):
        if receipt[key] != case[key] or data[key] != case[key]:
            raise ValueError('Archived action-group identity changed')
    if (receipt['status'] != 'COMPLETE' or len(receipt['branches']) != 8 or
            {row['candidate_index'] for row in receipt['branches']} != set(range(8)) or
            any(row['status'] != 'COMPLETE' or not all(row['checks'].values())
                for row in receipt['branches'])):
        raise ValueError('Archived same-native-state receipt failed')
    expected_movements = 16 if case['city'] == 'cologne' else 12
    if (len(data['branches']) != 8 or len(data['Y']) != 8 or len(data['costs_450']) != 8 or
            len(data['movement_ids']) != expected_movements or
            len(set(data['movement_ids'])) != expected_movements or
            data['label_names'] != ['stopline_entries_10s', 'receiving_entries_10s'] or
            len(set(data['candidate_states'])) != 8):
        raise ValueError('Archived movement or candidate roster changed')
    lanes = data['controlled_lane_count']
    if not isinstance(lanes, int) or lanes <= 0:
        raise ValueError('Original controlled-lane cost divisor is missing')
    integral10, integral450 = [], []
    receipt_rows = {row['candidate_index']: row for row in receipt['branches']}
    for index, branch in enumerate(data['branches']):
        if (branch['candidate_index'] != index or branch['labels'] != data['Y'][index] or
                branch['preaction']['movement_ids'] != data['movement_ids'] or
                branch['cost'] != data['costs_450'][index] or
                abs(receipt_rows[index]['cost']-branch['cost']) > 1e-12):
            raise ValueError('Canonical response/cost/PP rows do not align')
        costs = branch['costs']
        if len(costs) != 450 or abs(mean(costs)-branch['cost']) > 1e-12:
            raise ValueError('Archived cost does not reproduce the full450s series')
        counts = [round(value*lanes) for value in costs]
        if any(abs(value*lanes-count) > 1e-6 for value, count in zip(costs, counts)):
            raise ValueError('Original per-second cost is not an integer halted-vehicle count')
        total = sum(counts)
        if abs(branch['cost']*450*lanes-total) > 1e-6:
            raise ValueError('Original mean cost does not reproduce stopped-vehicle seconds')
        integral10.append(sum(counts[:10]))
        integral450.append(total)
    result = audit_group(case['case_id'], case['city'], case['seed'], data['Y'],
                         data['costs_450'], data['reference_index'], lanes, integral10, integral450)
    result['provenance'] = dict(sumocfg=case['sumocfg'],
        resolved_checkpoint_sec=data['resolved_checkpoint_sec'],
        receipt_checks_passed=True, canonical_action_alignment=True,
        original_cost_grid_reproduced=True)
    return result


def run(store, output):
    protocol = json.loads((Path(__file__).resolve().parent/'protocol.json').read_text())
    spec = remote_json(store, protocol['input']['original_spec'])
    cases = spec['cases']
    if (len(cases) != 28 or len({row['case_id'] for row in cases}) != 28 or
            Counter(row['seed'] for row in cases) !=
            Counter({seed: 2 for seed in spec['source_seeds']+spec['target_seeds']})):
        raise ValueError('Original four-source/ten-target two-state roster changed')
    groups = []
    for case in cases:
        directory = Path(protocol['input']['collection_root'])/f'seed_{case["seed"]}'/case['case_id']
        receipt = remote_json(store, directory/'result.json')
        data = remote_json(store, directory/'group_data_server_only.json')
        groups.append(prepare_group(case, receipt, data))
        print(f'Audited {case["case_id"]}; all8 native actions retained', flush=True)
    result = dict(status='COMPLETE_EXISTING_RESPONSE_COST_AUDIT', protocol=protocol,
                  new_fits=0, new_simulations=0, chooser_evaluations=0,
                  archived_cost_labels_exposed_for_this_diagnostic=True,
                  source_and_target_kept_separate=True,
                  **summarize(groups))
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    with gzip.open(output/'aggregate_result.json.gz', 'wt') as stream:
        json.dump(result, stream, allow_nan=False)
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.store, args.output)
