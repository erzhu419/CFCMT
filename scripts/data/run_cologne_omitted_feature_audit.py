"""Read omitted bank features for the fixed V155E query-neighbor pairs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from cologne_omitted_feature_audit import audit_omitted_features

PROTOCOL = 'tsc-v155f-cologne-omitted-feature-audit-v1'
BOUND = ('source_root', 'sumocfg', 'model_path', 'training_group_ids', 'folds',
         'frozen_threshold', 'selected_policy', 'proposal_accounting')


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the frozen V155F inputs and source')
    if output.exists():
        raise FileExistsError(output)
    previous = json.loads(Path(protocol['previous_result']).read_text())
    if (previous['protocol'] != 'tsc-v155e-cologne-oof-neighborhood-v1'
            or previous['status'] != 'COMPLETE'
            or any(previous[key] != protocol[key] for key in BOUND)
            or previous['feature_names'] != protocol['original_model_feature_names']
            or previous['audit']['queries'] != protocol['expected_queries']
            or previous['refs']['native_bank'] != protocol['native_bank']
            or not all(previous['checks'].values())
            or not all(previous['audit']['checks'].values())):
        raise ValueError('The fixed E actions, labels, neighbors or model inputs changed')
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg,
        read_network_right_of_way_context)

    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, protocol['training_group_ids'])
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    absolute = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    contrast = build_action_contrast_dataset(absolute, reference_policy='phase_pressure',
                                            contrast_features=CONTRAST_FEATURES_V3)
    audit = audit_omitted_features(contrast, previous['audit'], protocol['folds'],
                                   protocol['inspected_feature_names'])
    if (audit['query_count'] != 48 or audit['pair_count'] != 144
            or not all(audit['checks'].values())):
        raise ValueError('All48 queries and144 original neighbor pairs are required')
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: protocol[key] for key in BOUND},
        'inspected_feature_names': protocol['inspected_feature_names'],
        'original_model_feature_names': protocol['original_model_feature_names'],
        'audit': audit,
        'checks': {'previous_e_result_and_roster_exact': True, 'original_b100_bank_complete': True},
        'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0,
        'threshold_retuned': False, 'neighbors_reselected': False,
        'refs': {key: protocol[key] for key in ('previous_result', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'queries': audit['query_count'],
                      'pairs': audit['pair_count'], 'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
