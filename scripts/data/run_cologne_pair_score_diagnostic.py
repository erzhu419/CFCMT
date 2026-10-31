"""Diagnose benefit signs and switch ordering from the frozen A OOF scores."""

import argparse
import json
from pathlib import Path
import time

from cologne_pair_score_diagnostic import diagnose_pair_scores
from cologne_pair_score_summary import summarize_pair_scores

PROTOCOL = 'tsc-v155j-cologne-pair-score-diagnostic-v1'
BOUND = ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids',
         'folds', 'original_feature_names', 'threshold')


def run(protocol, output):
    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0 or protocol['tolerance'] != 1e-12:
        raise ValueError('Use the fixed J score diagnostic and original zero threshold')
    source = json.loads(Path(protocol['source_result']).read_text())
    if (source['protocol'] != 'tsc-v155i-cologne-pre-mask-ranking-v1'
            or source['status'] != 'COMPLETE'
            or any(source[key] != protocol[key] for key in BOUND)
            or len(source['original_feature_names']) != 29
            or not all(source['checks'].values()) or not all(source['audit']['checks'].values())):
        raise ValueError('Require the completed I result with original A scores and native labels')
    audit = diagnose_pair_scores(source['audit']['rows'], protocol['training_group_ids'])
    summary = summarize_pair_scores(audit['rows'])
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE', **{key: protocol[key] for key in BOUND},
        'summary': summary, 'group_diagnostics': audit['rows'],
        'checks': {**audit['checks'], **summary['checks'], 'original_a_input_binding': True},
        'accounting': {'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0,
                       'additional_server_downloads': 0, 'threshold_retuned': False,
                       'execution': 'local cached JSON only', 'elapsed_sec': time.monotonic() - started},
        'refs': {'source_result': protocol['source_result'], 'prior_plan': protocol['prior_plan']},
        'limitations': protocol['limitations'] + [protocol['comparison_scope']]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': result['status'], 'accounting': result['accounting']}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    result = run(protocol, args.out)
