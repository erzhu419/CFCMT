"""Small description of the two measured states; no fitting or selection."""
import json
from pathlib import Path
import sys


def summarize(root):
    receipt = json.loads((root / 'result.json').read_text())
    if receipt['status'] != 'PASS_MEASUREMENT':
        raise ValueError('Do not summarize incomplete groups as complete measurements')
    rows = []
    for case in receipt['groups']:
        data = json.loads((root / case['case_id'] / 'group_data_server_only.json').read_text())
        labels, reference = data['Y'], data['reference_index']
        changed_actions, changed_cells = [], []
        for output in range(2):
            changed = [[row[output] != base[output] for row, base in zip(action, labels[reference])]
                       for i, action in enumerate(labels) if i != reference]
            changed_actions.append(sum(any(row) for row in changed))
            changed_cells.append(sum(sum(row) for row in changed))
        rows.append(dict(case_id=case['case_id'], city=case['city'], reference_index=reference,
            resolved_checkpoint_sec=case['resolved_checkpoint_sec'], **case['input_summary'],
            movements=len(data['movement_ids']), nonPP_actions_with_changed_response=changed_actions,
            nonPP_movement_cells_with_changed_response=changed_cells,
            output_order=['stopline_entries_10s', 'receiving_entries_10s']))
    return dict(status='DESCRIPTIVE_MEASUREMENT_SUMMARY', new_fits=0, new_simulations=0,
        cases=rows, scope='Two fixed states only; no source coverage or transfer-benefit claim.')


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
