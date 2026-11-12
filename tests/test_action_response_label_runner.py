"""Admission of same-state response/cost labels with their original units."""
from copy import deepcopy
from statistics import mean

import pytest

from experiments.tsc_action_response_label_audit_20261109 import run_audit as runner


def group():
    case = dict(case_id='test', city='cologne', seed=123, sumocfg='original.sumocfg')
    movements = [str(index) for index in range(16)]
    response = [[0, 0] for _ in movements]
    branches = []
    for index in range(8):
        costs = [index/8]*10+[1/8]*440
        branches.append(dict(candidate_index=index, labels=response,
            preaction={'movement_ids': movements}, cost=mean(costs), costs=costs))
    data = dict(**case, branches=branches, Y=[response for _ in branches],
        movement_ids=movements, candidate_states=[str(index) for index in range(8)],
        costs_450=[row['cost'] for row in branches], reference_index=5,
        controlled_lane_count=8, resolved_checkpoint_sec=600,
        label_names=['stopline_entries_10s', 'receiving_entries_10s'])
    # Execution receipts can be PP-first while stored group arrays are canonical.
    receipt = dict(**case, status='COMPLETE', branches=[dict(candidate_index=index,
        status='COMPLETE', checks={'same_native_state': True}, cost=branches[index]['cost'])
        for index in (5, 0, 1, 2, 3, 4, 6, 7)])
    return case, receipt, data


def test_pp_first_receipt_maps_to_canonical_data_and_integer_cost_units():
    case, receipt, data = group()
    result = runner.prepare_group(case, receipt, data)
    assert result['provenance']['canonical_action_alignment']
    assert result['provenance']['original_cost_grid_reproduced']


def test_cost_grid_and_candidate_mapping_drift_are_rejected():
    case, receipt, data = group()
    bad = deepcopy(data)
    bad['branches'][0]['costs'][0] += .1/8
    bad['branches'][0]['cost'] = mean(bad['branches'][0]['costs'])
    bad['costs_450'][0] = bad['branches'][0]['cost']
    receipt_bad = deepcopy(receipt)
    next(row for row in receipt_bad['branches'] if row['candidate_index'] == 0)['cost'] = bad['costs_450'][0]
    with pytest.raises(ValueError, match='integer halted-vehicle'):
        runner.prepare_group(case, receipt_bad, bad)
    bad = deepcopy(data)
    bad['branches'][0]['candidate_index'] = 5
    with pytest.raises(ValueError, match='Canonical response/cost/PP'):
        runner.prepare_group(case, receipt, bad)
