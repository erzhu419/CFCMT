"""Frozen expanded folds, common100 evaluation and native collection validity."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.data import fit_cologne_expanded_data_oof as fit
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def protocol():
    path = Path(__file__).parents[1] / 'cf_h2o/results/cluster/tsc_v155k_cologne_expanded_data_oof_20260910/protocol_v1.json'
    return json.loads(path.read_text())


def bank_fixture(groups):
    row_groups = [group for group in groups for _ in range(8)]
    values = np.arange(len(row_groups), dtype=float)
    return MechanismDataset(feature_names=('marker',), features=values[:, None],
        context_names=('context',), context=np.zeros((len(values), 1)),
        priors={'interval_cost': values.copy()}, targets={'interval_cost': values.copy()},
        domains=np.asarray(['cologne'] * len(values)), metadata={'action_group_ids': row_groups})


def test_all195_exclude_heldout_seed_but_evaluation_stays_original100():
    p = protocol()
    plans = fit.expanded_fold_plan(p['original_group_ids'], p['additional_group_ids'], p['folds'])
    bank = bank_fixture(p['original_group_ids'] + p['additional_group_ids'])
    evaluated = []
    for plan, wanted in zip(plans, (129, 131, 130)):
        training, evaluation = fit.fold_datasets(bank, plan)
        training_groups = set(training.metadata['action_group_ids'])
        evaluation_groups = set(evaluation.metadata['action_group_ids'])
        assert len(training_groups) == wanted and training.size == 8 * wanted
        assert all(fit.group_seed(group) != plan['heldout_seed'] for group in training_groups)
        assert not training_groups & set(plan['excluded_additional_group_ids'])
        assert evaluation_groups <= set(p['original_group_ids'])
        assert not evaluation_groups & set(p['additional_group_ids'])
        assert len(evaluation_groups) == plan['evaluation_groups']
        # The real subset operation preserves the exact original row values and costs.
        indices = [index for index, group in enumerate(bank.metadata['action_group_ids']) if group in evaluation_groups]
        assert np.array_equal(evaluation.targets['interval_cost'], bank.targets['interval_cost'][indices])
        evaluated.extend(evaluation_groups)
    assert len(evaluated) == len(set(evaluated)) == 100
    assert set(evaluated) == set(p['original_group_ids'])


def test_accidentally_reintroduced_new_heldout_group_is_rejected():
    p = protocol()
    plan = fit.expanded_fold_plan(p['original_group_ids'], p['additional_group_ids'], p['folds'])[0]
    plan['fit_group_ids'][0] = plan['excluded_additional_group_ids'][0]
    with pytest.raises(ValueError, match='heldout-seed fit labels'):
        fit.fold_datasets(bank_fixture(p['original_group_ids'] + p['additional_group_ids']), plan)


def test_original_only_fold_counts_cannot_silently_be_reused():
    p = protocol()
    p['folds'][0]['expanded_fit_groups'] = 66
    with pytest.raises(ValueError, match='129/131/130'):
        fit.expanded_fold_plan(p['original_group_ids'], p['additional_group_ids'], p['folds'])


def records_fixture():
    p = protocol()
    old = [{'group_id': group, 'seed': fit.group_seed(group), 'candidate_states': [str(i) for i in range(8)],
            'label_costs': [1., .5, 1., 1., 1., 1., 1., 1.], 'reference_index': 0,
            'selected_index': 0, 'predicted_advantage': 0.} for group in p['original_group_ids']]
    new = deepcopy(old)
    for row in new:
        if row['seed'] == 2027:
            row.update(selected_index=1, predicted_advantage=.1)
    return p, old, new


def test_common100_keeps_complete_denominator_and_equal_seed_costs():
    p, old, new = records_fixture()
    matched, comparison = fit.compare_common_records(old, new, p['original_group_ids'])
    assert len(matched) == comparison['groups'] == 100
    assert comparison['model']['accepted_overrides'] == 34
    assert comparison['model']['mean_cost'] == pytest.approx((.5 + 1. + 1.) / 3)
    assert comparison['previous_model']['mean_cost'] == comparison['phase_pressure']['mean_cost'] == 1.


@pytest.mark.parametrize('change', ['new_group_in_evaluation', 'changed_original_label'])
def test_primary_evaluation_cannot_gain_new_groups_or_changed_original_labels(change):
    p, old, new = records_fixture()
    if change == 'new_group_in_evaluation':
        new[0]['group_id'] = p['additional_group_ids'][0]
    else:
        new[0]['label_costs'][0] += .01
    with pytest.raises(ValueError):
        fit.compare_common_records(old, new, p['original_group_ids'])


def collection_fixture(tmp_path):
    p = protocol()
    for spec in p['collection_shards']:
        spec['result'] = str(tmp_path / f"seed{spec['seed']}_shard{spec['shard']}.json")
        cell = {'protocol': fit.PROTOCOL, 'status': 'COMPLETE', 'source_root': p['source_root'], 'sumocfg': p['sumocfg'],
                'seed': spec['seed'], 'shard': spec['shard'], 'bank': spec['bank'],
                'requested_groups': spec['group_ids'], 'retained_groups': spec['group_ids'],
                'rows': 8 * len(spec['group_ids']), 'invalid_groups': [], 'raw_collision_events': 1}
        Path(spec['result']).write_text(json.dumps(cell))
    return p


def test_complete_colliding_labels_are_retained(tmp_path):
    p = collection_fixture(tmp_path)
    cells, failures = fit.collection_gate(p)
    assert failures == [] and len(cells) == 6
    assert sum(len(cell['retained_groups']) for cell in cells) == 95
    assert sum(cell['rows'] for cell in cells) == 760
    assert sum(cell['raw_collision_events'] for cell in cells) == 6


@pytest.mark.parametrize('change', ['missing_shard', 'invalid_branch', 'missing_group'])
def test_incomplete_native_labels_block_data_admission(tmp_path, change):
    p = collection_fixture(tmp_path)
    path = Path(p['collection_shards'][0]['result'])
    if change == 'missing_shard':
        path.unlink()
    else:
        cell = json.loads(path.read_text())
        if change == 'invalid_branch':
            cell['invalid_groups'] = [cell['requested_groups'][0]]
            cell['status'] = 'INVALID'
        else:
            cell['retained_groups'].pop()
            cell['rows'] -= 8
        path.write_text(json.dumps(cell))
    _, failures = fit.collection_gate(p)
    assert len(failures) == 1
