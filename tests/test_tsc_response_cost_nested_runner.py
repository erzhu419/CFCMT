"""Toy-only nested split, fit-budget and cost/response leakage acceptance."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tsc_response_cost_readout_20261109 import nested_runner as n


def toy_data(monkeypatch):
    monkeypatch.setattr(n.projection, 'DYNAMICS_FEATURE_NAMES', ('supply', 'exposure', 'interaction'))
    monkeypatch.setattr(n.projection, 'READER_FEATURE_NAMES', ('nominal', 'pred_N', 'pred_H'))
    monkeypatch.setattr(n.projection, '_reader_g0', lambda inputs: np.asarray(inputs['g0']))
    monkeypatch.setattr(n.projection, 'project_dynamics',
                        lambda inputs: (inputs['lane_ids'], np.asarray(inputs['x'], dtype=float)))
    monkeypatch.setattr(n.projection, 'project_predicted_state', lambda inputs, delta: delta)
    monkeypatch.setattr(n.projection, 'project_reader', lambda inputs, predicted:
                        np.array([inputs['nominal'], predicted[:, 0].mean(), predicted[:, 1].mean()]))
    protocol = json.loads((Path(n.__file__).parent/'protocol.json').read_text())
    budget, groups = protocol['proposed_native_budget'], []
    for city, key in (('hangzhou', 'source_seeds'), ('jinan', 'target_seeds')):
        for ordinal, seed in enumerate(budget[key]):
            for anchor in budget['anchors']:
                branches = []
                for action in range(8):
                    exposure = (action-3)/7
                    x = np.array([[.2+.01*ordinal+.001*lane, exposure,
                                   exposure*(.2+.01*ordinal+.001*lane)] for lane in range(12)])
                    y = np.column_stack((.2*x[:, 0]+.3*x[:, 1], -.1*x[:, 0]+.1*x[:, 2],
                                         .3*x[:, 1]-.2*x[:, 2], .1*x[:, 0]))
                    lanes = [f'lane{lane}' for lane in range(12)]
                    branch = dict(candidate_index=action, inputs=dict(lane_ids=lanes, x=x.tolist(),
                        nominal=exposure, g0=[ordinal, anchor]),
                        targets=dict(lane_ids=lanes, columns=list(n.TARGET_COLUMNS), y=y.tolist()), cost=None)
                    if city == 'jinan':
                        branch['cost'] = 1+.01*ordinal+.001*(anchor/600)+.02*(action-2)**2
                    branches.append(branch)
                groups.append(dict(city=city, seed=seed, anchor=anchor, pp_index=3, branches=branches))
    return dict(groups=groups), protocol


def test_exact_92_fit_ledger_and_whole_seed_inner_oof_provenance(monkeypatch):
    dataset, protocol = toy_data(monkeypatch)
    events = []
    result = n.run(dataset, protocol, on_fit=lambda ledger, model: events.append(deepcopy(ledger[-1])))
    assert result['logical_fits'] == 92
    assert result['fit_counts'] == dict(source_prior=2, inner_target_dynamics=60,
                                       outer_target_dynamics=15, target_cost_reader=15)
    assert len(events) == 184 and sum(event['status'] == 'STARTED' for event in events) == 92
    for fit in result['fit_ledger']:
        assert fit['status'] == 'COMPLETE'
        assert len(fit['training_seeds']) == len(set(fit['training_seeds']))
        assert not set(fit['training_seeds']) & set(fit['held_out_seeds'])
        expected = 4 if fit['role'] == 'source_prior' else 6 if fit['role'] == 'inner_target_dynamics' else 8
        assert fit['weight_mass'] == pytest.approx(expected)
        if fit['role'] == 'target_cost_reader':
            assert fit['rows'] == 8*2*7
    assert len(result['records']) == 20*3
    assert len(result['oof_manifest']) == 5*16*3
    for row in result['oof_manifest']:
        assert row['seed'] not in row['response_training_seeds']
        assert not set(row['response_training_seeds']) & set(row['outer_held_seeds'])
        assert not set(row['reader_training_seeds']) & set(row['outer_held_seeds'])
    assert all(row['predicted_delta_costs'][row['pp_index']] == 0 for row in result['records'])
    assert result['source_450_costs_used'] is False


def test_outer_held_response_and_cost_changes_do_not_change_that_folds_predictions(monkeypatch):
    dataset, protocol = toy_data(monkeypatch)
    original = n.run(dataset, protocol)
    changed = deepcopy(dataset)
    held = set(protocol['nested_validation']['outer'][0])
    for group in changed['groups']:
        if group['city'] == 'jinan' and group['seed'] in held:
            for branch in group['branches']:
                branch['targets']['y'] = (np.asarray(branch['targets']['y'])+1000).tolist()
                branch['cost'] += 20
    updated = n.run(changed, protocol)
    before = [row for row in original['records'] if row['outer_fold'] == 0]
    after = [row for row in updated['records'] if row['outer_fold'] == 0]
    assert len(before) == len(after) == 12
    for left, right in zip(before, after):
        np.testing.assert_allclose(left['predicted_delta_costs'], right['predicted_delta_costs'], rtol=0, atol=1e-12)
        assert left['selected_action'] == right['selected_action']
        assert right['selected_cost']-left['selected_cost'] == pytest.approx(20)


def test_placebo_swaps_whole_nonPP_joint_tensors_and_t0_state_drift_is_not_fitted(monkeypatch):
    dataset, protocol = toy_data(monkeypatch)
    source = [group for group in n.prepare(dataset, protocol) if group.city == 'hangzhou']
    permuted, mappings = n.permute_source(source)
    for group, mapping in zip(source, mappings):
        np.testing.assert_array_equal(permuted[group.identity][group.pp], group.Y[group.pp])
        for action, donor in zip(mapping['recipient_actions'], mapping['donor_actions']):
            np.testing.assert_array_equal(permuted[group.identity][action], group.Y[donor])
    changed = deepcopy(dataset)
    changed['groups'][0]['branches'][1]['inputs']['g0'][0] += 1
    with pytest.raises(ValueError, match='t0 reader'):
        n.prepare(changed, protocol)


def test_paired_seed_interval_uses_t_df9_and_keeps_exact_zero_ties():
    seeds = list(range(10))
    left, right = {seed: 2-.1*seed for seed in seeds}, {seed: 2 for seed in seeds}
    summary = n.paired_summary(left, right, seeds)
    differences = np.array([-.1*seed for seed in seeds])
    radius = n.T95_DF9*differences.std(ddof=1)/np.sqrt(10)
    assert summary['mean_difference'] == pytest.approx(differences.mean())
    assert summary['approximate_paired_seed_95_t'] == pytest.approx([differences.mean()-radius, differences.mean()+radius])
    tied = n.paired_summary(right, right, seeds)
    assert tied['approximate_paired_seed_95_t'] == [0, 0]
