"""Toy-only acceptance for the fixed five-fit measured-response diagnostic."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.tsc_measured_response_readout_20261109 import run_readout as r


class SourceNotRead(dict):
    def __getitem__(self, key):
        if key in ('branches', 'cost', 'targets', 'inputs'):
            raise AssertionError('Source features and labels must not enter this diagnostic')
        return super().__getitem__(key)


def fixture_data():
    protocol = json.loads((Path(r.__file__).parent/'protocol.json').read_text())
    groups = [SourceNotRead(city='hangzhou', seed=1, anchor=600)]
    for ordinal, seed in enumerate(protocol['target_seeds']):
        for anchor in protocol['anchors']:
            nodes = [dict(lane_id=f'lane{lane}', N=5+.2*ordinal+.03*lane,
                          H=2+.01*lane, M=4+.04*lane, occupancy=.1+.001*lane,
                          storage=20.) for lane in range(12)]
            branches = []
            for action in range(8):
                inputs = dict(lane_nodes=deepcopy(nodes),
                    global_nodes={node['lane_id']: deepcopy(node) for node in nodes},
                    receiving_nodes={'receiver': dict(N=3, H=1, M=2, occupancy=.1, storage=20.)},
                    request_differs=action != 3,
                    nominal_executor10=dict(snapshot=dict(mode='green',
                        green_elapsed_sec=10.+action, remaining_sec=0., pending_all_red_sec=0.),
                        minimum_green_sec=30., feasible_states=['metadata']*8))
                for lane, node in enumerate(inputs['lane_nodes']):
                    color = (action+lane) % 7
                    node['nominal_signal_fractions'] = [[float(index == color) for index in range(7)]]*10
                y = [[.004*(action-3)*(ordinal+1)+.0001*lane,
                      -.002*(action-3), .006*(action-3), .001*(action-3)] for lane in range(12)]
                branches.append(dict(candidate_index=action, inputs=inputs,
                    targets=dict(lane_ids=[node['lane_id'] for node in nodes],
                                 columns=list(r.TARGET_COLUMNS), y=y),
                    cost=1+.01*ordinal+.001*anchor/600+.002*(action-2)**2))
            groups.append(dict(city='jinan', seed=seed, anchor=anchor, pp_index=3, branches=branches))
    return dict(groups=groups), protocol


def completed_models(events):
    return {entry['outer_fold']: model for entry, model in events if model is not None}


def run_with_models(dataset, protocol):
    events = []
    result = r.run(dataset, protocol, on_fit=lambda ledger, model:
                   events.append((deepcopy(ledger[-1]), model)))
    return result, events


def test_five_fits_use_112_nonpp_rows_eight_seed_mass_and_training_only_rms(monkeypatch):
    dataset, protocol = fixture_data()
    monkeypatch.setattr(r.projection, 'project_dynamics', lambda *_:
                        pytest.fail('No dynamics projection or source model is needed'))
    fitted_rows = []
    real_fit = r.fit_reader

    def capture_fit(X, y, w, alpha=1):
        fitted_rows.append((np.array(X), np.array(w)))
        return real_fit(X, y, w, alpha=alpha)

    monkeypatch.setattr(r, 'fit_reader', capture_fit)
    result, events = run_with_models(dataset, protocol)
    assert result['logical_fits'] == 5
    assert len(result['fit_ledger']) == 5
    assert len(events) == 10 and len(fitted_rows) == 5
    groups = r.prepare(dataset, protocol)
    for entry, (X, weights) in zip(result['fit_ledger'], fitted_rows):
        held = protocol['outer_folds'][entry['outer_fold']]
        train = [group for group in groups if group.seed not in held]
        assert entry['status'] == 'COMPLETE'
        assert entry['training_seeds'] == sorted({group.seed for group in train})
        assert entry['held_out_seeds'] == held
        assert len(entry['training_seeds']) == 8 and entry['rows'] == 112
        assert entry['weight_mass'] == pytest.approx(8)
        assert X.shape == (112, 211)
        np.testing.assert_array_equal(weights, np.full(112, 1/14))
        expected = np.concatenate([group.features[[a for a in range(8) if a != group.pp]] for group in train])
        np.testing.assert_array_equal(X, expected)
        scale = np.sqrt((weights[:, None]*X**2).sum(axis=0)/weights.sum())
        scale[scale == 0] = 1
        np.testing.assert_allclose(completed_models(events)[entry['outer_fold']].scale, scale, rtol=0, atol=0)
    assert len(result['records']) == 20
    assert all(row['predicted_delta_costs'][row['pp_index']] == 0 for row in result['records'])


def test_outer_held_costs_cannot_change_predictions_and_held_features_cannot_change_rms():
    dataset, protocol = fixture_data()
    original, original_events = run_with_models(dataset, protocol)
    held = set(protocol['outer_folds'][0])
    changed = deepcopy(dataset)
    for group in changed['groups']:
        if group['city'] == 'jinan' and group['seed'] in held:
            for branch in group['branches']:
                branch['cost'] += 10*(branch['candidate_index']+1)**2
    updated, _ = run_with_models(changed, protocol)
    before = [row for row in original['records'] if row['outer_fold'] == 0]
    after = [row for row in updated['records'] if row['outer_fold'] == 0]
    assert len(before) == len(after) == 4
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left['predicted_delta_costs'], right['predicted_delta_costs'])
        assert left['selected_action'] == right['selected_action']
        assert right['selected_cost']-left['selected_cost'] == pytest.approx(10*(left['selected_action']+1)**2)
    # Measured held responses are legitimate oracle inputs; they may change held
    # scores, but must never enter this fold's fit or its training RMS.
    changed = deepcopy(dataset)
    for group in changed['groups']:
        if group['city'] == 'jinan' and group['seed'] in held:
            for branch in group['branches']:
                branch['targets']['y'] = (np.asarray(branch['targets']['y'])+10).tolist()
    _, changed_events = run_with_models(changed, protocol)
    for field in ('scale', 'coef'):
        np.testing.assert_array_equal(getattr(completed_models(original_events)[0], field),
                                      getattr(completed_models(changed_events)[0], field))


def test_measured_phi_has_one_projection_for_train_and_held_with_exact_pp_zero():
    dataset, protocol = fixture_data()
    groups = r.prepare(dataset, protocol)
    assert len(groups) == 20
    rows = {(row['seed'], row['anchor']): row for row in dataset['groups'] if row['city'] == 'jinan'}
    for group in groups:
        branches = rows[group.seed, group.anchor]['branches']
        pp_delta = np.asarray(branches[group.pp]['targets']['y'])
        np.testing.assert_array_equal(group.features[group.pp], np.zeros(211))
        for action, branch in enumerate(branches):
            # All toy measured states remain inside the frozen physical bounds.
            # Phi's first column is therefore the observed mean rhoN change.
            delta = np.asarray(branch['targets']['y'])
            assert group.features[action, -16] == pytest.approx((delta[:, 0]-pp_delta[:, 0]).mean(), abs=1e-15)
    changed = deepcopy(dataset)
    held = set(protocol['outer_folds'][0])
    for row in changed['groups']:
        if row['city'] == 'jinan' and row['seed'] in held:
            row['branches'][0]['targets']['y'][0][0] += .12
    updated = {group.identity: group for group in r.prepare(changed, protocol)}
    for group in groups:
        other = updated[group.identity]
        np.testing.assert_array_equal(group.features[:, :195], other.features[:, :195])
        if group.seed in held:
            assert other.features[0, -16]-group.features[0, -16] == pytest.approx(.01)
        else:
            np.testing.assert_array_equal(group.features, other.features)
