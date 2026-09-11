from copy import deepcopy
import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.calibrate_cologne_rigid_threshold import threshold_metrics
from scripts.data.cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting


def fixture():
    groups = [f'cologne1:seed{seed}:{index}:tls' for index, seed in enumerate([2027, 3037, 4047])]
    states = [f'phase{i}' for i in range(8)]
    costs, scores = np.arange(8, dtype=float), np.asarray([1., 0., .2, 2., 3., 4., 5., 6.])
    records = [group_record(group, states, costs, scores, 0, selected)
               for group, selected in zip(groups, [1, 2, 0])]
    # PP is index 0. Group 0 top-1 stays; group 1 top-1 switches despite delta_switch=0.
    features = np.tile(np.column_stack([np.arange(8, 0, -1), [1, 0, 1, 1, 1, 1, 1, 1],
                                       [0, -1, 0, 0, 0, 0, 0, 0]]), (3, 1))
    bank = MechanismDataset(
        feature_names=('green_q', 'switch_indicator', 'delta_switch_indicator'), features=features,
        context_names=(), context=np.zeros((24, 0)), priors={},
        targets={key: np.tile(costs, 3) for key in ('interval_cost', 'prefix_mean_cost_450s')},
        domains=np.asarray(['cologne1'] * 24),
        metadata={'action_group_ids': np.repeat(groups, 8).tolist(), 'candidate_states': states * 3})
    return records, bank, groups


def test_mask_retains_all_groups_and_labels_without_selecting_second_best():
    records, bank, groups = fixture()
    original = deepcopy(records)
    effective, diagnostic = apply_no_stay_to_oof(records[::-1], bank, groups)
    assert [r['group_id'] for r in effective] == groups
    assert [r['selected_index'] for r in effective] == [0, 2, 0]
    assert effective[0]['original_selected_index'] == 1
    assert effective[0]['original_predicted_advantage'] == 1
    assert effective[0]['rigid_cost'] == effective[0]['phase_pressure_cost']
    assert effective[0]['predicted_advantage'] == effective[0]['actual_advantage'] == 0
    assert not effective[1]['model_stay_vetoed']  # Reads absolute switch_indicator, not delta.
    assert all(r['label_costs'] == old['label_costs'] and r['predicted_scores'] == old['predicted_scores']
               for r, old in zip(effective, original))
    assert records == original
    assert diagnostic['groups'] == 3 and diagnostic['candidate_rows'] == 24
    assert diagnostic['model_stay_vetoed_groups'] == 1


@pytest.mark.parametrize('mismatch', ['candidate', 'label', 'reference', 'roster'])
def test_alignment_mismatch_rejected(mismatch):
    records, bank, groups = fixture()
    if mismatch == 'candidate':
        records[0]['candidate_states'][0:2] = records[0]['candidate_states'][1::-1]
    elif mismatch == 'label':
        records[0]['label_costs'][0] += 1
    elif mismatch == 'reference':
        records[0]['reference_index'] = 1
    else:
        records.pop()
    with pytest.raises(ValueError):
        apply_no_stay_to_oof(records, bank, groups)


@pytest.mark.parametrize('threshold,rejected,vetoed,accepted', [
    (None, 2, 0, 0), (1., 2, 0, 0), (.8, 1, 1, 0), (0., 0, 1, 1)])
def test_accounting_applies_threshold_before_veto(threshold, rejected, vetoed, accepted):
    effective, _ = apply_no_stay_to_oof(*fixture())
    result = proposal_accounting(effective, threshold)
    assert result['groups'] == 3 and result['raw_proposals'] == 2
    assert result['threshold_rejected'] == rejected
    assert result['stay_veto_count'] == vetoed
    assert result['accepted_overrides'] == accepted
    assert result['groups'] == result['original_reference_actions'] + rejected + vetoed + accepted
    assert accepted == threshold_metrics(effective, threshold)['accepted_overrides']
    assert sum(r['groups'] for r in result['per_seed'].values()) == 3
