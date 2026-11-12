"""Independent prefix-factor arithmetic on handwritten caches; 0F/optimizer."""
from copy import deepcopy
from collections import Counter
from math import exp, log

import numpy as np
import pytest

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_prefix_family_decomposition_20261111 import readout


FAMILIES = ('brake_residual', 'start_permission', 'yield_permission', 'prefocal_lane_change')


def phi(x, y):
    return [1., x, y]+[0.]*35


def payload():
    features = [phi(.5, -.2), phi(-.4, .8), phi(.25, .1), phi(-.2, .3), phi(.75, .7)]
    # An unreferenced source-table entry tests that prefix slices keep IDs.
    factors = [dict(kind='bernoulli', actor='neighbor', feature_id=4, block=0, bit=1)]
    focal, other = {}, {}
    for group in range(3):
        for block in range(3):
            for bit in (0, 1):
                focal[group, block, bit] = len(factors)
                factors.append(dict(kind='bernoulli', actor='focal', feature_id=group,
                                    block=block, bit=bit))
    for block in range(3):
        for bit in (0, 1):
            other[block, bit] = len(factors)
            factors.append(dict(kind='bernoulli', actor='neighbor', feature_id=3,
                                block=block, bit=bit))
    categorical = {}
    for choice in range(3):
        categorical[choice] = len(factors)
        factors.append(dict(kind='categorical', actor='neighbor', feature_id=3,
                            allowed=[0, 1, 2], choice=choice))
    groups = [dict(feature_id=0, allowed=[0, 1, 2]),
              dict(feature_id=1, allowed=[0, 1]),
              dict(feature_id=2, allowed=[1, 2])]
    particles = []
    for index in range(8):
        group = index % 3
        focal_bits = ((index & 1), (index >> 1) & 1, (index >> 2) & 1)
        other_bits = ((index >> 1) & 1, (index >> 2) & 1, (index+1) & 1)
        ids = [focal[group, block, focal_bits[block]] for block in range(3)]
        ids += [other[block, other_bits[block]] for block in range(3)]
        ids += [categorical[group]]
        particles.append(dict(sample_id=index, factor_ids=ids, focal_group=group,
            focal_targets={('left', 'stay', 'right')[label]: 'lane'+str(label)
                           for label in groups[group]['allowed']},
            outcome=dict(survived=index != 0, physical_speed_mps=1.+focal_bits[0],
                started=bool(focal_bits[1]), entry_event=bool(focal_bits[2]),
                lane_change='stay', lane_id='lane1', position_m=10.)))
    return dict(focal_id='focal', features=features, factors=factors, focal_groups=groups,
        particles=particles, old_stopped=True, eligibility=dict(motion=True, start=True,
            yield_entry=True, lane_change=True), stats=dict(physical_calls=0))


def candidate():
    theta = np.zeros((5, 38))
    theta[:, :3] = ((.7, .4, -.2), (-1., .2, .1), (.5, -.3, .2),
                    (-1.1, .5, -.2), (.8, -.4, .3))
    return theta


def direct_ratios(data, theta):
    """Independent probability ratios, with no production grouping helper."""
    result = {family: np.zeros(len(data['particles'])) for family in FAMILIES}
    roles = {family: {role: np.zeros(len(data['particles'])) for role in ('focal', 'other')}
             for family in FAMILIES}
    features = np.asarray(data['features'])
    for index, row in enumerate(data['particles']):
        for factor_id in row['factor_ids']:
            factor = data['factors'][factor_id]
            feature = features[factor['feature_id']]
            if factor['kind'] == 'bernoulli':
                block = factor['block']
                probability = 1./(1.+exp(-float(theta[block] @ feature)))
                ratio = 2.*(probability if factor['bit'] else 1.-probability)
                family = FAMILIES[block]
            else:
                scores = np.asarray([theta[3] @ feature, 0., theta[4] @ feature])
                allowed = factor['allowed']
                chosen_probability = exp(float(scores[factor['choice']]))/sum(exp(float(scores[j])) for j in allowed)
                ratio = len(allowed)*chosen_probability
                family = 'prefocal_lane_change'
            value = log(ratio)
            role = 'focal' if factor['actor'] == data['focal_id'] else 'other'
            result[family][index] += value
            roles[family][role][index] += value
    return result, roles


def direct_weight_readout(data, logw):
    weights = np.exp(logw-np.max(logw));weights /= weights.sum()
    survived = np.asarray([row['outcome']['survived'] for row in data['particles']])
    conditional = weights*survived;conditional /= conditional.sum()
    probability = np.zeros(3);availability = np.zeros(3)
    for weight,row in zip(conditional,data['particles']):
        allowed = data['focal_groups'][row['focal_group']]['allowed']
        probability[allowed] += weight/len(allowed)
        availability[allowed] += weight
    return dict(probability=probability,availability=availability,global_ess=1./sum(weights**2),
        survivor_ess=1./sum(conditional**2),global_max=float(weights.max()),survivor_max=float(conditional.max()))


def test_four_factor_families_and_two_actor_roles_reconstruct_old_log_weights():
    data,theta = payload(),candidate()
    original = deepcopy(data)
    expected,roles = direct_ratios(data,theta)
    with physical_forbidden():
        pool = ParticleLaw(data)
        actual = readout.factor_family_ratios(pool,theta)
        old = pool._evaluate(theta)[0]
    assert tuple(actual['family_log_ratios']) == FAMILIES
    for family in FAMILIES:
        assert actual['family_log_ratios'][family] == pytest.approx(expected[family],abs=1e-12)
        for role in ('focal','other'):
            assert actual['role_log_ratios'][family][role] == pytest.approx(roles[family][role],abs=1e-12)
        assert (actual['role_log_ratios'][family]['focal']+
                actual['role_log_ratios'][family]['other']) == pytest.approx(expected[family],abs=1e-12)
    assert sum(actual['family_log_ratios'].values()) == pytest.approx(old,abs=1e-12)
    assert data == original


def test_brake_bit_sign_and_categorical_joint_normalizer_are_not_swapped():
    data,theta = payload(),candidate()
    with physical_forbidden():
        pool = ParticleLaw(data)
        actual = readout.factor_family_ratios(pool,theta)['family_log_ratios']
    expected,unused = direct_ratios(data,theta)
    assert actual['brake_residual'] == pytest.approx(expected['brake_residual'],abs=1e-12)
    assert actual['prefocal_lane_change'] == pytest.approx(expected['prefocal_lane_change'],abs=1e-12)
    # The same actor/feature with positive z favors bit1 (the brake choice).
    theta = np.zeros((5,38));theta[0,0] = 1.
    chosen = deepcopy(data)
    for index,row in enumerate(chosen['particles']):
        row['factor_ids'] = [next(j for j,f in enumerate(chosen['factors']) if
            f['kind']=='bernoulli' and f['actor']=='focal' and f['feature_id']==0 and
            f['block']==0 and f['bit']==index%2)]
    with physical_forbidden():
        brake = readout.factor_family_ratios(ParticleLaw(chosen),theta)['family_log_ratios']['brake_residual']
    assert brake[1] > 0. > brake[0]
    assert brake[1]-brake[0] == pytest.approx(1.,abs=1e-12)


def test_exactly_ten_weight_arrays_keep_only_and_leave_one_out_definitions():
    expected,unused = direct_ratios(payload(),candidate())
    views = readout.diagnostic_weight_arrays(expected)
    assert set(views) == {'zero','full'}|{'only_'+g for g in FAMILIES}|{'without_'+g for g in FAMILIES}
    assert views['zero'] == pytest.approx(np.zeros(8))
    full = sum(expected.values())
    assert views['full'] == pytest.approx(full)
    for family in FAMILIES:
        assert views['only_'+family] == pytest.approx(expected[family])
        assert views['without_'+family] == pytest.approx(full-expected[family])


def test_prefocal_categorical_uses_each_recorded_mask_including_forced_stay():
    data,theta = payload(),candidate()
    for factor in data['factors']:
        if factor['kind']=='categorical':
            factor['allowed'] = {0:[0,1],1:[1],2:[1,2]}[factor['choice']]
    expected,unused = direct_ratios(data,theta)
    with physical_forbidden():
        actual = readout.factor_family_ratios(ParticleLaw(data),theta)
    assert actual['family_log_ratios']['prefocal_lane_change'] == pytest.approx(
        expected['prefocal_lane_change'],abs=1e-12)
    for index,row in enumerate(data['particles']):
        if row['focal_group']==1:
            assert actual['family_log_ratios']['prefocal_lane_change'][index] == pytest.approx(0.,abs=1e-12)


def test_full_readout_keeps_zero_focal_class_table_and_matches_mf0_not_mff():
    data,theta = payload(),candidate()
    expected,unused = direct_ratios(data,theta)
    logw = sum(expected.values())
    direct = direct_weight_readout(data,logw)
    with physical_forbidden():
        pool = ParticleLaw(data)
        zero_table = pool._evaluate(np.zeros_like(theta))[2]
        result = readout.weight_readout(pool,logw,0,zero_table)
        changed_both = pool.marginal(theta,'lane_change',0)
    assert result['probabilities'] == pytest.approx(direct['probability'],abs=1e-12)
    assert result['mask_availability'] == pytest.approx(direct['availability'],abs=1e-12)
    assert result['native_nll']['value'] == pytest.approx(-log(direct['probability'][0]),abs=1e-12)
    assert abs(exp(-result['native_nll']['value'])-changed_both['likelihood']) > 1e-3
    assert sum(result['probabilities']) == pytest.approx(1.,abs=1e-12)


def test_decomposition_driver_uses_zero_focal_table_for_all_ten_diagnostics():
    data,theta = payload(),candidate()
    original = deepcopy(data)
    theta_before = theta.copy()
    families,unused = direct_ratios(data,theta)
    full = direct_weight_readout(data,sum(families.values()))
    zero = direct_weight_readout(data,np.zeros(8))
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data),theta,0)
    assert len(result['views']) == 10
    assert result['views']['zero']['probabilities'] == pytest.approx(zero['probability'],abs=1e-12)
    assert result['views']['full']['probabilities'] == pytest.approx(full['probability'],abs=1e-12)
    assert data == original
    assert np.array_equal(theta,theta_before)


def test_global_and_survivor_normalizers_keep_ended_mass_out_of_class_probability():
    data,theta = payload(),candidate()
    logw = np.asarray([50., .1, -.2, .7, -.4, .2, -.3, .5])
    expected = direct_weight_readout(data,logw)
    with physical_forbidden():
        pool = ParticleLaw(data)
        result = readout.weight_readout(pool,logw,2,pool._evaluate(np.zeros_like(theta))[2])
    assert result['global_ess'] == pytest.approx(expected['global_ess'],abs=1e-12)
    assert result['survivor_ess'] == pytest.approx(expected['survivor_ess'],abs=1e-12)
    assert result['max_normalized_weight'] == pytest.approx(expected['global_max'],abs=1e-12)
    assert result['max_survivor_normalized_weight'] == pytest.approx(expected['survivor_max'],abs=1e-12)
    assert result['probabilities'] == pytest.approx(expected['probability'],abs=1e-12)
    assert result['global_ess'] < 1.001 < result['survivor_ess']


def test_mask_zero_is_infinite_and_stay_remains_constant_under_concentrated_weights():
    data,theta = payload(),np.zeros((5,38))
    for group in data['focal_groups']:group['allowed'] = [1]
    logw = np.asarray([0.,50.,0.,0.,0.,0.,0.,0.])
    with physical_forbidden():
        pool = ParticleLaw(data);table = pool._evaluate(theta)[2]
        impossible = readout.weight_readout(pool,logw,0,table)
        stay = readout.weight_readout(pool,logw,1,table)
    assert impossible['native_nll']['state'] == 'positive_infinity'
    assert impossible['native_mask_availability'] == 0.
    assert impossible['mask_availability'] == pytest.approx([0.,1.,0.])
    assert stay['native_nll']['value'] == pytest.approx(0.,abs=1e-12)
    assert stay['probabilities'] == pytest.approx([0.,1.,0.],abs=1e-12)
    assert stay['global_ess'] < 1.001


def test_no_survivor_is_undefined_not_a_zero_probability_native_event():
    data,theta = payload(),candidate()
    for row in data['particles']:row['outcome']['survived'] = False
    with physical_forbidden():
        pool = ParticleLaw(data)
        result = readout.weight_readout(pool,np.zeros(8),0,pool._evaluate(np.zeros_like(theta))[2])
    assert not result['defined']
    assert result['native_nll']['state'] == 'undefined'
    assert result['native_mask_availability'] is None
    assert result['survivor_ess'] == 0.


def test_families_with_no_recorded_active_factors_remain_zero_arrays():
    data,theta = payload(),candidate()
    for row in data['particles']:
        row['factor_ids'] = [j for j in row['factor_ids'] if data['factors'][j]['kind']=='bernoulli'
                             and data['factors'][j]['block']==0]
    with physical_forbidden():
        result = readout.factor_family_ratios(ParticleLaw(data),theta)
    for family in FAMILIES[1:]:
        assert result['family_log_ratios'][family] == pytest.approx(np.zeros(8),abs=1e-12)
    assert len(readout.diagnostic_weight_arrays(result['family_log_ratios'])) == 10


def test_prefix_slice_preserves_original_factor_and_feature_ids():
    data,theta = payload(),candidate()
    sliced = dict(data,particles=data['particles'][:4])
    expected,unused = direct_ratios(sliced,theta)
    with physical_forbidden():
        pool = ParticleLaw(sliced)
        actual = readout.factor_family_ratios(pool,theta)
    assert pool.data['factors'] is data['factors']
    assert pool.data['features'] is data['features']
    assert [row['sample_id'] for row in pool.rows] == list(range(4))
    assert all(0 not in row['factor_ids'] for row in pool.rows)
    for family in FAMILIES:
        assert actual['family_log_ratios'][family] == pytest.approx(expected[family],abs=1e-12)


def test_role_means_use_the_named_survivor_and_native_mask_condition():
    data,theta = payload(),candidate()
    families,roles = direct_ratios(data,theta)
    full = sum(families.values())
    survived = np.asarray([row['outcome']['survived'] for row in data['particles']])
    allowed = np.asarray([0 in data['focal_groups'][row['focal_group']]['allowed']
                          for row in data['particles']])
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data),theta,0)
    for point,logs in (('zero',np.zeros(8)),('full',full)):
        survivor_weights = np.exp(logs[survived]-np.max(logs[survived]))
        survivor_weights /= survivor_weights.sum()
        for population,selected in (('all_survived',survived),
            ('native_allowed',survived & allowed),('native_disallowed',survived & ~allowed)):
            chosen = np.exp(logs[selected]-np.max(logs[selected]));chosen /= chosen.sum()
            actual = result['role_contributions'][point][population]
            assert actual['particle_count'] == int(selected.sum())
            assert actual['survivor_conditioned_mass'] == pytest.approx(
                float(survivor_weights[selected[survived]].sum()),abs=1e-12)
            for family in FAMILIES:
                for role in ('focal','other'):
                    assert actual['families'][family][role]['mean_log_ratio'] == pytest.approx(
                        float(chosen @ roles[family][role][selected]),abs=1e-12)


def test_panel_summary_preserves_infinity_and_missing_reference_denominators():
    data,theta = payload(),candidate()
    impossible = deepcopy(data)
    for group in impossible['focal_groups']:group['allowed'] = [1]
    no_survivor = deepcopy(data)
    for row in no_survivor['particles']:row['outcome']['survived'] = False
    with physical_forbidden():
        normal = readout.decompose_pool(ParticleLaw(data),theta,0)
        infinite = readout.decompose_pool(ParticleLaw(impossible),theta,0)
        undefined = readout.decompose_pool(ParticleLaw(no_survivor),theta,0)
    first = dict(y=0,views=normal['views'])
    summary = readout.summarize_records([first,dict(y=0,views=infinite['views'])],2)
    for name in summary['views']:
        assert summary['views'][name]['native_nll_mean']['state'] == 'positive_infinity'
        expected_mass = normal['views'][name]['native_mask_availability']/2.
        assert summary['views'][name]['native_mask_mass_mean']['value'] == pytest.approx(expected_mass)
        assert summary['by_label']['positive']['views'][name]['native_nll_mean']['state'] == 'positive_infinity'
    missing = readout.summarize_records([first],2)
    assert not missing['complete_reference_count']
    assert all(v['native_nll_mean']['state']=='undefined' for v in missing['views'].values())
    failed = readout.summarize_records([first,dict(y=0,views=undefined['views'])],2)
    assert all(v['native_nll_mean']['state']=='undefined' for v in failed['views'].values())


def identity_receipts():
    cases,refs = [],[]
    seeds = (76116901,76116902,76116903,76116904)
    for seed_index,seed in enumerate(seeds):
        for head in ('motion','start','yield_entry','lane_change'):
            for ordinal in range(32):
                index = len(cases)
                ref = dict(seed=seed,time_sec=index,packet_id='car'+str(index))
                y = 1 if head=='lane_change' else 0
                if head=='lane_change' and ordinal==0 and seed_index<3:
                    y = 0 if seed_index==2 else 2
                if head=='lane_change' and ordinal==0 and seed_index==2:
                    ref.update(time_sec=1362,packet_id='jinan_3x4_real_1178')
                split = 'development' if seed_index==3 else 'train'
                obs = dict(ref,head=head,y=y,manifest_ordinal=ordinal,split=split)
                cases.append(dict(case_index=index,reference=ref,
                    particle_path='case'+str(index)+'.json.gz',observations=[obs]))
                refs.append(dict(obs,case_index=index,nll=1.+index/1000.))
    panels = {panel:dict(references=[dict(r) for r in refs if r['split']==panel])
              for panel in ('train','development')}
    fit = dict(zero=deepcopy(panels),final=deepcopy(panels))
    previous = []
    for ref in refs:
        if ref['head']=='lane_change':
            previous.append(dict(reference={k:ref[k] for k in ('seed','time_sec','packet_id')},
                panel=ref['split'],case_index=ref['case_index'],head='lane_change',
                manifest_ordinal=ref['manifest_ordinal'],y=ref['y'],particle_size=256))
    pilot = []
    for index in range(15):
        ref = dict(seed=seeds[index%4],time_sec=2000+index,packet_id='aux'+str(index))
        is_lc = index<9 or index==14
        y = 1 if index==14 else 2 if index==5 else 0
        if index==6:
            ref = dict(seed=76116903,time_sec=1362,packet_id='jinan_3x4_real_1178')
        obs = dict(head='lane_change' if is_lc else 'motion',y=y,manifest_ordinal=100+index)
        pilot.append(dict(case_index=index,reference=ref,
            readouts={'1024':{'0':dict(observed_heads=[obs])}}))
        if is_lc:
            for size in (256,1024):
                previous.append(dict(reference=ref,panel='auxiliary',case_index=index,
                    head='lane_change',manifest_ordinal=100+index,y=y,particle_size=size))
    return (dict(status='PASS_FROZEN_PARTICLE_TRAINING_POOLS',cases=cases),fit,
        dict(status='PASS_BOUNDED_REAL_PARTICLE_PILOT',cases=pilot),dict(records=previous))


def test_selector_preserves_all148_readouts_and_the_one_declared_panel_overlap():
    inputs = identity_receipts()
    before = deepcopy(inputs)
    rows = readout.select_references(*inputs)
    assert len(rows)==148
    assert Counter((r['panel'],r['particle_size']) for r in rows) == Counter({
        ('train',256):96,('development',256):32,('auxiliary',256):10,('auxiliary',1024):10})
    overlap = [r for r in rows if r['reference']['packet_id']=='jinan_3x4_real_1178']
    assert len(overlap)==3
    assert {(r['panel'],r['particle_size']) for r in overlap} == {
        ('train',256),('auxiliary',256),('auxiliary',1024)}
    assert all(r['head']=='lane_change' for r in rows)
    assert inputs == before


@pytest.mark.parametrize('change',('duplicate_identity','changed_reference'))
def test_selector_rejects_count_preserving_previous_identity_changes(change):
    inputs = identity_receipts()
    records = inputs[-1]['records']
    if change=='duplicate_identity':
        records[-1] = deepcopy(records[-2])
    else:
        records[0]['reference']['packet_id'] = 'different_current_case'
    with pytest.raises(ValueError):
        readout.select_references(*inputs)
