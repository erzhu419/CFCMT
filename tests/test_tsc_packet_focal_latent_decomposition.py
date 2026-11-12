"""Independent fixed-payload checks: no kernel, sampler, physics or optimizer."""
from copy import deepcopy
from math import exp, log

import numpy as np
import pytest

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_focal_latent_decomposition_20261111 import readout


def feature(x, z):
    return [1., x, z]+[0.]*35


def payload():
    """Different focal masks correlate with a nonfocal LC choice."""
    groups = [dict(feature_id=0, allowed=[0, 1, 2]),
              dict(feature_id=1, allowed=[0, 1]),
              dict(feature_id=2, allowed=[1, 2])]
    factors = [dict(kind='categorical', actor='other', feature_id=3,
                    allowed=[0, 1, 2], choice=choice) for choice in range(3)]
    factors.extend(dict(kind='bernoulli', actor='focal', feature_id=3,
                        block=0, bit=bit) for bit in (0, 1))
    rows = []
    for group, count in enumerate((2, 3, 3)):
        for unused in range(count):
            index = len(rows)
            rows.append(dict(sample_id=index, factor_ids=[group, 3+index % 2],
                focal_group=group,
                focal_targets={('left', 'stay', 'right')[label]: 'lane'+str(label)
                               for label in groups[group]['allowed']},
                outcome=dict(survived=index != 0, physical_speed_mps=2.,
                    started=False, entry_event=False, lane_change='stay',
                    lane_id='lane1', position_m=10.)))
    return dict(focal_id='focal', features=[feature(.5, -.2), feature(-.7, .8),
        feature(.2, .4), feature(.1, -.4)], factors=factors, focal_groups=groups,
        particles=rows, old_stopped=False, eligibility=dict(motion=True, start=False,
            yield_entry=True, lane_change=True), stats=dict(physical_calls=0))


def candidate():
    theta = np.zeros((5, 38))
    theta[0, 0] = .4
    theta[3, :3] = (-1.2, .8, -.5)
    theta[4, :3] = (.6, -.4, .2)
    return theta


def direct_values(data, theta):
    """Direct probabilities, independent of ParticleLaw._evaluate/SNIS code."""
    vectors = np.asarray(data['features'])
    other = vectors[3]
    scores = np.array([theta[3] @ other, 0., theta[4] @ other])
    probabilities = np.exp(scores)/np.exp(scores).sum()
    bit_probability = 1./(1.+exp(-float(theta[0] @ other)))
    weights = np.asarray([3.*probabilities[row['factor_ids'][0]]*
        2.*(bit_probability if row['factor_ids'][1] == 4 else 1.-bit_probability)
        for row in data['particles']])
    survived = np.asarray([row['outcome']['survived'] for row in data['particles']])
    weights[~survived] = 0.
    weights /= weights.sum()
    classes = []
    for group in data['focal_groups']:
        phi = vectors[group['feature_id']]
        score = np.array([theta[3] @ phi, 0., theta[4] @ phi])
        mass = np.zeros(3)
        allowed = group['allowed']
        mass[allowed] = np.exp(score[allowed])/np.exp(score[allowed]).sum()
        classes.append(mass)
    classes = np.asarray(classes)
    ids = np.asarray([row['focal_group'] for row in data['particles']])
    return weights, classes[ids], vectors[[data['focal_groups'][i]['feature_id'] for i in ids]]


def test_four_cells_independently_cross_focal_table_and_latent_weights():
    data, theta = payload(), candidate()
    original = deepcopy(data)
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data), theta, 0)
    w0, p0, unused = direct_values(data, np.zeros_like(theta))
    wf, pf, unused = direct_values(data, theta)
    expected = {'00': w0 @ p0, '0f': w0 @ pf,
                'f0': wf @ p0, 'ff': wf @ pf}
    for key, values in expected.items():
        row = result['combinations'][key]
        assert row['probabilities'] == pytest.approx(values, abs=1e-13)
        assert row['native_probability'] == pytest.approx(values[0], abs=1e-13)
        assert row['native_nll']['value'] == pytest.approx(-log(values[0]), abs=1e-12)
    assert data == original
    # Global LC coefficients affect both the nonfocal prefix and focal class.
    cleared = theta.copy()
    cleared[3:] = 0.
    with physical_forbidden():
        wrong = ParticleLaw(data).marginal(cleared, 'lane_change', 0)['likelihood']
    assert abs(result['combinations']['f0']['native_probability']-wrong) > 1e-3


def test_endpoints_reproduce_original_marginals_and_every_class_normalizes():
    pool, theta = ParticleLaw(payload()), candidate()
    with physical_forbidden():
        result = readout.decompose_pool(pool, theta, 2)
        old = pool.marginal(np.zeros_like(theta), 'lane_change', 2)
        new = pool.marginal(theta, 'lane_change', 2)
    for cell in result['combinations'].values():
        assert sum(cell['probabilities']) == pytest.approx(1., abs=1e-13)
        assert cell['normalization_error'] < 1e-12
    assert result['combinations']['00']['native_nll']['value'] == pytest.approx(old['nll'], abs=1e-12)
    assert result['combinations']['ff']['native_nll']['value'] == pytest.approx(new['nll'], abs=1e-12)


def test_both_telescoping_orders_close_and_retain_a_real_interaction():
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(payload()), candidate(), 0)
    values = {key:row['native_nll']['value'] for key,row in result['combinations'].items()}
    decomposition = result['decomposition']
    total = values['ff']-values['00']
    assert decomposition['total']['value'] == pytest.approx(total)
    first = decomposition['focal_then_latent']
    second = decomposition['latent_then_focal']
    assert first['focal']['value'] == pytest.approx(values['0f']-values['00'])
    assert first['latent']['value'] == pytest.approx(values['ff']-values['0f'])
    assert second['latent']['value'] == pytest.approx(values['f0']-values['00'])
    assert second['focal']['value'] == pytest.approx(values['ff']-values['f0'])
    assert first['sum']['value'] == pytest.approx(total)
    assert second['sum']['value'] == pytest.approx(total)
    interaction = values['ff']-values['f0']-values['0f']+values['00']
    assert abs(interaction) > 1e-3
    assert decomposition['interaction']['value'] == pytest.approx(interaction)


def test_mask_zero_stays_zero_and_infinite_differences_remain_undefined():
    data = payload()
    for group in data['focal_groups']:
        group['allowed'] = [1]
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data), candidate(), 0)
    for cell in result['combinations'].values():
        assert cell['probabilities'] == pytest.approx([0., 1., 0.])
        assert cell['native_probability'] == cell['native_mask_availability'] == 0.
        assert cell['native_nll']['state'] == 'positive_infinity'
    assert result['decomposition']['total']['state'] == 'undefined'
    assert result['decomposition']['interaction']['state'] == 'undefined'


def test_no_survivor_is_undefined_instead_of_zero_supported_mass():
    data = payload()
    for row in data['particles']:
        row['outcome']['survived'] = False
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data), candidate(), 0)
    for cell in result['combinations'].values():
        assert cell['native_nll']['state'] == 'undefined'
        assert cell['native_probability'] is None
        assert cell['native_mask_availability'] is None


def test_constant_stay_remains_exact_under_concentrated_latent_weights():
    data = payload()
    data['particles'] = [deepcopy(data['particles'][1]) for unused in range(64)]
    for index,row in enumerate(data['particles']):
        row['outcome']['survived'] = True
        row['factor_ids'] = [0 if index == 0 else 2, 3]
    for group in data['focal_groups']:
        group['allowed'] = [1]
    theta = np.zeros((5, 38))
    theta[3, 0],theta[4, 0] = 10.,-10.
    with physical_forbidden():
        pool = ParticleLaw(data)
        result = readout.decompose_pool(pool, theta, 1)
        ess = pool.summary(theta)['global_ess']
    assert ess < 1.001
    for cell in result['combinations'].values():
        assert cell['native_probability'] == pytest.approx(1., abs=1e-13)
        assert cell['native_nll']['value'] == pytest.approx(0., abs=1e-13)
    assert result['decomposition']['total']['value'] == pytest.approx(0., abs=1e-13)


def test_scores_and_all38_contributions_use_the_correct_allowed_denominator():
    data, theta = payload(),candidate()
    with physical_forbidden():
        result = readout.decompose_pool(ParticleLaw(data), theta, 0)
    allowed = np.asarray([0 in data['focal_groups'][row['focal_group']]['allowed']
                          for row in data['particles']])
    for name,point in (('zero',np.zeros_like(theta)),('candidate',theta)):
        weights,unused,features = direct_values(data,point)
        availability = float(weights[allowed].sum())
        for population,mask in (('all_survived',np.ones(len(weights),dtype=bool)),
                                ('native_allowed',allowed)):
            chosen = weights*mask
            chosen /= chosen.sum()
            phi = chosen @ features
            for direction,block in (('left',3),('right',4)):
                detail = result['feature_details'][name][population][direction]
                score = result['score_summary'][name][population][direction]
                contributions = phi*theta[block]
                assert len(detail['weighted_mean_contribution']) == 38
                assert detail['weighted_mean_phi'] == pytest.approx(phi, abs=1e-13)
                assert detail['weighted_mean_contribution'] == pytest.approx(contributions, abs=1e-13)
                assert sum(detail['weighted_mean_contribution']) == pytest.approx(score['weighted_mean_logit'], abs=1e-12)
        for cell in ('00','0f') if name=='zero' else ('f0','ff'):
            assert result['combinations'][cell]['native_mask_availability'] == pytest.approx(availability)


def primary_receipts():
    cases,refs = [],[]
    for seed_index,seed in enumerate((76116901,76116902,76116903,76116904)):
        for head in ('motion','start','yield_entry','lane_change'):
            for ordinal in range(32):
                index = len(cases)
                current = dict(seed=seed,time_sec=index,packet_id='car'+str(index))
                y = 0 if head!='lane_change' else (0 if seed_index==2 and ordinal==0 else
                    2 if seed_index in (0,1) and ordinal==0 else 1)
                if head=='lane_change' and seed_index==2 and ordinal==0:
                    current.update(time_sec=1362,packet_id='jinan_3x4_real_1178')
                obs = dict(current,head=head,y=y,manifest_ordinal=ordinal,
                    split='development' if seed_index==3 else 'train')
                cases.append(dict(case_index=index,reference=current,
                    particle_path='case'+str(index)+'.json.gz',observations=[obs]))
                refs.append(dict(obs,case_index=index,nll=1.+index/1000.))
    bysplit={split:dict(references=[dict(ref) for ref in refs if ref['split']==split])
             for split in ('train','development')}
    return dict(status='PASS_FROZEN_PARTICLE_TRAINING_POOLS',cases=cases),dict(
        zero=deepcopy(bysplit),final=deepcopy(bysplit))


def test_primary_selection_keeps_all96_training_and32_development_lc_references():
    particles,fit = primary_receipts()
    rows = readout.select_primary_references(particles,fit)
    assert len(rows)==128
    assert all(row['head']=='lane_change' for row in rows)
    assert sum(int(row['reference']['seed'])==76116904 for row in rows)==32
    assert len({(row['case_index'],row['head'],row['manifest_ordinal']) for row in rows})==128
    assert sum(row['y']!=1 for row in rows)==3
    overlap=[row for row in rows if row['reference']['packet_id']=='jinan_3x4_real_1178']
    assert len(overlap)==1 and overlap[0]['y']==0


def test_duplicate_identity_with_unchanged_counts_is_rejected():
    particles,fit = primary_receipts()
    cases=[case for case in particles['cases'] if case['observations'][0]['head']=='lane_change']
    cases[1]['case_index']=cases[0]['case_index']
    cases[1]['observations'][0]['manifest_ordinal']=cases[0]['observations'][0]['manifest_ordinal']
    with pytest.raises(ValueError):
        readout.select_primary_references(particles,fit)


def test_original_receipt_label_mismatch_is_rejected_without_relabeling():
    particles,fit = primary_receipts()
    row=next(ref for ref in fit['final']['train']['references'] if ref['head']=='lane_change')
    row['y']=1-row['y']
    with pytest.raises(ValueError):
        readout.select_primary_references(particles,fit)


def auxiliary_receipts():
    particles,fit=primary_receipts()
    shared=next(case['reference'] for case in particles['cases']
                if case['reference']['packet_id']=='jinan_3x4_real_1178')
    cases,records=[],[]
    for index in range(15):
        head='lane_change' if index<9 or index==14 else 'yield_entry'
        y=(1 if index==14 else 2 if index==5 else 0) if head=='lane_change' else 1
        current=dict(shared) if index==6 else dict(seed=76116901,time_sec=100+index,
                                                  packet_id='diagnostic'+str(index))
        ordinal=0 if index==6 else 100+index
        observation=dict(head=head,y=y,manifest_ordinal=ordinal)
        cases.append(dict(case_index=index,reference=current,status='PASS_REAL_PARTICLE_CASE',
            readouts={'1024':{'0':dict(observed_heads=[observation])}}))
        records.append(dict(case_index=index,reference=current,head=head,y=y,
            manifest_ordinal=ordinal,points={
                point:{size:dict(nll=1.+index/100.+offset,loss=dict(state='finite',value=1.+index/100.+offset))
                       for size in ('256','1024')}
                for point,offset in (('zero',0.),('candidate',.2))}))
    return particles,fit,dict(status='PASS_BOUNDED_REAL_PARTICLE_PILOT',cases=cases),dict(auxiliary=records)


def test_auxiliary_all10_refs_keep_the_single_training_overlap_in_both_panels():
    particles,fit,pilot,stability=auxiliary_receipts()
    training=readout.select_primary_references(particles,fit)
    auxiliary=readout.select_auxiliary_references(pilot,stability)
    assert len(training)==128 and len(auxiliary)==10
    assert sum(row['y']!=1 for row in auxiliary)==9
    def key(row):
        ref=row['reference']
        return ref['seed'],ref['time_sec'],ref['packet_id']
    assert len(set(map(key,training)) & set(map(key,auxiliary)))==1
    assert len(training)+len(auxiliary)==138
    assert [row['case_index'] for row in auxiliary]==list(range(9))+[14]


def test_auxiliary_label_identity_mismatch_is_rejected():
    unused,unused_fit,pilot,stability=auxiliary_receipts()
    stability['auxiliary'][0]['y']=2
    with pytest.raises(ValueError):
        readout.select_auxiliary_references(pilot,stability)


def test_panel_summary_keeps_infinite_cases_and_never_shrinks_missing_count():
    bad=payload()
    for group in bad['focal_groups']:
        group['allowed']=[1]
    with physical_forbidden():
        good_result=readout.decompose_pool(ParticleLaw(payload()),candidate(),0)
        bad_result=readout.decompose_pool(ParticleLaw(bad),candidate(),0)
    rows=[dict(good_result,y=0),dict(bad_result,y=0)]
    complete=readout.summarize_records(rows,2)
    assert complete['complete_reference_count'] is True
    for loss in complete['combination_native_nll_means'].values():
        assert loss['state']=='positive_infinity'
    assert complete['decomposition_means']['total']['state']=='undefined'
    assert complete['by_label']['positive']['n']==2
    missing=readout.summarize_records(rows,3)
    assert missing['complete_reference_count'] is False
    assert all(loss['state']=='undefined' for loss in missing['combination_native_nll_means'].values())


def test_direction_allowed_scores_are_independent_of_the_observed_stay_label():
    data,theta=payload(),candidate()
    with physical_forbidden():
        result=readout.decompose_pool(ParticleLaw(data),theta,1)
    for latent,point in (('zero',np.zeros_like(theta)),('candidate',theta)):
        weights,unused,features=direct_values(data,point)
        native=result['score_summary'][latent]['native_allowed']
        assert native['mass']==pytest.approx(1.)
        for direction,label,block in (('left',0,3),('right',2,4)):
            allowed=np.asarray([label in data['focal_groups'][row['focal_group']]['allowed']
                                for row in data['particles']])
            mass=float(weights[allowed].sum())
            selected=weights*allowed/mass
            phi=selected @ features
            summary=result['score_summary'][latent][direction+'_allowed']
            detail=result['feature_details'][latent][direction+'_allowed'][direction]
            assert summary['mass']==pytest.approx(mass)
            assert mass<1.
            assert detail['weighted_mean_phi']==pytest.approx(phi,abs=1e-13)
            assert summary[direction]['weighted_mean_logit']==pytest.approx(float(phi @ theta[block]),abs=1e-12)
