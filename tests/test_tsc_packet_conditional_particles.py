"""Small checks only: no invocation of the full fixed Monte Carlo panel."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import json
import pickle

import numpy as np
import pytest

from experiments.tsc_packet_conditional_cache_20261110.verify import parameter_probes
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.synthetic import cases
from experiments.tsc_packet_conditional_particles_20261110.fixtures import pressure_case
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw, make_sampler
from experiments.tsc_packet_conditional_particles_20261110.verify import (
    completed_receipts, consistency, directional_check, error_summary, initialize_worker,
    physical_forbidden, worker_kernel,
)


@pytest.fixture(scope='module')
def small_pools():
    result = {}
    kernels = cases()
    for name in ('yielding_entry','cap4_lateral_earlier_neighbour','stopped_open'):
        kernel,focal = kernels[name]
        before = deepcopy(kernel.state)
        sampler = make_sampler(kernel,focal,max_stochastic=4)
        result[name] = sampler.draw(32,20261110)
        assert kernel.state==before
    return result


@pytest.mark.parametrize('count',[5,10])
def test_fixed_pressure_geometry_really_exceeds_four_actor_domain(count):
    kernel,focal=pressure_case(count)
    selected=select_stochastic(kernel,focal,max_stochastic=count)
    assert len(selected['participants'])==count
    assert selected['participants'][-1]==focal
    assert select_stochastic(kernel,focal,max_stochastic=4)['status']=='OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP'


def test_finite_entry_normalizer_has_a_nonzero_score_and_correct_gradient(small_pools):
    pool=small_pools['yielding_entry']
    theta=parameter_probes()[0]
    with physical_forbidden():
        for label in (0,1):
            row=pool.marginal(theta,'yield_entry',label,n=7)
            assert np.linalg.norm(row['denominator_score'])>1e-6
            count,error=directional_check(pool,theta,('yield_entry',label),7)
            if row['supported']:
                assert count==6 and error<1e-7


def test_same_weight_mixture_preserves_joint_marginals_without_new_physics(small_pools):
    pool=small_pools['cap4_lateral_earlier_neighbour']
    with physical_forbidden():
        for theta in parameter_probes():
            for n in (7,32):
                error,summary=consistency(pool,theta,pool.data['eligibility'],n)
                assert error<1e-10 and summary['global_ess']<=n*(1+1e-12)
                assert abs(sum(pool.marginal(theta,'lane_change',label,n=n)['likelihood']
                               for label in range(3))-1)<1e-10
            count,error=directional_check(pool,theta,('class_speed',1,1.),32)
            assert count==6 and error<1e-7


def test_missing_particle_support_is_infinite_not_probability_floor(small_pools):
    data=small_pools['stopped_open'].to_dict()
    data['particles']=[row for row in data['particles'] if row['outcome']['started']==0]
    assert data['particles']
    pool=ParticleLaw(data)
    with physical_forbidden():
        row=pool.marginal(parameter_probes()[0],'start',1)
    assert row['defined'] and not row['supported']
    assert row['likelihood']==0. and np.isneginf(row['log_likelihood'])
    assert np.isposinf(row['nll']) and row['gradient'] is None


def test_particle_payload_has_no_kernel_or_duplicated_background_frames(small_pools):
    def check(value):
        if isinstance(value,dict):
            assert not {'kernel','packets','projected_packets','full_frame'}&value.keys()
            for child in value.values():
                check(child)
        elif isinstance(value,list):
            for child in value:
                check(child)
    for pool in small_pools.values():
        check(pool.to_dict())
        assert pool.stats['physical_calls']==32


def test_failed_replicate_is_retained_in_overall_error_summary():
    common=dict(seed=1,global_ess=5.,survivor_ess=5.,posterior_ess=3.,maximum_normalized_weight=.2)
    observations=[dict(common,replicate=0,supported=True,failure=None,nll_error=.1,
                       gradient_error=np.ones((5,38))*.01),
                  dict(common,replicate=1,supported=False,failure='ZERO_NUMERATOR')]
    oracle=dict(likelihood=.2,gradient=np.zeros((5,38)))
    result=error_summary(oracle,observations)
    assert result['overall_errors_infinite'] and result['nll_rmse'] is None
    assert result['zero_numerator_replicates']==1 and len(result['failed_replicates'])==1
    assert result['finite_subset_only']['replicates']==1


def _worker_metadata(task):
    name,index,output=task
    kernel,focal=worker_kernel(name)
    return dict(name=name,index=index,output=str(output),focal=focal,
                signals=dict(kernel.signals))


def test_signalled_kernel_is_constructed_inside_worker_from_primitive_task(tmp_path):
    task=('yielding_entry',5,tmp_path)
    pickle.dumps(task)
    with ProcessPoolExecutor(max_workers=1,initializer=initialize_worker) as executor:
        row=executor.submit(_worker_metadata,task).result()
    assert row['name']=='yielding_entry' and row['index']==5
    assert row['signals']=={'J':'g'} and row['focal']=='ego'


def test_resume_reuses_completed_shape_without_physics_and_rejects_partial_work(tmp_path):
    row=dict(case='stopped_open',case_index=3,status='PASS_FIXED_PARTICLE_CASE',
        replicates=32,particle_sizes=[256,1024],parameter_probes=2,
        particle_physical_calls=32768,reference_physical_calls=12,
        metrics={str(n):{str(t):{'motion:1.0':dict(replicates=32)} for t in range(2)} for n in (256,1024)})
    (tmp_path/'stopped_open_result.json').write_text(json.dumps(row))
    summary=dict(requested_cases=15,replicates=32,particle_sizes=[256,1024],
        failed_case_particle_physical_calls=0,failed_case_reference_physical_calls=0,
        cases=[row],completed_cases=1,particle_physical_calls=32768,reference_physical_calls=12)
    (tmp_path/'result.json').write_text(json.dumps(summary))
    with physical_forbidden():
        reused=completed_receipts(tmp_path)
    assert len(reused)==1 and reused[0]['case']=='stopped_open'
    assert reused[0]['reused_from']==str(tmp_path)
    summary['failed_case_particle_physical_calls']=1
    (tmp_path/'result.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='partially evaluated'):
        completed_receipts(tmp_path)
