"""Integration regressions found during the kernel's first implementation."""
import pytest

from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    lane, connection, road_graph, laws, packet, executor_seed,
    initial_observation, coefficients, competing_approaches,
)


def build(static, target_laws, packets, seeds, coeffs=None):
    return PacketKernel(static,initial_observation(static,packets,executors=seeds),seeds,
                        target_laws,coefficients=coeffs or coefficients(),random_seed=0)


def test_directed_g_g_priority_wins_over_packet_insertion_order():
    static,target_laws=competing_approaches()
    seeds={'J':executor_seed(('gg',),clearance=('Ia','Ib'))}
    # b is inserted first and has a shorter ETA, but response[b,a] makes it yield.
    kernel=build(static,target_laws,[packet('minor','b',19.9,10),packet('major','a',19,10)],seeds)
    kernel.step()
    assert kernel.state['packets']['major']['lane_id'] in ('Ia','out_a')
    assert kernel.state['packets']['minor']['lane_id']=='b'


def test_illegal_lateral_choice_is_masked_before_sampling():
    static=road_graph([lane('r0',100,edge='r',index=0),lane('r1',100,edge='r',index=1)],[],cost_lanes=['r0','r1'])
    kernel=build(static,laws(static),[packet('p','r0',20,2)],{},
                 coeffs={name:[0.]*38 for name in ('motion','start','yield_entry','lane_left','lane_right')})
    kernel.noise.step_uniforms=lambda key,time:dict(start=.45,**{'yield':.45},lane=.45)
    kernel.step()
    # Left and stay each have probability1/2 after masking missing right.
    # Sampling the unmasked three-way law would choose stay at .45.
    assert kernel.state['packets']['p']['lane_id']=='r1'


def test_receiving_merge_updates_internal_conflict_clearing_reservation():
    static=road_graph(
        [lane('a',20),lane('b',20),lane('c',20),
         lane('Ia',2,internal=True),lane('Ib',2,internal=True),lane('Ic',10,internal=True),
         lane('out',100),lane('outc',100)],
        [connection('a','out','Ia',tls='J',link=0),
         connection('b','out','Ib',tls='J',link=1),
         connection('c','outc','Ic',tls='J',link=2),
         connection('Ia','out'),connection('Ib','out'),connection('Ic','outc')],
        foes=((1,2),(2,1)),yields=((2,1),))
    target=laws(static,intents={'a':{'out':1.},'b':{'out':1.},'c':{'outc':1.},
                               'out':{'END':1.},'outc':{'END':1.}})
    seeds={'J':executor_seed(('GGg',),clearance=('Ia','Ib','Ic'))}
    kernel=build(static,target,[packet('merge','Ia',1.9,.1),packet('blocked','Ib',1,15),
                               packet('foe','c',15,6)],seeds)
    kernel.step()
    assert kernel.state['packets']['merge']['lane_id']=='out'
    assert kernel.state['packets']['blocked']['lane_id']=='Ib'
    assert kernel.state['packets']['blocked']['speed_mps']==0
    # The first optimistic reservation cleared at (1+5)/18=1/3 sec.
    # The receiving merge stopped that packet, so foe entry at5/9 sec is blocked.
    assert kernel.state['packets']['foe']['lane_id']=='c'


def test_different_first_actions_keep_the_same_future_boundary_scenario():
    static,target=competing_approaches(with_foes=False)
    target.update(due_rate_per_sec=.6,entry_probs={'a':1.})
    seeds={'J':executor_seed(('Gr','rG'),yellow=1.,all_red=1.,min_green=1.)}
    packets=[packet('initial','a',19,6)]
    first=build(static,target,packets,seeds)
    second=build(static,target,packets,seeds)
    rows_a=[first.step(initial_requests={'J':'Gr'})]
    rows_b=[second.step(initial_requests={'J':'rG'})]
    for _ in range(19):
        rows_a.append(first.step());rows_b.append(second.step())
    assert [row['due_ids'] for row in rows_a]==[row['due_ids'] for row in rows_b]
    assert any(a['events']!=b['events'] for a,b in zip(rows_a,rows_b))
