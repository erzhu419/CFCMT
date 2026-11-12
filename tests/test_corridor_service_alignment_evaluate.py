"""Endpoint preservation, aligned seconds and factual-label score interpretation."""
import gzip
import json

import numpy as np

from experiments.tsc_corridor_service_alignment_20261109 import evaluate_alignment as e


def test_old_projection_uses_last_endpoint_not_service_second_eleven():
    values=np.zeros((5,14));values[:,10]=9;values[:,-1]=2
    old=e.old_forecasts({'road':values})['road']
    assert old.shape==(5,11)
    assert np.all(old[:,:10]==0)
    assert np.all(old[:,-1]==2)


def test_aligned_choice_uses_post_ten_second_service_and_original_ties():
    values=np.zeros((5,14));values[:,-1]=1;values[:,12]=1;values[1:3,11]=1
    row=dict(base=[1,0],protected={'road':[0,(1<<11)|(1<<12)]},gp=0)
    actions=e.choose_aligned(row,{'road':values})
    assert actions['T']==0  # queue versus one arrival-credit tie retains first request
    assert actions['S']==actions['P']==1
    assert actions['history']==actions['analytic']==0
    assert actions['GP']==0


def test_label_service_score_uses_observed_detector_ten_and_nominal_tail():
    row=dict(t=20,base=[5,3],protected={'road':[0,1<<12]})
    flights=[];outcomes={}
    for vid,status,end in [('a','straight',30),('b','straight',31),('c','other',21),('d','censored',3600)]:
        f=dict(edge='road',vid=vid,entry_s=0,age_s=20,tail_s=2.7)
        flights.append(f);outcomes[e.release_id(f)]=dict(edge='road',vid=vid,entry_s=0,end_s=end,status=status)
    scores,actual=e.label_service_scores(row,flights,outcomes)
    np.testing.assert_array_equal(scores,[5,4])
    assert actual['road']==1


def write_rows(path,records):
    with gzip.open(path,'wt') as stream:
        for record in records:stream.write(json.dumps(record)+'\n')


def test_previous_pp_request_updates_on_fallback_rows_and_old_outputs_reproduce(tmp_path,monkeypatch):
    mask10=sum(1<<i for i in range(5,10))
    flight=dict(edge='road',vid='a',entry_s=0,age_s=20,tail_s=2.7)
    def row(t,pp,flights):
        return dict(t=t,tls='signal',states=['Gr','rG'],pp=pp,gp=0,base=[0,0],protected={'road':[0,mask10]},
                    flights=flights,spillback=False,unmatched=False)
    data=[row(0,1,[]),row(10,0,[]),row(20,1,[flight])]
    write_rows(tmp_path/'decisions.jsonl.gz',data)
    write_rows(tmp_path/'labels.jsonl.gz',[dict(edge='road',vid='a',entry_s=0,end_s=30,status='straight')])
    calls=[]
    def extend(row,previous_request,horizon):
        calls.append(previous_request)
        assert previous_request=='Gr'  # t=10 had no flights but its PP request is current now
        return dict(row,base=[0,0],protected={'road':[0,mask10|(7<<10)]})
    monkeypatch.setattr(e,'extend_window',extend)
    class Predictor:
        def forecast(self,row):
            if not row['flights']:return {}
            values=np.zeros((5,14));values[:,12:]=1
            return {'road':values}
    class OldPolicy:
        def forecast(self,row):return {'road':True} if row['flights'] else None
    pairs={f'{a}_vs_{b}':0 for a,b in e.PAIRS}
    previous=dict(counts=dict(decisions=3,unlocked_decisions=3,locked_decisions=0,inflight_decisions=1,spillback_decisions=0,unmatched_decisions=0),
        scopes={s:dict(counts={'scored_edge_epochs':1},mae={a:0.0 for a in e.ARMS},action_differences=pairs,
                       action_differences_unlocked=pairs) for s in e.SCOPES})
    record=e.evaluate_hour('jinan',1,0,tmp_path,Predictor(),OldPolicy(),previous)
    assert calls==['Gr']
    for scope in e.SCOPES:
        assert record['old_reproduction'][scope]['all_request_pairs']
        assert record['scopes'][scope]['alignment_changes']['S']==1
        assert record['scopes'][scope]['label_score_differences']['S_vs_T']==0
        assert record['scopes'][scope]['action_differences']['S_vs_T']==0
