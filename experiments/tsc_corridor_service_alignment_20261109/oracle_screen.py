"""Perfect PP-arrival labels bound the frozen nominal service score, not traffic cost."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
for path in (HERE,HERE.parent/'tsc_corridor_residual_transfer_20261109',
             HERE.parent/'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0,str(path))
from arrival_policy import ArrivalPolicy,fit_hours as old_fit
from model_gate import hour,load_training,release_id,rows,verify_protocol
from residual_model import HistogramHead,fit_hours,update_history
from aligned_predictor import AlignedPredictor
from evaluate_alignment import ARMS,PAIRS,SCOPES,choose_aligned,label_service_scores
from service_window import extend_window

CONTROLLERS=(*ARMS,'GP')
SOURCE_PAIRS=('S_vs_T','S_vs_P','S_vs_GP')


def oracle_gain(scores,action):
    scores=np.round(np.asarray(scores,dtype=float),12)
    oracle=int(np.argmax(scores));gap=float(scores[oracle]-scores[action])
    return dict(strict_score_gain_sum=gap,positive_epochs=int(gap>0),
                tie_only_change=int(gap==0 and oracle!=action))


def recover_actions(base,logged):
    actions=dict(base,S=base['T'],P=base['T'])
    if logged is not None:
        if any(logged[a]!=base[a] for a in ('T','history','analytic','GP')):
            raise ValueError('Logged source request does not match restored target/physical controls')
        actions.update(S=logged['S'],P=logged['P'])
    return actions


def _panel():
    return dict(differences={f'{a}_vs_{b}':0 for a,b in PAIRS},
                unlocked={f'{a}_vs_{b}':0 for a,b in PAIRS},
                source_delta=Counter({key:0.0 for key in SOURCE_PAIRS}),
                gains={arm:Counter() for arm in CONTROLLERS},counts=Counter())


def check_reproduction(panel,expected,label):
    if any(panel['counts'][key]!=expected['counts'][key] for key in ('scored_decisions','scored_unlocked_decisions')):
        raise ValueError(f'Frozen oracle scoring population changed: {label}')
    if (panel['differences']!=expected['action_differences']
            or panel['unlocked']!=expected['action_differences_unlocked']):
        raise ValueError(f'Frozen aligned request pairs changed: {label}')
    if any(abs(panel['source_delta'][key]-expected['label_score_difference_sums'][key])>1e-12 for key in SOURCE_PAIRS):
        raise ValueError(f'Original PP-label service score differences changed: {label}')


def screen_hour(directory,city,seed,block,target,old_policy,logged,expected):
    outcomes={release_id(r):r for r in rows(directory/'labels.jsonl.gz')}
    predictor=AlignedPredictor(target,HistogramHead(),HistogramHead())
    previous,counts={},Counter();panels={s:_panel() for s in SCOPES}
    for row in rows(directory/'decisions.jsonl.gz'):
        t,tls,gp=row['t'],row['tls'],row['gp'];unlocked=len(row['states'])>1
        counts['decisions']+=1;counts['unlocked_decisions']+=unlocked
        observable=bool(row['flights'] and not row['spillback'] and not row['unmatched'] and t+10<=3600)
        enabled=dict(original_common_scored=observable and bool(old_policy.forecast(row)),all_observable_risk=observable)
        aligned=None
        if observable and unlocked:
            forecasts=predictor.forecast(row)
            aligned=extend_window(row,previous.get(tls),horizon=13)
            base=choose_aligned(aligned,forecasts)
        else:
            update_history(row,predictor.history)
            base={a:gp for a in CONTROLLERS}
        # All actual requests are fixed before this row's future labels are read.
        actions={}
        for scope,active in enabled.items():
            key=scope,city,seed,t,tls
            actions[scope]=recover_actions(base,logged.get(key)) if active else {a:gp for a in CONTROLLERS}
            if not active and key in logged:
                raise ValueError('A logged source disagreement is outside its frozen scoring population')
        label_scores=None
        if aligned is not None:
            label_scores,_=label_service_scores(aligned,row['flights'],outcomes)
        for scope,active in enabled.items():
            panel=panels[scope];chosen=actions[scope]
            for a,b in PAIRS:
                key=f'{a}_vs_{b}';different=chosen[a]!=chosen[b]
                panel['differences'][key]+=different
                panel['unlocked'][key]+=different and unlocked
            if active:
                panel['counts']['scored_decisions']+=1
                panel['counts']['scored_unlocked_decisions']+=unlocked
                if label_scores is not None:
                    for arm in CONTROLLERS:
                        panel['gains'][arm].update(oracle_gain(label_scores,chosen[arm]))
                    for other in ('T','P','GP'):
                        panel['source_delta']['S_vs_'+other]+=float(label_scores[chosen['S']]-label_scores[chosen[other]])
        previous[tls]=row['states'][row['pp']]
    if any(counts[key]!=expected['counts'][key] for key in counts):
        raise ValueError(f'The frozen decision roster changed: {city}/{seed}')
    result=dict(city=city,seed=seed,block=block,counts=dict(counts),scopes={})
    for scope,panel in panels.items():
        check_reproduction(panel,expected['scopes'][scope],f'{city}/{seed}/{scope}')
        result['scopes'][scope]=dict(counts=dict(panel['counts']),request_pairs_reproduced=True,
            PP_label_score_differences_reproduced=True,source_score_delta_sums=dict(panel['source_delta']),
            controllers={arm:dict(strict_score_gain_sum=g.get('strict_score_gain_sum',0.0),
                positive_epochs=g.get('positive_epochs',0),tie_only_change=g.get('tie_only_change',0),
                mean_score_gain_all_decisions=g.get('strict_score_gain_sum',0.0)/counts['decisions'])
                for arm,g in panel['gains'].items()})
    return result


def run(root,cohort_file,aligned_dir,output):
    cohort=json.loads(cohort_file.read_text())['cities'];verify_protocol(cohort)
    expected=json.loads((aligned_dir/'development/result.json').read_text())
    log=aligned_dir/'development/aligned_request_disagreements_server_only.jsonl.gz'
    logged={(r['scope'],r['city'],r['seed'],r['t'],r['tls']):r['actions'] for r in rows(log)}
    source_curves={}
    for city,spec in cohort.items():
        labels=load_training(root,'donor',city,spec['donor'])
        source_curves[city]=old_fit(labels),old_fit(labels,permute=True)
    cities={}
    for city,donor in (('hangzhou','jinan'),('jinan','hangzhou')):
        original={r['seed']:r for r in expected['cities'][city]['by_seed']};panel=[]
        for block,seeds in enumerate(cohort[city]['adapt']):
            target=fit_hours(root,'adapt',city,seeds)
            if dict(block=block,**target.metadata)!=expected['training']['target'][city][block]:
                raise ValueError('Frozen local adaptation evidence changed')
            policy=ArrivalPolicy(old_fit(load_training(root,'adapt',city,seeds)),*source_curves[donor])
            for seed in cohort[city]['dev'][2*block:2*block+2]:
                panel.append(screen_hour(hour(root,'dev',city,seed),city,seed,block,target,policy,logged,original[seed]))
                print(f'Oracle screen {city}/{seed}: all15 request pairs and label-score deltas reproduced',flush=True)
        summaries={}
        for scope in SCOPES:
            summaries[scope]={arm:dict(
                strict_score_gain_sum=sum(r['scopes'][scope]['controllers'][arm]['strict_score_gain_sum'] for r in panel),
                positive_epochs=sum(r['scopes'][scope]['controllers'][arm]['positive_epochs'] for r in panel),
                tie_only_change=sum(r['scopes'][scope]['controllers'][arm]['tie_only_change'] for r in panel),
                equal_seed_mean_score_gain=mean(r['scopes'][scope]['controllers'][arm]['mean_score_gain_all_decisions'] for r in panel))
                for arm in CONTROLLERS}
        cities[city]=dict(by_seed=panel,scopes=summaries)
    result=dict(status='COMPLETE_NOMINAL_ORACLE_SCREEN',previous_status=expected['status'],cities=cities,
        new_candidates=0,new_simulations=0,newly_collected_labels=0,reproduced_seed_hours=sum(len(c['by_seed']) for c in cities.values()),
        oracle_definition='First-in-order maximizer of the fixed13s nominal service score using PP-observed detector arrivals<=10s plus the same freeflow tail; locked/fallback relative gains are structural zero.',
        interpretation='Conditional information upper bound for this nominal PP-label score only; not a deployable policy or actual counterfactual cost. Original HOLD unchanged.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(result['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort','aligned-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.cohort,a.aligned_dir,a.output)
