from pathlib import Path
from collections import defaultdict
import csv,json,math
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'cf_h2o/results/cluster/tsc_v93_source_city_components_20260831'
OUT=Path(__file__).resolve().parents[1]/'source_data'
OUT.mkdir(parents=True,exist_ok=True)
SCENARIOS={'los_angeles':['la_1x4'],'jinan':['jinan_3x4_real','jinan_3x4_real_2000','jinan_3x4_real_2500']}
SEEDS=[1349594,3900758,5883745,6180752,6345095,7677617,8729267,9488138]
WAIT='mean_tripinfo_waiting_time'
metrics=['collision_incidents','starting_teleports','ending_teleports','system_vehicle_hours','completion_ratio']
all_rows={};metadata={};audits={}
for stage,directory in [('identity','source_identity_screen_v1'),('weight','source_weight_curve_v1'),('guard','source_guard_screen_v1')]:
 folder=BASE/directory
 audit=json.loads((folder/'audit.json').read_text());audits[stage]=audit
 files=sorted(folder.glob('*/_shards/shard_*.json')) if stage=='guard' else sorted((folder/'_shards').glob('shard_*.json'))
 rows=[]
 for path in files:
  data=json.loads(path.read_text());assert data.get('passed') is True,path
  for row in data['rows']:
   r=dict(row);r['profile']=path.parent.parent.name if stage=='guard' else 'raw';rows.append(r)
 grouped=defaultdict(list)
 for r in rows:grouped[(r['profile'],r['city'],r['candidate_key'],int(r['seed']))].append(r)
 expected_candidates=8 if stage=='identity' else 5 if stage=='weight' else 2
 expected_profiles=4 if stage=='guard' else 1
 assert len(grouped)==expected_candidates*expected_profiles*len(SCENARIOS)*len(SEEDS)
 means={}
 for key,rs in grouped.items():
  profile,city,candidate,seed=key
  assert seed in SEEDS
  assert sorted(r['scenario'] for r in rs)==sorted(SCENARIOS[city]),key
  weights=[r['source_city_weights'] for r in rs]
  assert all(w==weights[0] for w in weights)
  configs=[r['guard_config'] for r in rs]
  assert all(config==configs[0] for config in configs)
  waiting=sum(float(r['metrics'][WAIT]) for r in rs)/len(rs)
  item={'stage':stage,'city':city,'candidate_key':candidate,'seed':seed,'scenario_count':len(rs),'source_group':next(iter(weights[0]),'target_anchor'),'source_mass':sum(weights[0].values()),'requested_source_mass':float(candidate.removeprefix('selected_w').replace('p','.')) if stage=='weight' and candidate.startswith('selected_w') else sum(weights[0].values()),'guard_profile':profile,'risk_multiplier':configs[0]['risk_multiplier'] if configs[0] is not None else '', 'mean_waiting_sec':waiting}
  for metric in metrics:
   val=sum(float(r['metrics'][metric]) for r in rs)
   item[metric]=int(val) if metric in {'collision_incidents','starting_teleports','ending_teleports'} else val/len(rs)
  means[key]=item
 results=[]
 for key,item in sorted(means.items()):
  profile,city,candidate,seed=key
  target=means[(profile,city,'target_only',seed)]['mean_waiting_sec']
  raw=means[('raw',city,candidate,seed)]['mean_waiting_sec'] if stage=='guard' else item['mean_waiting_sec']
  item.update(target_anchor_waiting_sec=target,delta_vs_target_pct=(item['mean_waiting_sec']/target-1)*100,raw_same_candidate_waiting_sec=raw,delta_vs_raw_same_candidate_pct=(item['mean_waiting_sec']/raw-1)*100)
  results.append(item)
 # Compare every candidate's full eight-seed delta vector to its original audit.
 for city in SCENARIOS:
  if stage in {'identity','weight'}:
   for candidate,cell in audit['selections'][city]['grid'].items():
    computed=[means[('raw',city,candidate,seed)]['delta_vs_target_pct']/100 for seed in SEEDS]
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(computed,cell['seed_relative_deltas']))
  else:
   for profile,cell in audit['guard_selection_by_city'][city]['grid'].items():
    computed=[means[(profile,city,'selected_source',seed)]['delta_vs_raw_same_candidate_pct']/100 for seed in SEEDS]
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(computed,cell['seed_relative_deltas']))
   for profile,city_effects in audit['source_effect_by_profile'].items():
    for candidate,cell in city_effects[city]['grid'].items():
     computed=[means[(profile,city,candidate,seed)]['delta_vs_target_pct']/100 for seed in SEEDS]
     assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(computed,cell['seed_relative_deltas']))
 with (OUT/f'source_{stage}.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
 all_rows[stage]=results
 metadata[stage]={'audit':str((folder/'audit.json').relative_to(ROOT)),'shards':[str(p.relative_to(ROOT)) for p in files],'input_rollouts':len(rows),'exported_city_candidate_seed_rows':len(results),'candidate_count':expected_candidates,'profile_count':expected_profiles,'original_audit_delta_vectors_match':True,'output':f'source_{stage}.csv'}
metadata.update({'scope':'V93 target development screens; not held-out confirmation','seeds':SEEDS,'seeds_per_city':len(SEEDS),'city_scenarios':SCENARIOS,'aggregation':'For each city/candidate/seed, arithmetic mean waiting over all its scenarios (three Jinan demands; one Los Angeles demand). Relative change is formed within this matched city/seed, then averaged over seeds for plotting.','data_fields':{'source_mass':'Actual source-city prediction mass in the executed candidate; not alpha for the reference/pairwise component blend.','requested_source_mass':'Requested candidate-grid label; for Los Angeles the selected source is null so its actual source_mass remains zero even for selected_w1p0.','risk_multiplier':'Guard risk multiplier for this profile; empty means no guard, 0 is the mean-only guard. It is not an epistemic-dispersion coefficient learned from ensembles.','delta_vs_target_pct':'Relative waiting change against target_only run under the same guard profile, city, and seed.','delta_vs_raw_same_candidate_pct':'Guard-profile waiting change against the raw unguarded run of that same candidate, city, and seed. Only guard-screen rows vary in this field.','collision_incidents':'Sum across the city demand scenarios; other continuous metrics average across scenarios.'},'limitations':['All three stages are development-only and sequentially selected; they are not independent confirmation.','The source-weight screen varies Atlanta mass only in Jinan. Los Angeles has no selected source; all its actual source masses are zero.','The guard screen fixes the selected source weight (Jinan Atlanta mass one; Los Angeles zero) and varies raw/mean-only/risk0.25/risk0.5. It is not a full source-weight by guard-factor experiment.','No switch-cost removal or per-mechanism ablation exists in these records.','These are genuine rollout aggregates, not individual-vehicle waiting samples or microscopic time-position trajectories.'],'selected_identity_mass':audits['identity']['selected_weights_by_city'],'selected_final_mass':audits['weight']['selected_weights_by_city'],'selected_guard_by_city':audits['guard']['selected_guard_by_city']})
(OUT/'data_sources_screens.json').write_text(json.dumps(metadata,indent=2)+'\n')
for stage,rows in all_rows.items():
 print(stage,len(rows),'city means:')
 for city in SCENARIOS:
  combinations=sorted({(r['guard_profile'],r['candidate_key']) for r in rows if r['city']==city})
  for profile,candidate in combinations:
   rs=[r for r in rows if r['city']==city and r['guard_profile']==profile and r['candidate_key']==candidate]
   print(city,profile,candidate,'mass',rs[0]['source_mass'],'wait',sum(r['mean_waiting_sec'] for r in rs)/len(rs),'delta_target_pct',sum(r['delta_vs_target_pct'] for r in rs)/len(rs),'delta_raw_pct',sum(r['delta_vs_raw_same_candidate_pct'] for r in rs)/len(rs))
