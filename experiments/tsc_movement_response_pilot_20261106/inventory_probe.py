import json,re,pickle
from pathlib import Path
import numpy as np
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
from cf_h2o.traffic_signal.action_contrast import rule_reference_indices
inputs={'source_base_bank': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v154p_cologne_native_b100_20260909/fit_v1/bank.npz', 'target_bank': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v158_native_prefix_ranking_calibration_20260911/fit_v1/native_b100.npz', 'source_sumocfg': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v154d_cologne_bilateral_waiting_geometry_20260909/package_v1/cologne1.sumocfg', 'target_sumocfgs': {'jinan_3x4_real': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v53r49_external_network_repair_20260810/conversion_v9_task/full_networks_v9/jinan_3x4_real/jinan_3x4_real.sumocfg', 'jinan_3x4_real_2000': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v53r49_external_network_repair_20260810/conversion_v9_task/full_networks_v9/jinan_3x4_real_2000/jinan_3x4_real_2000.sumocfg', 'jinan_3x4_real_2500': '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v53r49_external_network_repair_20260810/conversion_v9_task/full_networks_v9/jinan_3x4_real_2500/jinan_3x4_real_2500.sumocfg'}, 'snapshot_root': '/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/423dc81960505f69f08e'}
cases=[];schemas={}
for city,path in [('cologne',inputs['source_base_bank']),('jinan',inputs['target_bank'])]:
 d=load_mechanism_dataset(Path(path));ids=np.asarray(d.metadata['action_group_ids']);refs=rule_reference_indices(d,policy='phase_pressure')
 roster=[]
 for group in dict.fromkeys(ids.tolist()):
  rows=np.flatnonzero(ids==group); first=int(rows[0]); seed=int(re.search(r':seed(\d+):',group).group(1)); checkpoint=int(d.metadata['row_times'][first]); scenario=d.metadata.get('row_scenarios',[d.metadata.get('scenario')]*d.size)[first]
  if city=='jinan' and scenario!='jinan_3x4_real':continue
  eligible=[j for j,i in enumerate(rows) if int(i)!=refs[group] and d.features[i,d.feature_names.index('switch_indicator')]>0]
  if d.features[first,d.feature_names.index('total_q')]>0 and eligible:roster.append((seed,checkpoint,group,rows,eligible[0],scenario))
 seed,checkpoint,group,rows,chosen,scenario=sorted(roster,key=lambda x:x[:3])[0]
 ref=list(rows).index(refs[group]); phases=[d.metadata['candidate_states'][i] for i in rows];costs=d.targets['interval_cost' if city=='cologne' else 'absolute_cost'][rows]
 case=dict(city=city,scenario=scenario,group_id=group,seed=seed,checkpoint_sec=checkpoint,tls_id=d.metadata['row_tls'][int(rows[0])],bank_path=path,sumocfg=inputs['source_sumocfg'] if city=='cologne' else inputs['target_sumocfgs'][scenario],candidate_states=phases,reference_index=ref,candidate_index=chosen,expected_costs={'pp':float(costs[ref]),'candidate':float(costs[chosen])},preaction_total_q=float(d.features[rows[0],d.feature_names.index('total_q')]),controlled_lane_count=8 if city=='cologne' else 144)
 if city=='cologne':
  case['interval_index']=(checkpoint-25200)//10
  case['native_source_record']=next(x for x in d.metadata.get('native_group_sources',[]) if x['group_id']==group)
 else:
  parent=Path(path).parents[1];case['parent_root']=str(parent)
  old=json.loads((parent/'development_v1'/f'seed_{seed}'/'result.json').read_text());traces=pickle.loads(Path(old['full_traces_server_only']).read_bytes());selected=traces[group]
  schemas['jinan_original_trace_group_keys']=list(selected)
  schemas['jinan_original_trace_baseline_keys']=list(selected.get('baseline_window',{}))
  case['original_trace_path']=old['full_traces_server_only']
 schemas[city]={'targets':list(d.targets),'feature_count':len(d.feature_names),'row_metadata_keys':list(d.metadata),'has_per_vehicle_step_movements':False}
 cases.append(case)
print(json.dumps({'cases':cases,'schemas':schemas,'new_fits':0,'new_branches':0}))
