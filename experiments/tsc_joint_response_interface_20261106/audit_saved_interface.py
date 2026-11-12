#!/usr/bin/env python3
"""Construct preaction graphs from saved data; no fitting or SUMO execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np

from inventory_sources import inventory
from static_context import read_static_context
from tsc_joint_response_interface import PROTOCOL, build_joint_response_interface


def run(root, output, source_configs):
    start=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    (output/'interfaces_server_only').mkdir()
    spec=json.loads((root/'protocol.json').read_text());rows=[];contexts={};networks={}
    for case in spec['cases']:
        cfg=case['sumocfg']
        if cfg not in contexts:
            contexts[cfg]=read_static_context(Path(cfg))
            networks[cfg]=inventory(Path(cfg),case['scenario'],case['city'])
        static=contexts[cfg]
        directory=root/'collection'/f"seed_{case['seed']}"/case['case_id']
        group=json.loads((directory/'group_data_server_only.json').read_text())
        obs=json.loads((directory/'first_pp/observations_server_only.json').read_text())
        clean={k:group[k] for k in ['case_id','movement_ids','reference_index','feature_names','X']}
        clean['branches']=[{'candidate_index':b['candidate_index'],'preaction':{
            k:b['preaction'][k] for k in ['nominal_states','nominal_request_accepted']}} for b in group['branches']]
        interface=build_joint_response_interface(clean,obs['frames'][0],obs['catalog'],obs['lane_geometry'],
                    static['lane_speed_limits'],conflict_relations=static['conflicts_by_tls'].get(case['tls_id']))
        lanes={n['lane_id']:n for n in interface['lane_nodes']}
        ref=group['reference_index'];X,Y=np.asarray(group['X']),np.asarray(group['Y'])
        count_index=group['feature_names'].index('incoming_vehicle_count')
        geometry_error=0.;count_error=0.
        for i,key in enumerate(group['movement_ids']):
            incoming=obs['catalog'][key]['incoming_lanes']
            count_error=max(count_error,abs(sum(lanes[l]['vehicle_count'] for l in incoming)-X[ref,i,count_index]))
        for lane,n in lanes.items():
            geometry_error=max(geometry_error,abs(n['length_m']-static['lane_lengths'][lane]))
        assert count_error==0 and geometry_error<1e-7
        aliases=np.zeros(2,dtype=int);separated=np.zeros(2,dtype=int)
        for a in range(8):
            if a==ref:continue
            different=interface['candidate_views'][a]['link_sequences']!=interface['candidate_views'][ref]['link_sequences']
            for m in range(X.shape[1]):
                if np.array_equal(X[a,m],X[ref,m]):
                    changed=Y[a,m]!=Y[ref,m];aliases+=changed;separated+=changed*different
        incoming=[n for n in lanes.values() if 'incoming' in n['roles']]
        cfg_inventory=networks[cfg]
        observed_min=X[ref,0,group['feature_names'].index('current_min_green_sec')]
        estimates=cfg_inventory['static_effective_min_green_sec_estimate']
        timing_in_range=bool(estimates['min']<=observed_min<=estimates['max'])
        assert timing_in_range
        record={'case_id':case['case_id'],'city':case['city'],'seed':case['seed'],
            'lane_nodes':len(lanes),'incoming_lane_nodes':len(incoming),'movement_nodes':len(interface['movement_nodes']),
            'link_nodes':len(interface['link_nodes']),'candidate_views':len(interface['candidate_views']),
            'missing_incoming_lane_measurements':sum(n['lane_measurements'] is None for n in incoming),
            'old_equal_input_different_response':aliases.tolist(),'separated_by_joint_signal_sequences':separated.tolist(),
            'conflict_status':interface['conflicts']['status'],
            'conflict_mapping_complete':interface['conflicts']['mapping_complete'],
            'full_conflict_context_complete':interface['conflicts'].get('full_context_complete',False),
            'foe_pairs':len(interface['conflicts']['foes'] or []),'response_pairs':len(interface['conflicts']['response'] or []),
            'external_foe_connection_count':interface['conflicts'].get('external_foe_connection_count'),
            'incoming_count_matches_original':count_error==0,'static_geometry_matches_saved':geometry_error<1e-7,
            'observed_executor_min_green_sec':float(observed_min),'static_timing_range_contains_observed':timing_in_range}
        rows.append(record)
        (output/'interfaces_server_only'/f"{case['case_id']}.json").write_text(json.dumps(interface,separators=(',',':'),allow_nan=False)+'\n')
    candidate_rows=[]
    for candidate in source_configs:
        cfg=Path(candidate['sumocfg']);entry=dict(candidate)
        if cfg.exists():
            entry.update(inventory(cfg,candidate['scenario'],candidate['city_group']))
            st=read_static_context(cfg)
            entry['conflict_mapping_complete_tls']=sum(v['mapping_complete'] for v in st['conflicts_by_tls'].values())
            entry['full_context_complete_tls']=sum(v.get('full_context_complete',False) for v in st['conflicts_by_tls'].values())
            entry['available']=True
        else:entry['available']=False
        candidate_rows.append(entry)
    city_summary={}
    for city in ['cologne','jinan']:
        selected=[r for r in rows if r['city']==city]
        city_summary[city]={'groups':len(selected),'candidate_views':sum(r['candidate_views'] for r in selected),
            'incoming_lane_nodes':sum(r['incoming_lane_nodes'] for r in selected),
            'missing_incoming_lane_measurements':sum(r['missing_incoming_lane_measurements'] for r in selected),
            'old_equal_input_different_response':np.sum([r['old_equal_input_different_response'] for r in selected],axis=0).tolist(),
            'separated_by_joint_signal_sequences':np.sum([r['separated_by_joint_signal_sequences'] for r in selected],axis=0).tolist(),
            'conflict_status_counts':dict(Counter(r['conflict_status'] for r in selected)),
            'full_conflict_context_complete_groups':sum(r['full_conflict_context_complete'] for r in selected)}
    result={'protocol':'joint-response-preaction-interface-audit-v1','interface_protocol':PROTOCOL,'status':'INTERFACE_AUDITED','new_fits':0,'new_simulations':0,
        'groups':len(rows),'candidate_views':sum(r['candidate_views'] for r in rows),'cities':city_summary,
        'case_results':rows,'current_exact_networks':list(networks.values()),'candidate_source_networks':candidate_rows,
        'graph_artifacts_bytes':sum(p.stat().st_size for p in (output/'interfaces_server_only').glob('*.json')),
        'elapsed_sec':time.monotonic()-start,'old_response_gate':'HOLD_RESPONSE',
        'scope':'Information availability and observed-label separation only. No learned gain, source admission, performance or closed-loop claim.',
        'collection_gap':'Direct t0 lane observations are absent for some incoming lanes; v2 recovers exact singleton incoming rows. Multi-lane aggregates are not split. Future collection should record all focal lanes directly.',
        'source_selection_boundary':'Physical static overlap only. Queue/demand overlap and operational validity remain unmeasured; canonical and hetero Hangzhou are one city family.'}
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-configs',type=Path,required=True)
    a=p.parse_args();r=run(a.root,a.output,json.loads(a.source_configs.read_text()))
    print(json.dumps({k:r[k] for k in ['status','groups','candidate_views','cities','elapsed_sec']}),flush=True)
