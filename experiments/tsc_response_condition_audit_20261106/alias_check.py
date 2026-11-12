"""Exact-input ambiguity in saved action groups; descriptive lower bound only."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);spec=json.loads((root/'protocol.json').read_text());out={}
for city in ['cologne','jinan']:
 rows=[]
 for case in spec['cases']:
  if case['city']!=city:continue
  p=root/'collection'/f"seed_{case['seed']}"/case['case_id']/'group_data_server_only.json'
  d=json.loads(p.read_text());X=np.asarray(d['X']);Y=np.asarray(d['Y']);ref=d['reference_index'];M=X.shape[1]
  identical=0;different=np.zeros(2,dtype=int);bound=np.zeros(2);ambiguous=np.zeros(2,dtype=int)
  for m in range(M):
   bins={}
   for a in range(8):
    if a==ref:continue
    bins.setdefault(tuple(X[a,m]),[]).append(a)
    if np.array_equal(X[a,m],X[ref,m]):
     identical+=1;different+=(Y[a,m]!=Y[ref,m])
   for features,indices in bins.items():
    delta=Y[indices,m]-Y[ref,m]
    prediction=np.zeros(2) if features==tuple(X[ref,m]) else np.median(delta,axis=0)
    bound+=np.abs(delta-prediction).sum(axis=0)
    ambiguous+=np.ptp(delta,axis=0)>0
  rows.append({'case_id':case['case_id'],'cells':7*M,'identical_to_PP_cells':identical,'different_truth_despite_identical_PP_X':different.tolist(),'varying_label_input_classes':ambiguous.tolist(),'contrast_mae_lower_bound':(bound/(7*M)).tolist()})
 out[city]={'groups':len(rows),'cells':sum(r['cells'] for r in rows),'identical_to_PP_cells':sum(r['identical_to_PP_cells'] for r in rows),'different_truth_despite_identical_PP_X':np.sum([r['different_truth_despite_identical_PP_X'] for r in rows],axis=0).tolist(),'varying_label_input_classes':np.sum([r['varying_label_input_classes'] for r in rows],axis=0).tolist(),'within_state_movement_contrast_mae_lower_bound':np.mean([r['contrast_mae_lower_bound'] for r in rows],axis=0).tolist()}
print(json.dumps({'status':'DESCRIPTIVE_ONLY','new_fits':0,'new_simulations':0,'output_order':spec['label_names'],'semantics':'Exact full37feature equality; lower bound allows arbitrary per-state/movement predictors, assigns each nonPP feature class median true contrast, and forces zero contrast for the PP feature class. This is an optimistic representation bound, not a fitted/usable predictor.','cities':out}))
