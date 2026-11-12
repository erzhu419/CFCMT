"""Locate information loss for exact-input, different-response candidate/PP pairs."""
import json
from pathlib import Path
import sys
import numpy as np
root=Path(sys.argv[1]);spec=json.loads((root/'protocol.json').read_text());results={}
for city in ['cologne','jinan']:
 counts={k:np.zeros(2,dtype=int) for k in ['different_truth_equal_X','own_nominal_sequence_equal','full_TLS_nominal_sequence_equal']};groups=set();seeds=set()
 for case in spec['cases']:
  if case['city']!=city:continue
  p=root/'collection'/f"seed_{case['seed']}"/case['case_id'];d=json.loads((p/'group_data_server_only.json').read_text());X,Y=np.asarray(d['X']),np.asarray(d['Y']);ref=d['reference_index']
  catalog=json.loads((p/'first_pp/observations_server_only.json').read_text())['catalog']
  for a in range(8):
   if a==ref:continue
   for m,key in enumerate(d['movement_ids']):
    changed=Y[a,m]!=Y[ref,m]
    if not np.array_equal(X[a,m],X[ref,m]) or not changed.any():continue
    counts['different_truth_equal_X']+=changed;groups.add(case['case_id']);seeds.add(case['seed'])
    tls=catalog[key]['tls_id'];links=catalog[key]['link_indices']
    actual=d['branches'][a]['preaction']['nominal_states'][tls];reference=d['branches'][ref]['preaction']['nominal_states'][tls]
    counts['own_nominal_sequence_equal']+=changed*all(x[i]==y[i] for x,y in zip(actual,reference) for i in links)
    counts['full_TLS_nominal_sequence_equal']+=changed*(actual==reference)
 results[city]={**{k:v.tolist() for k,v in counts.items()},'groups_with_any_alias':len(groups),'seeds_with_any_alias':len(seeds)}
print(json.dumps({'status':'DESCRIPTIVE_ONLY','new_fits':0,'new_simulations':0,'output_order':spec['label_names'],'cities':results,'scope':'Nominal sequences only. Use the separate actual-vs-nominal audit before inferring actual signal equivalence; no causal identification of omitted interaction mechanism.'}))
