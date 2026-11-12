#!/usr/bin/env python3
"""Pull compact completed shard summaries and the one short request trace."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
REMOTE='/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/evaluation'
sys.path.insert(0,'/home/erzhu419/mine_code/scheduleurm/skill')
import scheduler


def collect(nodes):
    paths=[]; receipt=[]
    for node in nodes:
        code=f'''from pathlib import Path
import json
r=Path({REMOTE!r})
s={{p.name:json.loads(p.read_text()) for p in r.glob('_shards/shard_*.json')}}
p=r/'jinan_3x4_real/seed_5622039/exact_reference/result.json'
t=json.loads(p.read_text())['metrics'].get('reference_policy_audit') if p.exists() else None
print(json.dumps(dict(summaries=s,reference_trace=t,completed=len(list(r.glob('**/result.json')))),separators=(',',':')))
'''
        rc,out,err=scheduler.run_on(node,'python3 -c '+shlex.quote(code),timeout=30,check=False)
        if rc:raise RuntimeError(err[-1000:])
        v=json.loads(out);receipt.append(dict(node=node,completed=v['completed'],shards=list(v['summaries'])))
        folder=HERE/'shards/_shards';folder.mkdir(parents=True,exist_ok=True)
        for name,summary in v['summaries'].items():
            p=folder/name
            if p.exists() and json.loads(p.read_text())!=summary:
                raise ValueError(f'Duplicate differing shard summary: {name}')
            p.write_text(json.dumps(summary,indent=2)+'\n');paths.append(str(p))
        if v['reference_trace']:
            (HERE/'exact_reference_unique_case_trace.json').write_text(json.dumps(v['reference_trace'],indent=2)+'\n')
    status=dict(collected_at_utc=datetime.now(timezone.utc).isoformat(),nodes=receipt,summary_paths=paths,storage='compact shard JSON and one short request trace only; raw tripinfo/results remain remote')
    (HERE/'collection.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nodes',nargs='+',default=['jtl110gpu2','jtl311linux'],choices=['jtl110gpu','jtl110gpu2','jtl311linux'])
    collect(p.parse_args().nodes)
