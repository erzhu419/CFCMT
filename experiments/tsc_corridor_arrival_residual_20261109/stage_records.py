"""Copy existing compressed PP records directly between servers; no local raw data."""
from pathlib import Path
import json
import shlex
import subprocess

SOURCE = '/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_corridor_arrival_transfer_20261107'
DEST = '/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109'

def stage():
    # The approved analysis host already has read access to the original data host.
    code = f"""from pathlib import Path
import json,subprocess
root=Path({DEST!r})/'data'
root.mkdir(parents=True,exist_ok=True)
sources=['zhengliang01@202.197.46.16:'+{SOURCE!r}+'/'+part for part in ['donor','adapt','dev']]
subprocess.run(['rsync','-a','--exclude=*.log',*sources,str(root)+'/'],check=True)
files=list(root.glob('*/*/*/*.jsonl.gz'))
print(json.dumps(dict(status='STAGED_EXISTING_RECORDS',hours=len(list(root.glob('*/*/*/decisions.jsonl.gz'))),compressed_bytes=sum(p.stat().st_size for p in files))))
"""
    completed = subprocess.run(['ssh','-o','BatchMode=yes','jtl110gpu2',
        shlex.join(['python3','-c',code])], capture_output=True,text=True,check=True,timeout=180)
    summary=json.loads(completed.stdout)
    summary.update(source=SOURCE,destination=DEST,analysis_server='jtl110gpu2',local_raw_data_bytes=0,new_simulations=0)
    Path(__file__).with_name('staging.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)

if __name__ == '__main__':
    stage()
