"""Check exact plan reconstruction on existing logs before scoring a candidate."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from service_window import extend_window


def run(root, cohort, output):
    cities=json.loads(cohort.read_text())['cities']
    results={}
    for city,spec in cities.items():
        counts=Counter();clearances=Counter();tails=set();seed_checks=[]
        for seed in spec['dev']:
            directory=root/'dev'/city/str(seed)
            receipt=json.loads((directory/'result.json').read_text())
            tails.update(meta['tail_s'] for meta in receipt['links'].values())
            previous={};n=0
            with gzip.open(directory/'decisions.jsonl.gz','rt') as stream:
                for line in stream:
                    row=json.loads(line);counts['decisions']+=1
                    if len(row['states'])>1:
                        extended=extend_window(row,previous.get(row['tls']),13)
                        clearances[str(extended['service_window']['clearance_sec'])]+=1
                        counts['unlocked_reconstructed']+=1;n+=1
                    previous[row['tls']]=row['states'][row['pp']]
            seed_checks.append(dict(seed=seed,reconstructed=n))
        if any(10+tail>13 for tail in tails):
            raise ValueError('Fixed13s preview does not cover all observed detector tails')
        results[city]=dict(counts=dict(counts),clearance_seconds=dict(clearances),tails=sorted(tails),seed_checks=seed_checks)
    output.write_text(json.dumps(dict(status='PASS_EXACT_RECONSTRUCTION',cities=results,new_simulations=0),indent=2)+'\n')
    print(json.dumps(results),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','cohort','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.cohort,a.output)
