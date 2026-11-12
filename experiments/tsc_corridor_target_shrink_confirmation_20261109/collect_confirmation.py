"""Collect exactly the six frozen confirmation seeds with the original collector."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys


def run(root, collector_dir, sumocfg, workers):
    protocol = json.loads((Path(__file__).resolve().parent/'protocol.json').read_text())
    sys.path.insert(0, str(collector_dir))
    from run_hours import collect
    seeds = [seed for block in protocol['seeds_by_block'] for seed in block]
    jobs = [('confirmation', 'hangzhou', str(sumocfg), seed) for seed in seeds]
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(collect, job, collector_dir, root) for job in jobs]
        for future in as_completed(futures):
            records.append(future.result())
    complete = len(records) == 6 and all(row['status'] == 'PASS_HOUR' for row in records)
    result = dict(status='PASS_CONFIRMATION_COLLECTION' if complete else 'HOLD_INCOMPLETE_CONFIRMATION',
                  scheduled=6, hours=sorted(records, key=lambda row: row['seed']))
    root.mkdir(parents=True, exist_ok=True)
    (root/'collection_result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'collector-dir', 'sumocfg'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    result = run(args.root, args.collector_dir, args.sumocfg, args.workers)
    raise SystemExit(0 if result['status'] == 'PASS_CONFIRMATION_COLLECTION' else 2)
