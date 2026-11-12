"""Reproduce the five rear-vehicle proposal bounds; server-only, zero fit."""
from pathlib import Path
import gzip
import json
from experiments.tsc_target_packet_training_20261109.conditional_features import prepare_feature_kernel, candidate_features
from experiments.tsc_target_packet_training_20261109.training_data import _candidates

old = Path('/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/target_packet_training')
out = old.parent/'target_packet_exclusion_audit/output'
traces = json.loads((out/'lane_neighbour_trace.json').read_text())
static = json.loads((old/'static_graph_server_only.json').read_text())
laws = json.loads((old/'target_laws_server_only.json').read_text())
result = []
for trace in traces:
    if trace['cause'] != 'rear':
        continue
    ref = trace['reference']
    seed, t = int(ref['seed']), int(ref['time_sec'])
    directory = old/'training'/f'seed{seed}'
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as stream:
        frame = json.loads(next(line for i, line in enumerate(stream) if i == t))
    with gzip.open(directory/'execution_server_only.jsonl.gz', 'rt') as stream:
        action = json.loads(next(line for i, line in enumerate(stream) if i == t))
    executor = json.loads((directory/'executor_seed_server_only.json').read_text())
    kernel = prepare_feature_kernel(static, frame, executor, laws, action['requests'], 76116999+seed*100000+t)
    pid, rows = trace['neighbour_id'], []
    for intent, prior in _candidates(kernel, pid).items():
        candidate = candidate_features(kernel, pid, intent)
        rows.append(dict(intent=intent, prior=prior, upper_mps=candidate['upper'],
                         nominal_residual_min_mps=max(candidate['upper']-candidate['b'], 0.)))
    actual = trace['neighbour_actual_next']['speed_mps']
    minimum = min(row['nominal_residual_min_mps'] for row in rows if row['prior'] > 0)
    result.append(dict(reference=ref, neighbour_id=pid, actual_speed_mps=actual,
                       minimum_over_intents_mps=minimum, below_all_proposal_means=actual < minimum,
                       candidates=rows))
(out/'rear_speed_envelope_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
