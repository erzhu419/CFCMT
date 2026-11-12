"""Posthoc decomposition of saved paired outputs; no model queries or fitting."""
import argparse
from collections import defaultdict
import gzip
import json
from math import fsum, isclose
from pathlib import Path


def delta_summary(values):
    total = fsum(values)
    return dict(records=len(values), delta_nll_sum=total,
        delta_nll_mean=total/len(values) if values else None,
        improved=sum(v < 0 for v in values), worsened=sum(v > 0 for v in values),
        unchanged=sum(v == 0 for v in values))


def first_examples(rows):
    ordered = sorted(rows, key=lambda r: (r['reference']['time_sec'],
        r['reference']['seed'], r['reference']['packet_id']))
    return [dict(reference=row['reference'], candidate_intents=row['candidate_intents'], prior=row['prior'],
        baseline=row['baseline'], candidate=row['candidate']) for row in ordered[:3]]


def run(output):
    result = json.loads((output/'result.json').read_text())
    if result['status'] != 'COMPLETE_FIXED_BETA_THREE_HEAD_AUDIT':
        raise ValueError('The complete frozen paired audit must precede this cached readout')
    groups, all_delta = defaultdict(list), []
    reduced_overprediction, speed_groups = defaultdict(list), defaultdict(list)
    motion_rows, action_rows = {}, {head: {} for head in ('start', 'yield_entry')}
    unsupported = {head: [] for head in ('start', 'yield_entry')}
    counts = defaultdict(int)
    for path in sorted(output.glob('seed*_paired_records_server_only.jsonl.gz')):
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                head, y = row['head'], row['reference']['y']
                ref = row['reference']
                state_key = (ref['seed'], ref['time_sec'], ref['packet_id'])
                counts[head] += 1
                before, after = row['baseline'], row['candidate']
                if head == 'motion':
                    mean0 = fsum(p*m for p, m in zip(row['prior'], before['mean_mps']))
                    mean1 = fsum(p*m for p, m in zip(row['prior'], after['mean_mps']))
                    error_group = 'above_y' if mean0 > y else 'below_y' if mean0 < y else 'equal_y'
                    shift_group = 'increased' if mean1 > mean0 else 'decreased' if mean1 < mean0 else 'unchanged'
                    delta = before['logp']-after['logp']
                    groups[error_group, shift_group].append(delta)
                    all_delta.append(delta)
                    speed_groups['zero_speed' if y == 0 else 'positive_speed'].append(delta)
                    motion_rows[state_key] = (y, mean0, mean1, delta)
                    if mean0 > y and mean1 < mean0:
                        final = 'below_y' if mean1 < y else 'above_y' if mean1 > y else 'equal_y'
                        reduced_overprediction[final].append(delta)
                elif y == 1 and before['logp'] is not None and after['logp'] is None:
                    unsupported[head].append(row)
                if head in action_rows:
                    action_rows[head][state_key] = y
    for head, saved in result['heads'].items():
        if counts[head] != saved['records']:
            raise ValueError('Cached head count does not reproduce the complete paired audit')
    motion = result['heads']['motion']
    if len(all_delta) != 40000 or len(all_delta) != motion['paired']['finite_in_both']:
        raise ValueError('All40000 motion rows must contribute; no selected-case subset is allowed')
    total = delta_summary(all_delta)
    if not isclose(total['delta_nll_sum'], motion['paired']['delta_nll_sum'], rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Cached motion delta does not reconcile with the primary full-audit total')
    for head, rows in unsupported.items():
        if len(rows) != result['heads'][head]['by_label']['1']['support_transition']['lost']:
            raise ValueError('Cached newly unsupported y1 count differs from the primary audit')
    cross = [dict(baseline_weighted_mean=error, candidate_weighted_mean_change=shift,
        **delta_summary(groups[error, shift])) for error in ('below_y', 'equal_y', 'above_y')
        for shift in ('decreased', 'unchanged', 'increased')]
    intersections = {}
    for head, labels in action_rows.items():
        common = motion_rows.keys() & labels.keys()
        per_label = {}
        for label in (0, 1):
            rows = [motion_rows[key] for key in common if labels[key] == label]
            per_label[str(label)] = dict(**delta_summary([r[3] for r in rows]),
                mean_native_speed=fsum(r[0] for r in rows)/len(rows) if rows else None,
                mean_baseline_potential_response=fsum(r[1] for r in rows)/len(rows) if rows else None,
                mean_candidate_potential_response=fsum(r[2] for r in rows)/len(rows) if rows else None)
        intersections[head] = dict(exact_state_intersection=len(common), motion_records=40000,
            coverage_fraction=len(common)/40000, by_observed_action=per_label)
    summary = dict(status='COMPLETE_CACHED_POPULATION_DECOMPOSITION', motion=dict(total=total, groups=cross),
        reduced_original_overprediction_by_final_mean={key: delta_summary(reduced_overprediction[key])
            for key in ('below_y', 'equal_y', 'above_y')},
        motion_by_observed_speed={key: delta_summary(speed_groups[key]) for key in ('zero_speed', 'positive_speed')},
        exact_cross_head_intersections=intersections,
        newly_unsupported_y1={head: dict(count=len(rows), first3=first_examples(rows))
            for head, rows in unsupported.items()},
        head_counts=dict(counts), group_rule='Strict stored-float comparisons of sum prior*accepted_mean; no prior renormalization or new grouping tolerance.',
        example_order='time_sec, seed, packet_id; first3 only, total counts include all rows',
        delta_sign='candidate minus baseline NLL; negative is better',
        interpretation='Posthoc descriptive decomposition of the same training records, not a causal mechanism test or independent validation.',
        new_queries=0, new_fits=0, optimizer_calls=0, new_native_starts=0)
    (output/'cached_decomposition.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    summary = run(parser.parse_args().output)
    print(json.dumps(dict(status=summary['status'], motion=summary['motion']['total'])), flush=True)
