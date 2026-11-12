"""One fixed-beta population readout on original references; no estimation."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
from math import isfinite
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_feature_kernel
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight, SEEDS
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_training_20261109.training_data import RESERVOIR_SEED, iter_transitions, _candidates
from .query import query_pair
from .stats import HEADS, distribution, log_probability, summarize_head


TOL = 1e-8  # Existing reference-reproduction tolerance, never a physical gap margin.
ARMS = ('baseline', 'candidate')
VECTORS = ('y', 'baseline_logp', 'candidate_logp', 'baseline_retained', 'candidate_retained',
           'baseline_max_mean', 'candidate_max_mean', 'baseline_max_physical', 'candidate_max_physical')


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [native(item) for item in value]
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save(path, value):
    Path(path).write_text(json.dumps(native(value), indent=2, allow_nan=False)+'\n')


def key(head, reference):
    return head, int(reference['time_sec']), reference['packet_id']


def verify_baseline(old, reference, intents, prior, components, logp):
    if old['reference'] != reference or old['candidate_intents'] != intents or old['prior'] != prior:
        raise ValueError('Frozen reference/label/intent order/prior changed')
    head = old['head']
    risk = [True if head == 'motion' else c['start_risk' if head == 'start' else 'yield_risk'] for c in components]
    if old['risk'] != risk:
        raise ValueError('Shared beta0 risk differs from the cached discrete baseline')
    for field, component_field in (('mean0_mps', 'mean_mps'), ('physical_ceiling_mps', 'physical_ceiling_mps')):
        if not np.allclose(old[field], [c[component_field] for c in components], rtol=0., atol=TOL):
            raise ValueError('Shared beta0 '+field+' differs from the cached discrete baseline')
    cached_logp = old['zero_log_probability']
    if (cached_logp is None and isfinite(logp)) or (cached_logp is not None and abs(logp-cached_logp) > TOL):
        raise ValueError('Shared beta0 label likelihood differs from the cached discrete baseline')


def summaries(arrays):
    return {head: summarize_head(head, **{field: arrays[head+'_'+field] for field in VECTORS}) for head in HEADS}


def inspect_seed(task):
    seed, cache, baseline, static, laws, reservoir, width, beta, output = task
    began = time.monotonic()
    directory = cache/'training'/f'seed{seed}'
    executors = json.loads((directory/'executor_seed_server_only.json').read_text())
    old = {}
    with gzip.open(baseline/f'seed{seed}_references_server_only.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['head'] in HEADS:
                k = key(row['head'], row['reference'])
                if k in old:
                    raise ValueError('Duplicate original reference')
                old[k] = row
    if len(old) != 30000:
        raise ValueError('Each seed needs all original30000 three-head rows')
    selected = defaultdict(lambda: defaultdict(list))
    for head in HEADS:
        for reference in reservoir[head]:
            selected[int(reference['time_sec'])][reference['packet_id']].append((head, reference))
    table = build_entity_geometry(RoadTopology(static), width)
    vectors = {head: {field: [] for field in VECTORS} for head in HEADS}
    counts, diagnostics = Counter(), {arm: Counter() for arm in ARMS}
    response_counts, scores, clips, response_changes = Counter(), [], [], []
    with gzip.open(output/f'seed{seed}_paired_records_server_only.jsonl.gz', 'xt', compresslevel=1) as stream:
        for frame, unused_next, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='population_progress', seed=seed, time_sec=t, paired_rows=counts['reference_rows'])), flush=True)
            if t not in selected:
                continue
            kernel = prepare_feature_kernel(static, frame, executors, laws, action['requests'],
                RESERVOIR_SEED+seed*100000+t, conflict_geometry=table)
            if kernel.signals != action['applied_states']:
                raise ValueError('The frozen current interval signals must reproduce')
            for pid, refs in selected[t].items():
                prior_map = _candidates(kernel, pid)
                intents, prior = list(prior_map), list(prior_map.values())
                components = {arm: [] for arm in ARMS}
                for intent in intents:
                    try:
                        pair = query_pair(kernel, pid, intent, beta)
                    except Exception as error:
                        save(output/f'seed{seed}_failed_query_server_only.json', dict(seed=seed, time_sec=t,
                            packet_id=pid, intent=intent, error=f'{type(error).__name__}: {error}'))
                        raise
                    for arm in ARMS:
                        components[arm].append(pair[arm])
                        for name, value in pair[arm]['diagnostics'].items():
                            if name not in ('focal_score', 'focal_clipped_residual_mps'):
                                diagnostics[arm][name] += value
                    scores.append(pair['candidate']['diagnostics']['focal_score'])
                    clips.append(pair['candidate']['diagnostics']['focal_clipped_residual_mps'])
                    for name, value in pair['comparison'].items():
                        if name.endswith('_mps'):
                            continue
                        response_counts[name] += value
                    response_changes.append(pair['comparison']['max_abs_accepted_motion_change_mps'])
                    counts['intent_queries'] += 1
                counts['focal_queries'] += 1
                for head, reference in refs:
                    logp = {arm: log_probability(head, reference['y'], prior, components[arm]) for arm in ARMS}
                    old_row = old.pop(key(head, reference))
                    try:
                        verify_baseline(old_row, reference, intents, prior, components['baseline'], logp['baseline'])
                    except Exception as error:
                        save(output/f'seed{seed}_baseline_mismatch_server_only.json', dict(reference=reference,
                            head=head, intent_order=intents, prior=prior, cached=old_row, actual=components['baseline'],
                            actual_logp=logp['baseline'] if isfinite(logp['baseline']) else None,
                            error=str(error)))
                        raise
                    row = dict(head=head, reference=reference, candidate_intents=intents, prior=prior)
                    data = vectors[head]
                    data['y'].append(reference['y'])
                    for arm in ARMS:
                        risks = [True if head == 'motion' else c['start_risk' if head == 'start' else 'yield_risk'] for c in components[arm]]
                        positive = [i for i, weight in enumerate(prior) if weight > 0]
                        data[arm+'_logp'].append(logp[arm])
                        data[arm+'_retained'].append(any(risks[i] for i in positive))
                        data[arm+'_max_mean'].append(max(components[arm][i]['mean_mps'] for i in positive))
                        data[arm+'_max_physical'].append(max(components[arm][i]['physical_ceiling_mps'] for i in positive))
                        row[arm] = dict(logp=logp[arm] if isfinite(logp[arm]) else None, risk=risks,
                            mean_mps=[c['mean_mps'] for c in components[arm]],
                            physical_ceiling_mps=[c['physical_ceiling_mps'] for c in components[arm]])
                    stream.write(json.dumps(native(row), allow_nan=False, separators=(',', ':'))+'\n')
                    counts['reference_rows'] += 1
    if old or any(len(vectors[head]['y']) != 10000 for head in HEADS):
        raise ValueError('Every original reference must be consumed exactly once')
    arrays = {head+'_'+field: np.asarray(values) for head, data in vectors.items() for field, values in data.items()}
    np.savez_compressed(output/f'seed{seed}_arrays_server_only.npz', **arrays)
    detail = dict(seed=seed, heads=summaries(arrays), counts=dict(counts),
        baseline_reproduced_reference_rows=counts['reference_rows'], query_dependency_diagnostics=diagnostics,
        paired_dependency_counts=response_counts, candidate_focal_score_distribution=distribution(scores),
        candidate_focal_clipped_residual_distribution=distribution(clips),
        query_max_absolute_accepted_response_change=distribution(response_changes),
        diagnostic_denominator='Repeated focal-intent queries and their requested dependencies, not unique vehicles.',
        elapsed_sec=time.monotonic()-began)
    save(output/f'seed{seed}_summary.json', detail)
    print(json.dumps(dict(stage='population_seed_complete', seed=seed, paired_rows=counts['reference_rows'], elapsed_sec=detail['elapsed_sec'])), flush=True)
    return detail


def run(cache, baseline, lp, preflight, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    try:
        solver = json.loads((lp/'summary.json').read_text())
        if not solver['solver_success']:
            raise ValueError('The sole frozen support solve must have returned a candidate')
        beta = np.asarray(json.loads((lp/'candidate_beta_server_only.json').read_text())['beta'], dtype=float)
        if beta.shape != (38,) or not np.isfinite(beta).all():
            raise ValueError('The existing38 coefficients are required without re-estimation')
        width, unused_proof = verify_entity_preflight(preflight)
        static, unused_width_source = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        reservoirs = json.loads((cache/'training_reservoir_server_only.json').read_text())
        if sorted(map(int, reservoirs)) != SEEDS or any(len(reservoirs[str(seed)][head]) != 10000
                for seed in SEEDS for head in (*HEADS, 'lane_change')):
            raise ValueError('The complete original160000-reference manifest must remain unchanged')
        tasks = [(seed, cache, baseline, static, laws, reservoirs[str(seed)], width, beta, output) for seed in SEEDS]
        completed = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(inspect_seed, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
        archives = [np.load(output/f'seed{seed}_arrays_server_only.npz') for seed in SEEDS]
        arrays = {name: np.concatenate([archive[name] for archive in archives]) for name in archives[0].files}
        heads = summaries(arrays)
        result = dict(status='COMPLETE_FIXED_BETA_THREE_HEAD_AUDIT', heads=heads,
            seeds=sorted(completed, key=lambda row: row['seed']),
            evaluated_reference_records=sum(h['records'] for h in heads.values()),
            baseline_reproduced_reference_rows=sum(c['baseline_reproduced_reference_rows'] for c in completed),
            original_manifest_records=160000, unevaluated_new_lc_likelihood_records=40000,
            new_unsupported_labels=sum(h['support_transition']['lost'] for h in heads.values()),
            model_admitted=False, original_training_hold_retained=True,
            full190_objective_evaluated=False, calibration_data_used=False,
            estimand='Fixed-beta shared longitudinal accepted responses and dynamically recomputed three-head working likelihood on original training references; no sequential LC population likelihood or old joint-optimizer objective.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIXED_BETA_POPULATION_TECHNICAL', error=f'{type(error).__name__}: {error}', model_admitted=False)
    result.update(new_fits=0, optimizer_calls=0, new_native_starts=0, new_nuisance_estimations=0,
        free_rollouts=0, elapsed_sec=time.monotonic()-began)
    save(output/'result.json', result)
    print(json.dumps(dict(status=result['status'], elapsed_sec=result['elapsed_sec'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'baseline', 'lp', 'preflight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache, args.baseline, args.lp, args.preflight, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'COMPLETE_FIXED_BETA_THREE_HEAD_AUDIT' else 2)
