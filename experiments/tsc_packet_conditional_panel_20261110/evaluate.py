"""One frozen-zero real panel, after a complete synthetic/static budget gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
from math import isfinite, log
import multiprocessing
from pathlib import Path
import resource
import time
import traceback

from experiments.tsc_packet_conditional_domain_20261110.register import RESERVOIR_SEED
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from .initialize import prepare_zero_kernel
from .preflight import PHYSICAL_CEILING, LEAF_CEILING, reference_key, save
from .stream import stream_zero_law


CONTROLS = {'jinan_3x4_real_491', 'jinan_3x4_real_728', 'jinan_3x4_real_1477'}


def inspect_case(task):
    cache, preflight, width_file, registered, label, output = task
    ref = registered['reference']
    seed, t, focal = ref['seed'], int(ref['time_sec']), ref['packet_id']
    directory = output/f'seed{seed}_t{t}'
    directory.mkdir(exist_ok=False)
    began = time.monotonic()
    save(directory/'started.json', dict(reference=ref, started_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate='New conditional model allzero, exactlyone candidate',
        physical_limit=registered['symbolic_budget']['physical_resolver_calls_upper_bound'],
        leaf_limit=registered['symbolic_budget']['lateral_leaf_branches_upper_bound']))
    law = None
    try:
        current = json.loads((preflight/registered['current_cache']).read_text())
        frame, action = current['frame'], current['action']
        static, unused = load_widths(json.loads((cache/'static_graph_server_only.json').read_text()))
        laws = json.loads((cache/'target_laws_server_only.json').read_text())
        executors = json.loads((cache/'training'/f'seed{seed}'/'executor_seed_server_only.json').read_text())
        width, unused = verify_entity_preflight(width_file)
        geometry = build_entity_geometry(RoadTopology(static), width)
        kernel = prepare_zero_kernel(static, frame, executors, laws, action, RESERVOIR_SEED+seed*100000+t, geometry)
        selected = select_stochastic(kernel, focal, max_stochastic=4)
        saved_selection = json.loads((preflight/f'seed{seed}_t{t}_selection_server_only.json').read_text())
        if json.loads(json.dumps(selected)) != saved_selection['selection']:
            raise ValueError('Response initialization changed the frozen geometric selection')
        if any(kernel.state['packets'][key]['_movement_key'] != binding['movement_key']
               for key, binding in saved_selection['internal_bindings'].items()):
            raise ValueError('Response initialization changed a current internal commitment')
        last_report = [time.monotonic()]
        def progress(counts):
            if counts.get('complete') or time.monotonic()-last_report[0] >= 25.:
                save(directory/'progress.json', counts)
                print(json.dumps(dict(stage='fixed_zero_progress', reference=ref, **counts)), flush=True)
                last_report[0] = time.monotonic()
        budget = registered['symbolic_budget']
        law = stream_zero_law(kernel, focal, observed_speed=label['native_speed_mps'],
            physical_limit=budget['physical_resolver_calls_upper_bound'],
            leaf_limit=budget['lateral_leaf_branches_upper_bound'], progress=progress)
        save(directory/'focal_law_server_only.json', law)
        y = int(label['lane_change'])
        probabilities = law['conditional_lc_probabilities']
        density = law['observed_speed_likelihood']
        joint = law['conditional_class_speed_likelihoods']
        normalized = (abs(law['total_probability']-1.) <= 1e-10 and
            abs(law['survivor_probability']+law['end_probability']-1.) <= 1e-10 and
            probabilities is not None and abs(sum(probabilities)-1.) <= 1e-10 and
            law['survivor_observation_total'] is not None and abs(law['survivor_observation_total']-1.) <= 1e-10 and
            all(abs(sum(values.values())-1.) <= 1e-10 for values in law['event_marginals'].values()) and
            joint is not None and density['defined'] and abs(sum(joint)-density['likelihood']) <= 1e-10)
        supported = probabilities is not None and probabilities[y] > 0. and joint is not None and joint[y] > 0.
        status = ('PASS_FIXED_ZERO_CASE' if normalized and supported else
                  'DEFINED_ZERO_OBSERVED_SUPPORT' if normalized else 'HOLD_NORMALIZATION_FAILURE')
        row = dict(status=status, reference=ref, historical_control=focal in CONTROLS,
            frame_actor_count=law['frame_actor_count'], stochastic_participants=law['stochastic_participants'],
            stochastic_actor_count=law['stochastic_actor_count'], physical_calls=law['physical_calls'],
            leaf_visits=law['leaf_visits'], focal_atoms=len(law['atoms']),
            total_probability=law['total_probability'], survivor_probability=law['survivor_probability'],
            end_probability=law['end_probability'], conditional_lc_probabilities=probabilities,
            observed_lc_label=y, observed_lc_probability=probabilities[y] if probabilities else None,
            native_speed_mps=label['native_speed_mps'], observed_speed_likelihood=density,
            conditional_class_speed_likelihoods=joint, observed_class_speed_likelihood=joint[y] if joint else None,
            observed_class_speed_nll=-log(joint[y]) if joint and joint[y] > 0 else None,
            event_marginals=law['event_marginals'], eligibility=law['eligibility'][focal],
            observation_total=law['survivor_observation_total'],
            normalized=normalized, observed_support=supported,
            worker_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            elapsed_sec=time.monotonic()-began)
    except Exception as error:
        (directory/'error_server_only.txt').write_text(traceback.format_exc())
        row = dict(status='HOLD_FIXED_ZERO_CASE_UNDEFINED', reference=ref,
            error=f'{type(error).__name__}: {error}',
            physical_calls=law['physical_calls'] if law is not None else getattr(error, 'physical_calls', 0),
            leaf_visits=law['leaf_visits'] if law is not None else getattr(error, 'leaf_visits', 0),
            branch_context=getattr(error, 'branch_context', None),
            lateral_context=getattr(error, 'lateral_action_context', None),
            normalized=False, observed_support=False, elapsed_sec=time.monotonic()-began,
            worker_process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.)
    save(directory/'summary.json', row)
    print(json.dumps(dict(stage='fixed_zero_case_complete', reference=ref, status=row['status'],
        physical_calls=row['physical_calls'], leaf_visits=row['leaf_visits'], elapsed_sec=row['elapsed_sec'])), flush=True)
    return row


def run(cache, preflight, labels_file, width_file, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    completed = []
    try:
        frozen = json.loads((preflight/'summary.json').read_text())
        if (frozen['status'] != 'PASS_FROZEN_CAP4_FULL9_BUDGET' or not frozen['complete9_domain_registered'] or
                frozen['full_frame_forward_calls_upper_bound'] > PHYSICAL_CEILING or frozen['lateral_leaf_upper_bound'] > LEAF_CEILING):
            raise ValueError('The complete frozen9-case budget must pass before any response')
        refs = {reference_key(row['reference']) for row in frozen['cases']}
        source_labels = json.loads(labels_file.read_text())['cases']
        labels = {reference_key(row['reference']): dict(lane_change=int(row['reference']['y']),
            native_speed_mps=row['native_motion_label_mps']) for row in source_labels if reference_key(row['reference']) in refs}
        if len(refs) != 9 or set(labels) != refs or not all(isfinite(row['native_speed_mps']) for row in labels.values()):
            raise ValueError('Every frozen native9-case label must match the pre-registered state')
        save(output/'panel_started.json', dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
            fixed_candidate_count=1, requested_states=9, full_forward_upper=frozen['full_frame_forward_calls_upper_bound'],
            leaf_upper=frozen['lateral_leaf_upper_bound'], new_fits=0, new_native_starts=0))
        tasks = iter((cache, preflight, width_file, row, labels[reference_key(row['reference'])], output) for row in frozen['cases'])
        stopped = False
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {}
            for unused in range(workers):
                task = next(tasks, None)
                if task is not None:
                    pending[pool.submit(inspect_case, task)] = task[3]['reference']
            while pending:
                done, unused = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = future.result()
                    completed.append(row)
                    del pending[future]
                    if row['status'].startswith('HOLD_'):
                        stopped = True
                if not stopped:
                    for unused in range(workers-len(pending)):
                        task = next(tasks, None)
                        if task is None:
                            break
                        pending[pool.submit(inspect_case, task)] = task[3]['reference']
        actual_physical = sum(row['physical_calls'] for row in completed)
        actual_leaves = sum(row['leaf_visits'] for row in completed)
        if actual_physical > frozen['full_frame_forward_calls_upper_bound'] or actual_leaves > frozen['lateral_leaf_upper_bound']:
            raise ArithmeticError('The panel exceeded its frozen symbolic budget')
        passed = len(completed) == 9 and all(row['status'] == 'PASS_FIXED_ZERO_CASE' for row in completed)
        controls = {row['reference']['packet_id']: row['observed_support'] for row in completed if row['reference']['packet_id'] in CONTROLS}
        result = dict(status='PASS_FIXED_ZERO_CONDITIONAL_SUPPORT_PANEL' if passed else 'HOLD_FIXED_ZERO_CONDITIONAL_PANEL',
            cases=sorted(completed, key=lambda row: reference_key(row['reference'])),
            registered_references=[row['reference'] for row in frozen['cases']],
            not_completed_references=[row['reference'] for row in frozen['cases'] if reference_key(row['reference'])
                not in {reference_key(done['reference']) for done in completed}],
            completed_case_attempts=len(completed), requested_states=9,
            positive_observed_lc_cases=sum(row['observed_support'] for row in completed),
            all_completed_cases_normalized=all(row['normalized'] for row in completed),
            historical_controls=controls, historical_controls_all_supported=len(controls) == 3 and all(controls.values()),
            actual_full_forward_calls=actual_physical, actual_lateral_leaf_visits=actual_leaves,
            frozen_full_forward_upper_bound=frozen['full_frame_forward_calls_upper_bound'],
            frozen_lateral_leaf_upper_bound=frozen['lateral_leaf_upper_bound'], stopped_after_domain_or_technical_failure=stopped,
            fixed_candidate_count=1, model_admitted=False, calibration_data_used=False,
            scope='Selected training9-case support/normalization and computational check of one new fixed-zero conditional law. No independent prediction, full-reservoir, source-transfer or control claim.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_FIXED_ZERO_CONDITIONAL_PANEL_TECHNICAL', error=f'{type(error).__name__}: {error}',
            cases=completed, model_admitted=False)
    result.update(new_native_starts=0, new_fits=0, new_nuisance_estimations=0, optimizer_calls=0,
        free_rollouts=0, elapsed_sec=time.monotonic()-began,
        memory_metric='Linux worker-process ru_maxrss; may include earlier states handled by the same worker.')
    save(output/'summary.json', result)
    print(json.dumps({key: result[key] for key in ('status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'preflight', 'labels-file', 'width-file', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    result = run(args.cache, args.preflight, args.labels_file, args.width_file, args.output, args.workers)
    raise SystemExit(0 if result['status'] == 'PASS_FIXED_ZERO_CONDITIONAL_SUPPORT_PANEL' else 2)
