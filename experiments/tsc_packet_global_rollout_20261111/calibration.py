"""Native PP collection and free-rollout calibration of the frozen global law."""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
import json
from pathlib import Path

from experiments.tsc_network_dense_measurement_v2_20261109.collect_dense import due_schedule
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import read_static
from experiments.tsc_packet_entity_footprint_20261110.diagnose import verify_entity_preflight
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_protocol_20261109.calibration_gates import evaluate_calibration
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation
from experiments.tsc_target_packet_training_20261109 import evaluate as existing
from experiments.tsc_target_packet_training_20261109.collect import collect_one

from .kernel import PacketKernel


def read(path):
    return json.loads(path.read_text())


def save(path, result):
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def load_model(base, point='new_lc'):
    folder = {'new_lc':'target_packet_lc_stratified_refit', 'old_lc':'target_packet_lc_refit'}[point]
    fitted = read(base/folder/'fit/coefficients_server_only.json')
    return dict(coefficients=fitted['coefficients'],
                target_laws=read(base/'target_packet_training/target_laws_server_only.json'),
                training_seeds=[76116901, 76116902, 76116903, 76116904])


def kernel_factory(base, static):
    width, _ = verify_entity_preflight(base/'target_packet_entity_footprint/entity_input_preflight.json')
    widths_static = static if all(lane.get('width_m') is not None for lane in static['lanes'].values()) else load_widths(static)[0]
    geometry = build_entity_geometry(RoadTopology(widths_static), width)
    return partial(PacketKernel, conflict_geometry=geometry)


def collect(base, output, *, seeds=existing.CALIBRATION_SEEDS):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context

    directory = output/'collection'
    directory.mkdir(exist_ok=False)
    cfg = base/'repaired_action_cost_pilot/inputs/jinan/jinan_3x4_real.sumocfg'
    static = read_static(cfg)
    static['conflicts_by_tls'] = read_static_context(cfg)['conflicts_by_tls']
    due, _ = due_schedule(cfg)
    api = load_libsumo()
    existing.require(libsumo_version(api) == '1.22.0', 'Frozen SUMO runtime mismatch')
    result = dict(status='COLLECTING', new_native_starts=0, new_fits=0, seeds=[])
    for seed in seeds:
        seed_directory = directory/f'seed{seed}'
        seed_directory.mkdir()
        case = dict(seed=seed, role='calibration', horizon_sec=1800, sumocfg=str(cfg))
        receipt = dict(seed=seed, status='STARTED')
        result['new_native_starts'] += 1
        result['seeds'].append(receipt)
        save(directory/'result.json', result)
        try:
            receipt.update(collect_one(api, v3, case, seed_directory, static, due))
        except Exception as exc:
            receipt.update(status='HOLD_COLLECTION', error=f'{type(exc).__name__}: {exc}')
        finally:
            try:
                api.close()
            except Exception:
                pass
        save(seed_directory/'result.json', receipt)
        result['status'] = ('HOLD_COLLECTION' if receipt['status'] != 'COMPLETE' else
                            'COMPLETE' if len(result['seeds']) == len(seeds) else
                            'COLLECTING')
        save(directory/'result.json', result)
        print(json.dumps(dict(stage='collection', seed=seed, status=receipt['status'],
                              completed_steps=receipt.get('completed_steps'), error=receipt.get('error'))), flush=True)
        if receipt['status'] != 'COMPLETE':
            break
    return result


def panel(base, output, workers, *, resume=False, seeds=existing.CALIBRATION_SEEDS,
          protocol=None, factory=None, one_step=True):
    collection = output/'collection'
    receipt = read(collection/'result.json')
    existing.require(receipt['status'] == 'COMPLETE' and
                     tuple(row['seed'] for row in receipt['seeds']) == tuple(seeds),
                     'Both frozen native trajectories are required')
    directory = output/'calibration'
    directory.mkdir(exist_ok=resume)
    static = read(output/'static_graph_server_only.json')
    training_static = read(base/'target_packet_training/static_graph_server_only.json')
    for field in ('lanes', 'successors', 'controlled_lane_ids', 'pp_lane_by_link'):
        existing.require(static[field] == training_static[field], f'Calibration static field {field} changed')
    static, _ = load_widths(static)
    model = load_model(base)
    protocol = read(base/'target_packet_training/protocol.json') if protocol is None else protocol
    factory = kernel_factory(base, static) if factory is None else factory
    planned = len(seeds)*len(existing.ANCHORS)*len(existing.SCENARIOS)
    tasks, draws, truth, native_inputs = deque(), [], {}, {}
    for seed in seeds:
        seed_dir = collection/f'seed{seed}'
        frames = {frame['time_s']:frame for frame in existing.json_lines(seed_dir/'frames_server_only.jsonl.gz')}
        executors = read(seed_dir/'executor_seed_server_only.json')
        existing.require(set(frames) == set(range(1801)),
                         'Calibration clocks do not cover the full frozen horizon')
        if one_step:
            execution = {row['time_sec']:row for row in existing.json_lines(seed_dir/'execution_server_only.jsonl.gz')}
            existing.require(set(execution) == set(range(1800)),
                             'Calibration execution clocks do not cover the full frozen horizon')
            native_inputs[seed] = dict(frames=frames, execution=execution, executor_seeds=executors)
        for anchor in existing.ANCHORS:
            initial = initialize_observation(frames[anchor], anchor)
            truth[seed, anchor] = existing._truth_window(static, frames, seed, anchor)
            for scenario in existing.SCENARIOS:
                path = directory/f'seed{seed}_t{anchor}_scenario{scenario}_server_only.json'
                if resume and path.is_file():
                    row = read(path)
                    existing.require(row['status'] == 'COMPLETE' and row['structural_passed'] and
                                     row['checked_steps'] == existing.HORIZON and
                                     (row['seed'], row['anchor_sec'], row['scenario_seed']) == (seed, anchor, scenario),
                                     f'Cannot resume from an incomplete draw {path.name}')
                    draws.append(row)
                else:
                    tasks.append((static, initial, executors, model, seed, anchor, scenario))
        del frames
    stopped = False
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while pending or (tasks and not stopped):
            while tasks and not stopped and len(pending) < workers:
                task = tasks.popleft()
                pending[pool.submit(existing._run_draw, task, factory)] = task[4:]
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                seed, anchor, scenario = pending.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = dict(seed=seed, anchor_sec=anchor, scenario_seed=scenario,
                               status='FAILED', structural_passed=False, checked_steps=0,
                               error=f'{type(exc).__name__}: {exc}')
                save(directory/f'seed{seed}_t{anchor}_scenario{scenario}_server_only.json', row)
                draws.append(row)
                stopped |= row['status'] != 'COMPLETE'
                print(json.dumps(dict(stage='free450', seed=seed, anchor=anchor, scenario=scenario,
                                      status=row['status'], checked_steps=row['checked_steps'], error=row.get('error'))),
                      flush=True)
    structural = len(draws) == planned and all(row['structural_passed'] for row in draws)
    numerical = conditional = None
    if structural:
        windows = [dict(seed=seed, anchor_sec=anchor, actual=actual,
                        predicted_scenarios=[row for row in draws if (row['seed'], row['anchor_sec']) == (seed, anchor)])
                   for (seed, anchor), actual in truth.items()]
        numerical = evaluate_calibration(windows, protocol)
        if one_step:
            conditional = existing._one_step_panel(static, native_inputs, model, factory)
    conditional_pass = (conditional is not None and conditional['status'] == 'PASS_CONDITIONAL_DIAGNOSTIC'
                        if one_step else None)
    numerical_pass = numerical is not None and numerical['status'] == 'PASS_NUMERICAL_CALIBRATION'
    result = dict(status='PASS_TARGET_PACKET_TECHNICAL_CALIBRATION'
                  if structural and numerical_pass and (conditional_pass or not one_step)
                  else 'HOLD_TARGET_PACKET_CALIBRATION',
                  checks=dict(all_scenarios_structurally_complete=structural,
                              numerical_calibration_passed=numerical_pass,
                              conditional_records_complete=conditional_pass),
                  numerical_calibration=numerical, one_step=conditional,
                  planned_draws=planned, completed_draws=sum(row['status'] == 'COMPLETE' for row in draws),
                  checked_free_model_seconds=sum(row['checked_steps'] for row in draws),
                  draws=[{key: row.get(key) for key in ('seed', 'anchor_sec', 'scenario_seed', 'status',
                                                        'checked_steps', 'I450', 'C450', 'error')} for row in draws],
                  new_fits=0)
    save(directory/'result.json', result)
    print(json.dumps(dict(stage='calibration', status=result['status'], checks=result['checks'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 17), default=4)
    parser.add_argument('--resume', action='store_true', help='Reuse saved complete model draws')
    parser.add_argument('stage', choices=('collect', 'calibrate'))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    if args.stage == 'collect':
        return 0 if collect(args.base, args.output)['status'] == 'COMPLETE' else 2
    result = panel(args.base, args.output, args.workers, resume=args.resume)
    return 0 if result['checks']['all_scenarios_structurally_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
