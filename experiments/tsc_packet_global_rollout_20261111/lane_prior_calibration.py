"""Conditional fresh 450s calibration of the training-only lane intent prior."""

import argparse
from functools import partial
import json
from pathlib import Path

from . import calibration
from .lane_prior import LanePriorKernel, estimate_lane_prior


SEEDS = (76116931, 76116932)
WORKERS = 12
STATIC_FIELDS = ('lanes', 'successors', 'controlled_lane_ids', 'pp_lane_by_link',
                 'incoming_lanes_by_tls', 'conflicts_by_tls')
existing = calibration.existing
read, save = calibration.read, calibration.save


def log(stage, **record):
    print(json.dumps(dict(stage=stage, **record)), flush=True)


def frozen_protocol(base):
    protocol = read(base/'target_packet_training/protocol.json')
    spec = protocol['calibration']
    if (tuple(spec['anchors_sec']) != existing.ANCHORS
            or tuple(spec['scenario_random_seeds']) != existing.SCENARIOS
            or spec['horizon_sec'] != existing.HORIZON
            or spec['model_scenarios'] != len(existing.SCENARIOS)
            or protocol['cost']['lane_count'] != 144):
        raise ValueError('The existing anchors, model scenarios and 450s gates must be used')
    prior_seeds = set(protocol['cohorts']['training_seeds']) | set(protocol['cohorts']['calibration_seeds'])
    if prior_seeds.intersection(SEEDS):
        raise ValueError('Fresh native seeds must be distinct from original training and calibration seeds')
    protocol['cohorts']['calibration_seeds'] = list(SEEDS)
    return protocol


def preflight(base, archive, short_result, output):
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='HOLD_PREFLIGHT', new_native_starts=0, numerical_optimizations=0,
                   nuisance_law_estimations=0, error=None)
    try:
        short = read(short_result)
        if (short['status'] != 'COMPLETE_SHORT_INTENT_DIAGNOSTIC'
                or not short['gate']['candidates']['lane_conditioned']['pass_both_seeds']):
            raise ValueError('Observable lane conditioning must improve both short-window errors on each seed')
        protocol = frozen_protocol(base)
        archived = read(archive/'static_graph_server_only.json')
        training = read(base/'target_packet_training/static_graph_server_only.json')
        if any(archived[field] != training[field] for field in STATIC_FIELDS):
            raise ValueError('Archived network and training network differ')
        static, _ = calibration.load_widths(archived)
        prior = estimate_lane_prior(base, static)
        if tuple(prior['training_seeds']) != tuple(protocol['cohorts']['training_seeds']):
            raise ValueError('The lane prior must use exactly the frozen training seeds')
        save(output/'lane_prior_server_only.json', prior)
        save(output/'frozen_protocol.json', protocol)
        save(output/'reference_static_server_only.json', archived)
        receipt.update(status='PREFLIGHT_COMPLETE', nuisance_law_estimations=prior['nuisance_law_estimations'],
                       lane_probability_cells=len(prior['lane_probabilities']), training_seeds=prior['training_seeds'],
                       training_counts=prior['training_counts'], short_result=str(short_result),
                       short_gate=short['gate']['candidates']['lane_conditioned'],
                       fresh_native_seeds=list(SEEDS))
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
    save(output/'preflight_result.json', receipt)
    log('lane_prior_preflight', status=receipt['status'], error=receipt['error'])
    return receipt


def calibrate(base, output):
    native = read(output/'collection/result.json')
    static = read(output/'static_graph_server_only.json')
    reference = read(output/'reference_static_server_only.json')
    if any(static[field] != reference[field] for field in STATIC_FIELDS):
        raise ValueError('Fresh native measurement changed the frozen network or signal roster')
    protocol = read(output/'frozen_protocol.json')
    prior = read(output/'lane_prior_server_only.json')
    static, _ = calibration.load_widths(static)
    geometry = calibration.kernel_factory(base, static).keywords['conflict_geometry']
    factory = partial(LanePriorKernel, conflict_geometry=geometry, lane_probabilities=prior['lane_probabilities'])
    result = calibration.panel(base, output, WORKERS, seeds=SEEDS, protocol=protocol,
                               factory=factory, one_step=False)
    result.update(new_native_starts=native['new_native_starts'],
                  native_seconds=sum(row['completed_steps'] for row in native['seeds']),
                  nuisance_law_estimations=prior['nuisance_law_estimations'], action_stage_launched=False)
    save(output/'calibration/result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--short-result', type=Path)
    parser.add_argument('stage', choices=('preflight', 'collect', 'calibrate'))
    args = parser.parse_args()
    if args.stage == 'preflight':
        if args.archive is None or args.short_result is None:
            parser.error('preflight requires --archive and --short-result')
        return 0 if preflight(args.base, args.archive, args.short_result, args.output)['status'] == 'PREFLIGHT_COMPLETE' else 2
    if args.stage == 'collect':
        if read(args.output/'preflight_result.json')['status'] != 'PREFLIGHT_COMPLETE':
            raise ValueError('Complete lane-prior preflight before fresh native collection')
        return 0 if calibration.collect(args.base, args.output, seeds=SEEDS)['status'] == 'COMPLETE' else 2
    return 0 if calibrate(args.base, args.output)['checks']['all_scenarios_structurally_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
