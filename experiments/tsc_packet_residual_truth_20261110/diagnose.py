"""Read three frozen residuals from existing native frames; execute no model."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import traceback

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from .receiving_trace import explain_receiving
from .zone_truth import observe_zone_truth


def observed_path(before, after, packet_id):
    for event in after['events']:
        if event['kind'] == 'lane_transition' and event.get('vehicle_id') == packet_id:
            return event['observed_lane_path']
    return [before['packets'][packet_id]['lane_id']]


def read_window(directory, start, end):
    frames = {}
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as stream:
        for line in stream:
            frame = json.loads(line)
            t = int(frame['time_s'])
            if start <= t <= end:
                frames[t] = frame
            if t >= end:
                break
    if set(frames) != set(range(start, end+1)):
        raise ValueError('The fixed native observation window must be consecutive')
    return frames


def inspect_case(task):
    case, protocol, cache, entity, static = task
    seed, t, pid = case['seed'], case['time_sec'], case['packet_id']
    print(json.dumps(dict(stage='residual_read_begin', seed=seed, time_sec=t)), flush=True)
    probes = json.loads((entity/'output'/f'seed{seed}_internal_probes_server_only.json').read_text())
    probe, = [row for row in probes if row['time_sec'] == t and row['packet_id'] == pid]
    if not probe['candidate_bad']:
        raise ValueError('The query must be an unchanged remaining entity-footprint residual')
    topology = RoadTopology(static)
    frames = read_window(cache/'training'/f'seed{seed}', t-4, t+8)
    before, after = frames[t], frames[t+1]
    packet = before['packets'][pid]
    if (packet['lane_id'] != probe['current_lane'] or
            abs(packet['position_m']-probe['current_position_m']) > 1e-8 or
            abs(after['packets'][pid]['speed_mps']-probe['actual_next_speed_mps']) > 1e-8):
        raise ValueError('Native state and measured speed must match the frozen probe')
    connection = topology.by_key[probe['paired']['connection_key']]
    path = connection['path'][connection['path'].index(packet['lane_id']):]
    receiving = explain_receiving(topology, packet, before['packets'], path, after['packets'],
        connection=connection, observed_ego_path=observed_path(before, after, pid),
        observed_neighbour_paths={event['vehicle_id']: event['observed_lane_path']
            for event in after['events'] if event['kind'] == 'lane_transition'})
    if abs(receiving['current_bound_m']-probe['paired']['candidate']['path_bound_m']) > 1e-8:
        raise ValueError('The current receiving explanation must reproduce the frozen path bound')
    result = dict(seed=seed, time_sec=t, packet_id=pid,
        observation_window_sec=[t-4, t+8], frozen_probe=probe,
        connection=connection, model_path=list(path), receiving_trace=receiving,
        current_speed_mps=packet['speed_mps'], next_speed_mps=after['packets'][pid]['speed_mps'],
        observed_lane_path=observed_path(before, after, pid), path_bound_reproduced=True)
    spatial = protocol['spatial_case']
    if (seed, t, pid) == (spatial['seed'], spatial['time_sec'], spatial['packet_id']):
        result['zone_truth'] = {}
        for role, selected_id, connection_key, zone in [
            ('ego', pid, spatial['connection_key'], spatial['ego_zone_m']),
            ('foe', spatial['foe_packet_id'], spatial['foe_connection_key'], spatial['foe_zone_m'])]:
            snapshots = [dict(time_sec=clock, packet=frame['packets'].get(selected_id))
                         for clock, frame in sorted(frames.items())]
            result['zone_truth'][role] = observe_zone_truth(
                topology, topology.by_key[connection_key], zone, snapshots)
        ego_truth, foe_truth = result['zone_truth']['ego'], result['zone_truth']['foe']
        entry = ego_truth['front_entry_interval_sec']
        clear = foe_truth['tail_clear_interval_sec']
        order = ('unresolved_censored' if entry is None or clear is None else
                 'foe_clear_before_ego_entry' if clear[1] <= entry[0] else
                 'ego_entry_before_foe_clear' if entry[1] <= clear[0] else
                 'unresolved_overlapping_observation_intervals')
        result['zone_order'] = dict(ego_front_entry_interval_sec=entry,
            foe_tail_clear_interval_sec=clear, observed_order=order,
            both_bodies_observed_on_model_zones_at_sec=[a['time_sec'] for a, b in
                zip(ego_truth['frames'], foe_truth['frames'])
                if a['body_on_zone'] is True and b['body_on_zone'] is True])
    print(json.dumps(dict(stage='residual_read_complete', seed=seed, time_sec=t)), flush=True)
    return result


def run(cache, entity, protocol_path, output):
    protocol = json.loads(protocol_path.read_text())
    cases = [protocol['spatial_case'], *protocol['nonspatial_cases']]
    if len(cases) != 3 or protocol['new_native_starts'] != 0 or protocol['new_fits'] != 0:
        raise ValueError('Only the three frozen zero-fit, zero-native cases are in scope')
    static = json.loads((cache/'static_graph_server_only.json').read_text())
    output.mkdir(exist_ok=False)
    try:
        tasks = [(case, protocol, cache, entity, static) for case in cases]
        with ProcessPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(inspect_case, tasks))
        result = dict(status='COMPLETE_RESIDUAL_OBSERVATION', cases=results,
            new_native_starts=0, new_fits=0, new_nuisance_estimations=0, free_rollouts=0,
            calibration_data_used=False, source_data_used=False,
            timing_convention='Observed events are intervals (previous_frame,current_frame]; no subsecond interpolation.',
            use='Posthoc diagnosis only; future packets are never predictor inputs.')
    except Exception as error:
        (output/'error_server_only.txt').write_text(traceback.format_exc())
        result = dict(status='HOLD_RESIDUAL_READ_ERROR', error=f'{type(error).__name__}: {error}',
            new_native_starts=0, new_fits=0)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--entity-base', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.cache_base, args.entity_base, args.protocol, args.output)
    raise SystemExit(0 if result['status'] == 'COMPLETE_RESIDUAL_OBSERVATION' else 2)
