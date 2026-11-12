"""Zero-simulation replay of the preserved first dense-measurement attempt."""
import argparse
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from packet_observer import ATOL, PARAM_GETTERS, PacketObserver


def run(base):
    base = Path(base)
    static = json.loads((base/'static_graph_server_only.json').read_text())
    observer, current = PacketObserver(static), {}

    class ReplayVehicle:
        def getIDList(self):
            return current['native_active_ids']

        def __getattr__(self, name):
            fields = dict(getLaneID='lane_id', getLanePosition='position_m', getSpeed='speed_mps',
                getAcceleration='acceleration_mps2', getLateralLanePosition='lateral_position_m',
                getLateralSpeed='lateral_speed_mps', getAllowedSpeed='allowed_speed_mps',
                isStopped='is_stopped', getTypeID='type_id')
            if name in fields:
                return lambda vid: current['packets'][vid][fields[name]]
            if name in PARAM_GETTERS.values():
                key = next(key for key, getter in PARAM_GETTERS.items() if getter == name)
                return lambda vid: current['packets'][vid]['parameters'][key]
            raise AssertionError('Unrecorded or forbidden getter: '+name)

    simulation_fields = dict(getDepartedIDList='departed_ids', getArrivedIDList='arrived_ids',
        getStartingTeleportIDList='starting_teleport_ids', getEndingTeleportIDList='ending_teleport_ids',
        getCollidingVehiclesIDList='collision_ids', getPendingVehicles='native_pending_ids')
    lane_fields = dict(getLastStepVehicleIDs='ids', getLastStepVehicleNumber='N',
        getLastStepHaltingNumber='H', getLastStepOccupancy='O', getLastStepMeanSpeed='mean_speed_mps')
    api = SimpleNamespace(vehicle=ReplayVehicle(), simulation=SimpleNamespace(
        getTime=lambda: current['time_s'], **{getter: (lambda key=key: current[key])
            for getter, key in simulation_fields.items()}), lane=SimpleNamespace(
        **{getter: (lambda lane, key=key: current['native_lanes'][lane][key])
           for getter, key in lane_fields.items()}))
    stats = dict(frames=0, lane_evaluations=0, before=dict(sensor_failed_frames=0, max_O_error=0., unresolved_count=0),
                 after=dict(sensor_failed_frames=0, max_O_error=0., unresolved_count=0))
    case, start_time, end_time = None, None, None
    raw_path = base/'measurement/first_pp/frames_server_only.jsonl.gz'
    with gzip.open(raw_path, 'rt') as stream:
        for line in stream:
            current = json.loads(line)
            if stats['frames'] == 0:
                # Only this initialization uses the already causal prefix trail.
                observer.previous = current['packets']
                observer.previous_time = current['time_s']
                start_time = current['time_s']
                new_lanes = observer._reconstruct(observer.previous)
                new_unresolved = current['unresolved']
            else:
                old = observer.previous.get('jinan_3x4_real_963')
                # Getter adapters expose only current observations, never saved
                # future body_trail or reconstructed_lanes.
                result = observer.capture(api, current['time_s'], current['role'])
                new_lanes, new_unresolved = result['reconstructed_lanes'], result['unresolved']
                if current['time_s'] == 644:
                    packet, lane = result['packets']['jinan_3x4_real_963'], ':intersection_4_1_21_0'
                    case = dict(vehicle_id=packet['packet_id'],
                        previous_front=dict(lane=old['lane_id'], position_m=old['position_m']),
                        current_front=dict(lane=packet['lane_id'], position_m=packet['position_m']),
                        length_m=packet['parameters']['length_m'],
                        tail_m=packet['parameters']['length_m']-packet['position_m'],
                        affected_lane_length_m=static['lanes'][lane]['length_m'],
                        native_O=current['native_lanes'][lane]['O'],
                        before_O=current['reconstructed_lanes'][lane]['O'], after_O=new_lanes[lane]['O'],
                        native_body_accounting_trail=packet['body_trail'],
                        front_transition=next(event for event in result['events']
                            if event['kind'] == 'lane_transition' and event['vehicle_id'] == packet['packet_id']))
            for name, lanes, unknown in [('before', current['reconstructed_lanes'], current['unresolved']),
                                         ('after', new_lanes, new_unresolved)]:
                failed = False
                for lane, native in current['native_lanes'].items():
                    error = abs(native['O']-lanes[lane]['O'])
                    stats[name]['max_O_error'] = max(stats[name]['max_O_error'], error)
                    failed |= (any(native[key] != lanes[lane][key] for key in ('N', 'H', 'ids')) or
                               error > ATOL or abs(native['mean_speed_mps']-lanes[lane]['mean_speed_mps']) > ATOL)
                stats[name]['sensor_failed_frames'] += int(failed)
                stats[name]['unresolved_count'] += len(unknown)
            stats['frames'] += 1
            stats['lane_evaluations'] += len(current['native_lanes'])
            end_time = current['time_s']
    passed = stats['after']['sensor_failed_frames'] == stats['after']['unresolved_count'] == 0
    return dict(status='PASS_RECORDED_SENSOR_REPLAY' if passed else 'HOLD_RECORDED_SENSOR_REPLAY',
        new_native_starts=0, new_fits=0, original_HOLD_result_unchanged=True,
        full_450_sec_validated=False, start_time_sec=start_time, end_time_sec=end_time,
        horizon_observed_sec=end_time-start_time, requested_horizon_sec=450, stats=stats,
        causal_case=case, path_semantics=dict(
            observed_lane_path='Past front movement inferred from measured endpoints, unique topology and SUMO move-before-lanechange order.',
            body_trail='Current native partial-occupancy accounting path after lanechange remapping; it does not assert that the front traversed these lanes.'),
        original_observer_snapshot='packet_observer_native_attempt_v1.py', raw_frames=str(raw_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base)
    args.output.write_text(json.dumps(result, allow_nan=False, indent=2)+'\n')
    print(json.dumps(result, allow_nan=False))
