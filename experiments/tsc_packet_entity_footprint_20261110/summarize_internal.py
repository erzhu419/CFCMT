"""Small evidence readout of the saved paired probes; no model execution."""
import argparse
import json
from pathlib import Path
from statistics import mean, median


def key(row):
    return row['seed'], row['time_sec'], row['packet_id']


def read_probes(base):
    return {key(row): row for path in sorted(Path(base).glob('seed*_internal_probes_server_only.json'))
            for row in json.loads(path.read_text())}


def summarize(old_output, entity_output):
    old, new = read_probes(old_output), read_probes(entity_output)
    if old.keys() != new.keys() or len(new) != 253:
        raise ValueError('The same complete 253 internal references are required')
    remaining, old_new_bad, future_new_bad = [], [], []
    for ref, row in new.items():
        prior = old[ref]
        detail = row['paired']['candidate']
        originally_bad = prior['paired']['baseline']['upper']+1e-8 < row['actual_next_speed_mps']
        added_by_lane = row['previously_bad'] and not originally_bad
        if added_by_lane:
            old_new_bad.append(ref)
            old_detail = prior['paired']['candidate']
            zones = [zone for foe in old_detail['spatial_details'] for zone in foe['active_zones']
                     if abs(zone['limit_m']-old_detail['spatial_limit_m']) <= 1e-8]
            if zones and all(zone['foe_time']['start_sec'] > 1. for zone in zones):
                future_new_bad.append(ref)
        if not row['candidate_bad']:
            continue
        remaining.append(dict(seed=ref[0], time_sec=ref[1], packet_id=ref[2],
            current_lane=row['current_lane'], current_position_m=row['current_position_m'],
            actual_next_speed_mps=row['actual_next_speed_mps'],
            baseline_upper_mps=row['baseline_upper_mps'], candidate_upper_mps=detail['upper'],
            deficit_mps=row['actual_next_speed_mps']-detail['upper'],
            already_below_before_spatial=originally_bad,
            binding='spatial_conflict' if detail['spatial_limit_m'] < detail['raw_upper_mps']-1e-8
                    else 'nonspatial_motion_or_receiving',
            candidate_detail=detail))
    deficits = [row['deficit_mps'] for row in remaining]
    return dict(status='COMPLETE_SAVED_PROBE_READOUT', probes=len(new),
        lane_added_bad=len(old_new_bad),
        lane_added_bad_fixed=sum(not new[ref]['candidate_bad'] for ref in old_new_bad),
        lane_added_bad_remaining=sum(new[ref]['candidate_bad'] for ref in old_new_bad),
        lane_added_future_zone_cases=len(future_new_bad),
        lane_added_future_zone_cases_still_bad=sum(new[ref]['candidate_bad'] for ref in future_new_bad),
        remaining_count=len(remaining), mean_deficit_mps=mean(deficits) if deficits else 0.,
        median_deficit_mps=median(deficits) if deficits else 0.,
        maximum_deficit_mps=max(deficits, default=0.), remaining_cases=remaining,
        new_native_starts=0, new_fits=0,
        interpretation='Only saved paired probes are read. A zero-residual configured ceiling is not full parameter-space support. Residual geometry/time attribution requires the actual trajectories through the same local zone.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-output', type=Path, required=True)
    parser.add_argument('--entity-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.old_output, args.entity_output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'remaining_cases'}))
