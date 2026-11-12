"""Compact descriptive readout from completed server-only feature arrays."""
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURES


def run(base):
    names = [name for name, divisor in FEATURES]
    divisors = np.asarray([divisor for name, divisor in FEATURES])
    all_deficits, groups, seeds, examples = [], Counter(), [], []
    for seed in (76116901, 76116902, 76116903, 76116904):
        with np.load(base/'output_v2'/f'seed{seed}_full_features_server_only.npz') as data:
            y, prior = data['motion_y'], data['motion_prior']
            ceilings = np.where(prior > 0, data['motion_physical_ceiling_mps'], -np.inf)
            component = np.argmax(ceilings, axis=1)
            maximum = ceilings[np.arange(len(y)), component]
            deficit = y-maximum
            selected = np.flatnonzero(deficit > 1e-8)
            raw = data['motion_X'][np.arange(len(y)), component]*divisors
            allowed = raw[:, names.index('allowed_speed_mps')]
            velocity = raw[:, names.index('speed_mps')]
            accel = raw[:, names.index('accel_mps2')]
            local_groups = Counter()
            for index in selected:
                above_allowed = y[index] > allowed[index]+1e-8
                above_accel = y[index] > velocity[index]+accel[index]+1e-8
                label = ('above_both_current_basic_bounds' if above_allowed and above_accel else
                         'above_current_allowed_only' if above_allowed else
                         'above_current_v_plus_accel_only' if above_accel else
                         'within_both_basic_bounds_other_proposal_limit')
                local_groups[label] += 1
            groups.update(local_groups)
            values = deficit[selected].tolist()
            all_deficits.extend(values)
            seeds.append(dict(seed=seed, count=len(values), groups=dict(local_groups),
                median_deficit_mps=float(np.median(values)) if values else 0.,
                max_deficit_mps=max(values, default=0.)))
            largest = sorted(selected, key=lambda index: deficit[index], reverse=True)[:3]
            refs = {}
            index = 0
            with gzip.open(base/'output_v2'/f'seed{seed}_references_server_only.jsonl.gz', 'rt') as stream:
                for line in stream:
                    row = json.loads(line)
                    if row['head'] == 'motion':
                        if index in largest:
                            if abs(row['reference']['y']-y[index]) > 1e-8:
                                raise ValueError('The saved motion reference must match its feature row')
                            refs[index] = row['reference']
                        index += 1
            for index in largest:
                features = {name: float(raw[index, names.index(name)]) for name in (
                    'speed_mps', 'allowed_speed_mps', 'accel_mps2', 'decel_mps2',
                    'stopline_distance_m', 'net_leader_gap_m', 'leader_speed_mps',
                    'leader_present', 'internal_lane')}
                signals = [name[7:] for name in names if name.startswith('signal_') and raw[index, names.index(name)] > .5]
                examples.append(dict(reference=refs[index], observed_mps=float(y[index]),
                    max_proposal_mps=float(maximum[index]), deficit_mps=float(deficit[index]),
                    best_component_features=features, best_component_signals=signals))
    result = dict(status='COMPLETE_SAVED_ARRAY_ENVELOPE_SUMMARY', count=len(all_deficits),
        fraction_of_motion_reservoir=len(all_deficits)/40000,
        mean_deficit_mps=float(np.mean(all_deficits)), median_deficit_mps=float(np.median(all_deficits)),
        p90_deficit_mps=float(np.quantile(all_deficits, .9)), max_deficit_mps=max(all_deficits),
        groups=dict(groups), by_seed=seeds,
        largest_examples=sorted(examples, key=lambda row: row['deficit_mps'], reverse=True)[:3],
        interpretation='Descriptive training-array readout. Current allowed/acceleration exceedances are not native safety violations. Other proposal limits include following, signals and finite paths; these arrays alone do not distinguish them. Gaussian sigma1 likelihood remains finite.',
        new_native_starts=0, new_fits=0)
    (base/'motion_envelope_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run(Path('/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/target_packet_mean_alignment'))
