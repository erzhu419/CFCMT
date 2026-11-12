"""Zero-fit audit of saved choices, reader blocks and archived target costs."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numeric_projection as projection
from prior_ridge import DynamicsModel, ReaderModel, Scaler, predict, reader_predict

BLOCKS = dict(nominal=slice(0, 15), global_nominal=slice(15, 195), predicted_state=slice(195, 211))


def _load(path, dynamics):
    with np.load(path) as values:
        if dynamics:
            return DynamicsModel(Scaler(values['mean'], values['scale']), values['coef'], values['intercept'],
                                 bool(values['single_output'].item()))
        return ReaderModel(values['scale'], values['coef'])


def run(base):
    base = Path(base)
    result = json.loads((base/'nested_fit/result.json').read_text())
    data = json.loads((base/'training_dataset_server_only.json').read_text())
    groups = [group for group in data['groups'] if group['city'] == 'jinan']
    models = {}
    for entry in result['fit_ledger']:
        if entry['role'] in ('outer_target_dynamics', 'target_cost_reader'):
            models[entry['outer_fold'], entry['arm'], entry['role']] = _load(
                base/'nested_fit'/entry['model_file'], entry['role'] == 'outer_target_dynamics')
    rows = []
    for group in sorted(groups, key=lambda item: (item['seed'], item['anchor'])):
        identity = f"jinan/seed{group['seed']}/anchor{group['anchor']}"
        records = {row['arm']: row for row in result['records'] if row['group_id'] == identity}
        branches = sorted(group['branches'], key=lambda row: row['candidate_index'])
        actual = [branch['cost'] for branch in branches]
        pp = group['pp_index']
        arm_rows = {}
        for arm, record in records.items():
            dynamics = models[record['outer_fold'], arm, 'outer_target_dynamics']
            reader = models[record['outer_fold'], arm, 'target_cost_reader']
            absolute = []
            for branch in branches:
                inputs = branch['inputs']
                unused_lanes, X = projection.project_dynamics(inputs)
                state = projection.project_predicted_state(inputs, predict(dynamics, X))
                absolute.append(projection.project_reader(inputs, state))
            features = np.asarray(absolute)-absolute[pp]
            scores = reader_predict(reader, features)
            if not np.allclose(scores, record['predicted_delta_costs'], atol=1e-12, rtol=0):
                raise ValueError('Frozen reader reconstruction differs from saved held-out scores')
            chosen = record['selected_action']
            blocks = {name: float((features[chosen, columns]/reader.scale[columns])@reader.coef[columns])
                      for name, columns in BLOCKS.items()}
            if not np.isclose(sum(blocks.values()), scores[chosen], atol=1e-12, rtol=0):
                raise ValueError('Reader block contributions do not sum to the saved score')
            arm_rows[arm] = dict(selected_action=chosen, predicted_delta=float(scores[chosen]),
                actual_delta=record['cost_minus_pp'], blocks=blocks,
                max_score_change_from_T=max(abs(a-b) for a, b in zip(scores, records['T']['predicted_delta_costs'])))
        rows.append(dict(group_id=identity, seed=group['seed'], anchor=group['anchor'], pp_index=pp,
            action_agreement=len({row['selected_action'] for row in arm_rows.values()}) == 1,
            PP_cost=actual[pp], observed_min_cost=min(actual), PP_is_observed_min=actual[pp] == min(actual),
            observed_min_minus_PP=min(actual)-actual[pp], arms=arm_rows))
    seeds = sorted({row['seed'] for row in rows})
    per_seed = [dict(seed=seed, PP_cost=mean(row['PP_cost'] for row in rows if row['seed'] == seed),
        observed_min_cost=mean(row['observed_min_cost'] for row in rows if row['seed'] == seed),
        observed_min_minus_PP=mean(row['observed_min_minus_PP'] for row in rows if row['seed'] == seed)) for seed in seeds]
    summary = dict(status='DESCRIPTIVE_ZERO_FIT', new_fits=0, new_simulations=0,
        saved_score_reconstruction=True, state_count=len(rows), seed_count=len(seeds),
        action_agreement_states=sum(row['action_agreement'] for row in rows),
        mean_costs=result['mean_costs'], comparisons=result['comparisons'], response_mae=result['response_mae'],
        observed_label_bound=dict(mean_cost=mean(row['observed_min_cost'] for row in per_seed),
            mean_minus_PP=mean(row['observed_min_minus_PP'] for row in per_seed),
            PP_minimum_states=sum(row['PP_is_observed_min'] for row in rows), per_seed=per_seed),
        arm_diagnostics={arm: dict(mean_predicted_selected_delta=mean(row['arms'][arm]['predicted_delta'] for row in rows),
            mean_actual_selected_delta=mean(row['arms'][arm]['actual_delta'] for row in rows),
            harmful_states=sum(row['arms'][arm]['actual_delta'] > 0 for row in rows),
            beneficial_states=sum(row['arms'][arm]['actual_delta'] < 0 for row in rows),
            benefit_sum=sum(max(-row['arms'][arm]['actual_delta'], 0) for row in rows),
            harm_sum=sum(max(row['arms'][arm]['actual_delta'], 0) for row in rows),
            mean_reader_blocks={name: mean(row['arms'][arm]['blocks'][name] for row in rows) for name in BLOCKS})
            for arm in ('T', 'S', 'P')}, records=rows,
        semantics='Block scores are algebraic contributions; observed minima are descriptive label bounds, not deployable choices.')
    (base/'nested_fit/zero_fit_diagnostic.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    output = run(args.base)
    print(json.dumps({key: output[key] for key in ('status', 'state_count', 'action_agreement_states',
                                                  'observed_label_bound', 'arm_diagnostics')}, allow_nan=False))
