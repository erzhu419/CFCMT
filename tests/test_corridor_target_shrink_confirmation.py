"""Fresh confirmation keeps its candidate fixed and tests both source contrasts."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from experiments.tsc_corridor_target_shrink_confirmation_20261109 import run_confirmation as r
from experiments.tsc_corridor_target_shrink_confirmation_20261109 import collect_confirmation as c


def panel(losses, placebo=.19, target=.2):
    return [dict(seed=seed, block=index//2, scopes={'all_observable_risk': {
        'proper_brier': {'S': loss, 'T': target, 'P': placebo}}})
        for index, (seed, loss) in enumerate(zip(range(6), losses))]


def test_confirmation_requires_both_contrasts_and_reports_small_sample_uncertainty():
    assert r.inference(panel([.095, .1, .105, .098, .102, .1]))['status'] == 'PROBABILITY_INCREMENT_SUPPORTED'
    variable = [.0, .3, .15, .15, .15, .15]
    result = r.inference(panel(variable, placebo=.17, target=.16))
    assert result['status'] == 'DIRECTION_REPLICATED'
    assert all(value['approximate_95pct_interval'][1] > 0 for value in result['contrasts'].values())
    assert r.inference(panel(variable, placebo=.14, target=.16))['status'] == 'INCONCLUSIVE'


def test_collection_schedules_only_the_six_new_hangzhou_hours(tmp_path, monkeypatch):
    jobs = []

    def collect(job, collector_dir, root):
        jobs.append(job)
        return dict(seed=job[3], status='PASS_HOUR')

    monkeypatch.setitem(sys.modules, 'run_hours', SimpleNamespace(collect=collect))
    result = c.run(tmp_path/'data', tmp_path/'collector', tmp_path/'frozen.sumocfg', 3)
    assert result['status'] == 'PASS_CONFIRMATION_COLLECTION'
    assert len(jobs) == 6
    assert sorted(job[3] for job in jobs) == list(range(76113701, 76113707))
    assert all(job[:2] == ('confirmation', 'hangzhou') for job in jobs)


def test_restore_all_three_frozen_heads_before_reading_new_labels(tmp_path, monkeypatch):
    cohort_path = Path('experiments/tsc_corridor_residual_transfer_20261109/cohort.json')
    cohort = json.loads(cohort_path.read_text())['cities']
    protocol = json.loads((r.HERE/'protocol.json').read_text())
    frozen = dict(status='ALL_TARGET_WEIGHTS_FROZEN', development_hours_read=0,
        training=dict(source={'jinan': {'real': {'kind': 'real'}, 'placebo': {'kind': 'placebo'}}},
                      target={'hangzhou': [dict(metadata=dict(block=block, seeds=seeds),
                          calibration={'weights': {'S': protocol['weights_S'][block],
                                                   'P': protocol['weights_P'][block]}})
                          for block, seeds in enumerate(cohort['hangzhou']['adapt'])]}))
    frozen_file = tmp_path/'training.json'
    frozen_file.write_text(json.dumps(frozen))
    restored, evaluated = [], []

    def fit_target(root, part, city, seeds):
        assert part == 'adapt' and city == 'hangzhou' and not evaluated
        restored.append(seeds)
        return SimpleNamespace(metadata={'seeds': seeds})

    output = tmp_path/'output'

    def evaluate(city, seed, block, directory, prediction, reference, policy, previous):
        assert previous is None and len(restored) == 3
        receipt = json.loads((output/'model_restoration.json').read_text())
        assert receipt['new_calibration_fits'] == receipt['confirmation_label_hours_read'] == 0
        assert len(receipt['blocks']) == 3
        assert prediction.weights == frozen['training']['target']['hangzhou'][block]['calibration']['weights']
        evaluated.append(seed)
        return dict(seed=seed, block=block, scopes={'all_observable_risk': {
            'proper_brier': {'T': .2, 'P': .19, 'S': .1}}})

    monkeypatch.setattr(r, 'fit_source', lambda *args: tuple(SimpleNamespace(metadata=value)
        for value in frozen['training']['source']['jinan'].values()))
    monkeypatch.setattr(r, 'fit_hours', fit_target)
    monkeypatch.setattr(r, 'load_training', lambda *args: [])
    monkeypatch.setattr(r, 'old_fit', lambda *args, **kwargs: None)
    monkeypatch.setattr(r, 'ArrivalPolicy', lambda *args: None)
    monkeypatch.setattr(r, 'hour', lambda root, part, city, seed: root/part/city/str(seed))
    monkeypatch.setattr(r, 'ShrinkPredictor', lambda target, source, placebo, weights: SimpleNamespace(weights=weights))
    monkeypatch.setattr(r, 'evaluate_hour', evaluate)
    monkeypatch.setattr(r, 'summarize', lambda rows: {'by_seed': rows})
    result = r.run(tmp_path, tmp_path, cohort_path, frozen_file, output)
    assert evaluated == list(range(76113701, 76113707))
    assert result['new_calibration_fits'] == result['learned_controller_closed_loop_runs'] == 0
    assert result['old_two_city_status'] == 'HOLD_TARGET_SHRINK_DEVELOPMENT'
