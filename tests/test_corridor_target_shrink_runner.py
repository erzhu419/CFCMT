"""The entire training panel must be frozen before any development evaluation."""
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.tsc_corridor_target_shrink_20261109 import run_shrink as runner


def test_all_six_block_weights_written_before_first_dev_label(tmp_path, monkeypatch):
    cohort_file = Path('experiments/tsc_corridor_residual_transfer_20261109/cohort.json')
    cohort = json.loads(cohort_file.read_text())['cities']
    training = dict(source={}, target={city: [] for city in cohort})
    for city in cohort:
        training['source'][city] = dict(real={'city': city}, placebo={'city': city, 'permuted': True})
        for block, seeds in enumerate(cohort[city]['adapt']):
            training['target'][city].append(dict(block=block, city=city, seeds=seeds))
    previous = dict(training=training,
                    cities={city: {'by_seed': [{'seed': seed} for seed in spec['dev']]}
                            for city, spec in cohort.items()})
    previous_file = tmp_path/'previous.json'
    previous_file.write_text(json.dumps(previous))
    fitted, evaluated = [], []

    def fit_source(root, part, city, seeds):
        assert part == 'donor' and seeds == cohort[city]['donor']
        return tuple(SimpleNamespace(metadata=value) for value in training['source'][city].values())

    def fit_block(root, city, seeds, source, placebo):
        assert not evaluated
        assert source.metadata['city'] != city
        fitted.append((city, tuple(seeds)))
        return SimpleNamespace(metadata=dict(city=city, seeds=seeds)), {'weights': {'S': 0., 'P': .25}}

    output = tmp_path/'output'

    def evaluate(city, seed, block, directory, predictor, old, policy, prior):
        assert len(fitted) == 6
        frozen = json.loads((output/'training_result.json').read_text())
        assert frozen['development_hours_read'] == 0
        assert sum(len(rows) for rows in frozen['training']['target'].values()) == 6
        assert predictor.weights == {'S': 0., 'P': .25}
        assert old.weights == {'S': 1., 'P': 1.}
        assert prior['seed'] == seed
        evaluated.append((city, seed))
        return dict(city=city, seed=seed, block=block)

    monkeypatch.setattr(runner, 'fit_source', fit_source)
    monkeypatch.setattr(runner, 'fit_block', fit_block)
    monkeypatch.setattr(runner, 'hour', lambda root, part, city, seed: root/part/city/str(seed))
    monkeypatch.setattr(runner, 'load_training', lambda *args: [])
    monkeypatch.setattr(runner, 'old_fit', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'ArrivalPolicy', lambda *args: None)
    monkeypatch.setattr(runner, 'ShrinkPredictor',
                        lambda target, source, placebo, weights: SimpleNamespace(weights=weights))
    monkeypatch.setattr(runner, 'evaluate_hour', evaluate)
    monkeypatch.setattr(runner, 'summarize', lambda rows: dict(
        prediction_pass=False, ranking_pass=False, by_seed=rows))
    result = runner.run(tmp_path, cohort_file, previous_file, output)
    assert len(evaluated) == len(set(evaluated)) == 12
    assert result['status'] == 'HOLD_TARGET_SHRINK_DEVELOPMENT'
    assert result['training_frozen_before_development']
    assert result['independent_confirmation'] == result['new_simulations'] == 0
