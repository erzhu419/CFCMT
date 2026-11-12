"""Small gates and cost accounting; the root runs the 512-forward gate once."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.tsc_packet_conditional_particles_20261110.fixtures import cases
from experiments.tsc_packet_discrete_braking_20261110 import motion
from experiments.tsc_packet_joint_law_definition_20261110.law import OutsidePrototype
from experiments.tsc_packet_real_particle_pilot_20261110 import sampler as implementation
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key, select_cases
from experiments.tsc_packet_real_particle_pilot_20261110.evaluate import (
    case_seed, counters_from_sampler, load_labels, primary_response_budget, probability_error,
)
from experiments.tsc_packet_real_particle_pilot_20261110 import evaluate
from experiments.tsc_packet_real_particle_pilot_20261110.synthetic import feature_only, numeric_memo_only


def test_physical_budget_is_checked_before_a_new_forward():
    kernel, focal = cases()['stopped_open']
    sampler = implementation.make_sampler(kernel, focal, physical_limit=0, feature_limit=10)
    with feature_only(), pytest.raises(OutsidePrototype):
        sampler.draw(1, 20261111)
    assert sampler.stats['physical_calls'] == 0
    assert sampler.stats['canonical_feature_evaluations'] == 0


def test_feature_budget_preserves_the_partial_feature_count_without_a_forward():
    kernel, focal = cases()['cap4_lateral']
    sampler = implementation.make_sampler(kernel, focal, physical_limit=1, feature_limit=1)
    with feature_only(), pytest.raises(OutsidePrototype):
        sampler.draw(1, 20261111)
    assert sampler.stats['physical_calls'] == 0
    assert sampler.stats['canonical_feature_evaluations'] == 1


def test_switching_goal_contexts_keeps_numeric_memo_and_one_physical_view():
    kernel, focal = cases()['intent_and_survival']
    sampler = implementation.make_sampler(kernel, focal, physical_limit=1, feature_limit=10)
    with feature_only():
        for goal, probability in sampler.goals[0]:
            sampler._configuration([goal])
            numeric_memo_only(sampler)
        sampler._configuration([sampler.goals[0][0][0]])
    assert len(sampler.configurations) == 2
    assert sampler.stats['canonical_feature_evaluations'] == 2
    assert sampler.stats['physical_calls'] == 0
    assert sampler.stats['retained_current_contexts'] == sampler.stats['peak_retained_current_contexts'] == 1


def test_failed_third_forward_keeps_started_and_partial_consumed_counts():
    """Two completed bodies and one failing constructor are not reported as0."""
    kernel, focal = cases()['stopped_open']
    sampler = implementation.make_sampler(kernel, focal, physical_limit=4, feature_limit=4)
    original = motion.MotionResolver.__init__
    attempts = []
    def initialize(resolver, *args, **kwargs):
        attempts.append(True)
        if len(attempts) == 3:
            raise OutsidePrototype('synthetic third forward obstruction')
        return original(resolver, *args, **kwargs)
    with patch.object(motion.MotionResolver, '__init__', initialize), \
            pytest.raises(OutsidePrototype, match='third forward'):
        sampler.draw(4, 20261111)
    assert len(attempts) == sampler.stats['physical_calls'] == 3
    assert sampler.stats['canonical_feature_evaluations'] == 1
    receipt = counters_from_sampler(sampler)
    assert receipt['physical_calls'] == 3 and receipt['canonical_feature_evaluations'] == 1


def test_completed_gate_retains_all_frozen_and_unprepared_cases():
    path = Path('experiments/tsc_packet_real_particle_pilot_20261110/synthetic_result.json')
    assert path.exists(), 'Run the declared512-forward synthetic gate once before this receipt check'
    result = json.loads(path.read_text())
    assert result['status'] == 'PASS_LIGHT_SAMPLER_EQUIVALENCE'
    assert result['explicit_full_forward_calls'] == 512
    assert result['declared_synthetic_cases'] == 15 and result['unprepared_cases'] == 1
    assert len(result['cases']) == 16
    assert result['cases']['cap5_focal_last']['stochastic_actor_count'] == 5
    assert result['cases']['cap10_motion_oracle']['stochastic_actor_count'] == 10
    assert result['maximum_probability_difference'] < 1e-12
    assert result['maximum_gradient_difference'] < 1e-12
    assert result['real_record_response_queries'] == result['new_native_starts'] == result['new_fits'] == 0


def census_metadata():
    originals, rows = [], []
    for index, count in enumerate((2, 3, 2, 2, 2, 4, 2, 3, 2)):
        reference = dict(seed=76116901, time_sec=index, packet_id='old'+str(index))
        originals.append(reference)
        rows.append(dict(reference, required_actor_count=count,
            required_participants=['old'+str(index)+'_'+str(j) for j in range(count)],
            frame_actor_count=100, status='REGISTERED_SYMBOLIC_UPPER_BOUND', y=0))
    for count in range(5, 11):
        # The later time in the earlier seed wins lexicographically.
        for seed, t, prefix in ((76116902, 1, 'laterseed'), (76116901, 100+count, 'chosen')):
            rows.append(dict(seed=seed, time_sec=t, packet_id=prefix+str(count),
                required_actor_count=count,
                required_participants=[prefix+str(count)+'_'+str(j) for j in range(count)],
                frame_actor_count=100, status='OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP', y=0))
    return originals, rows


def test_cohort_uses_original_order_then_current_size_and_lexicographic_first():
    originals, rows = census_metadata()
    chosen = select_cases(originals, reversed(rows))
    assert [row['reference'] for row in chosen[:9]] == originals
    assert [row['required_actor_count'] for row in chosen[9:]] == list(range(5, 11))
    assert [row['reference']['packet_id'] for row in chosen[9:]] == ['chosen'+str(j) for j in range(5, 11)]
    assert [row['case_index'] for row in chosen] == list(range(15))
    assert len({reference_key(row['reference']) for row in chosen}) == 15
    assert sum(row['required_actor_count'] for row in chosen) == 67


def test_cohort_selection_does_not_depend_on_or_export_stored_labels():
    originals, rows = census_metadata()
    before = select_cases(originals, rows)
    for index, row in enumerate(rows):
        row['y'] = index % 2
    after = select_cases(originals, rows)
    assert before == after
    assert '"y"' not in json.dumps(after)


def test_missing_declared_size_holds_without_substitution():
    originals, rows = census_metadata()
    with pytest.raises(ValueError, match='each required size5-10'):
        select_cases(originals, [row for row in rows if row['required_actor_count'] != 10])


def test_dispatch_order_does_not_change_frozen_particle_seed_indices():
    originals, rows = census_metadata()
    selected = select_cases(originals, rows)
    ordered = sorted(selected, key=lambda row: (-row['required_actor_count'], row['case_index']))
    seeds = {reference_key(row['reference']): case_seed(row['case_index']) for row in ordered}
    assert [seeds[reference_key(row['reference'])] for row in selected] == [
        20261111+100000*index for index in range(15)]


def test_label_recovery_matches_exact_tuples_and_preserves_head_ordinals(tmp_path):
    old = dict(seed=76116901, time_sec=462., packet_id='old')
    extra = dict(seed=76116902, time_sec=10., packet_id='extra')
    unrelated = dict(seed=76116901, time_sec=463., packet_id='old')
    manifest = {'76116901': {'motion': [dict(old, y=10.5)],
        'lane_change': [dict(unrelated, y=2), dict(old, y=0)]},
        '76116902': {'start': [dict(extra, y=1)]}}
    manifest_path, exact_path = tmp_path/'manifest.json', tmp_path/'exact.json'
    manifest_path.write_text(json.dumps(manifest))
    exact_path.write_text(json.dumps({'cases': [dict(reference=old,
        native_speed_mps=10.5, observed_lc_label=0)]}))
    selected = [dict(reference=extra, case_index=9), dict(reference=old, case_index=0)]
    labels = load_labels(manifest_path, selected, exact_path)
    observed = labels[reference_key(old)]['observations']
    assert observed == [dict(head='motion', y=10.5, manifest_ordinal=0),
                        dict(head='lane_change', y=0, manifest_ordinal=1)]
    assert labels[reference_key(old)]['native_speed_mps'] == 10.5
    assert labels[reference_key(extra)]['observations'] == [dict(head='start', y=1, manifest_ordinal=0)]
    assert 'exact' not in labels[reference_key(extra)]
    manifest['76116901']['lane_change'][1]['y'] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='native LC labels differ'):
        load_labels(manifest_path, selected, exact_path)


def test_unregistered_feature_resolver_is_rejected_before_its_constructor():
    class NoParticles:
        stats = {'physical_calls': 0}
    with primary_response_budget(NoParticles(), 1) as accounting:
        with pytest.raises(RuntimeError, match='unregistered'):
            motion.MotionResolver(None, None, None, None)
    assert accounting['resolver_constructions'] == 0
    assert accounting['forbidden_feature_or_extra_resolver_attempts'] == 1


def test_saved_positive_zero_sample_is_reported_as_infinite_error():
    missed = probability_error(0., .1)
    assert missed['missed_known_positive'] and missed['nll_error_infinite']
    assert missed['nll_error'] is None and not missed['structural_zero_false_positive']
    structural = probability_error(.1, 0.)
    assert structural['structural_zero_false_positive'] and not structural['missed_known_positive']


def test_receipt_collector_recovers_unreturned_summary_without_double_counting(tmp_path):
    reference = dict(seed=76116901, time_sec=10, packet_id='a')
    returned = dict(case_index=0, reference=reference, physical_calls=1024,
                    canonical_feature_evaluations=40)
    written = dict(returned, case_index=1, reference=dict(reference, packet_id='b'))
    for row in (returned, written):
        directory = tmp_path/f"case{row['case_index']:02d}_seed76116901_t10"
        directory.mkdir()
        (directory/'summary.json').write_text(json.dumps(row))
    pending = tmp_path/'case02_seed76116901_t10'
    pending.mkdir()
    (pending/'started.json').write_text(json.dumps(dict(reference=reference, case_index=2)))
    (pending/'progress.json').write_text(json.dumps(dict(physical_calls=123,
        canonical_feature_evaluations=456)))
    completed, unknown = evaluate.collect_receipts(tmp_path, [returned])
    assert [row['case_index'] for row in completed] == [0, 1]
    assert sum(row['physical_calls'] for row in completed) == 2048
    assert len(unknown) == 1 and unknown[0]['case_index'] == 2
    assert unknown[0]['physical_calls_lower_bound'] == 123
    assert unknown[0]['canonical_feature_evaluations_lower_bound'] == 456
    assert unknown[0]['consumed_counters_exact'] is False


def test_technical_driver_failure_preserves_unknown_costs_as_lower_bounds(tmp_path, monkeypatch):
    originals, metadata = census_metadata()
    selected = select_cases(originals, metadata)
    preflight = tmp_path/'preflight'
    preflight.mkdir()
    (preflight/'summary.json').write_text(json.dumps(dict(status='PASS_FROZEN_REAL_PARTICLE_COHORT',
        cases=selected, particle_full_forward_ceiling=15360, canonical_feature_ceiling=68608)))
    proof = tmp_path/'gate.json'
    proof.write_text(json.dumps(dict(status='PASS_LIGHT_SAMPLER_EQUIVALENCE')))
    output = tmp_path/'output'
    labels = {reference_key(row['reference']): dict(observations=[]) for row in selected}
    monkeypatch.setattr(evaluate, 'load_labels', lambda *args: labels)

    class FailedDriver:
        def __init__(self, *args, **kwargs):
            self.submitted = 0
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, task):
            self.submitted += 1
            row = task[3]
            reference = row['reference']
            directory = output/f"case{row['case_index']:02d}_seed{reference['seed']}_t{reference['time_sec']}"
            directory.mkdir()
            if self.submitted == 1:
                (directory/'summary.json').write_text(json.dumps(dict(case_index=row['case_index'],
                    reference=reference, status='PASS_REAL_PARTICLE_CASE', physical_calls=1024,
                    canonical_feature_evaluations=40)))
                return object()
            (directory/'started.json').write_text(json.dumps(dict(case_index=row['case_index'],
                reference=reference)))
            (directory/'progress.json').write_text(json.dumps(dict(physical_calls=123,
                canonical_feature_evaluations=456)))
            raise RuntimeError('synthetic driver transport failure')

    monkeypatch.setattr(evaluate, 'ProcessPoolExecutor', FailedDriver)
    result = evaluate.run(tmp_path/'cache', preflight, proof, tmp_path/'exact.json',
                          tmp_path/'width.json', output, workers=4)
    assert result['status'] == 'HOLD_REAL_PARTICLE_PILOT_TECHNICAL'
    assert result['consumed_counters_exact'] is False
    assert result['actual_full_forward_calls'] is None
    assert result['actual_canonical_feature_evaluations'] is None
    assert result['full_forward_calls_lower_bound'] == 1147
    assert result['canonical_feature_evaluations_lower_bound'] == 496
    assert len(result['cases']) == 1 and len(result['incomplete_started_cases']) == 1
