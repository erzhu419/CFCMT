"""Synthetic common-random-scenario alignment, with no simulator or fitting."""
from statistics import mean, pvariance

import pytest

from experiments.tsc_target_packet_kernel_20261109.scenario_noise import ScenarioNoise


def test_boundary_draws_do_not_depend_on_packet_decision_or_exit_activity():
    quiet = ScenarioNoise(76117001, ['a', 'b'], 600)
    busy = ScenarioNoise(76117001, ['b', 'a'], 600)
    for second in range(20):
        busy.step_uniforms('a', 600+second)
        busy.step_uniforms('a', 600+second)
        if second % 3 == 0:
            busy.step_uniforms('b', 600+second)
        for _ in range(second):
            busy.intent_uniform('a')
        count = quiet.poisson_due(2.3)
        assert busy.poisson_due(2.3) == count
        for index in range(count):
            packet_id = f'due_{second}_{index}'
            quiet.register(packet_id, 600+second)
            busy.register(packet_id, 600+second)
            assert busy.entry_uniform() == quiet.entry_uniform()
            assert busy.profile_index(17) == quiet.profile_index(17)
            assert busy.intent_uniform(packet_id) == quiet.intent_uniform(packet_id)


def test_skipped_pending_seconds_match_dense_absolute_time_and_cache_current_bundle():
    dense = ScenarioNoise(7, ['pending'], 600)
    skipped = ScenarioNoise(7, ['pending'], 600)
    for second in range(11):
        last = dense.step_uniforms('pending', 600+second)
    assert skipped.step_uniforms('pending', 610) == last
    assert skipped.step_uniforms('pending', 610) == last
    assert set(last) == {'start', 'yield', 'lane'}
    assert skipped.step_uniforms('pending', 611) == dense.step_uniforms('pending', 611)
    with pytest.raises(ValueError, match='backwards'):
        skipped.step_uniforms('pending', 610)


def test_each_packet_kth_intent_is_independent_of_other_packets_and_decision_clock():
    left = ScenarioNoise(11, ['p', 'q'], 100)
    right = ScenarioNoise(11, ['q', 'p'], 100)
    for road_entry in range(8):
        for _ in range(road_entry*3):
            right.intent_uniform('q')
        right.step_uniforms('p', 100+10*road_entry)
        assert right.intent_uniform('p') == left.intent_uniform('p')
    left.register('new_due', 200)
    right.register('new_due', 200)
    assert left.intent_uniform('new_due') == right.intent_uniform('new_due')
    assert left.step_uniforms('new_due', 210) == right.step_uniforms('new_due', 210)


def test_independent_entry_profile_streams_and_poisson_sampler_moments():
    first, second = ScenarioNoise(99, [], 0), ScenarioNoise(99, [], 0)
    for _ in range(100):
        first.profile_index(13)
    assert first.entry_uniform() == second.entry_uniform()
    assert first.poisson_due(0) == 0
    counts = [first.poisson_due(2.3) for _ in range(10000)]
    assert mean(counts) == pytest.approx(2.3, abs=.08)
    assert pvariance(counts) == pytest.approx(2.3, abs=.15)
