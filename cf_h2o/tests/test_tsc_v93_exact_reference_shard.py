import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import run_tsc_v93_exact_reference_shard as worker
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTEXT_NAMES_V3, _absolute_target_candidates, _pressure_candidate_v3,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import LocalTransitionState
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import PhaseCandidate, TlsPhaseInfo
from cf_h2o.traffic_signal.action_contrast import rule_reference_indices
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json'
LAUNCH = ROOT / 'cf_h2o/results/cluster/tsc_v93_source_city_components_20260831/disjoint_confirmation_v1/launch.json'


def test_three_shards_retain_original_224_runs():
    _, seeds = worker.load_cohort(LAUNCH, SPEC)
    shards = [worker.shard_identities(seeds, shard_index=i, shard_count=3) for i in range(3)]
    identities = [x for shard in shards for x in shard]
    assert len(identities) == len(set(identities)) == 224
    assert {seed for _, seed in identities} == set(seeds)


@pytest.mark.parametrize('occupancy,mode,expected', [
    ((0.8, 0.1), 'green', 'rG'),
    ((0.1, 0.1), 'green', 'Gr'),
    ((0.8, 0.1), 'yellow', 'rG'),
])
def test_exact_baseline_matches_cfcmt_reference_including_ties_and_clearance(occupancy, mode, expected):
    info = TlsPhaseInfo(
        'tls', ((('in_a', 'out_a', ''),), (('in_b', 'out_b', ''),)),
        (PhaseCandidate(0, 'Gr', 30.0, 1), PhaseCandidate(1, 'rG', 30.0, 1)),
        ('in_a', 'in_b'),
    )
    values = {'in_a': 5.0, 'in_b': 5.0}
    state = LocalTransitionState(
        tls_id='tls', info=info, q_by_lane=values, veh_by_lane=values,
        speed_by_lane=values, occ_by_lane={'in_a': .1, 'in_b': .1},
        down_q_by_lane=values, down_occ_by_lane=dict(zip(info.incoming_lanes, occupancy)),
        context=np.zeros(len(CONTEXT_NAMES_V3)), sim_time=1880.0,
    )
    executor = SafePhaseExecutor(
        sumo_api=object(), tls_id='tls',
        timings=[PhaseTiming('Gr', 5., 3., 1.), PhaseTiming('rG', 5., 3., 1.)],
        initial_state='Gr', initial_green_elapsed_sec=10.,
    )
    if mode == 'yellow':
        executor.mode = 'yellow'
        executor.target_green_state = 'rG'
    spec = PressurePolicySpec(downstream_occupancy_weight=.02)
    baseline = _pressure_candidate_v3(state, executor, spec, control_interval_sec=10)
    candidates, absolute = _absolute_target_candidates(
        state, executor, control_interval_sec=10, prediction_horizon_sec=60,
    )
    reference = next(iter(rule_reference_indices(absolute, policy=spec).values()))
    assert baseline.state == candidates[reference].state == expected
    if mode == 'green' and occupancy[0] != occupancy[1]:
        pp = _pressure_candidate_v3(state, executor, 'phase_pressure', control_interval_sec=10)
        assert pp.state != baseline.state


def test_worker_uses_frozen_reference_without_guard_or_learning_override(tmp_path, monkeypatch):
    launch, seeds = worker.load_cohort(LAUNCH, SPEC)
    received = {}
    def fake_rollout(**kwargs):
        received.update(kwargs)
        return {
            'city': 'jinan', 'runtime_policy': 'selected_source_prior',
            'evaluation_horizon_sec': 3600, 'control_interval_sec': 10, 'warmup_sec': 60,
            'metrics': {**{k: 0 for k in worker.METRICS}, 'reference_policy_audit': {
                'reference_policy_spec': worker.REFERENCE_SPEC, 'decision_requests': 4320,
                'post_warmup_decision_requests': 4248, 'request_trace': [],
            }},
        }
    monkeypatch.setattr(worker, 'run_external_closed_loop_rollout', fake_rollout)
    row = worker._run_one({'identity': ('jinan_3x4_real', seeds[0]), 'allowed_seeds': seeds,
                          'results_root': str(tmp_path), 'rollout_kwargs': worker.rollout_kwargs(launch)})
    spec = received['diagnostic_spec']
    assert spec.source_policy == 'selected_source_prior'
    assert spec.coordination_mode == 'direct' and spec.guard_config is None
    assert 'method_runtime_override' not in received
    assert received['expected_sumo_version'] == '1.22.0'
    assert str(received['conversion_root']).endswith('full_networks_v8')
    assert row['reference_post_warmup_decision_requests'] == 4248
    assert json.loads(Path(row['result_path']).read_text())['reference_comparison']['reference_key'] == worker.REFERENCE_KEY


def test_progress_reports_start_and_every_completed_rollout(tmp_path, monkeypatch, capsys):
    identities = (('la_1x4', 5622039), ('jinan_3x4_real', 5622039))
    monkeypatch.setattr(worker, 'shard_identities', lambda *_args, **_kwargs: identities)
    class Pool:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def imap_unordered(self, function, payloads, chunksize):
            for payload in payloads:
                scenario, seed = payload['identity']
                yield {'scenario': scenario, 'seed': seed}
    monkeypatch.setattr(worker.mp, 'get_context', lambda *_args: SimpleNamespace(Pool=lambda **_kwargs: Pool()))
    result = worker.main([
        '--parent-launch', str(LAUNCH), '--confirmation-spec', str(SPEC),
        '--results-root', str(tmp_path), '--shard-index', '0', '--shard-count', '3', '--workers', '2',
    ])
    assert result == 0
    lines = capsys.readouterr().out.splitlines()
    progress = [json.loads(line.split(' ', 1)[1]) for line in lines if line.startswith('SCHEDULEURM_PROGRESS ')]
    assert [row['current'] for row in progress] == [0, 1, 2]
    assert all(row['total'] == 2 and row['unit'] == 'rollout' for row in progress)
    assert all(row['elapsed_seconds'] >= 0 for row in progress)
