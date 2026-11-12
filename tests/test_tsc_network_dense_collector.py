"""Passive prefix timing and the dense pilot's eight-start/first-failure contract."""
import json
import sys
from types import SimpleNamespace

import pytest

from experiments.tsc_network_dense_measurement_20261109 import collect_dense as collector
from scripts.cluster import collect_tsc_physical_response_trial as original


def prefix_setup():
    api = SimpleNamespace(time=0)
    api.simulation = SimpleNamespace(getTime=lambda: api.time)
    events, requests = [], []

    class Executor:
        elapsed = 0

        def feasible_states_now(self):
            return tuple('abcdefgh') if api.time >= 30 else ('a',)

        def request(self, state):
            requests.append((api.time, state))
            events.append(('request', api.time, state))

        def advance(self, elapsed):
            self.elapsed += elapsed
            events.append(('advance', api.time, self.elapsed))

    executor = Executor()

    def advance(*, sumo_api, executors, actions, control_interval_sec,
                collision_mode, safety_ledger):
        assert collision_mode == 'audit'
        for tls in sorted(actions):
            executors[tls].request(actions[tls])
        for _ in range(control_interval_sec):
            events.append(('step', api.time))
            api.time += 1
            for ex in executors.values():
                ex.advance(1)
        return []

    v3 = SimpleNamespace(
        _start_sumo=lambda *args: None,
        _tls_phase_infos=lambda api: {'A': None},
        build_signal_routing_graph=lambda *args: None,
        _scenario_context_v3=lambda **kwargs: None,
        build_safe_phase_executors=lambda *args: {'A': executor},
        _new_safety_ledger=lambda: {},
        _finalize_safety_ledger=lambda ledger: {'no_teleport_passed': True},
        _read_graph_states_v3=lambda api, *args: {'A': api.time},
        _rule_candidate=lambda state, ex, policy: SimpleNamespace(state='abcdefgh'[(state//10) % 8]),
        _advance_with_actions=advance)
    return api, v3, executor, events, requests


def test_prefix_instrumentation_preserves_original_requests_steps_and_postadvance_capture():
    case = dict(sumocfg='unused.sumocfg', seed=76116801, tls_id='A',
                checkpoint_sec=20, checkpoint_search_sec=60)
    plain_api, plain_v3, plain_executor, plain_events, plain_requests = prefix_setup()
    original.native_prefix(plain_api, plain_v3, case)
    api, v3, executor, events, requests = prefix_setup()
    captures = []

    def capture(observed_api, timestamp, role):
        assert observed_api is api and timestamp == api.time == executor.elapsed
        assert role == 'prefix_observation'
        captures.append(timestamp)
        return {'checks': {'passive': True}, 'time_s': timestamp}

    *unused, prefix = collector.native_prefix(api, v3, case, SimpleNamespace(capture=capture))
    assert events == plain_events and requests == plain_requests
    assert requests == [(0, 'a'), (10, 'b'), (20, 'c')]
    assert api.time == plain_api.time == executor.elapsed == plain_executor.elapsed == 30
    assert captures == list(range(31))
    assert prefix['samples'] == 31 and prefix['failed_prefix_samples'] == 0
    assert prefix['last_frame']['time_s'] == 30


@pytest.mark.parametrize('fail_pairing', [False, True])
def test_first_pp_counts_within_eight_and_first_pairing_failure_preserves_partial(
        tmp_path, monkeypatch, fail_pairing):
    calls, closes = [], []

    def compare(a, b, tolerance):
        assert tolerance == 0
        if fail_pairing and len(calls) == 2:
            raise ValueError('pairing audit failed after native branch completion')
        return {'passed': a == b}

    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks',
                        SimpleNamespace(compare_physical=compare))

    def branch(api, v3, case, requested, output, static, sealed_due):
        assert ('resolved_checkpoint_sec' not in case if requested is None
                else case['resolved_checkpoint_sec'] == 620)
        calls.append(requested)
        # A completed native trace must survive a later pairing/audit failure.
        (output/'partial_server_only.json').write_text('{"completed_steps":450}\n')
        index = 5 if requested is None else requested
        return dict(status='COMPLETE', candidate_index=index, reference_index=5,
            candidate_states=list('abcdefgh'), resolved_checkpoint_sec=620,
            completed_steps=450, checks={'complete': True},
            snapshot={'pending_ids': [], 'vehicles': ['same']})

    monkeypatch.setattr(collector, 'collect_branch', branch)
    api = SimpleNamespace(close=lambda: closes.append(1))
    output = tmp_path/'measurement'
    result = collector.collect_case(api, None, {'tls_id': 'A', 'seed': 76116801}, output, {}, {})
    if fail_pairing:
        assert result['status'] == 'HOLD_DENSE_MEASUREMENT'
        assert calls == [None, 0] and len(closes) == result['new_native_starts'] == 2
        failed = json.loads((output/'action_0'/'result.json').read_text())
        assert failed['status'] == 'HOLD_DENSE_MEASUREMENT'
        assert 'pairing audit failed' in failed['error']
        assert json.loads((output/'action_0'/'partial_server_only.json').read_text()) == {'completed_steps': 450}
        assert not (output/'action_1').exists()
    else:
        assert result['status'] == 'PASS_DENSE_MEASUREMENT'
        assert calls == [None, 0, 1, 2, 3, 4, 6, 7]
        assert len(closes) == result['new_native_starts'] == result['native_start_budget'] == 8
        assert sorted(row['candidate_index'] for row in result['branches']) == list(range(8))
    assert json.loads((output/'result.json').read_text()) == result
