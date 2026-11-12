"""Synthetic current-conflict tracing; no native records or fitting."""
from copy import deepcopy
import gzip
import json

from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    competing_approaches, executor_seed, packet,
)
from experiments.tsc_target_packet_training_20261109 import conditional_features as baseline
from experiments.tsc_packet_conflict_audit_20261110.audit import (
    interval_overlap, run, select_focus, summarize, trace_current,
)


def test_current_trace_separates_eligible_and_strictly_overlapping_reservations():
    static, laws = competing_approaches()
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    packets = {p['packet_id']: p for p in (
        packet('ego', 'b', 19, 2), packet('near', 'a', 19, 2), packet('far', 'a', 1, 0))}
    frame = dict(time_s=0, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(row['snapshot'])) for tls, row in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    kernel = baseline.prepare_feature_kernel(static, frame, seeds, laws, {'J': 'Gg'}, 0)
    old_state, old_connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    result = trace_current(kernel, 'ego', 'out_b')
    assert result['admission']['conflict'] is True
    assert len(result['blockers']) == 2
    overlap = [row for row in result['blockers'] if row['is_overlapping']]
    assert len(overlap) == 1 and overlap[0]['packet_id'] == 'near'
    assert overlap[0]['ego_yields_to_foe'] is True
    assert overlap[0]['foe_yields_to_ego'] is False
    assert overlap[0]['internal'] is False
    assert overlap[0]['raw_zero_speed_mps'] >= overlap[0]['capped_zero_speed_mps']
    assert kernel.state == old_state and kernel._connections == old_connections
    assert interval_overlap((0., 1.), (1., 2.)) == 0


def test_exact_focus_excludes_different_connections_raw_reach_and_permitted_rows(tmp_path):
    out = tmp_path/'output'
    out.mkdir()
    actual = dict(key='actual', to_edge='out')

    def row(index, same=True, conflict=True, risk=False, y=1):
        return dict(reference=dict(seed=1, time_sec=index, packet_id=f'p{index}', y=y),
            feature_seed=index, prior={'out': 1.},
            paired=[dict(intent='out', connection=dict(key='actual' if same else 'other'),
                         candidate=dict(yield_risk=risk, conflict=conflict))],
            posthoc_actual_entry=dict(match=dict(matches=[actual])))

    records = [row(i) for i in range(374)]
    records += [row(i+374, same=False) for i in range(547)]
    records += [row(921+i, conflict=False) for i in range(2)]
    records += [row(923, same=False, conflict=False), row(924, risk=True), row(925, y=0)]
    with gzip.open(out/'seed1_pairs_server_only.jsonl.gz', 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record)+'\n')
    selected, categories = select_focus(tmp_path)
    assert len(selected[1]) == 374
    assert categories['conflict:different_connection'] == 547
    assert categories['raw_reach:same_connection'] == 2
    assert selected[1][0]['reference']['packet_id'] == 'p0'


def test_summary_counts_only_true_blockers_and_keeps_mixed_truth():
    def blocker(internal, verdict, road='same', overlap=True):
        result = dict(internal=internal, is_overlapping=overlap)
        if overlap:
            result['observed_truth'] = dict(connection_verdict=verdict, road_target_verdict=road,
                query_plus1_snapshot=dict(occupied_internal_lanes=['I']))
        return result
    records = [dict(poststep_observed_ego=dict(front_internal=True),
        blockers=[blocker(True, 'same'), blocker(False, 'different'), blocker(False, 'same', overlap=False)])]
    result = summarize(records)
    assert result['reservation_counts']['eligible'] == 3
    assert result['reservation_counts']['overlapping'] == 2
    assert result['case_blocker_types']['mixed_internal_and_approach'] == 1
    assert result['case_physical_truth_groups']['mixed_truth_verdicts'] == 1
    assert result['blocker_road_target_truth'] == {'same': 2}


def test_unreproduced_selection_stops_with_a_zero_fit_receipt(tmp_path):
    result = run(tmp_path/'absent_cache', tmp_path/'empty_pairs', tmp_path/'new_output', workers=1)
    assert result['status'] == 'HOLD_CONFLICT_AUDIT'
    assert result['new_native_starts'] == result['new_fits'] == 0
    assert json.loads((tmp_path/'new_output/result.json').read_text()) == result
