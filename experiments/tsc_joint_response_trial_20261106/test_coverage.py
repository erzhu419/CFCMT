import importlib.util
import json
from pathlib import Path


spec_loader = importlib.util.spec_from_file_location('coverage_summary', Path(__file__).with_name('summarize_coverage.py'))
coverage = importlib.util.module_from_spec(spec_loader)
spec_loader.loader.exec_module(coverage)


def fixture(tmp_path):
    spec = dict(protocol=coverage.PROTOCOL, source_seeds=list(range(4)), target_seeds=list(range(10, 20)), cases=[])
    for city, seeds in [('hangzhou', spec['source_seeds']), ('jinan', spec['target_seeds'])]:
        for seed in seeds:
            for time in (600, 1800):
                case = dict(case_id=f'{city}_{seed}_{time}', city=city, seed=seed)
                spec['cases'].append(case)
                directory = tmp_path / f'seed_{seed}' / case['case_id']
                directory.mkdir(parents=True)
                receipt = dict(**case, status='COMPLETE', branches=[dict(candidate_index=i, status='COMPLETE', checks={'native_state': True}) for i in range(8)])
                # No group-data, Y, costs, nominal states or actual future frames exist.
                graph = dict(case_id=case['case_id'], information_time_sec=time,
                    lane_nodes=[dict(lane_id='a', vehicle_count=3, lane_measurements={'halted': 2}, speed_limit_mps=10, distances_to_lane_end_m=[10, 50, 150]),
                                dict(lane_id='b', vehicle_count=1, lane_measurements={'halted': 0}, speed_limit_mps=10, distances_to_lane_end_m=[180])],
                    movement_nodes=[dict(incoming_lanes=['a', 'a']), dict(incoming_lanes=['a', 'b'])])
                (directory / 'result.json').write_text(json.dumps(receipt))
                (directory / 'joint_interface_server_only.json').write_text(json.dumps(graph))
    return spec


def test_complete_summary_needs_only_t0_inputs_and_receipt(tmp_path):
    spec = fixture(tmp_path)
    result = coverage.run(spec, tmp_path)
    assert result['status'] == 'COVERAGE_REVIEW_REQUIRED'
    assert result['complete_groups'] == 28 and result['new_fits'] == result['new_simulations'] == 0
    assert result['group_support']['vehicles'] == dict(source=dict(count=8, min=4, median=4., max=4), target=dict(count=20, min=4, median=4., max=4), target_above_source_max=0)
    assert len(json.dumps(result, separators=(',', ':'))) < 30000


def test_shared_and_duplicate_lanes_are_counted_once_per_unit(tmp_path):
    result = coverage.run(fixture(tmp_path), tmp_path)
    row = result['groups'][0]
    assert row['supply'] == [4, 2, 2, .5]
    assert row['incoming_lane_count'] == 2 and row['lane_halted'] == [2, 0]
    assert row['max_lane_halted'] == 2 and row['lanes_with_halted'] == 1
    assert row['movement_supply'] == [[3, 2, 2, 2 / 3], [4, 2, 2, .5]]


def test_incomplete_group_is_not_zero_or_coverage_pass(tmp_path):
    spec = fixture(tmp_path)
    case = spec['cases'][0]
    directory = tmp_path / f"seed_{case['seed']}" / case['case_id']
    (directory / 'result.json').write_text(json.dumps(dict(**case, status='INVALID', branches=[])))
    (directory / 'joint_interface_server_only.json').unlink()
    result = coverage.run(spec, tmp_path)
    assert result['status'] == 'HOLD_COLLECTION' and result['complete_groups'] == 27
    assert result['incomplete'] == [{'case_id': case['case_id'], 'status': 'INVALID'}]
    assert all(row['case_id'] != case['case_id'] for row in result['groups'])
    assert 'group_support' not in result and 'movement_support' not in result
