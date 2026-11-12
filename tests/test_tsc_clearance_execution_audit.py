from copy import deepcopy

from scripts.cluster.tsc_clearance_execution_audit import audit_execution, read_clearance_lanes


def fixture(counts, actual=None, yellow_seconds=5):
    nominal = ['Yr'] * yellow_seconds + ['rr'] + ['rG'] * (9 - yellow_seconds)
    actual = nominal if actual is None else actual
    observation = dict(preaction={'nominal_states': {'tls': nominal}},
        lane_geometry={':j_0_0': {'length_m': 20}},
        frames=[dict(offset_sec=t, vehicles={f'v{i}': {'lane_id': ':j_0_0'} for i in range(counts[t])},
                     applied_states={} if t == 0 else {'tls': actual[t - 1]}) for t in range(11)])
    return observation, nominal


def test_four_extensions_use_frames_six_to_nine_not_seven_to_ten():
    counts = [0] * 6 + [1, 1, 1, 1, 0]
    actual = ['Yr'] * 5 + ['rr'] * 5
    obs, nominal = fixture(counts, actual)
    before = deepcopy(obs)
    result = audit_execution(obs, nominal, [':j_0_0'])
    assert result['matches'] and result['expected_states'] == actual
    assert result['nominal_mismatch_steps'] == result['clearance_extension_steps'] == [6, 7, 8, 9]
    assert result['mismatch_steps'] == [] and result['clearance_counts'] == counts
    assert obs == before


def test_green_stays_green_after_first_empty_clearance_sample():
    counts = [0] * 6 + [1, 1, 0, 1, 1]
    actual = ['Yr'] * 5 + ['rr'] * 3 + ['rG'] * 2
    obs, nominal = fixture(counts, actual)
    result = audit_execution(obs, nominal, [':j_0_0'])
    assert result['matches'] and result['clearance_extension_steps'] == [6, 7]
    assert result['expected_states'][8:] == ['rG', 'rG']


def test_unexplained_pre_release_and_late_red_are_rejected():
    obs, nominal = fixture([0] * 11)
    obs['frames'][3]['applied_states']['tls'] = 'rr'
    result = audit_execution(obs, nominal, [':j_0_0'])
    assert not result['matches'] and result['mismatch_steps'] == [2]
    obs, nominal = fixture([0] * 11)
    obs['frames'][7]['applied_states']['tls'] = 'rr'
    result = audit_execution(obs, nominal, [':j_0_0'])
    assert not result['matches'] and result['mismatch_steps'] == [6]


def test_nominal_equal_and_three_second_yellow_supported():
    obs, nominal = fixture([0] * 11)
    assert audit_execution(obs, nominal, [':j_0_0'])['matches']
    obs, _ = fixture([1] * 11, ['Gr'] * 10)
    nominal = ['Gr'] * 10
    assert audit_execution(obs, nominal, [':j_0_0'], tls='tls')['matches']
    counts = [0] * 4 + [1, 0, 0, 0, 0, 0, 0]
    actual = ['Yr'] * 3 + ['rr'] * 2 + ['rG'] * 5
    obs, nominal = fixture(counts, actual, yellow_seconds=3)
    result = audit_execution(obs, nominal, [':j_0_0'])
    assert result['matches'] and result['clearance_extension_steps'] == [4]


def test_missing_clearance_coverage_cannot_explain_a_delay():
    obs, nominal = fixture([0] * 6 + [1] * 5, ['Yr'] * 5 + ['rr'] * 5)
    result = audit_execution(obs, nominal, [':j_0_0', ':j_1_0'])
    assert not result['matches'] and result['clearance_counts'] is None
    assert result['missing_clearance_lanes'] == [':j_1_0']


def test_static_clearance_expands_all_matching_junction_lanes(tmp_path):
    (tmp_path / 'net.xml').write_text('''<net>
      <edge id=":junction_name_0"><lane id=":junction_name_0_0"/></edge>
      <edge id=":junction_name_12"><lane id=":junction_name_12_0"/></edge>
      <edge id=":other_0"><lane id=":other_0_0"/></edge>
      <connection tl="tls" via=":junction_name_0_0"/>
      <connection tl="other" via=":other_0_0"/>
      </net>''')
    cfg = tmp_path / 'test.sumocfg'
    cfg.write_text('<configuration><input><net-file value="net.xml"/></input></configuration>')
    assert read_clearance_lanes(cfg, 'tls') == [':junction_name_0_0', ':junction_name_12_0']
