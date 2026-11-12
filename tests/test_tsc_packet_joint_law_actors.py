"""LC closure must retain arrivals/far actors and expose exact-budget failure."""
from copy import deepcopy

from experiments.tsc_packet_joint_law_definition_20261110.actors import select_component
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import lane, packet, road_graph
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


def topology():
    return RoadTopology(road_graph([
        lane('a0', 1000, edge='a', index=0), lane('a1', 1000, edge='a', index=1),
        lane('b0', 1000, edge='b', index=0), lane('b1', 1000, edge='b', index=1),
        lane('I', 20, internal=True), lane('one', 1000)], []))


def test_whole_model_projected_road_retains_far_actors_in_original_order():
    current = {row['packet_id']: row for row in [
        packet('focal', 'a0', 20), packet('far', 'a1', 950), packet('other_road', 'b0', 21)]}
    original = deepcopy(current)
    order = {'far': 0, 'other_road': 1, 'focal': 2}
    result = select_component(topology(), current, deepcopy(current), order, 'focal')
    assert result['status'] == 'CLOSED_ORDINARY_ROAD'
    assert result['participants'] == ['far', 'focal']
    assert result['receiving_intent_ids'] == []
    assert current == original


def test_crossroad_arrival_is_included_and_requires_receiving_road_intent():
    current = {row['packet_id']: row for row in [
        packet('focal', 'a0', 20), packet('arrival', 'I', 19), packet('elsewhere', 'b0', 21)]}
    projected = deepcopy(current)
    projected['arrival'].update(lane_id='a1', position_m=2.)
    result = select_component(topology(), current, projected,
                              {'arrival': 0, 'focal': 1, 'elsewhere': 2}, 'focal')
    assert result['participants'] == ['arrival', 'focal']
    assert result['receiving_intent_ids'] == ['arrival']


def test_component_overflow_is_not_silently_truncated_or_conditioned_away():
    current = {str(i): packet(str(i), 'a'+str(i % 2), 30.*i+10) for i in range(4)}
    result = select_component(topology(), current, current, {str(i): i for i in range(4)}, '0', max_actors=3)
    assert result['status'] == 'OBSTRUCTION_COMPONENT_EXCEEDS_CAP'
    assert result['required_participants'] == ['0', '1', '2', '3']
    assert result['required_actor_count'] == 4
    assert result['participants'] == []


def test_incomplete_frame_must_not_hide_arrivals_or_longitudinal_dependencies():
    current = {'focal': packet('focal', 'a0', 20), 'other': packet('other', 'a1', 30)}
    result = select_component(topology(), current, {'focal': current['focal']},
                              {'focal': 0, 'other': 1}, 'focal')
    assert result['status'] == 'OBSTRUCTION_INCOMPLETE_LONGITUDINAL_FRAME'
    assert result['missing_ids'] == ['other']


def test_ended_internal_and_single_lane_focals_have_deterministic_stay_factor():
    for lane_id in ('I', 'one'):
        current = {'focal': packet('focal', lane_id, 5)}
        result = select_component(topology(), current, current, {'focal': 0}, 'focal')
        assert result['status'] == 'CLOSED_NO_LC' and result['focal_stay']
        assert result['participants'] == []
    current = {'focal': packet('focal', 'a0', 999)}
    ended = select_component(topology(), current, {}, {'focal': 0}, 'focal', ended_ids=['focal'])
    assert ended['status'] == 'CLOSED_NO_LC' and ended['reason'] == 'focal_ended'
