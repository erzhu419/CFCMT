from copy import deepcopy
from types import SimpleNamespace

from scripts.cluster.tsc_joint_response_observer import JointResponseObserver
from scripts.cluster.tsc_movement_response import MovementResponseObserver


class GetterAPI:
    """Read-only fixture deliberately has no simulator step or signal setter."""

    def __init__(self):
        self.rows = {'v': ('in_0', 'in')}
        self.measured = []
        self.measurements = {
            'in_0': (1, .25, 0.), 'in_1': (0, 0., 11.11),
            'out_0': (0, .1, 7.), 'other_in_0': (0, 0., 9.),
            'other_out_0': (0, 0., 9.)}
        self.lane = SimpleNamespace(
            getEdgeID=lambda lane: lane.rsplit('_', 1)[0],
            getLinks=lambda lane: [], getLength=lambda lane: 600.,
            getLastStepVehicleIDs=lambda lane: [v for v, row in self.rows.items() if row[0] == lane],
            getLastStepVehicleNumber=self.vehicle_count,
            getLastStepHaltingNumber=lambda lane: self.measurements[lane][0],
            getLastStepOccupancy=lambda lane: self.measurements[lane][1],
            getLastStepMeanSpeed=lambda lane: self.measurements[lane][2])
        self.vehicle = SimpleNamespace(
            getIDList=lambda: list(self.rows), getRoute=lambda v: ('in', 'out'),
            getRouteIndex=lambda v: int(self.rows[v][1] == 'out'),
            getLaneID=lambda v: self.rows[v][0], getRoadID=lambda v: self.rows[v][1],
            getLanePosition=lambda v: 599. if self.rows[v][1] == 'in' else 1.)
        self.simulation = SimpleNamespace(getArrivedIDList=lambda: [],
            getStartingTeleportIDList=lambda: [])

    def vehicle_count(self, lane):
        self.measured.append(lane)
        return sum(row[0] == lane for row in self.rows.values())


def infos():
    return {
        'tls': SimpleNamespace(controlled_links=[
            [('in_0', 'out_0', '')], [('in_1', 'out_0', '')]]),
        'other': SimpleNamespace(controlled_links=[
            [('other_in_0', 'other_out_0', '')]])}


def test_t0_records_each_focal_input_and_output_lane_directly():
    api = GetterAPI()
    observer = JointResponseObserver(api, infos(), 600, focal_tls='tls')
    observer.sample(600)
    actual = observer.frames[0]['focal_lane_measurements']
    assert set(actual) == {'in_0', 'in_1', 'out_0'}
    assert actual['in_0'] == dict(vehicles=1, halted=1, occupancy_raw=.25, mean_speed_mps=0.)
    assert actual['in_1'] == dict(vehicles=0, halted=0, occupancy_raw=0., mean_speed_mps=11.11)
    assert api.measured == ['in_0', 'in_1', 'out_0']
    assert actual['out_0'] == observer.frames[0]['receiving_lanes']['out_0']


def test_extra_t0_measurements_leave_parent_event_and_receiving_accounting_unchanged():
    api = GetterAPI()
    base = MovementResponseObserver(api, infos(), 600, windows=(1,))
    joint = JointResponseObserver(api, infos(), 600, windows=(1,), focal_tls='tls')
    for time_sec, lane in [(600, 'in_0'), (601, 'out_0')]:
        api.rows = {'v': (lane, lane.rsplit('_', 1)[0])}
        base.sample(time_sec, {'tls': 'G'})
        joint.sample(time_sec, {'tls': 'G'})
    stripped = deepcopy(joint.frames)
    stripped[0].pop('focal_lane_measurements')
    assert stripped == base.frames
    assert joint.events == base.events
    assert [event['kind'] for event in joint.events] == ['stopline_entry', 'receiving_entry']
    assert joint.summary() == base.summary()
    assert joint.summary()['complete']


def test_future_samples_do_not_update_or_repeat_preaction_measurements():
    api = GetterAPI()
    observer = JointResponseObserver(api, infos(), 600, windows=(1,), focal_tls='tls')
    observer.sample(600)
    original = deepcopy(observer.frames[0]['focal_lane_measurements'])
    calls_at_t0 = list(api.measured)
    api.rows = {'v': ('out_0', 'out')}
    api.measurements['in_0'] = (0, 0., 11.11)
    observer.sample(601)
    assert observer.frames[0]['focal_lane_measurements'] == original
    assert 'focal_lane_measurements' not in observer.frames[1]
    assert api.measured == calls_at_t0
