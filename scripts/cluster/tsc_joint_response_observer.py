"""Direct focal-lane measurements at the native preaction checkpoint."""
from __future__ import annotations

try:
    from .tsc_movement_response import MovementResponseObserver
except ImportError:
    from tsc_movement_response import MovementResponseObserver


class JointResponseObserver(MovementResponseObserver):
    """Preserve movement accounting and add direct t0 input/output lane data."""

    def __init__(self, api, infos, checkpoint_sec, windows=(10,), *, focal_tls):
        super().__init__(api, infos, checkpoint_sec, windows=windows)
        self.focal_tls = str(focal_tls)
        self.focal_lanes = sorted({lane for movement in self.movements.values()
            if movement['tls_id'] == self.focal_tls
            for role in ('incoming_lanes', 'outgoing_lanes') for lane in movement[role]})
        if not self.focal_lanes:
            raise ValueError('The focal TLS has no observed movements')

    def sample(self, time_sec, applied_states=None):
        super().sample(time_sec, applied_states=applied_states)
        frame = self.frames[-1]
        if frame['offset_sec'] == 0:
            api = self.api
            frame['focal_lane_measurements'] = {lane: dict(
                vehicles=int(api.lane.getLastStepVehicleNumber(lane)),
                halted=int(api.lane.getLastStepHaltingNumber(lane)),
                occupancy_raw=float(api.lane.getLastStepOccupancy(lane)),
                mean_speed_mps=float(api.lane.getLastStepMeanSpeed(lane)))
                for lane in self.focal_lanes}
