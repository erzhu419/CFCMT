"""Safe phase execution for externally controlled SUMO traffic lights.

SUMO stops advancing a signal program after ``setRedYellowGreenState`` is
called.  Any external controller using that API must therefore implement its
own clearance transitions.  This module is the sole phase-write path for the
new benchmark and applies the same minimum-green, yellow, and all-red rules to
learned and non-learned controllers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


ACTIVE_CHARS = frozenset("GgsO")
YELLOW_CHARS = frozenset("yY")


@dataclass(frozen=True)
class PhaseTiming:
    """Execution timing attached to one feasible green phase."""

    state: str
    min_green_sec: float
    yellow_sec: float
    all_red_sec: float


@dataclass
class PhaseExecutionAudit:
    """Counters needed to audit whether a guard actually changes actions."""

    requests: int = 0
    accepted_requests: int = 0
    same_phase_requests: int = 0
    rejected_busy: int = 0
    rejected_min_green: int = 0
    switches: int = 0
    green_seconds: float = 0.0
    yellow_seconds: float = 0.0
    all_red_seconds: float = 0.0
    occupancy_clearance_extensions: int = 0
    occupancy_clearance_extension_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class SafePhaseExecutor:
    """State machine that executes feasible phases with explicit clearance.

    A request can start only while the executor is in a green phase and that
    phase has satisfied its minimum-green duration.  Switching first retires
    every currently permissive signal, then applies an all-red interval, and
    only then activates the requested target phase.
    """

    def __init__(
        self,
        *,
        sumo_api: Any,
        tls_id: str,
        timings: Sequence[PhaseTiming],
        initial_state: str,
        initial_green_elapsed_sec: float = 0.0,
        clearance_lane_ids: Sequence[str] = (),
        clearance_poll_sec: float = 1.0,
    ) -> None:
        if not timings:
            raise ValueError("SafePhaseExecutor requires at least one feasible phase")
        state_lengths = {len(item.state) for item in timings}
        if len(state_lengths) != 1 or len(initial_state) not in state_lengths:
            raise ValueError("all phase states and initial_state must have equal length")
        if len({item.state for item in timings}) != len(timings):
            raise ValueError("phase states must be unique")

        self.sumo_api = sumo_api
        self.tls_id = str(tls_id)
        self.timings = {item.state: item for item in timings}
        self.audit = PhaseExecutionAudit()
        self.clearance_lane_ids = tuple(
            dict.fromkeys(str(lane_id) for lane_id in clearance_lane_ids if str(lane_id))
        )
        self.clearance_poll_sec = max(float(clearance_poll_sec), 1e-3)

        self.mode = "green"
        self.current_state = str(initial_state)
        self.current_green_state = self._closest_feasible_state(initial_state)
        self.target_green_state = self.current_green_state
        self.green_elapsed_sec = max(float(initial_green_elapsed_sec), 0.0)
        self.remaining_sec = 0.0
        self._pending_all_red_sec = 0.0

    @property
    def is_switching(self) -> bool:
        return self.mode != "green"

    @property
    def feasible_states(self) -> tuple[str, ...]:
        return tuple(self.timings)

    @property
    def can_switch(self) -> bool:
        if self.is_switching:
            return False
        timing = self.timings[self.current_green_state]
        return self.green_elapsed_sec + 1e-9 >= timing.min_green_sec

    def feasible_states_now(self) -> tuple[str, ...]:
        """Return the action set that can be executed at the current instant."""

        if self.is_switching:
            return (self.target_green_state,)
        if not self.can_switch:
            return (self.current_green_state,)
        return self.feasible_states

    def projected_clearance_sec(self, target_state: str) -> float:
        """Return yellow plus all-red time for a feasible requested switch."""

        target_state = str(target_state)
        if target_state not in self.timings:
            raise KeyError(f"unknown phase state for {self.tls_id}: {target_state!r}")
        if target_state == self.current_green_state and not self.is_switching:
            return 0.0
        current = self.timings[self.current_green_state]
        target = self.timings[target_state]
        yellow = max(float(current.yellow_sec), float(target.yellow_sec), 0.0)
        all_red = max(float(current.all_red_sec), float(target.all_red_sec), 0.0)
        return yellow + all_red

    def request(self, target_state: str) -> bool:
        """Request a feasible phase and return whether a switch was accepted."""

        target_state = str(target_state)
        self.audit.requests += 1
        if target_state not in self.timings:
            raise KeyError(f"unknown phase state for {self.tls_id}: {target_state!r}")
        if self.is_switching:
            self.audit.rejected_busy += 1
            return False
        if target_state == self.current_green_state:
            self.audit.same_phase_requests += 1
            return False

        current_timing = self.timings[self.current_green_state]
        if not self.can_switch:
            self.audit.rejected_min_green += 1
            return False

        self.audit.accepted_requests += 1
        self.audit.switches += 1
        self.target_green_state = target_state
        target_timing = self.timings[target_state]
        yellow_state = _clearance_yellow_state(self.current_state)
        yellow_sec = max(float(current_timing.yellow_sec), float(target_timing.yellow_sec), 0.0)
        self._pending_all_red_sec = max(
            float(current_timing.all_red_sec),
            float(target_timing.all_red_sec),
            0.0,
        )

        if yellow_sec > 0.0 and any(char in YELLOW_CHARS for char in yellow_state):
            self._set_state(yellow_state)
            self.mode = "yellow"
            self.remaining_sec = yellow_sec
        elif self._pending_all_red_sec > 0.0:
            self._enter_all_red()
        else:
            self._enter_target_green()
        return True

    def take_control(self) -> None:
        """Freeze SUMO's native program at the executor's tracked state."""

        self._set_state(self.current_state)

    def advance(self, elapsed_sec: float) -> None:
        """Advance the state machine after SUMO has simulated ``elapsed_sec``."""

        remaining = max(float(elapsed_sec), 0.0)
        while remaining > 1e-12:
            if self.mode == "green":
                self.green_elapsed_sec += remaining
                self.audit.green_seconds += remaining
                return

            consumed = min(remaining, self.remaining_sec)
            if self.mode == "yellow":
                self.audit.yellow_seconds += consumed
            elif self.mode == "all_red":
                self.audit.all_red_seconds += consumed
            self.remaining_sec -= consumed
            remaining -= consumed

            if self.remaining_sec > 1e-12:
                return
            if self.mode == "yellow" and self._pending_all_red_sec > 0.0:
                self._enter_all_red()
            elif self.mode == "all_red" and self._clearance_is_occupied():
                self.remaining_sec = self.clearance_poll_sec
                self.audit.occupancy_clearance_extensions += 1
                self.audit.occupancy_clearance_extension_seconds += self.clearance_poll_sec
            else:
                self._enter_target_green()

    def snapshot(self) -> dict[str, Any]:
        return {
            "tls_id": self.tls_id,
            "mode": self.mode,
            "current_state": self.current_state,
            "current_green_state": self.current_green_state,
            "target_green_state": self.target_green_state,
            "green_elapsed_sec": float(self.green_elapsed_sec),
            "remaining_sec": float(self.remaining_sec),
            "pending_all_red_sec": float(self._pending_all_red_sec),
            "audit": self.audit.to_dict(),
        }

    def restore(self, snapshot: Mapping[str, Any], *, write_state: bool = True) -> None:
        """Restore a runtime snapshot after a SUMO state reload.

        SUMO state files retain the online signal-program reference but do not
        reliably serialize phase-state mutations made through
        ``setRedYellowGreenState``. Counterfactual branching must therefore
        restore this state machine and explicitly rewrite the saved signal
        state after every ``loadState``.
        """

        if str(snapshot.get("tls_id")) != self.tls_id:
            raise ValueError(f"snapshot belongs to a different TLS: {snapshot.get('tls_id')!r}")
        mode = str(snapshot["mode"])
        current_state = str(snapshot["current_state"])
        current_green_state = str(snapshot["current_green_state"])
        target_green_state = str(snapshot["target_green_state"])
        if mode not in {"green", "yellow", "all_red"}:
            raise ValueError(f"unknown executor mode in snapshot: {mode!r}")
        if current_green_state not in self.timings or target_green_state not in self.timings:
            raise ValueError("snapshot references a phase unavailable to this executor")
        if len(current_state) != len(self.current_state):
            raise ValueError("snapshot signal state has a different width")

        self.mode = mode
        self.current_state = current_state
        self.current_green_state = current_green_state
        self.target_green_state = target_green_state
        self.green_elapsed_sec = max(float(snapshot["green_elapsed_sec"]), 0.0)
        self.remaining_sec = max(float(snapshot["remaining_sec"]), 0.0)
        self._pending_all_red_sec = max(float(snapshot.get("pending_all_red_sec", 0.0)), 0.0)
        audit_values = dict(snapshot.get("audit", {}))
        self.audit = PhaseExecutionAudit(
            **{
                name: audit_values.get(name, field.default)
                for name, field in PhaseExecutionAudit.__dataclass_fields__.items()
            }
        )
        if write_state:
            self._set_state(current_state)

    def _closest_feasible_state(self, state: str) -> str:
        if state in self.timings:
            return state
        return max(self.timings, key=lambda candidate: _active_overlap(state, candidate))

    def _set_state(self, state: str) -> None:
        self.sumo_api.trafficlight.setRedYellowGreenState(self.tls_id, state)
        self.current_state = state

    def _clearance_is_occupied(self) -> bool:
        for lane_id in self.clearance_lane_ids:
            try:
                if float(self.sumo_api.lane.getLastStepVehicleNumber(lane_id)) > 0.0:
                    return True
            except Exception:
                continue
        return False

    def _enter_all_red(self) -> None:
        self._set_state("r" * len(self.current_state))
        self.mode = "all_red"
        self.remaining_sec = self._pending_all_red_sec
        self._pending_all_red_sec = 0.0

    def _enter_target_green(self) -> None:
        self._set_state(self.target_green_state)
        self.mode = "green"
        self.current_green_state = self.target_green_state
        self.green_elapsed_sec = 0.0
        self.remaining_sec = 0.0
        self._pending_all_red_sec = 0.0


def build_safe_phase_executors(
    sumo_api: Any,
    infos: Mapping[str, Any],
    *,
    default_min_green_sec: float = 5.0,
    default_yellow_sec: float = 3.0,
    default_all_red_sec: float = 1.0,
) -> dict[str, SafePhaseExecutor]:
    """Build one executor per TLS from the active SUMO program logic."""

    executors: dict[str, SafePhaseExecutor] = {}
    for tls_id in sorted(infos):
        info = infos[tls_id]
        logics = list(sumo_api.trafficlight.getAllProgramLogics(tls_id))
        phases = list(logics[0].phases) if logics else []
        yellow_values = [
            _phase_duration(phase)
            for phase in phases
            if any(char in YELLOW_CHARS for char in str(getattr(phase, "state", "")))
        ]
        all_red_values = [
            _phase_duration(phase)
            for phase in phases
            if _is_all_red(str(getattr(phase, "state", "")))
        ]
        yellow_sec = float(median(yellow_values)) if yellow_values else float(default_yellow_sec)
        all_red_sec = float(median(all_red_values)) if all_red_values else float(default_all_red_sec)
        phase_by_state = {str(getattr(phase, "state", "")): phase for phase in phases}
        clearance_lane_ids = _junction_clearance_lane_ids(
            sumo_api,
            info.controlled_links,
        )
        timings = []
        for candidate in info.candidates:
            phase = phase_by_state.get(str(candidate.state))
            min_green = _phase_min_green(phase, candidate, default_min_green_sec)
            timings.append(
                PhaseTiming(
                    state=str(candidate.state),
                    min_green_sec=min_green,
                    yellow_sec=yellow_sec,
                    all_red_sec=all_red_sec,
                )
            )

        initial_state = str(sumo_api.trafficlight.getRedYellowGreenState(tls_id))
        try:
            initial_elapsed = float(sumo_api.trafficlight.getSpentDuration(tls_id))
        except Exception:
            initial_elapsed = float(default_min_green_sec)
        executor = SafePhaseExecutor(
            sumo_api=sumo_api,
            tls_id=str(tls_id),
            timings=timings,
            initial_state=initial_state,
            initial_green_elapsed_sec=initial_elapsed,
            clearance_lane_ids=clearance_lane_ids,
        )
        executor.take_control()
        executors[str(tls_id)] = executor
    return executors


def aggregate_phase_audits(executors: Mapping[str, SafePhaseExecutor]) -> dict[str, float | int]:
    fields = tuple(PhaseExecutionAudit.__dataclass_fields__)
    return {
        field: sum(
            getattr(executors[tls_id].audit, field)
            for tls_id in sorted(executors)
        )
        for field in fields
    }


def _junction_clearance_lane_ids(
    sumo_api: Any,
    controlled_links: Sequence[Any],
) -> tuple[str, ...]:
    direct_via_lanes = []
    for link_group in controlled_links:
        for link in link_group or ():
            if len(link) >= 3 and str(link[2]):
                direct_via_lanes.append(str(link[2]))
    prefixes = {
        f"{via_lane.rsplit('_', 2)[0]}_"
        for via_lane in direct_via_lanes
        if via_lane.startswith(":") and via_lane.count("_") >= 2
    }
    expanded = list(direct_via_lanes)
    try:
        all_lane_ids = tuple(str(value) for value in sumo_api.lane.getIDList())
    except Exception:
        all_lane_ids = ()
    expanded.extend(
        lane_id
        for lane_id in sorted(all_lane_ids)
        if any(lane_id.startswith(prefix) for prefix in prefixes)
    )
    return tuple(dict.fromkeys(expanded))


def _phase_duration(phase: Any) -> float:
    return max(float(getattr(phase, "duration", 0.0) or 0.0), 0.0)


def _phase_min_green(phase: Any, candidate: Any, default: float) -> float:
    if phase is not None:
        min_dur = getattr(phase, "minDur", None)
        if min_dur is not None and float(min_dur) > 0.0:
            return float(min_dur)
    duration = max(float(getattr(candidate, "duration", 0.0) or 0.0), 0.0)
    if duration > 0.0:
        return max(1.0, min(float(default), duration))
    return max(float(default), 1.0)


def _is_all_red(state: str) -> bool:
    return bool(state) and not any(char in ACTIVE_CHARS or char in YELLOW_CHARS for char in state)


def _clearance_yellow_state(state: str) -> str:
    chars = []
    for char in state:
        if char == "G":
            chars.append("y")
        elif char == "g":
            chars.append("y")
        else:
            chars.append("r")
    return "".join(chars)


def _active_overlap(left: str, right: str) -> int:
    return sum(
        int(a in ACTIVE_CHARS and b in ACTIVE_CHARS)
        for a, b in zip(left, right)
    )
