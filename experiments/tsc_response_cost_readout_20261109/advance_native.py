"""Native10s source responses or450s target costs with exact step accounting."""
from collections import Counter
from math import isfinite



def _executors(executors):
    return {tls: dict(snapshot=executor.snapshot(), feasible_states=list(executor.feasible_states_now()),
                     minimum_green_sec=float(executor.timings[executor.current_green_state].min_green_sec))
            for tls, executor in sorted(executors.items())}


def _endpoint(api, executors, lane_ids, start):
    offset = float(api.simulation.getTime())-start
    return dict(offset_sec=offset, time_sec=float(api.simulation.getTime()),
        role='preaction_input' if offset == 0 else 'future_measurement_label',
        executors=_executors(executors),
        lanes={lane: dict(vehicle_count=int(api.lane.getLastStepVehicleNumber(lane)),
                         halted_count=int(api.lane.getLastStepHaltingNumber(lane)),
                         occupancy_raw=float(api.lane.getLastStepOccupancy(lane)),
                         mean_speed_mps=float(api.lane.getLastStepMeanSpeed(lane))) for lane in lane_ids})


def advance_horizon(api, v3, infos, graph, context, executors, initial_actions, lanes, ledger, observer, horizon_sec):
    """Root owns the prefix, initial observer sample, raw persistence and API close.

    An invalid step stops this measurement and returns its partial trace. Actual
    signal rows describe the state before each step, never the next executor state.
    """
    if horizon_sec not in (10, 450):
        raise ValueError("The frozen source/target horizons are10 and450 seconds")
    endpoints_at = (0, 10) if horizon_sec == 10 else (0, 10, 30, 60, 450)
    start = float(api.simulation.getTime())
    lanes = tuple(lanes)
    if not lanes or set(initial_actions) != set(executors) or set(infos) != set(executors):
        raise ValueError('A measured native interval needs controlled lanes and one request for every TLS')
    lane_ids = sorted(set(lanes) | set(observer.lane_geometry))
    active0 = sorted(api.vehicle.getIDList())
    pending0 = sorted(api.simulation.getPendingVehicles())
    endpoints = {'0': _endpoint(api, executors, lane_ids, start)}
    halted, costs, trace, signals, population_steps = [], [], [], [], []
    departed, arrived = [], []
    observed_samples, error = 0, None
    try:
        for step in range(1, horizon_sec+1):
            offset = step-1
            if float(api.simulation.getTime()) != start+offset:
                raise ValueError('Native measurement clock skipped a simulation step')
            if offset % 10 == 0:
                if offset == 0:
                    actions = dict(initial_actions)
                else:
                    states = v3._read_graph_states_v3(api, infos, context, graph)
                    actions = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
                               for tls in sorted(executors)}
                row = dict(offset_sec=offset, time_sec=start+offset, role='execution_measurement_label',
                    requests=actions, before=_executors(executors), request_returns={})
                trace.append(row)
                for tls in sorted(executors):
                    row['request_returns'][tls] = bool(executors[tls].request(actions[tls]))
                row['after'] = _executors(executors)
            applied = {tls: str(api.trafficlight.getRedYellowGreenState(tls)) for tls in sorted(executors)}
            signals.append(dict(interval_start_offset_sec=offset, interval_end_offset_sec=step,
                time_sec=start+offset, role='execution_measurement_label', applied_states=applied))
            raw = v3._advance_with_actions(sumo_api=api, executors=executors, actions={}, control_interval_sec=1,
                cost_lanes=lanes, collision_mode='audit', safety_ledger=ledger)
            if len(raw) != 1 or not isfinite(raw[0]) or raw[0] < 0 or abs(raw[0]-round(raw[0])) > 1e-6:
                raise ValueError('Native step did not return one integer controlled-lane halting count')
            if float(api.simulation.getTime()) != start+step:
                raise ValueError('Native measurement step did not advance exactly one second')
            halted.append(int(round(raw[0])))
            costs.append(halted[-1]/len(lanes))
            step_departed = sorted(api.simulation.getDepartedIDList())
            step_arrived = sorted(api.simulation.getArrivedIDList())
            departed.extend(step_departed); arrived.extend(step_arrived)
            population_steps.append(dict(offset_sec=step, departed_ids=step_departed, arrived_ids=step_arrived,
                active_count=len(api.vehicle.getIDList()), pending_count=len(api.simulation.getPendingVehicles())))
            if step <= 10:
                observer.sample(float(api.simulation.getTime()), applied_states=applied)
                observed_samples += 1
            if step in endpoints_at:
                endpoints[str(step)] = _endpoint(api, executors, lane_ids, start)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    active_end = sorted(api.vehicle.getIDList())
    pending_end = sorted(api.simulation.getPendingVehicles())
    final_time = float(api.simulation.getTime())
    integrals = {str(end): sum(halted[:end]) if len(halted) >= end else None for end in endpoints_at}
    segments = {f'{a}_{b}': integrals[str(b)]-integrals[str(a)]
                if integrals[str(a)] is not None and integrals[str(b)] is not None else None
                for a, b in zip(endpoints_at, endpoints_at[1:])}
    full = len(halted) == horizon_sec
    count_closure = len(active_end) == len(active0)+len(departed)-len(arrived)
    expected = Counter(active0)
    expected.update(departed); expected.subtract(arrived)
    identity_closure = expected == Counter(active_end)
    population = dict(initial_active_ids=active0, final_active_ids=active_end,
        initial_pending_ids=pending0, final_pending_ids=pending_end,
        departed_ids=departed, arrived_ids=arrived, per_second=population_steps,
        initial_active_count=len(active0), final_active_count=len(active_end),
        departed_count=len(departed), arrived_count=len(arrived),
        count_closure=count_closure, identity_closure=identity_closure)
    checks = dict(no_execution_error=error is None, full_cost_horizon=full,
        native_clock_exact=final_time == start+horizon_sec,
        request_epochs_complete=[row['offset_sec'] for row in trace] == list(range(0, horizon_sec, 10)) and
            all(set(row['request_returns']) == set(executors) and 'after' in row for row in trace),
        actual_signal_steps_complete=len(signals) == horizon_sec,
        population_steps_complete=len(population_steps) == horizon_sec,
        movement_observer_ten_poststeps=observed_samples == 10,
        endpoint_states_complete=set(endpoints) == {str(end) for end in endpoints_at},
        population_count_closure=count_closure, population_identity_closure=identity_closure,
        integral_closure=full and all(value is not None for value in segments.values()) and
            sum(segments.values()) == integrals[str(horizon_sec)],
        cost_unit_closure=full and abs(sum(costs)*len(lanes)-integrals[str(horizon_sec)]) <= 1e-6)
    return dict(start_time_sec=start, end_time_sec=final_time, completed_steps=len(halted),
        controlled_lane_count=len(lanes), cost=integrals[str(horizon_sec)]/(horizon_sec*len(lanes)) if full and horizon_sec == 450 else None,
        costs=costs, halted_counts=halted, integrals=integrals, segments=segments,
        execution_trace=trace, actual_signals=signals, actual_signals_10=signals[:10],
        endpoint_states=endpoints, population_ledger=population, checks=checks, error=error,
        horizon_sec=horizon_sec, cost_semantics='450-second mean halted vehicles per controlled lane for target only')
