"""Audit saved signal execution against occupancy clearance; not model inputs."""
from pathlib import Path
from xml.etree import ElementTree as ET


def read_clearance_lanes(sumocfg, tls):
    """Match SafePhaseExecutor's direct-via plus junction-prefix expansion."""
    config = Path(sumocfg)
    root = ET.parse(config).getroot()
    network = ET.parse(config.parent / root.find('./input/net-file').get('value')).getroot()
    via = {row.get('via') for row in network.findall('connection')
           if row.get('tl') == tls and row.get('via')}
    prefixes = {f"{lane.rsplit('_', 2)[0]}_" for lane in via
                if lane.startswith(':') and lane.count('_') >= 2}
    expanded = {row.get('id') for row in network.findall('./edge/lane')
                if any(row.get('id').startswith(prefix) for prefix in prefixes)}
    return sorted(via | expanded)


def _release_index(nominal):
    """Recognize the one-switch sequence used at eight-action eligible epochs."""
    release = next((i for i, state in enumerate(nominal) if any(c in 'GgsO' for c in state)), None)
    if release is None or release == 0:
        return None
    target = nominal[release]
    red = 'r' * len(target)
    if (any(state != target for state in nominal[release:]) or any(c in 'yY' for c in target)
            or nominal[release - 1] != red):
        return None
    yellow_end = next((i for i, state in enumerate(nominal[:release]) if state == red), None)
    if (yellow_end is None or yellow_end == 0 or not any(c in 'yY' for c in nominal[0])
            or any(c not in 'yYr' for c in nominal[0])
            or any(state != nominal[0] for state in nominal[:yellow_end])
            or any(state != red for state in nominal[yellow_end:release])):
        return None
    return release


def audit_execution(observation, nominal_states, clearance_lanes, *, tls=None):
    """Reconstruct the supported saved ten-step clearance transition.

    frame[t] records occupancy after advance at t; this is also the occupancy
    queried by that advance. Its result determines the signal used for step t,
    saved in frame[t+1].applied_states. Once green begins, later occupancy does
    not switch it back to red. Future occupancy is used only for this audit.
    """
    if tls is None:
        keys = list(observation['preaction']['nominal_states'])
        if len(keys) != 1:
            raise ValueError('Specify the focal TLS for a multi-signal observation')
        tls = keys[0]
    nominal = list(nominal_states)
    frames = observation['frames']
    if len(nominal) != 10 or [f['offset_sec'] for f in frames] != list(range(11)):
        return dict(expected_states=None, matches=False, mismatch_steps=[], nominal_mismatch_steps=[],
            clearance_counts=None, clearance_extension_steps=[], missing_clearance_lanes=[],
            explanation='Incomplete ten-step observation; no execution equivalence established.')
    actual = [frame['applied_states'][tls] for frame in frames[1:]]
    nominal_mismatches = [i for i, (a, b) in enumerate(zip(nominal, actual)) if a != b]
    clearance = set(clearance_lanes)
    missing = sorted(clearance - set(observation['lane_geometry']))
    counts = None if missing else [sum(v['lane_id'] in clearance for v in frame['vehicles'].values())
                                    for frame in frames]
    result = dict(expected_states=nominal, matches=not nominal_mismatches,
        mismatch_steps=nominal_mismatches, nominal_mismatch_steps=nominal_mismatches,
        clearance_counts=counts, clearance_extension_steps=[], missing_clearance_lanes=missing,
        explanation='Actual signals equal the nominal ten-step sequence.')
    if not nominal_mismatches:
        return result
    if missing:
        result.update(expected_states=None,
            explanation='Some executor clearance lanes were not observed; extensions cannot be established.')
        return result
    release = _release_index(nominal)
    if release is None:
        result.update(expected_states=None,
            explanation='Signal difference is outside the supported yellow/all-red/target-green transition.')
        return result
    expected = list(nominal)
    target, red = nominal[release], 'r' * len(nominal[release])
    released = False
    for step in range(release, 10):
        if not released and counts[step] > 0:
            expected[step] = red
            result['clearance_extension_steps'].append(step)
        else:
            released = True
            expected[step] = target
    mismatches = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    result.update(expected_states=expected, matches=not mismatches, mismatch_steps=mismatches,
        explanation='Every signal difference is explained by the saved occupancy-driven all-red extension.'
            if not mismatches else 'Actual signals differ from the occupancy-clearance execution rule.')
    return result
