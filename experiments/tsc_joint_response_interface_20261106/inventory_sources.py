#!/usr/bin/env python3
"""Static source geometry/program inventory; no demand, outcomes or simulation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median
from xml.etree import ElementTree as ET


def span(values):
    values = list(values)
    return dict(min=min(values), median=median(values), max=max(values)) if values else None


def min_green(phase, candidate):
    # SUMO defaults absent minDur to duration; explicit nonpositive values reach
    # SafePhaseExecutor's project fallback. First program, no runtime override.
    value = float(phase.get('minDur', phase.get('duration', '0')))
    duration = max(float(candidate.get('duration', '0')), 0.)
    return value if value > 0 else max(1., min(5., duration)) if duration > 0 else 5.


def inventory(sumocfg, scenario, city_group):
    config = ET.parse(sumocfg).getroot()
    network = (sumocfg.parent / config.find('./input/net-file').get('value')).resolve()
    root = ET.parse(network).getroot()
    lanes = {(edge.get('id'), lane.get('index')): lane
             for edge in root.findall('edge') for lane in edge.findall('lane')}
    destinations, movements, widths, tls_lanes = defaultdict(set), {}, defaultdict(int), defaultdict(set)
    for c in root.findall('connection'):
        if c.get('tl') is None:
            continue
        tls = c.get('tl'); widths[tls] = max(widths[tls], int(c.get('linkIndex')) + 1)
        if c.get('from').startswith(':'):
            continue
        lane = (c.get('from'), c.get('fromLane'))
        destinations[lane].add(c.get('to')); tls_lanes[tls].add(lane)
        key = (tls, c.get('from'), c.get('to'))
        m = movements.setdefault(key, {'incoming': set(), 'outgoing': set(), 'links': set()})
        m['incoming'].add(lane); m['outgoing'].add((c.get('to'), c.get('toLane'))); m['links'].add(int(c.get('linkIndex')))
    dedicated = {key: all(len(destinations[lane]) == 1 for lane in m['incoming']) for key, m in movements.items()}
    programs = {}
    for logic in root.findall('tlLogic'):
        programs.setdefault(logic.get('id'), logic)
    stable_counts, effective, yellow, all_red, missing_durations = [], [], [], [], []
    explicit_positive = explicit_nonpositive = absent = multiple_program_tls = 0
    for tls, logic in programs.items():
        phases = logic.findall('phase'); last_phase = {p.get('state'): p for p in phases}; candidates = {}
        multiple_program_tls += sum(p.get('id') == tls for p in root.findall('tlLogic')) > 1
        for phase in phases:
            state = phase.get('state'); duration = float(phase.get('duration', '0'))
            if any(c in 'yY' for c in state): yellow.append(duration)
            if state and not any(c in 'GgsOyY' for c in state): all_red.append(duration)
            if len(state) == widths.get(tls) and any(c in 'Gg' for c in state) and not any(c in 'yY' for c in state):
                candidates.setdefault(state, phase)
        stable_counts.append(len(candidates))
        for state, candidate in candidates.items():
            phase = last_phase[state]; effective.append(min_green(phase, candidate))
            if 'minDur' not in phase.attrib:
                absent += 1; missing_durations.append(float(phase.get('duration', '0')))
            elif float(phase.get('minDur')) > 0: explicit_positive += 1
            else: explicit_nonpositive += 1
    overlap_tls = [tls for tls in sorted(tls_lanes)
        if any(dedicated[key] for key in movements if key[0] == tls)
        and any(400 <= float(lanes[lane].get('length')) <= 800 for lane in tls_lanes[tls])]
    return dict(scenario=scenario, city_group=city_group, sumocfg=str(sumocfg), net_file=str(network),
        version_role='network_at_recorded_sumocfg_path',
        network_xml_version=root.get('version'), tls_count=len(programs), controlled_tls_count=len(tls_lanes),
        external_controlled_movement_count=len(movements), external_controlled_incoming_lane_count=len(destinations),
        dedicated_incoming_lane_count=sum(len(v) == 1 for v in destinations.values()),
        dedicated_incoming_lane_fraction=sum(len(v) == 1 for v in destinations.values()) / len(destinations),
        dedicated_movement_count=sum(dedicated.values()), dedicated_movement_fraction=sum(dedicated.values())/len(movements),
        incoming_length_m=span(float(lanes[l].get('length')) for l in destinations),
        incoming_speed_limit_mps=span(float(lanes[l].get('speed')) for l in destinations),
        movement_link_count=span(len(m['links']) for m in movements.values()),
        movement_outgoing_lane_count=span(len(m['outgoing']) for m in movements.values()),
        first_program_unique_stable_phase_count=span(stable_counts), first_program_stable_phase_total=sum(stable_counts),
        stable_phase_minDur_explicit_positive_count=explicit_positive,
        stable_phase_minDur_explicit_nonpositive_count=explicit_nonpositive,
        stable_phase_minDur_absent_count=absent, absent_minDur_phase_duration_sec=span(missing_durations),
        static_effective_min_green_sec_estimate=span(effective), first_program_yellow_phase_duration_sec=span(yellow),
        first_program_all_red_phase_duration_sec=span(all_red), tls_with_multiple_programs=multiple_program_tls,
        tls_with_dedicated_movement_and_target_length_lane=overlap_tls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); manifest = args.manifest.resolve(); spec = json.loads(manifest.read_text())
    result = dict(protocol='static-source-physical-coverage-inventory-v1', new_fits=0, new_simulations=0,
        manifest=str(manifest), source_scenario_count=len(spec['scenarios']),
        dedicated_semantics='An external incoming lane has exactly one distinct outgoing EDGE across controlled connections; a movement is dedicated only if all its incoming lanes are dedicated.',
        timing_semantics='Static estimate assumes first program and no additional-file/runtime overrides. Absent minDur uses SUMO duration default; explicit nonpositive minDur uses project fallback. XML duration is not unconditionally the executor minimum.',
        overlap_semantics='TLS IDs satisfy both at least one dedicated movement and at least one controlled incoming lane400–800m. These may be different approaches. Observed target-range overlap only, no source selection or admission score.',
        scope='Existing18source-network static metadata only; no demand scans, trajectories, outcome ranking or operational admission.',
        scenarios=[inventory((manifest.parent/r['sumocfg']).resolve(), r['scenario'], r['city_group']) for r in spec['scenarios']])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'source_scenarios':len(result['scenarios']), 'new_fits':0, 'new_simulations':0,
        'output':str(args.output), 'output_bytes':args.output.stat().st_size}))
