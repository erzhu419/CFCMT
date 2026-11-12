"""Static movement and same-junction priority audit. Run with SUMO tools on PYTHONPATH."""

from collections import defaultdict
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import sumolib

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / 'tsc_joint_response_interface_20261106'))
from inventory_sources import min_green
from static_context import _conflicts


def stable_phases(program, width):
    phases = program.findall('phase')
    last = {p.get('state'): p for p in phases}
    seen = set()
    for i, phase in enumerate(phases):
        state = phase.get('state')
        if (
            state in seen
            or len(state) != width
            or 'y' in state.lower()
            or not any(signal in 'Gg' for signal in state)
        ):
            continue
        seen.add(state)
        yield len(seen) - 1, i, state, min_green(last[state], phase)


def movement(c):
    return '|'.join((c.getTLSID(), c.getFrom().getID(), c.getTo().getID()))


def exit_type(c, p):
    if c.getToLane() == p.getToLane():
        return 'same_lane'
    if c.getTo() == p.getTo():
        return 'same_edge_other_lane'
    return 'different_edge'


def actual_priority(own, others):
    """Keep physical connections distinct when several use the same TLS index."""
    junction = own.getJunction()
    i = own.getJunctionIndex()
    for other in others:
        if other.getJunction() != junction or movement(other) == movement(own):
            continue
        j = other.getJunctionIndex()
        if (
            i >= 0
            and j >= 0
            and junction.forbids(other, own)
            and (junction.areFoes(i, j) or junction.areFoes(j, i))
        ):
            yield other


def inspect_tls(tls, program, connections, context):
    dests = defaultdict(set)
    by_move = defaultdict(list)
    for c in connections:
        dests[c.getFromLane()].add(c.getTo().getID())
        by_move[movement(c)].append(c)
    phases = list(stable_phases(program, 1 + max(c.getTLLinkIndex() for c in connections)))
    attrs = {}
    for m, entries in by_move.items():
        lanes = sorted({c.getFromLane() for c in entries}, key=lambda l: l.getID())
        attrs[m] = dict(
            turn=''.join(sorted({c.getDirection() for c in entries})),
            dedicated=all(len(dests[l]) == 1 for l in lanes),
            incoming_lanes=[l.getID() for l in lanes],
            incoming_length_m=[l.getLength() for l in lanes],
            incoming_speed_mps=[l.getSpeed() for l in lanes],
        )
    priority = {
        c: list(actual_priority(c, connections))
        for c in connections
        if any(state[c.getTLLinkIndex()] in 'gs' for _, _, state, _ in phases)
    }
    projected_pairs = {tuple(pair) for pair in context['response']}
    modes = {}
    for mode in ('g', 's'):
        exposed, rows = set(), []
        for candidate, phase_index, state, minimum in phases:
            for m, entries in by_move.items():
                own = [
                    c
                    for c in entries
                    if state[c.getTLLinkIndex()] == mode and (mode == 'g' or c.getDirection() == 'r')
                ]
                if not own:
                    continue
                exposed.add(m)
                pairs = {(c, p) for c in own for p in priority[c] if state[p.getTLLinkIndex()] == 'G'}
                if not pairs:
                    continue
                # Detect orientation or index-space differences with the existing graph reader.
                for c, p in pairs:
                    assert (c.getTLLinkIndex(), p.getTLLinkIndex()) in projected_pairs
                rows.append(
                    dict(
                        m=m,
                        candidate=candidate,
                        phase=phase_index,
                        min_green_sec=minimum,
                        pairs=sorted(
                            {
                                (movement(p), exit_type(c, p), c.getTLLinkIndex(), p.getTLLinkIndex())
                                for c, p in pairs
                            }
                        ),
                    )
                )
        if exposed:
            modes[mode] = {'exposed': sorted(exposed), 'rows': rows}
    return dict(
        tls_id=tls,
        program_id=program.get('programID'),
        candidate_count=len(phases),
        full_context_complete=context['full_context_complete'],
        external_foe_connection_count=context['external_foe_connection_count'],
        external_response_connection_count=context['external_response_connection_count'],
        missing_controlled_request_count=context['missing_controlled_request_count'],
        unresolved_conflicting_request_count=context['unresolved_conflicting_request_count'],
        attrs=attrs,
        modes=modes,
    )


def summarize(rows, name):
    stats, witnesses, movements = {}, [], {}
    all_attributes = {m: a for tls in rows for m, a in tls["attrs"].items()}
    # Count movements once regardless of repeated candidate states or signal indices.
    for mode in ('g', 's'):
        counts = defaultdict(set)
        keyed = defaultdict(set)
        examples = {}
        for t in rows:
            if mode not in t['modes']:
                continue
            attrs = t['attrs']
            data = t['modes'][mode]
            counts['exposed'].update(data['exposed'])
            for m in data['exposed']:
                keyed['exposed_turn_' + attrs[m]['turn']].add(m)
            for row in data['rows']:
                m = row['m']
                counts['yielding_to_simultaneous_G_foe'].add(m)
                if not t['full_context_complete']:
                    continue
                counts['complete'].add(m)
                if attrs[m]['dedicated']:
                    counts['complete_own_dedicated'].add(m)
                    if all(attrs[p]['dedicated'] for p, _, _, _ in row['pairs']):
                        counts['complete_all_dedicated'].add(m)
                        if t['candidate_count'] == 8:
                            counts['eight_candidate_complete_all_dedicated'].add(m)
                for p, geometry, i, j in row['pairs']:
                    direction = f"{attrs[m]['turn']}>{attrs[p]['turn']}"
                    key = f'{direction}/{geometry}'
                    keyed[key].add((m, p))
                    if attrs[m]['dedicated'] and attrs[p]['dedicated']:
                        keyed['dedicated_pair/' + key].add((m, p))
                    sample = dict(
                        mode=mode,
                        own=m,
                        priority=p,
                        geometry=geometry,
                        tls=t['tls_id'],
                        candidate=row['candidate'],
                        phase=row['phase'],
                        min_green_sec=row['min_green_sec'],
                        signal_links=[i, j],
                        both_dedicated=attrs[m]['dedicated'] and attrs[p]['dedicated'],
                    )
                    # Keep one compact physical witness per turn/geometry/dedication in non-SaltLake assets,
                    # and per unique movement pair for the two eight-action SaltLake candidates.
                    exemplar = (
                        (m, p, geometry)
                        if name.startswith('saltlake')
                        else (key, attrs[m]['dedicated'], attrs[p]['dedicated'])
                    )
                    examples.setdefault(exemplar, sample)
        for example in examples.values():
            for m in (example['own'], example['priority']):
                movements[m] = all_attributes[m]
        witnesses.extend(examples.values())
        stats[mode] = dict(
            movement_counts={
                k: len(counts[k])
                for k in (
                    'exposed',
                    'yielding_to_simultaneous_G_foe',
                    'complete',
                    'complete_own_dedicated',
                    'complete_all_dedicated',
                    'eight_candidate_complete_all_dedicated',
                )
            },
            turn_and_exit_counts={k: len(v) for k, v in sorted(keyed.items())},
        )
    return dict(
        stats=stats,
        tls=[{k: v for k, v in row.items() if k not in ('attrs', 'modes')} for row in rows if row['modes']],
        witness_movements=movements,
        witnesses=witnesses,
    )


def inspect_network(scenario):
    net_path = Path(scenario['net_file'])
    root = ET.parse(net_path).getroot()
    requests = {
        (j.get('id'), int(r.get('index'))): r.attrib
        for j in root.findall('junction')
        for r in j.findall('request')
    }
    net = sumolib.net.readNet(str(net_path), withInternal=True, withFoes=True)
    connections = [c for e in net.getEdges(withInternal=True) for l in e.getLanes() for c in l.getOutgoing()]
    contexts = _conflicts(connections, requests, net_path)
    controlled = defaultdict(list)
    for c in connections:
        if c.getTLSID():
            controlled[c.getTLSID()].append(c)
    programs = {}
    for p in root.findall('tlLogic'):
        programs.setdefault(p.get('id'), p)
    rows = [inspect_tls(t, p, controlled[t], contexts[t]) for t, p in programs.items()]
    return dict(
        scenario=scenario['scenario'],
        source_cfg=scenario['sumocfg'],
        net_file=str(net_path),
        **summarize(rows, scenario['scenario']),
    )


if __name__ == '__main__':
    inventory = json.loads(
        (EXPERIMENTS / 'tsc_joint_response_interface_20261106/source_inventory.json').read_text()
    )
    signal = json.loads(
        (EXPERIMENTS / 'tsc_joint_response_trial_20261106/source_signal_inventory.json').read_text()
    )
    active = {
        r['scenario']
        for r in signal['scenarios']
        if any(c in r['first_program_stable_signal_chars'] for c in ('g', 's'))
    }
    out = dict(
        scope='Existing first-program source network geometry and signals; zero simulation, no demand or response measurements.',
        modes={
            'g': 'Any external permissive-green movement',
            's': 'External right turn with stop-then-yield s signal',
        },
        counts='Movement = TLS|incoming edge|outgoing edge. Paired counts deduplicate ordered movement pairs per turn and exit type. Exposed turns count movements. The same pair can have different geometries through different physical connections.',
        geometry='Based on directed junction-request priority and physical foe bits for the exact two connections active as g/s and G; distinguishes same exit lane, different lane of same exit edge, and different exit edges. Complete uses full prior graph-context gate including any uncontrolled conflicts. Turn direction s in direction pairs means straight. Signal mode s means stop before turning right.',
        both_dedicated='Both movements use only incoming lanes with one controlled destination edge. All-dedicated movement counts require this for all protected counterparts in at least one phase.',
        new_fits=0,
        new_simulations=0,
        input_scenarios=len(inventory['scenarios']),
        screened_out_no_g_or_s=[r['scenario'] for r in inventory['scenarios'] if r['scenario'] not in active],
        scenarios=[inspect_network(s) for s in inventory['scenarios'] if s['scenario'] in active],
    )
    (HERE / 'source_static.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    for s in out['scenarios']:
        print(s['scenario'], {mode: r['movement_counts'] for mode, r in s['stats'].items()})
