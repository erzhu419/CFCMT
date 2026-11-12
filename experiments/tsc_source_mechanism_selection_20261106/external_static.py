"""Read the configured Bologna and MoST networks and signal programs, without simulation."""
import json
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import sumolib
from source_static import actual_priority, exit_type, movement, stable_phases

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / 'H2Oplus/downloads/traffic_signal_external'


def inspect(cfg):
    config = ET.parse(cfg)
    network = cfg.parent / config.find('./input/net-file').get('value')
    root = ET.parse(network)
    files = config.find('./input/additional-files').get('value').split(',')
    programs = [(network, p) for p in root.findall('tlLogic')]
    for f in files:
        path = cfg.parent / f
        programs.extend((path, p) for p in ET.parse(path).findall('tlLogic'))
    net = sumolib.net.readNet(str(network), withInternal=True, withFoes=True, withPedestrianConnections=True)
    widths = defaultdict(int)
    for c in root.iter('connection'):
        if c.get('tl'):
            tls = c.get('tl'); widths[tls] = max(widths[tls], int(c.get('linkIndex'))+1)
    by_tls = defaultdict(list)
    destinations = defaultdict(set)
    for edge in net.getEdges():
        if edge.getFunction() in ('internal', 'crossing', 'walkingarea'):
            continue
        for lane in edge.getLanes():
            for c in lane.getOutgoing():
                destinations[lane].add(c.getTo().getID())
                if c.getTLSID():
                    by_tls[c.getTLSID()].append(c)
    out = []
    for origin, program in programs:
        tls = program.get('id')
        phases = list(stable_phases(program,widths[tls]))
        moves = defaultdict(list)
        for c in by_tls[tls]:
            moves[movement(c)].append(c)
        own_candidates = [c for c in by_tls[tls] if c.getDirection() == 'r' and any(s[c.getTLLinkIndex()]=='g' for _,_,s,_ in phases)]
        opponents = [c for c in by_tls[tls] if c.getDirection() == 's']
        pairs = defaultdict(set)
        winners = []
        unmapped = sum(c.getJunctionIndex() < 0 for c in own_candidates + opponents)
        for c in own_candidates:
            for p in actual_priority(c,opponents):
                kind = exit_type(c,p)
                on = [k for k,_,s,_ in phases if s[c.getTLLinkIndex()]=='g' and s[p.getTLLinkIndex()]=='G']
                if not on or kind == 'different_edge':
                    continue
                m,n = movement(c),movement(p)
                pairs['co_service'].add((m,n))
                off = [k for k,_,s,_ in phases if s[c.getTLLinkIndex()]=='g' and s[p.getTLLinkIndex()]!='G']
                if off:
                    pairs['switch'].add((m,n))
                dedicated = all(len(destinations[x.getFromLane()])==1 for x in moves[m]+moves[n])
                if not dedicated:
                    continue
                pairs['both_dedicated'].add((m,n))
                if off:
                    pairs['both_dedicated_and_switch'].add((m,n))
                winners.append(dict(own=m,priority=n,outgoing=kind,links=[c.getTLLinkIndex(),p.getTLLinkIndex()],
                    on=on,off=off,all_own_links_g_on=[k for k,_,s,_ in phases if k in on and all(s[a.getTLLinkIndex()]=='g' for a in moves[m])],
                    all_own_links_g_off=[k for k,_,s,_ in phases if k in off and all(s[a.getTLLinkIndex()]=='g' for a in moves[m])]))
        out.append(dict(tls=tls, program=program.get('programID'), origin=origin.name, stable_actions=len(phases),
            g_right_movements=len({movement(c) for c in own_candidates}), unmapped_right_g_or_straight_connections=unmapped,
            same_exit_g_right_to_G_straight_pairs=len(pairs['co_service']), both_dedicated_pairs=len(pairs['both_dedicated']),
            same_exit_pairs_with_stable_priority_switch=len(pairs['switch']),
            both_dedicated_with_stable_priority_switch=len(pairs['both_dedicated_and_switch']), witnesses=winners))
    return dict(config=str(cfg.relative_to(ASSETS)), network=str(network.relative_to(ASSETS)), programs_examined=len(out),
        programs_by_origin={p.name:dict(Counter(row['program'] for row in out if row['origin']==p.name)) for p in sorted({f for f,_ in programs})},
        tls_programs_with_external_g_right=sum(r['g_right_movements']>0 for r in out),
        unmapped_right_g_or_straight_connections=sum(r['unmapped_right_g_or_straight_connections'] for r in out),
        matched_tls_programs=[r for r in out if r['same_exit_g_right_to_G_straight_pairs']])


if __name__=='__main__':
    bol = ASSETS/'sumo-scenarios/bologna'
    most = ASSETS/'MoSTScenario'
    cfgs = sorted(bol.glob('*/run.sumocfg'))+[most/'scenario/most.sumocfg',most/'tools/most.test.sumocfg']
    print(json.dumps(dict(scope='All configured network and additional tlLogic definitions in four Bologna configs and the two distinct MoST networks. MoST most.out/most.traci share the scenario network and no additional tlLogic. Most tools duarouter shares the test network. Each TLS/program is independent. Candidate selection in the current control code uses TraCI getAllProgramLogics()[0]; runtime order is unmeasured. Static co-service requires actual junction forbids/foe and an identical outgoing edge. Stable switch requires a second stable candidate with same physical right g and same straight not G. No traffic/outcomes inspected.', results=[inspect(f) for f in cfgs],new_simulations=0,new_fits=0)))
