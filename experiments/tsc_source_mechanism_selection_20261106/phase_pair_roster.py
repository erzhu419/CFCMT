"""Freeze paired candidate indices from t0 supply and nominal signal plans only."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def signals(graph, move):
    timelines = {}
    for view in graph['candidate_views']:
        all_links = {view['link_sequences'][str(i)] for i in move['link_indices']}
        if len(all_links) != 1:
            raise ValueError(f"Mixed signal within {move['movement_id']} at candidate {view['candidate_index']}")
        timeline, = all_links
        assert len(timeline) == 10
        timelines[view['candidate_index']] = timeline
    assert set(timelines) == set(range(8))
    return timelines


def read_roster(protocol, collection):
    cases = [c for c in protocol['cases'] if c['city'] == 'jinan']
    cfg, = {Path(c['sumocfg']) for c in cases}
    network = cfg.parent / ET.parse(cfg).find('./input/net-file').get('value')
    turns = defaultdict(set)
    for c in ET.parse(network).iter('connection'):
        if c.get('tl') and not c.get('from').startswith(':'):
            turns[c.get('tl'),c.get('from'),c.get('to')].add(c.get('dir'))
    missing,selected = [],[]
    for case in cases:
        graph = json.loads((collection/f"seed_{case['seed']}"/case['case_id']/'joint_interface_server_only.json').read_text())
        tls = graph['focal_tls']
        assert graph['case_id']==case['case_id'] and tls==case['tls_id'] and graph['conflicts']['full_context_complete']
        assert Path(graph['conflicts']['source']).resolve()==network.resolve()
        lane_rows = {l['lane_id']:l for l in graph['lane_nodes']}
        supplied = defaultdict(list)
        for m in graph['movement_nodes']:
            direction, = turns[tls,m['incoming_edge'],m['outgoing_edge']]
            lanes = [lane_rows[l] for l in m['incoming_lanes']]
            n = sum(l['near_stopline_vehicle_count'] for l in lanes)
            if direction in ('r','s') and n:
                assert len(lanes)==1 and not lanes[0]['shared_turn_lane']
                supplied[direction].append((m,n))
        for own,own_count in supplied['r']:
            own_states = signals(graph,own)
            responses = {j for i,j in graph['conflicts']['response'] if i in own['link_indices']}
            eligible = []
            has_joint_supply = False
            for opponent,opp_count in supplied['s']:
                if not responses.intersection(opponent['link_indices']):
                    continue
                opp_states = signals(graph,opponent)
                exposure = {c:{t for t,(a,b) in enumerate(zip(own_states[c],opp_states[c])) if a=='g' and b=='G'}
                            for c in own_states}
                has_joint_supply |= any(exposure.values())
                for high,low in permutations(sorted(own_states),2):
                    if own_states[high] != own_states[low]:
                        continue
                    gap = len(exposure[high])-len(exposure[low])
                    if gap<=0:
                        continue
                    assert exposure[high]-exposure[low]
                    eligible.append(((-gap, high, low, opponent['movement_id']), dict(
                        case_id=case['case_id'], seed=case['seed'], tls_id=tls,
                        own_movement=own['movement_id'], priority_movement=opponent['movement_id'],
                        own_near=own_count, priority_near=opp_count,
                        high_candidate=high, low_candidate=low, own_sequence_both=own_states[high],
                        priority_sequence_high=opp_states[high], priority_sequence_low=opp_states[low],
                        priority_exposure_s_high=len(exposure[high]), priority_exposure_s_low=len(exposure[low]), gap_s=gap)))
            if eligible:
                _, row = min(eligible,key=lambda item:item[0]); row['eligible_oriented_pairs_with_opponent']=len(eligible)
                selected.append(row)
            elif has_joint_supply:
                missing.append(dict(case_id=case['case_id'],own_movement=own['movement_id']))
    return dict(status='ROSTER_LOCKED_PRE_RESPONSE', source_protocol=protocol['protocol'], collection=str(collection),
        purpose='Target-only readout roster, formed before reading responses.',
        selection='One group/right movement per row. Both dedicated right/straight movement inlets have positive t0 near supply; the right yields to the straight in the same-TLS response relation. Nominal right sequence is identical on all links and across selected candidates for all ten seconds. Count straight G seconds while own right is g, maximizing high-minus-low seconds then lowest (high candidate index, low candidate index, priority movement ID). We verify uniform signals across all links of each evaluated movement. Bounds are candidate signal plans, without a future clearance oracle.',
        target_groups_scanned=len(cases), double_supplied_group_rights=len(missing)+len(selected), eligible=len(selected),
        eligible_groups=len({r['case_id'] for r in selected}), eligible_seeds=len({r['seed'] for r in selected}),
        missing=missing, gap_s_counts=dict(Counter(r['gap_s'] for r in selected)), rows=selected,
        new_simulations=0,new_fits=0,response_values_read=0)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('protocol', type=Path);p.add_argument('collection',type=Path)
    args=p.parse_args();print(json.dumps(read_roster(json.loads(args.protocol.read_text()),args.collection)))
