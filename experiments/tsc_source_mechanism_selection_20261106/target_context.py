"""Summarize supplied permissive turn service from frozen Jinan preaction graphs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from xml.etree import ElementTree


def span(values):
    return [min(values), max(values)] if values else None


def audit(protocol, collection):
    cases = [c for c in protocol['cases'] if c['city'] == 'jinan']
    cfg, = {Path(c['sumocfg']) for c in cases}
    network = cfg.parent / ElementTree.parse(cfg).find('./input/net-file').get('value')
    directions = defaultdict(set)
    exits = defaultdict(set)
    for c in ElementTree.parse(network).iter('connection'):
        if c.get('tl') and not c.get('from').startswith(':'):
            key = c.get('tl'), int(c.get('linkIndex'))
            directions[key].add(c.get('dir'))
            exits[key].add((c.get('to'), c.get('toLane')))
    turns = defaultdict(list)
    physical_pairs = defaultdict(set)
    switches = defaultdict(set)
    switch_examples = {}
    for case in cases:
        path = collection / f"seed_{case['seed']}" / case['case_id'] / 'joint_interface_server_only.json'
        graph = json.loads(path.read_text())
        assert graph['case_id'] == case['case_id'] and graph['focal_tls'] == case['tls_id']
        tls = graph['focal_tls']
        assert graph['conflicts']['full_context_complete']
        assert Path(graph['conflicts']['source']).resolve() == network.resolve()
        lanes = {lane['lane_id']: lane for lane in graph['lane_nodes']}
        states = sorted({state for v in graph['candidate_views'] for state in v['nominal_states']})
        moves = graph['movement_nodes']
        links_to_lanes = defaultdict(set)
        for m in moves:
            for link in m['link_indices']:
                links_to_lanes[link].update(m['incoming_lanes'])
        for m in moves:
            own = set(m['link_indices'])
            direction, = {d for link in own for d in directions[tls, link]}
            own_lanes = [lanes[key] for key in m['incoming_lanes']]
            # The frozen target has singleton inlets. For this graph the feature
            # mapping link -> movement inlet is also an exact physical lane join.
            assert len(own_lanes) == 1
            lane, = own_lanes
            entry = dict(case=case['case_id'], seed=case['seed'], tls=tls,
                         movement=m['movement_id'], turn=direction, dedicated=not lane['shared_turn_lane'],
                         inlet_length_m=lane['length_m'], speed_mps=lane['speed_limit_mps'],
                         current_min_green_s=graph['executor_context']['current_min_green_sec'],
                         own_near=lane['near_stopline_vehicle_count'], g=False, priority=[], foe=[], examples=[])
            responses = graph['conflicts']['response']
            foes = graph['conflicts']['foes']
            for state in states:
                g_links = {link for link in own if state[link] == 'g'}
                if not g_links:
                    continue
                assert g_links == own, 'target movement mixes g with another service character'
                entry['g'] = True
                if not entry['own_near']:
                    continue
                priority_links = {j for i,j in responses if i in g_links and j not in own and state[j]=='G'}
                foe_links = {j for i,j in foes if i in g_links} | {i for i,j in foes if j in g_links}
                serving_foes = {i for i in foe_links-own if state[i] in 'GgsO'}
                def near(links):
                    ids = {lane_id for i in links for lane_id in links_to_lanes[i]}
                    return sum(lanes[k]['near_stopline_vehicle_count'] for k in ids)
                priority_near = near(priority_links)
                entry['priority'].append(priority_near)
                entry['foe'].append(near(serving_foes))
                supplied = {j for j in priority_links if near({j}) > 0}
                for i,j in responses:
                    if i not in g_links or j not in supplied:
                        continue
                    our_exit, = exits[tls,i]
                    other_exit, = exits[tls,j]
                    other_direction, = directions[tls,j]
                    exit_type = ('same_lane' if our_exit == other_exit else
                                 'same_edge_different_lane' if our_exit[0] == other_exit[0] else 'different_edge')
                    physical_pairs[direction,other_direction,exit_type].add(
                        (case['seed'], case['case_id'], m['movement_id'], i, j))
                for other in moves:
                    indices = set(other['link_indices']) & supplied
                    if not indices:
                        continue
                    other_direction, = {d for j in indices for d in directions[tls,j]}
                    ids = other['incoming_lanes']
                    entry['examples'].append(dict(priority_movement=other['movement_id'], priority_turn=other_direction,
                        priority_dedicated=all(not lanes[k]['shared_turn_lane'] for k in ids),
                        priority_lane_near=near(indices), total_priority_near=priority_near,
                        nominal_state=state))
            if entry['examples']:
                # Compare the same movement pair and second across candidates at one t0.
                related = {j for i,j in responses if i in own and j not in own}
                opposing_ids = {e['priority_movement'] for e in entry['examples']}
                for opponent in (m for m in moves if m['movement_id'] in opposing_ids):
                    opponent_links = related.intersection(opponent['link_indices'])
                    other_dir, = {d for j in opponent_links for d in directions[tls,j]}
                    key = (direction, other_dir)
                    for t in range(graph['horizon_sec']):
                        by_priority = {True: [], False: []}
                        for view in graph['candidate_views']:
                            state = view['nominal_states'][t]
                            if all(state[i] == 'g' for i in own):
                                chars = ''.join(sorted({state[j] for j in opponent_links}))
                                by_priority['G' in chars].append((view['candidate_index'], chars))
                        if by_priority[True] and by_priority[False]:
                            switches[key].add((case['seed'], case['case_id'], m['movement_id']))
                            switch_examples.setdefault(key, dict(case=case['case_id'], own=m['movement_id'],
                                opponent=opponent['movement_id'], nominal_offset_sec=t,
                                candidate_with_G=by_priority[True][0], candidate_without_G=by_priority[False][0]))
                            break
            turns[direction].append(entry)
    metrics = {}
    all_matched = []
    pair_rows = defaultdict(list)
    for direction, entries in sorted(turns.items()):
        served = [r for r in entries if r['g']]
        supplied = [r for r in served if r['own_near'] > 0]
        matched = [r for r in supplied if max(r['priority'])>0]
        all_matched.extend(matched)
        metrics[direction] = dict(total_group_movements=len(entries), g_group_movements=len(served),
            supplied_g_group_movements=len(supplied), supplied_g_and_priority_G_group_movements=len(matched),
            supplied_g_groups=len({r['case'] for r in supplied}), supplied_g_seeds=len({r['seed'] for r in supplied}),
            supplied_g_and_priority_G_groups=len({r['case'] for r in matched}),
            supplied_g_and_priority_G_seeds=len({r['seed'] for r in matched}),
            supplied_g_dedicated_group_movements=sum(r['dedicated'] for r in supplied),
            supplied_g_own_near_minmax=span([r['own_near'] for r in supplied]),
            supplied_g_foe_servable_near_minmax=span([n for r in supplied for n in r['foe']]),
            supplied_g_priority_near_minmax=span([n for r in supplied for n in r['priority']]),
            supplied_g_inlet_length_m_minmax=span([r['inlet_length_m'] for r in supplied]),
            supplied_g_inlet_speed_mps_minmax=span([r['speed_mps'] for r in supplied]),
            supplied_g_current_min_green_s_minmax=span([r['current_min_green_s'] for r in supplied]),
            simultaneous_supplied_priority_turns=sorted({x['priority_turn'] for r in matched for x in r['examples']}))
        for r in matched:
            for x in r['examples']:
                pair_rows[direction, x['priority_turn']].append((r,x))
    pairs=[]
    for (direction,other), items in sorted(pair_rows.items()):
        r,x = items[0]
        pairs.append(dict(yielding_turn=direction, priority_turn=other,
            group_movements=len({(r['case'],r['movement']) for r,x in items}),
            groups=len({r['case'] for r,x in items}), seeds=len({r['seed'] for r,x in items}),
            both_dedicated=all(r['dedicated'] and x['priority_dedicated'] for r,x in items),
            same_second_own_g_both_opponent_G_vs_no_G=dict(
                group_yielding_movements=len(switches[direction,other]),
                groups=len({v[1] for v in switches[direction,other]}),
                seeds=len({v[0] for v in switches[direction,other]}),
                example=switch_examples.get((direction,other))),
            example={**{key:r[key] for key in ('case','tls','movement','own_near')}, **x}))
    return dict(protocol=protocol['protocol'], collection=str(collection), target_network=str(network),
        groups=len(cases), seeds=len({c['seed'] for c in cases}), tls=len({c['tls_id'] for c in cases}),
        group_movements=sum(map(len,turns.values())), context_by_turn=metrics, supplied_turn_pairs=pairs,
        supplied_physical_pair_relations=[dict(yielding_turn=own, priority_turn=other, outgoing=exit_type,
            seeds=len({v[0] for v in observations}), groups=len({v[1] for v in observations}),
            group_yielding_movements=len({v[1:3] for v in observations}),
            group_link_pairs=len(observations))
            for (own,other,exit_type), observations in sorted(physical_pairs.items())],
        any_supplied_g_and_priority_G_groups=len({r['case'] for r in all_matched}),
        any_supplied_g_and_priority_G_seeds=len({r['seed'] for r in all_matched}),
        any_supplied_g_and_priority_G_tls=len({r['tls'] for r in all_matched}),
        scope='Counts deduplicate candidate actions/stable seconds per group and movement. Near supply counts distinct incoming-lane vehicles within 10s at lane speed limit at t0. G/g here refer to nominal candidate signals during the next 10s. Direction is the native external connection dir. Directed response is the saved graph same-TLS junction-request relation [yielding link, priority link]. Static possible conflicts and observed lane supply do not identify realized encounters, routes or discharge. Pair subsets can overlap.',
        new_simulations=0,new_fits=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('protocol', type=Path)
    parser.add_argument('collection', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.protocol.read_text()), args.collection), ensure_ascii=False))
