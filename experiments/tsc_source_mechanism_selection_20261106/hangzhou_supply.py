"""Read only frozen Hangzhou t0 graphs and the network used to generate them."""

from collections import defaultdict
import json
from pathlib import Path
from xml.etree import ElementTree as ET
import sys


def audit(protocol, collection):
    cases = [c for c in protocol['cases'] if c['city'] == 'hangzhou']
    (cfg,) = {Path(c['sumocfg']) for c in cases}
    net = (cfg.parent / ET.parse(cfg).find('./input/net-file').get('value')).resolve()
    topology = defaultdict(dict)
    for c in ET.parse(net).iter('connection'):
        if c.get('tl') and not c.get('from').startswith(':'):
            mapping = topology[c.get('tl')]
            link = int(c.get('linkIndex'))
            assert link not in mapping
            mapping[link] = (c.get('dir'), c.get('to'), c.get('toLane'))
    pairs = []
    for case in cases:
        path = collection / f"seed_{case['seed']}" / case['case_id'] / 'joint_interface_server_only.json'
        g = json.loads(path.read_text())
        tls = case['tls_id']
        assert g['case_id'] == case['case_id'] and g['focal_tls'] == tls
        assert g['conflicts']['full_context_complete'] and Path(g['conflicts']['source']).resolve() == net
        links = topology[tls]
        movement_lanes = {m['movement_id']: m['incoming_lanes'] for m in g['movement_nodes']}
        assert all(len(v) == 1 for v in movement_lanes.values())
        lanes = {l['lane_id']: l for l in g['lane_nodes']}
        movement_dir = {
            m['movement_id']: {links[i][0] for i in m['link_indices']} for m in g['movement_nodes']
        }
        assert all(len(d) == 1 for d in movement_dir.values())
        directed = {tuple(p) for p in g['conflicts']['response']}
        foes = {tuple(p) for p in g['conflicts']['foes']}
        states = {s for v in g['candidate_views'] for s in v['nominal_states']}
        for r in g['movement_nodes']:
            if movement_dir[r['movement_id']] != {'r'}:
                continue
            (right,) = (lanes[l] for l in r['incoming_lanes'])
            for s in g['movement_nodes']:
                if movement_dir[s['movement_id']] != {'s'} or s['outgoing_edge'] != r['outgoing_edge']:
                    continue
                (straight,) = (lanes[l] for l in s['incoming_lanes'])
                assert not right['shared_turn_lane'] and not straight['shared_turn_lane']
                merge = {
                    (i, j)
                    for i in r['link_indices']
                    for j in s['link_indices']
                    if links[i][1:] == links[j][1:]
                }
                assert merge
                same_edge = {(i, j) for i in r['link_indices'] for j in s['link_indices']}

                def relation(slot):
                    return [
                        (i, j) for i, j in slot if (i, j) in directed and ((i, j) in foes or (j, i) in foes)
                    ]

                pairs.append(
                    dict(
                        case=case['case_id'],
                        seed=case['seed'],
                        tls=tls,
                        right_movement=r['movement_id'],
                        straight_movement=s['movement_id'],
                        near=[right['near_stopline_vehicle_count'], straight['near_stopline_vehicle_count']],
                        native_directed_same_lane=bool(relation(merge)),
                        native_directed_same_edge=bool(relation(same_edge)),
                        actual_nominal_same_lane_GG=any(
                            st[i] == st[j] == 'G' for st in states for i, j in merge
                        ),
                        actual_nominal_directed_same_lane_GG=any(
                            st[i] == st[j] == 'G' for st in states for i, j in relation(merge)
                        ),
                        own_nominal_chars=''.join(
                            sorted({st[i] for st in states for i in r['link_indices']})
                        ),
                    )
                )

    def tally(selected):
        return dict(
            group_pairs=len(selected),
            groups=len({r['case'] for r in selected}),
            seeds=len({r['seed'] for r in selected}),
        )

    supplied = [r for r in pairs if min(r['near']) > 0]
    return dict(
        protocol=protocol['protocol'],
        collection=str(collection),
        net_file=str(net),
        groups=len(cases),
        native_nominal_right_chars_seen=sorted({c for r in pairs for c in r['own_nominal_chars']}),
        same_exit_with_same_lane_possible=tally(pairs),
        right_supplied=tally([r for r in pairs if r['near'][0] > 0]),
        same_exit_cosupplied=tally(supplied),
        actual_nominal_G_and_G_among_cosupplied=tally(
            [r for r in supplied if r['actual_nominal_same_lane_GG']]
        ),
        static_right_yields_to_straight_same_lane=tally([r for r in pairs if r['native_directed_same_lane']]),
        cosupplied_static_right_yields_to_straight_same_lane=tally(
            [r for r in supplied if r['native_directed_same_lane']]
        ),
        cosupplied_static_right_yields_to_straight_same_edge=tally(
            [r for r in supplied if r['native_directed_same_edge']]
        ),
        cosupplied_static_directed_same_lane_and_nominal_GG=tally(
            [r for r in supplied if r['actual_nominal_directed_same_lane_GG']]
        ),
        cosupplied_cases=supplied,
        scope='Observed frozen t0 inlet supply within 10 s at lane speed limit. Movement-pair counted once per state. Raw XML directs and exit lane identify right and straight. Relation requires the native directed response and physical foe; signal characters are from original nominal timelines. Geometric co-supply includes both vehicles regardless of light. No realized yield or response outcomes evaluated.',
        new_simulations=0,
        new_fits=0,
    )


if __name__ == '__main__':
    print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]))))
