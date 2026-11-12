"""Static lane and controlled same-TLS conflict context; no SUMO execution."""
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


def _conflicts(connections, requests, source):
    """Map junction request bits to TLS links using public Connection methods.

    response[i,j] means physical connection i yields to j. Request bit zero is
    the rightmost character. Joined TLSs can span distinct junction index spaces.
    """
    indexed, controlled = defaultdict(list), defaultdict(list)
    for c in connections:
        key = (c.getJunction().getID(), c.getJunctionIndex())
        if key[1] >= 0: indexed[key].append(c)
        if c.getTLSID(): controlled[c.getTLSID()].append((c, key))
    result = {}
    for tls, own in controlled.items():
        pairs = {'foes': set(), 'response': set()}
        external = {'foes': set(), 'response': set()}
        missing, unresolved = set(), set()
        for c, key in own:
            row = requests.get(key)
            if key[1] < 0 or row is None or any(name not in row for name in pairs):
                missing.add(key)
                continue
            for name in pairs:
                for other_index, bit in enumerate(reversed(row[name])):
                    if bit != '1': continue
                    other_key = (key[0], other_index)
                    others = indexed.get(other_key, [])
                    if not others:
                        unresolved.add(other_key)
                        continue
                    for other in others:
                        if other.getTLSID() == tls:
                            pairs[name].add((c.getTLLinkIndex(), other.getTLLinkIndex()))
                        else:
                            external[name].add(other_key)
        mapping_complete = not missing and not unresolved
        result[tls] = dict(foes=[list(p) for p in sorted(pairs['foes'])],
            response=[list(p) for p in sorted(pairs['response'])],
            indexing='TLS link indices; union physical pairs within junction',
            response_orientation='[i,j] means i yields to j',
            mapping_complete=mapping_complete, source=str(source),
            external_foe_connection_count=len(external['foes']),
            external_response_connection_count=len(external['response']),
            missing_controlled_request_count=len(missing), unresolved_conflicting_request_count=len(unresolved),
            full_context_complete=mapping_complete and not external['foes'] and not external['response'],
            scope='controlled sameTLS pairs')
    return result


def read_static_context(sumocfg):
    """Read existing SUMO XML with sumolib; never start a simulation.

    Lane dictionaries include internal lanes. mapping_complete concerns the
    known controlled subset; external/uncontrolled conflicts separately make
    full_context_complete false. Missing requests never count as empty context.
    """
    import sumolib
    sumocfg = Path(sumocfg)
    config = ET.parse(sumocfg).getroot()
    network = (sumocfg.parent / config.find('./input/net-file').get('value')).resolve()
    xml = ET.parse(network).getroot()
    requests = {(j.get('id'), int(r.get('index'))): dict(r.attrib)
                for j in xml.findall('junction') for r in j.findall('request')}
    net = sumolib.net.readNet(str(network), withInternal=True, withFoes=True)
    lanes = [lane for edge in net.getEdges(withInternal=True) for lane in edge.getLanes()]
    connections = [c for lane in lanes for c in lane.getOutgoing()]
    return dict(lane_speed_limits={lane.getID(): lane.getSpeed() for lane in lanes},
                lane_lengths={lane.getID(): lane.getLength() for lane in lanes},
                conflicts_by_tls=_conflicts(connections, requests, network), netfile=str(network))
