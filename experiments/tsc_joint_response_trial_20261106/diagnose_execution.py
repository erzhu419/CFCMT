"""Explain recorded signal differences from saved clearance occupancy; no replay."""
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def run(spec, collection_root):
    findings = []
    for case in spec['cases']:
        directory = collection_root / f"seed_{case['seed']}" / case['case_id']
        receipt = directory / 'result.json'
        if not receipt.exists():
            continue
        result = json.loads(receipt.read_text())
        for execution_index, branch in enumerate(result['branches']):
            if branch['status'] != 'HOLD_EXECUTION':
                continue
            folder = 'first_pp' if execution_index == 0 else f"action_{branch['candidate_index']}"
            obs = json.loads((directory / folder / 'observations_server_only.json').read_text())
            cfg = Path(case['sumocfg']); config = ET.parse(cfg).getroot()
            net = ET.parse(cfg.parent / config.find('./input/net-file').get('value')).getroot()
            via = {c.get('via') for c in net.findall('connection') if c.get('tl') == case['tls_id'] and c.get('via')}
            prefixes = {f"{lane.rsplit('_', 2)[0]}_" for lane in via if lane.startswith(':') and lane.count('_') >= 2}
            clearance = via | {lane.get('id') for lane in net.findall('./edge/lane')
                               if any(lane.get('id').startswith(prefix) for prefix in prefixes)}
            mismatches = branch['audit']['nominal_actual_mismatch_step_indices']
            frames = {int(f['offset_sec']): f for f in obs['frames']}
            counts = [sum(v['lane_id'] in clearance for v in frames[t]['vehicles'].values()) for t in range(11)]
            blockers = {v for t in mismatches for v, row in frames[t]['vehicles'].items() if row['lane_id'] in clearance}
            traces = {v: [[t, frames[t]['vehicles'][v]['lane_id'], frames[t]['vehicles'][v]['position_m']]
                          for t in range(11) if v in frames[t]['vehicles']] for v in sorted(blockers)}
            pre = obs['preaction']; feature = pre['feature_names'].index('clearance_vehicle_count')
            missing = sorted(clearance - set(obs['lane_geometry']))
            explained = not missing and all(counts[t] > 0 and set(branch['audit']['actual_states'][t]) == {'r'}
                                             for t in mismatches)
            findings.append(dict(case_id=case['case_id'], city=case['city'], candidate_index=branch['candidate_index'],
                mismatch_step_indices=mismatches, resolved_checkpoint_sec=result['resolved_checkpoint_sec'],
                clearance_lane_count=len(clearance), unobserved_clearance_lanes=missing,
                clearance_vehicle_count_by_offset=counts,
                t0_clearance_count_matches_feature=all(row[feature] == counts[0] for row in pre['features']),
                occupied_at_every_divergent_release=explained,
                blocker_traces=traces,
                conclusion='Observed clearance occupancy supports all-red extensions at every divergent release.'
                    if explained else 'Not fully explained by the saved clearance observations.'))
    return dict(protocol=spec['protocol'], new_fits=0, new_simulations=0, findings=findings,
        scope='Posthoc execution diagnosis only; future occupancy is not a predictor and the frozen collection gate is unchanged.')
