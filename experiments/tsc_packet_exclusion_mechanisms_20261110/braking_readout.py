"""Fixed-six algebraic comparison; never changes the fitted/forward operator."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from .braking_terms import discrete_safe_speed


def run(base, cache):
    result = json.loads((base/'output/result.json').read_text())
    selected = [case for case in result['cases'] if case['head'] in ('motion', 'yield_entry')]
    by_seed = defaultdict(list)
    for case in selected:
        by_seed[case['reference']['seed']].append(case)
    rows = []
    for seed, cases in sorted(by_seed.items()):
        details = {case['case_id']: case for case in json.loads(
            (base/'output'/f'seed{seed}_cases_server_only.json').read_text())}
        by_time = defaultdict(list)
        for case in cases:
            by_time[case['reference']['time_sec']].append(case)
        with gzip.open(cache/'training'/f'seed{seed}'/'frames_server_only.jsonl.gz', 'rt') as stream:
            for line in stream:
                frame = json.loads(line)
                t = frame['time_s']
                for case in by_time.get(t, []):
                    pid = case['reference']['packet_id']
                    packet = frame['packets'][pid]
                    focal = case['focal_observation']['current']
                    if any(packet[key] != focal[key] for key in ('lane_id', 'position_m', 'speed_mps')):
                        raise ValueError('Current parameters must join the exact frozen focal snapshot')
                    parameters = packet['parameters']
                    components = []
                    for component in details[case['case_id']]['components']:
                        old = component['proposal']
                        leader = old['leader']
                        terms = dict(old['terms_mps'])
                        if leader['packet_id'] is not None:
                            terms['leader_safe'] = discrete_safe_speed(leader['net_gap_m'],
                                leader['speed_mps'], parameters['decel_mps2'],
                                leader['parameters']['decel_mps2'], parameters['tau_s'])
                        stop = old['signal_stop']
                        if stop['active']:
                            stop_terms = dict(stop['components_mps'])
                            stop_terms['stationary_stop_safe'] = discrete_safe_speed(
                                stop['distance_to_lane_end_m'], 0., parameters['decel_mps2'],
                                parameters['decel_mps2'], parameters['tau_s'])
                            terms['signal_stop'] = max(min(stop_terms.values()), 0.)
                        proposed = max(min(v for v in terms.values() if v is not None), 0.)
                        components.append(dict(intent=component['intent'], prior=component['prior'],
                            signal=component['signal'], compatible=component['compatible'],
                            connection_key=None if component['connection'] is None else component['connection']['key'],
                            eligible_current_g_entry=component['entry_evidence']['eligible_current_g_entry'],
                            baseline_P_mps=old['physical_proposal_mps'], baseline_mu0_mps=component['prediction']['mean0_mps'],
                            baseline_binding_terms=old['binding_terms'], leader=leader,
                            baseline_terms_mps=old['terms_mps'], algebraic_discrete_terms_mps=terms,
                            algebraic_discrete_P_mps=proposed,
                            algebraic_discrete_binding_terms=[name for name, v in terms.items() if v == proposed],
                            lane_end_remaining_m=stop['distance_to_lane_end_m'],
                            baseline_reaches_lane_end=old['physical_proposal_mps'] >= stop['distance_to_lane_end_m'],
                            algebraic_discrete_reaches_lane_end=proposed >= stop['distance_to_lane_end_m']))
                    # Native next state was already saved by diagnose.py. It is a readout only.
                    observed = case['focal_observation']['next']['speed_mps']
                    old_max = max(c['baseline_P_mps'] for c in components)
                    new_max = max(c['algebraic_discrete_P_mps'] for c in components)
                    rows.append(dict(case_id=case['case_id'], head=case['head'], reference=case['reference'],
                        parameters=parameters, type_id=packet['type_id'], native_next_speed_mps=observed,
                        baseline_max_P_mps=old_max, algebraic_discrete_max_P_mps=new_max,
                        native_minus_baseline_max_P_mps=observed-old_max,
                        native_minus_discrete_max_P_mps=observed-new_max, components=components))
                if t >= max(by_time):
                    break
    if len(rows) != 6:
        raise ValueError('Exactly the three selected entry and three selected motion cases are required')
    output = dict(status='COMPLETE_FIXED_SIX_ALGEBRAIC_BRAKING_COMPARISON', cases=rows,
        timestep_s=1, formula='D(v,b)=sum(n>=1,max(v-n*b,0)); tau*v+D(v,b)=max(g,0)+D(vL,bL)',
        scope='Only continuous following/active-stop speed terms replaced algebraically. No response re-resolution, entry mask recomputation, native simulator, fitting or production operator modification.',
        new_native_starts=0, new_fits=0, optimizer_calls=0, calibration_data_used=False)
    (base/'braking_comparison.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=output['status'], cases=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.cache)
