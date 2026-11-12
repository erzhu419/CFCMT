"""Table-only lane-envelope versus observed-entity-width comparison; zero fit."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import json
import multiprocessing
from pathlib import Path
import traceback
import xml.etree.ElementTree as ET

from experiments.tsc_packet_spatial_conflict_20261110 import conditional_features as shared
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110.geometry import build_conflict_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import (
    load_focus, load_widths, save, seed_context, iter_transitions, _candidates,
    RESERVOIR_SEED, RoadTopology, _empty_permission, update_permission,
)
from .geometry import build_entity_geometry


SEEDS = [76116901, 76116902, 76116903, 76116904]
TOL = 1e-8  # The frozen previous probe comparison tolerance.


def verify_entity_preflight(path):
    """Verify the completed frame scan against the actual configured vType XML."""
    proof = json.loads(Path(path).read_text())
    if proof['status'] != 'PASS_ENTITY_INPUT_PREFLIGHT' or proof['training_seeds'] != SEEDS:
        raise ValueError('The fixed four-seed input preflight must pass first')
    if (not proof['centerline_observed'] or proof['maximum_abs_lateral_position_m'] != 0
            or proof['maximum_abs_lateral_speed_mps'] != 0):
        raise ValueError('This entity table requires the observed centerline state support')
    cfg = Path(proof['sumocfg'])
    configured = set()
    configuration = ET.parse(cfg).getroot()
    for name in ('route-files', 'additional-files'):
        item = configuration.find('./input/'+name)
        if item is not None:
            configured.update((cfg.parent/value.strip()).resolve()
                              for value in item.get('value', '').split(',') if value.strip())
    xml_widths, sources = {}, {}
    for definition in proof['vtype_definitions']:
        source, type_id = Path(definition['source']), definition['id']
        if source.resolve() not in configured:
            raise ValueError('The vType source must belong to the actual configured input files')
        if source not in sources:
            sources[source] = {node.attrib['id']: node.attrib for node in ET.parse(source).iter('vType')}
        attributes = sources[source][type_id]
        if attributes.get('width') is None or float(attributes['width']) != float(definition['width']):
            raise ValueError('The explicit vType width differs from the completed input preflight')
        xml_widths[type_id] = float(attributes['width'])
    observed = proof['observed_type_counts']
    if not observed or any(count <= 0 for count in observed.values()):
        raise ValueError('Observed vehicle types are required')
    if any(type_id not in xml_widths or xml_widths[type_id] != proof['vehicle_width_by_type'][type_id]
           for type_id in observed):
        raise ValueError('Every observed type must have its matching explicit XML width')
    widths = {xml_widths[type_id] for type_id in observed}
    if len(widths) != 1 or next(iter(widths)) != proof['single_vehicle_width_m']:
        raise ValueError('The shared entity table requires one derived observed-type width')
    return next(iter(widths)), proof


def load_saved_probes(spatial_base):
    saved, bad, reduced = defaultdict(dict), 0, 0
    for path in sorted((Path(spatial_base)/'output').glob('seed*_internal_probes_server_only.json')):
        for row in json.loads(path.read_text()):
            key = int(row['time_sec']), row['packet_id']
            seed = int(row['seed'])
            if key in saved[seed]:
                raise ValueError('Every frozen internal probe must be unique')
            old = row['paired']['candidate']['upper']
            observed = row['actual_next_speed_mps']
            previous_bad = old+TOL < observed
            bad += int(previous_bad)
            reduced += int(old < row['paired']['baseline']['upper']-TOL)
            saved[seed][key] = dict(row, previously_bad=bool(previous_bad))
    if sum(map(len, saved.values())) != 253 or bad != 20 or reduced != 21:
        raise ValueError('The fixed 253 probes, 20 old violations and 21 reductions must reproduce')
    return dict(saved)


def paired_admission(kernel, packet_id, intent, old_table, entity_table):
    """Use one unchanged admission implementation; replace only its geometry table."""
    packet = kernel.state['packets'][packet_id]
    previous = deepcopy(kernel.state['intents'].get(packet_id))
    had_intent = packet_id in kernel.state['intents']
    old_connection = kernel._connections[packet_id]
    previous_table = kernel._spatial_geometry
    try:
        geometry = kernel.topology.lanes[packet['lane_id']]
        if not geometry['internal']:
            kernel.state['intents'][packet_id] = dict(road_id=geometry['edge_id'], choice=intent)
        connection, compatible = kernel._connection(packet)
        kernel._connections[packet_id] = connection, compatible
        view = seed_context._context_view(kernel, packet, connection, compatible)
        configure_geometry(kernel, old_table)
        before = shared._admission_detail(kernel, packet, connection, compatible, view)
        configure_geometry(kernel, entity_table)
        after = shared._admission_detail(kernel, packet, connection, compatible, view)
        distance = geometry['length_m']-packet['position_m']
        yielding = (not geometry['internal'] and connection is not None and compatible
                    and kernel._signal(connection) == 'g')
        for detail in (before, after):
            detail['yield_risk'] = bool(yielding and detail['upper'] > 0 and
                                        detail['upper'] >= distance and detail['admissible'])
        return dict(intent=intent, connection_key=None if connection is None else connection['key'],
                    baseline=before, candidate=after)
    finally:
        if had_intent:
            kernel.state['intents'][packet_id] = previous
        else:
            kernel.state['intents'].pop(packet_id, None)
        kernel._connections[packet_id] = old_connection
        configure_geometry(kernel, previous_table)


def _seed(task):
    seed, directory, static, laws, references, focus, saved_probes, width, output = task
    print(json.dumps(dict(stage='entity_pair_begin', seed=seed)), flush=True)
    topology = RoadTopology(static)
    old_table, entity_table = build_conflict_geometry(topology), build_entity_geometry(topology, width)
    executor_seeds = json.loads((directory/'executor_seed_server_only.json').read_text())
    by_time, probe_by_time = defaultdict(list), defaultdict(list)
    for reference in references:
        by_time[reference['time_sec']].append(reference)
    planned_keys = {(t+1, pid) for (t, pid), selected in focus.items() if selected['clean']}
    if set(saved_probes) != planned_keys:
        raise ValueError('The previous probe references must match the unchanged clean focus')
    for key in saved_probes:
        probe_by_time[key[0]].append(key[1])
    stats, focus_counts, probes, geometry_uses = _empty_permission(), Counter(), Counter(), Counter()
    probe_rows, fixed_bad, new_bad_refs, samples = [], [], [], 0
    with gzip.open(output/f'seed{seed}_pairs_server_only.jsonl.gz', 'xt', compresslevel=1) as stream:
        for frame, current, action in iter_transitions(directory):
            t = int(frame['time_s'])
            if t % 300 == 0:
                print(json.dumps(dict(stage='entity_pair_progress', seed=seed, time_sec=t)), flush=True)
            if t not in by_time and t not in probe_by_time:
                continue
            kernel = seed_context.prepare_feature_kernel(static, frame, executor_seeds, laws,
                action['requests'], RESERVOIR_SEED+seed*100000+t)
            configure_geometry(kernel, old_table)
            if kernel.signals != action['applied_states']:
                raise ValueError('The paired tables must use the observed current actuator signals')
            for ref in by_time[t]:
                pid, prior = ref['packet_id'], _candidates(kernel, ref['packet_id'])
                pairs = [paired_admission(kernel, pid, intent, old_table, entity_table) for intent in prior]
                update_permission(stats, ref['y'], list(prior.values()), pairs)
                record = dict(reference=ref, prior=prior, paired=pairs)
                selected = focus.get((t, pid))
                if selected is not None:
                    pair = next(row for row in pairs if row['intent'] == selected['intent'])
                    if pair['connection_key'] != selected['connection_key'] or not pair['baseline']['yield_risk']:
                        raise ValueError('The frozen old-spatial focus connection and permission must reproduce')
                    focus_counts['all_rows'] += 1
                    focus_counts['baseline_all_allowed'] += 1
                    focus_counts['candidate_all_allowed'] += int(pair['candidate']['yield_risk'])
                    if selected['clean']:
                        focus_counts['clean_rows'] += 1
                        focus_counts['baseline_clean_allowed'] += 1
                        focus_counts['candidate_clean_allowed'] += int(pair['candidate']['yield_risk'])
                    record['posthoc_focus'] = selected
                for pair in pairs:
                    for name in ('resolved_pairs', 'no_zone_pairs', 'unresolved_pairs'):
                        geometry_uses[name+'_visits'] += pair['candidate'].get(name, 0)
                stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                samples += 1
            for pid in probe_by_time[t]:
                saved = saved_probes[(t, pid)]
                packet, observed = frame['packets'].get(pid), current['packets'].get(pid)
                if (packet is None or observed is None or not kernel.topology.lanes[packet['lane_id']]['internal']
                        or packet['lane_id'] != saved['current_lane']
                        or abs(packet['position_m']-saved['current_position_m']) > TOL):
                    raise ValueError('The fixed internal probe current state must reproduce')
                connection, compatible = kernel._connections[pid]
                pair = paired_admission(kernel, pid, connection['to_edge'], old_table, entity_table)
                before, after, speed = pair['baseline']['upper'], pair['candidate']['upper'], observed['speed_mps']
                if (abs(before-saved['paired']['candidate']['upper']) > TOL
                        or abs(speed-saved['actual_next_speed_mps']) > TOL):
                    raise ValueError('The old lane-envelope probe ceiling and observed speed must reproduce')
                old_bad, new_bad = before+TOL < speed, after+TOL < speed
                if old_bad != saved['previously_bad']:
                    raise ValueError('The exact old-bad reference set must reproduce')
                probes['evaluated'] += 1
                probes['baseline_reduced_relative_receiving'] += int(before < saved['paired']['baseline']['upper']-TOL)
                probes['baseline_ceiling_below_observed_speed'] += int(old_bad)
                probes['candidate_ceiling_below_observed_speed'] += int(new_bad)
                probes['candidate_limit_raised_vs_lane'] += int(after > before+TOL)
                probes['candidate_limit_reduced_vs_lane'] += int(after < before-TOL)
                probes['old_bad_fixed'] += int(old_bad and not new_bad)
                probes['old_bad_remaining'] += int(old_bad and new_bad)
                probes['new_bad_previously_ok'] += int(new_bad and not old_bad)
                probes['candidate_zero_while_actual_moving'] += int(after < .1 <= speed)
                brief = dict(seed=seed, time_sec=t, packet_id=pid, baseline_upper_mps=before,
                    candidate_upper_mps=after, actual_next_speed_mps=speed,
                    previously_bad=bool(old_bad), candidate_bad=bool(new_bad))
                row = dict(brief, current_lane=packet['lane_id'], current_position_m=packet['position_m'], paired=pair)
                probe_rows.append(row)
                if old_bad:
                    fixed_bad.append(brief)
                elif new_bad:
                    new_bad_refs.append(brief)
    if samples != len(references) or len(probe_rows) != len(saved_probes):
        raise ValueError('All frozen yield and internal-probe references must be compared exactly once')
    probes['planned'] = len(saved_probes)
    save(output/f'seed{seed}_internal_probes_server_only.json', probe_rows)
    return dict(seed=seed, samples=samples, permission_by_label=stats, focus=dict(focus_counts),
        internal_probes=dict(probes), old_bad_case_outcomes=fixed_bad, new_bad_case_refs=new_bad_refs,
        geometry_uses=dict(geometry_uses),
        baseline_geometry_metadata={key: value for key, value in old_table.items() if key != 'pairs'},
        candidate_geometry_metadata={key: value for key, value in entity_table.items() if key != 'pairs'})


def summarize(results):
    permission, focus, probes, uses = _empty_permission(), Counter(), Counter(), Counter()
    old_bad, new_bad = [], []
    for result in results:
        for label, values in result['permission_by_label'].items():
            for name, value in values.items():
                permission[label][name] += value
        focus.update(result['focus'])
        probes.update(result['internal_probes'])
        uses.update(result['geometry_uses'])
        old_bad.extend(result.pop('old_bad_case_outcomes'))
        new_bad.extend(result.pop('new_bad_case_refs'))
    for values in permission.values():
        n = values['rows']
        values['baseline_permission_mass_mean'] = values['baseline_permission_mass_sum']/n
        values['candidate_permission_mass_mean'] = values['candidate_permission_mass_sum']/n
    reproduction = (permission['1']['rows'], permission['0']['rows'],
        permission['1']['baseline_any_count'], permission['0']['baseline_any_count'],
        focus['all_rows'], focus['baseline_all_allowed'], focus['clean_rows'], focus['baseline_clean_allowed'],
        probes['planned'], probes['evaluated'], probes['baseline_ceiling_below_observed_speed'],
        probes['baseline_reduced_relative_receiving'], len(old_bad))
    passed = reproduction == (9371, 30629, 9368, 29458, 374, 374, 253, 253, 253, 253, 20, 21, 20)
    return dict(status='COMPLETE_ENTITY_FOOTPRINT_DIAGNOSIS' if passed else 'HOLD_BASELINE_REPRODUCTION',
        baseline_reproduced=passed, baseline_reproduction_values=list(reproduction),
        new_native_starts=0, new_fits=0, new_nuisance_estimations=0, free_rollouts=0,
        permission_by_label=permission, focus=dict(focus), internal_probes=dict(probes),
        old_bad_case_outcomes=old_bad, new_bad_case_refs=new_bad, geometry_uses=dict(uses), by_seed=results,
        interpretation='The same frozen spatial admission/priority/time implementation uses two static geometry tables. Only capsule inflation is changed from lane width to the explicit observed-type entity width; paths and lane-width receiving prefixes remain fixed. Permissions are not entry predictions or control benefits; internal ceilings are not calibrated speed predictions.')


def run(cache_base, conflict_base, spatial_base, preflight, output, workers=4):
    cache_base, conflict_base, spatial_base, output = map(Path, (cache_base, conflict_base, spatial_base, output))
    output.mkdir(parents=True, exist_ok=False)
    try:
        width, proof = verify_entity_preflight(preflight)
        focus, saved = load_focus(conflict_base), load_saved_probes(spatial_base)
        reservoirs = json.loads((cache_base/'training_reservoir_server_only.json').read_text())
        original = json.loads((cache_base/'static_graph_server_only.json').read_text())
        static, width_source = load_widths(original)
        laws = json.loads((cache_base/'target_laws_server_only.json').read_text())
        tasks = [(seed, cache_base/'training'/f'seed{seed}', static, laws,
                  reservoirs[str(seed)]['yield_entry'], focus[seed], saved[seed], width, output) for seed in SEEDS]
        if workers == 1:
            results = [_seed(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
                results = list(pool.map(_seed, tasks))
        result = summarize(results)
        result.update(lane_width_source=width_source, entity_width_m=width,
            entity_width_source=dict(preflight=str(preflight), sumocfg=proof['sumocfg'],
                vehicle_width_by_type=proof['vehicle_width_by_type'], observed_type_counts=proof['observed_type_counts'],
                vtype_definitions=proof['vtype_definitions'], centerline_observed=proof['centerline_observed']),
            shared_admission_source=shared.__file__, seed_context_source=seed_context.__file__)
    except Exception as error:
        result = dict(status='HOLD_ENTITY_DIAGNOSTIC_ERROR', error=f'{type(error).__name__}: {error}',
                      new_native_starts=0, new_fits=0, free_rollouts=0)
        (output/'error_server_only.txt').write_text(traceback.format_exc())
    save(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-base', type=Path, required=True)
    parser.add_argument('--conflict-base', type=Path, required=True)
    parser.add_argument('--spatial-base', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    result = run(args.cache_base, args.conflict_base, args.spatial_base, args.preflight, args.output, args.workers)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['status'] == 'COMPLETE_ENTITY_FOOTPRINT_DIAGNOSIS' else 2)
