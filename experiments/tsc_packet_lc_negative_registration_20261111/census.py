"""Current-only census of the fixed three-seed LC reservoir population."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

from experiments.tsc_packet_conditional_census_20261110.census import selected_current_records
from experiments.tsc_packet_lane_change_coverage_20261111.census import (
    DISTANCE_TOLERANCE, SPEED_TOLERANCE, current_stratum,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


SEEDS = (76116901, 76116902, 76116903)
DIRECTIONS = {0: 'left', 2: 'right'}
BINS = ('00', '10', '01', '11')


def identity(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def _stratum_key(row):
    value = row['current_stratum']
    return tuple(value[:4]) + (tuple(value[4]),) + tuple(value[5:])


def assign_bins(population_rows):
    """Assign fixed fine-neighbourhood bins inside the training population.

    Only an observed positive's actual direction enters the corresponding
    positive index. Directional rows with no static adjacent stratum cannot
    match. Missing current quantities stop registration instead of entering00.
    """
    rows = [dict(row, directions={name:dict(value) for name, value in row['directions'].items()})
            for row in population_rows]
    positive_index = defaultdict(list)
    for row in rows:
        for value in row['directions'].values():
            if value['status'] not in ('RECORDED', 'NO_ADJACENT_CURRENT_STRATUM'):
                raise ValueError(f'Missing archived current stratum at {identity(row)}: {value["status"]}')
        if row['y'] != 1:
            name = DIRECTIONS[row['y']]
            value = row['directions'][name]
            if value['status'] == 'RECORDED':
                positive_index[(name, _stratum_key(value))].append((identity(row), value))
    for row in rows:
        near = {}
        for name, value in row['directions'].items():
            candidates = (positive_index[(name, _stratum_key(value))]
                          if value['status'] == 'RECORDED' else ())
            matches = [list(key) for key, other in candidates
                       if abs(value['speed_mps']-other['speed_mps']) <= SPEED_TOLERANCE
                       and abs(value['lane_end_distance_m']-other['lane_end_distance_m']) <= DISTANCE_TOLERANCE]
            value['matched_positive_keys'] = sorted(matches)
            near[name] = bool(matches)
        row.update(near_left=near['left'], near_right=near['right'],
                   bin=f'{int(near["left"])}{int(near["right"])}' if row['y'] == 1 else 'positive')
    return rows


def _domain_metadata(path, references):
    wanted = {identity(row) for row in references}
    counts = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = identity(row)
            if key not in wanted:
                continue
            if key in counts or not row['eligibility']['lane_change']:
                raise ValueError(f'Invalid existing LC domain at {key}')
            if row['required_actor_count'] is None:
                raise ValueError(f'Existing census lacks the current actor count at {key}')
            counts[key] = {name:row[name] for name in ('required_actor_count', 'required_participants',
                'frame_actor_count', 'eligibility', 'physical_calls_upper', 'leaf_visits_upper')}
    if counts.keys() != wanted:
        raise ValueError(f'Existing census misses {len(wanted-counts.keys())} population references')
    return counts


def _current_rows(frame, action, references, domains, topology):
    """Read strata from one archived current frame and its aligned action."""
    lanes = defaultdict(list)
    for packet in frame['packets'].values():
        lanes[packet['lane_id']].append(packet)
    result = []
    for reference in references:
        key = identity(reference)
        if frame['time_s'] != key[1] or action['time_sec'] != key[1]:
            raise ValueError(f'Current frame/action clocks differ at {key}')
        packet = frame['packets'][key[2]]
        row = dict(reference, **domains[key], directions={})
        for direction, name in DIRECTIONS.items():
            current = current_stratum(packet, direction, domains[key]['required_actor_count'], topology,
                                      lanes, action['applied_states'])
            if current is None:
                value = dict(status='NO_ADJACENT_CURRENT_STRATUM', current_stratum=None,
                    speed_mps=packet['speed_mps'], lane_end_distance_m=
                    topology.lanes[packet['lane_id']]['length_m']-packet['position_m'])
            else:
                stratum, speed, distance = current
                value = dict(status='RECORDED',
                    current_stratum=[*stratum[:4], list(stratum[4]), *stratum[5:]],
                    speed_mps=speed, lane_end_distance_m=distance)
            row['directions'][name] = value
        result.append(row)
    return result


def _coverage_summary(rows):
    covered = set()
    for row in rows:
        if row['y'] == 1:
            covered.update(tuple(key) for value in row['directions'].values()
                           for key in value['matched_positive_keys'])
    return {name:dict(positive_rows=sum(row['y'] == direction for row in rows),
                positives_with_population_stay_match=sum(identity(row) in covered for row in rows if row['y'] == direction))
            for direction, name in DIRECTIONS.items()}


def census_population(base, output):
    """Return all30000 compact rows and save their server-only census.

    The existing whole-J count is read from the completed conditional census.
    No kernel, canonical features, physical responses or904 states are queried.
    """
    began = time.monotonic()
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (base/'target_packet_training/training_reservoir_server_only.json').open() as stream:
        manifest = json.load(stream)
    references = {}
    for seed in SEEDS:
        rows = manifest[str(seed)]['lane_change']
        if (len(rows) != 10000 or len({identity(row) for row in rows}) != 10000
                or any(int(row['seed']) != seed or row['time_sec'] != int(row['time_sec'])
                       or row['y'] not in (0, 1, 2) for row in rows)):
            raise ValueError(f'The original LC reservoir population changed for seed{seed}')
        references[seed] = [dict(seed=seed, time_sec=int(row['time_sec']), packet_id=row['packet_id'],
                                y=row['y'], manifest_ordinal=ordinal)
                            for ordinal, row in enumerate(rows)]
    if Counter(row['y'] for rows in references.values() for row in rows) != {0:165, 1:29669, 2:166}:
        raise ValueError('The fixed30000-row LC population class counts changed')
    with (base/'target_packet_training/static_graph_server_only.json').open() as stream:
        topology = RoadTopology(json.load(stream))
    population, retrieval = [], {}
    for seed, rows in references.items():
        domains = _domain_metadata(base/'target_packet_conditional_census/output'/f'seed{seed}_cases_server_only.jsonl.gz', rows)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['time_sec']].append(row)
        frames, streamed = 0, 0
        archive = base/'target_packet_training/training'/f'seed{seed}'
        for frame, action, source_lines in selected_current_records(archive, grouped):
            population.extend(_current_rows(frame, action, grouped[int(frame['time_s'])], domains, topology))
            frames, streamed = frames+1, source_lines
        retrieval[str(seed)] = dict(current_frames_decoded=frames,
            current_action_line_pairs_streamed=streamed, original_domain_references=len(domains))
    population = assign_bins(sorted(population, key=identity))
    path = output/'population_server_only.jsonl.gz'
    with gzip.open(path, 'xt', compresslevel=1) as stream:
        for row in population:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    negatives = [row for row in population if row['y'] == 1]
    summary = dict(status='COMPLETE_CURRENT_ONLY_LC_NEGATIVE_CENSUS', population_rows=len(population),
        positive_rows=len(population)-len(negatives), stay_rows=len(negatives),
        class_counts=dict(Counter(row['y'] for row in population)),
        bins={name:sum(row['bin'] == name for row in negatives) for name in BINS},
        bins_by_seed={str(seed):{name:sum(row['seed'] == seed and row['bin'] == name for row in negatives)
                               for name in BINS} for seed in SEEDS},
        directional_status_counts={name:dict(Counter(row['directions'][name]['status'] for row in population))
                                   for name in DIRECTIONS.values()},
        possible_positive_coverage=_coverage_summary(population), retrieval=retrieval,
        speed_tolerance_mps=SPEED_TOLERANCE, lane_end_distance_tolerance_m=DISTANCE_TOLERANCE,
        matching='Original current_stratum, exact lane/direction, seed omitted; only observed positive direction.',
        population_file=path.name, new_full_forward_calls=0, new_canonical_feature_evaluations=0,
        new_native_starts=0, new_fits=0, optimizer_invocations=0, elapsed_sec=time.monotonic()-began)
    (output/'census_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return population, summary
