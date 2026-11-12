"""Current-only natural-frequency904 census and frozen stratified sample."""

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from experiments.tsc_packet_conditional_census_20261110.census import selected_current_records
from experiments.tsc_packet_lane_change_coverage_20261111.census import DISTANCE_TOLERANCE, SPEED_TOLERANCE
from experiments.tsc_packet_lc_negative_registration_20261111.census import (
    DIRECTIONS, _current_rows, _domain_metadata, _stratum_key, identity,
)
from experiments.tsc_packet_lc_negative_registration_20261111.sample import (
    BINS, RANDOM_SEED, _check_total, _weights, allocate_counts,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


SEED = 76116904
TRAINING_SEEDS = (76116901, 76116902, 76116903)
POPULATION_SIZE = 10000
POPULATION_CLASS_COUNTS = {0:56, 1:9883, 2:61}
STAY_BUDGET = 223


def assign_training_bins(population_rows, training_positives):
    """Use only observed901-903 positive directions to define904 strata."""
    index = defaultdict(list)
    for positive in training_positives:
        if positive['seed'] not in TRAINING_SEEDS or positive['y'] not in DIRECTIONS:
            raise ValueError('Only original901-903 positives may define the904 sampling bins')
        name = DIRECTIONS[positive['y']]
        value = positive['directions'][name]
        if value['status'] == 'RECORDED':
            index[(name, _stratum_key(value))].append((identity(positive), value))
        elif value['status'] != 'NO_ADJACENT_CURRENT_STRATUM':
            raise ValueError('The frozen training-positive current stratum is missing')
    rows = []
    for original in population_rows:
        row = dict(original, directions={name:dict(value) for name,value in original['directions'].items()})
        near = {}
        for name,value in row['directions'].items():
            if value['status'] not in ('RECORDED', 'NO_ADJACENT_CURRENT_STRATUM'):
                raise ValueError(f'Missing archived904 current stratum at {identity(row)}')
            candidates = index[(name, _stratum_key(value))] if value['status'] == 'RECORDED' else ()
            matches = [list(key) for key,other in candidates
                       if abs(value['speed_mps']-other['speed_mps']) <= SPEED_TOLERANCE
                       and abs(value['lane_end_distance_m']-other['lane_end_distance_m']) <= DISTANCE_TOLERANCE]
            value['matched_positive_keys'] = sorted(matches)
            near[name] = bool(matches)
        row.update(near_left=near['left'], near_right=near['right'],
                   bin=f'{int(near["left"])}{int(near["right"])}' if row['y'] == 1 else 'positive')
        rows.append(row)
    return rows


def select_cohort(population_rows):
    """Select all117 positives and223 SRS stays with denominator10000."""
    population = sorted(population_rows, key=identity)
    if (len(population) != POPULATION_SIZE
            or len({identity(row) for row in population}) != POPULATION_SIZE
            or any(row['seed'] != SEED or row['time_sec'] != int(row['time_sec']) for row in population)
            or Counter(row['y'] for row in population) != POPULATION_CLASS_COUNTS):
        raise ValueError('The original10000 distinct904 LC reservoir rows are required')
    if any((row['y'] == 1 and row['bin'] not in BINS)
           or (row['y'] != 1 and row['bin'] != 'positive') for row in population):
        raise ValueError('Stay rows require frozen training-defined bins and positive rows bin=positive')
    positives = [row for row in population if row['y'] != 1]
    selected = [dict(row, head='lane_change', split='regression', inclusion_probability=1.,
                     raw_weight=1., objective_weight=1./POPULATION_SIZE,
                     stratum_size=len(positives), sampled_stratum_size=len(positives),
                     selection_provenance='all_904_positive') for row in positives]
    bins = {name:[row for row in population if row['y'] == 1 and row['bin'] == name] for name in BINS}
    sizes = {name:len(rows) for name,rows in bins.items()}
    allocation = allocate_counts(sizes, STAY_BUDGET)
    receipts = {}
    for index,name in enumerate(BINS):
        size,take = sizes[name],allocation[name]
        probability,weight = (take/size if size else None), (size/take if take else None)
        if take:
            rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, SEED, index]))
            indices = rng.choice(size, size=take, replace=False)
            selected.extend(dict(bins[name][int(i)], head='lane_change', split='regression',
                inclusion_probability=probability, raw_weight=weight, objective_weight=weight/POPULATION_SIZE,
                stratum_size=size, sampled_stratum_size=take,
                selection_provenance='training_strata_random_904_stay') for i in indices)
        receipts[name] = dict(population_size=size, selected_count=take, inclusion_probability=probability,
            raw_weight=weight, objective_weight=weight/POPULATION_SIZE if take else None,
            random_substream=[RANDOM_SEED, SEED, index])
    selected.sort(key=identity)
    if len(selected) != 340 or Counter(row['y'] for row in selected) != {0:56,1:223,2:61}:
        raise ValueError('The frozen117-positive plus223-stay904 selection changed')
    totals = _weights(selected)
    for label,size in POPULATION_CLASS_COUNTS.items():
        _check_total(totals['raw_weight_by_class'][str(label)], size)
    _check_total(totals['raw_weight_sum'], POPULATION_SIZE)
    _check_total(totals['objective_weight_sum'], 1.)
    return selected, dict(status='PASS_NATURAL_FREQUENCY_904_COHORT', seed=SEED,
        sampling_unit='(seed,time_sec,packet_id) reservoir row', random_seed=RANDOM_SEED,
        bin_order=list(BINS), stay_budget=STAY_BUDGET, population_references=len(population),
        selected_references=len(selected), population_class_counts={str(k):v for k,v in POPULATION_CLASS_COUNTS.items()},
        selected_class_counts={str(k):sum(row['y'] == k for row in selected) for k in (0,1,2)},
        bins=receipts, **totals, objective='sum(raw_weight * native LC NLL) /10000',
        weighting_scope='Fixed training-defined bins, SRS without replacement, all904 positives retained; restore every904 bin total.',
        exposure='Internal natural-frequency regression on previously exposed904; not independent confirmation.')


def census_and_sample(base, output):
    """Read existing current states, freeze the340-row sample, and stop."""
    began = time.monotonic()
    base,output = Path(base),Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training_path = base/'target_packet_lc_negative_registration/registration/census/population_server_only.jsonl.gz'
    training_positives = []
    with gzip.open(training_path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['seed'] in TRAINING_SEEDS and row['y'] != 1:
                training_positives.append(row)
    if (len(training_positives) != 331 or len({identity(row) for row in training_positives}) != 331
            or Counter(row['y'] for row in training_positives) != {0:165,2:166}):
        raise ValueError('The original331 training positives are required for904 sampling strata')
    with (base/'target_packet_training/training_reservoir_server_only.json').open() as stream:
        rows = json.load(stream)[str(SEED)]['lane_change']
    if (len(rows) != POPULATION_SIZE or len({identity(row) for row in rows}) != POPULATION_SIZE
            or Counter(row['y'] for row in rows) != POPULATION_CLASS_COUNTS
            or any(row['seed'] != SEED or row['time_sec'] != int(row['time_sec']) for row in rows)):
        raise ValueError('The original10000-row904 LC population changed')
    references = [dict(seed=SEED, time_sec=int(row['time_sec']), packet_id=row['packet_id'],
                       y=row['y'], manifest_ordinal=ordinal) for ordinal,row in enumerate(rows)]
    domains = _domain_metadata(base/'target_packet_conditional_census/output'/f'seed{SEED}_cases_server_only.jsonl.gz', references)
    with (base/'target_packet_training/static_graph_server_only.json').open() as stream:
        topology = RoadTopology(json.load(stream))
    grouped = defaultdict(list)
    for row in references:
        grouped[row['time_sec']].append(row)
    population,decoded,streamed = [],0,0
    for frame,action,source_lines in selected_current_records(base/'target_packet_training/training'/f'seed{SEED}', grouped):
        population.extend(_current_rows(frame, action, grouped[int(frame['time_s'])], domains, topology))
        decoded,streamed = decoded+1,source_lines
    population = assign_training_bins(sorted(population, key=identity), training_positives)
    selected,summary = select_cohort(population)
    with gzip.open(output/'population_server_only.jsonl.gz', 'xt', compresslevel=1) as stream:
        for row in population:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    summary['census'] = dict(population_rows=len(population), training_positive_rows=len(training_positives),
        training_source=str(training_path), current_frames_decoded=decoded,
        current_action_line_pairs_streamed=streamed, original_domain_references=len(domains),
        directional_status_counts={name:dict(Counter(row['directions'][name]['status'] for row in population))
                                   for name in DIRECTIONS.values()},
        population_file='population_server_only.jsonl.gz', speed_tolerance_mps=SPEED_TOLERANCE,
        lane_end_distance_tolerance_m=DISTANCE_TOLERANCE,
        bin_definition='Proximity only to original901-903 observed positive directions; exact lane/direction and original strata, seed omitted.')
    summary.update(new_full_forward_calls=0,new_canonical_feature_evaluations=0,new_native_starts=0,
                   new_fits=0,optimizer_invocations=0,new_numerical_likelihood_readouts=0,elapsed_sec=time.monotonic()-began)
    (output/'cohort_server_only.json').write_text(json.dumps(dict(summary, references=selected), indent=2, allow_nan=False)+'\n')
    (output/'cohort_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return selected,summary
