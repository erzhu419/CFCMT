"""Frozen row-level stratified sample with natural LC reservoir weights."""

from collections import Counter
from math import fsum, isclose

import numpy as np


SEEDS = (76116901, 76116902, 76116903)
BINS = ('00', '10', '01', '11')
RANDOM_SEED = 20261111
STAYS_PER_SEED = 223
PER_SEED_RECORDS = 10000
POPULATION_SIZE = len(SEEDS) * PER_SEED_RECORDS
POPULATION_CLASS_COUNTS = {
    SEEDS[0]: {0: 51, 1: 9896, 2: 53},
    SEEDS[1]: {0: 51, 1: 9887, 2: 62},
    SEEDS[2]: {0: 63, 1: 9886, 2: 51},
}


def _identity(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def allocate_counts(population_sizes, budget=STAYS_PER_SEED):
    """Apply the registered capped allocation and fixed-order remainder rule."""
    if set(population_sizes) != set(BINS) or any(
            n != int(n) or n < 0 for n in population_sizes.values()):
        raise ValueError('Expected nonnegative integer sizes for all four bins')
    if budget != int(budget) or not 0 <= budget <= sum(population_sizes.values()):
        raise ValueError('The sample budget must fit the finite population')
    sizes = {name: int(population_sizes[name]) for name in BINS}
    nonempty = [name for name in BINS if sizes[name]]
    allocated = dict.fromkeys(BINS, 0)
    if not nonempty:
        return allocated
    base = int(budget) // len(nonempty)
    allocated.update({name: min(sizes[name], base) for name in nonempty})
    remaining = int(budget) - sum(allocated.values())
    while remaining:
        for name in BINS:
            if allocated[name] < sizes[name]:
                allocated[name] += 1
                remaining -= 1
                if not remaining:
                    break
    return allocated


def _check_total(actual, expected):
    if not isclose(actual, expected, rel_tol=0., abs_tol=1e-9):
        raise ValueError(f'Natural reservoir weight changed: {actual} versus {expected}')


def _weights(rows):
    return dict(
        raw_weight_by_class={str(y): fsum(r['raw_weight'] for r in rows if r['y'] == y)
                             for y in (0, 1, 2)},
        objective_weight_by_class={str(y): fsum(r['objective_weight'] for r in rows if r['y'] == y)
                                   for y in (0, 1, 2)},
        raw_weight_sum=fsum(r['raw_weight'] for r in rows),
        objective_weight_sum=fsum(r['objective_weight'] for r in rows),
    )


def select_cohort(population_rows):
    """Return the fixed 1000-row sample and its finite-population receipt.

    Rows have flat seed/time_sec/packet_id/y fields. Stay ``bin`` values are
    00,10,01,11 (left overlap, right overlap); positive rows have bin=positive.
    Census metadata is retained. Sampling never filters or deduplicates vehicles:
    each distinct (seed,time_sec,packet_id) reservoir row is a sampling unit.
    """
    population = sorted(list(population_rows), key=_identity)
    if (len(population) != POPULATION_SIZE
            or len({_identity(row) for row in population}) != POPULATION_SIZE
            or any(row['seed'] not in SEEDS or row['time_sec'] != int(row['time_sec'])
                   or row['y'] not in (0, 1, 2) for row in population)):
        raise ValueError('The original 30000 distinct training LC reservoir rows are required')
    if any((row['y'] == 1 and row['bin'] not in BINS)
           or (row['y'] != 1 and row['bin'] != 'positive') for row in population):
        raise ValueError('Stay rows need a fixed four-way bin; positive rows need bin=positive')

    selected, per_seed = [], {}
    for seed in SEEDS:
        rows = [row for row in population if row['seed'] == seed]
        counts = Counter(row['y'] for row in rows)
        if dict(counts) != POPULATION_CLASS_COUNTS[seed]:
            raise ValueError(f'The frozen reservoir class counts changed for seed {seed}')
        positives = [row for row in rows if row['y'] != 1]
        current = [dict(row, head='lane_change', split='train',
                        inclusion_probability=1., raw_weight=1.,
                        objective_weight=1./POPULATION_SIZE,
                        stratum_size=len(positives), sampled_stratum_size=len(positives),
                        selection_provenance='all_positive') for row in positives]
        bins = {name: [row for row in rows if row['y'] == 1 and row['bin'] == name]
                for name in BINS}
        sizes = {name: len(bins[name]) for name in BINS}
        allocation = allocate_counts(sizes)
        bin_receipts = {}
        for index, name in enumerate(BINS):
            size, take = sizes[name], allocation[name]
            probability = take/size if size else None
            weight = size/take if take else None
            if take:
                rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, seed, index]))
                indices = rng.choice(size, size=take, replace=False)
                current.extend(dict(bins[name][int(i)], head='lane_change', split='train',
                                    inclusion_probability=probability, raw_weight=weight,
                                    objective_weight=weight/POPULATION_SIZE,
                                    stratum_size=size, sampled_stratum_size=take,
                                    selection_provenance='current_strata_random_stay')
                               for i in indices)
            bin_receipts[name] = dict(population_size=size, selected_count=take,
                inclusion_probability=probability, raw_weight=weight,
                objective_weight=weight/POPULATION_SIZE if take else None,
                random_substream=[RANDOM_SEED, seed, index])
        totals = _weights(current)
        for y in (0, 1, 2):
            _check_total(totals['raw_weight_by_class'][str(y)], counts[y])
        _check_total(totals['raw_weight_sum'], PER_SEED_RECORDS)
        _check_total(totals['objective_weight_sum'], 1./len(SEEDS))
        per_seed[str(seed)] = dict(
            population_references=len(rows), selected_references=len(current),
            population_class_counts={str(y): counts[y] for y in (0, 1, 2)},
            selected_class_counts={str(y): sum(row['y'] == y for row in current) for y in (0, 1, 2)},
            bins=bin_receipts, **totals)
        selected.extend(current)

    selected.sort(key=_identity)
    counts = Counter(row['y'] for row in selected)
    if len(selected) != 1000 or dict(counts) != {0: 165, 1: 669, 2: 166}:
        raise ValueError('The frozen 331-positive plus 669-stay selection changed')
    totals = _weights(selected)
    _check_total(totals['raw_weight_sum'], POPULATION_SIZE)
    _check_total(totals['objective_weight_sum'], 1.)
    summary = dict(
        status='PASS_STRATIFIED_NEGATIVE_COHORT', training_seeds=list(SEEDS),
        sampling_unit='(seed,time_sec,packet_id) reservoir row', random_seed=RANDOM_SEED,
        bin_order=list(BINS), stay_budget_per_seed=STAYS_PER_SEED,
        population_references=len(population), selected_references=len(selected),
        selected_class_counts={str(y): counts[y] for y in (0, 1, 2)},
        per_seed=per_seed, **totals,
        objective='sum(raw_weight * native LC NLL) / 30000; identical gradient weights',
        weighting_scope='Fixed-bin SRS without replacement; all positive rows retained; restore every seed/bin natural total.',
    )
    return selected, summary
