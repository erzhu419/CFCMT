"""Register complete LC positives and the original sampled stay records."""

from collections import Counter
from math import fsum, isclose


SEEDS = (76116901, 76116902, 76116903)
PER_SEED_RECORDS = 10000
ORIGINAL_SAMPLE_SIZE = 32
CLASS_COUNTS = {0: 165, 1: 93, 2: 166}


def _identity(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def _check_weight_total(actual, expected):
    if not isclose(actual, expected, rel_tol=0., abs_tol=1e-9):
        raise ValueError(f'Class weighting changed the archive total: {actual} versus {expected}')


def select_cohort(manifest, original_cohort):
    """Return weighted references and the per-seed finite-population receipt.

    ``original_cohort`` is the saved first-fit dictionary with ``references``.
    Only the three training seeds' LC labels enter this registration.
    """
    original = [row for row in original_cohort['references']
                if row['head'] == 'lane_change' and row['split'] == 'train']
    if Counter(int(row['seed']) for row in original) != {
            seed: ORIGINAL_SAMPLE_SIZE for seed in SEEDS}:
        raise ValueError('The original LC training sample must have 32 references per seed')

    references, per_seed = [], {}
    original_labels = Counter()
    denominator = len(SEEDS) * PER_SEED_RECORDS
    for seed in SEEDS:
        rows = manifest[str(seed)]['lane_change']
        if (len(rows) != PER_SEED_RECORDS or len({_identity(row) for row in rows}) != PER_SEED_RECORDS
                or any(int(row['seed']) != seed or row['time_sec'] != int(row['time_sec'])
                       or row['y'] not in (0, 1, 2) for row in rows)):
            raise ValueError(f'The original 10000-record LC reservoir changed for seed {seed}')
        selected = [row for row in original if int(row['seed']) == seed]
        ordinals = {row['manifest_ordinal'] for row in selected}
        if len(ordinals) != ORIGINAL_SAMPLE_SIZE:
            raise ValueError(f'The original sample has duplicate ordinals in seed {seed}')
        for row in selected:
            ordinal = row['manifest_ordinal']
            if not 0 <= ordinal < len(rows) or _identity(rows[ordinal]) != _identity(row):
                raise ValueError(f'Original sampled ordinal or identity differs in seed {seed}')
            original_labels[rows[ordinal]['y']] += 1

        population = Counter(row['y'] for row in rows)
        sampled = Counter(rows[ordinal]['y'] for ordinal in ordinals)
        stay_count = sampled[1]
        if stay_count == 0:
            raise ValueError(f'The original sample has no stay observations in seed {seed}')
        stay_weight = population[1] / stay_count
        current = []
        for ordinal, row in enumerate(rows):
            label = row['y']
            if label == 1 and ordinal not in ordinals:
                continue
            positive = label != 1
            weight = 1. if positive else stay_weight
            provenance = ('original_label_blind_fit_stay' if not positive else
                          'all_positive_and_original_fit' if ordinal in ordinals else 'all_positive')
            current.append(dict(seed=seed, time_sec=int(row['time_sec']), packet_id=row['packet_id'],
                y=label, head='lane_change', split='train', manifest_ordinal=ordinal,
                raw_weight=weight, objective_weight=weight/denominator,
                inclusion_probability=1. if positive else stay_count/population[1],
                provenance=provenance))

        selected_counts = Counter(row['y'] for row in current)
        raw_by_class = {str(label): fsum(row['raw_weight'] for row in current if row['y'] == label)
                        for label in (0, 1, 2)}
        objective_by_class = {str(label): fsum(row['objective_weight'] for row in current if row['y'] == label)
                              for label in (0, 1, 2)}
        for label in (0, 1, 2):
            _check_weight_total(raw_by_class[str(label)], population[label])
        _check_weight_total(fsum(raw_by_class.values()), PER_SEED_RECORDS)
        _check_weight_total(fsum(objective_by_class.values()), 1./len(SEEDS))
        per_seed[str(seed)] = dict(
            original_class_counts={str(label): population[label] for label in (0, 1, 2)},
            original_fit_class_counts={str(label): sampled[label] for label in (0, 1, 2)},
            selected_class_counts={str(label): selected_counts[label] for label in (0, 1, 2)},
            selected_references=len(current), stay_raw_weight=stay_weight,
            stay_inclusion_probability=stay_count/population[1],
            raw_weight_by_class=raw_by_class, raw_weight_total=fsum(raw_by_class.values()),
            objective_weight_by_class=objective_by_class,
            objective_weight_total=fsum(objective_by_class.values()))
        references.extend(current)

    references.sort(key=_identity)
    selected_counts = Counter(row['y'] for row in references)
    if (dict(selected_counts) != CLASS_COUNTS or len(references) != 424
            or len({_identity(row) for row in references}) != 424
            or original_labels != {0: 1, 1: 93, 2: 2}):
        raise ValueError('The frozen 331-positive plus 93-stay cohort changed')
    raw_sum = fsum(row['raw_weight'] for row in references)
    objective_sum = fsum(row['objective_weight'] for row in references)
    _check_weight_total(raw_sum, denominator)
    _check_weight_total(objective_sum, 1.)
    receipt = dict(status='PASS_POSITIVE_COVERED_LC_COHORT', training_seeds=list(SEEDS),
        original_reservoir_records=denominator, selected_references=len(references),
        selected_class_counts={str(label): selected_counts[label] for label in (0, 1, 2)},
        original_fit_positive_references_retained=original_labels[0]+original_labels[2],
        per_seed=per_seed, raw_weight_sum=raw_sum, objective_weight_sum=objective_sum,
        objective='sum(weighted native LC NLL) / 30000; weights apply to gradients identically',
        weighting_scope='Condition on each original label-blind sample stay count. Restore the original LC reservoir class frequencies and equal seed means.')
    return references, receipt


def smoke_check():
    """Check the complete selection and weights on one synthetic reservoir."""
    manifest, old = {}, []
    positive_counts = {SEEDS[0]: (51, 53), SEEDS[1]: (51, 62), SEEDS[2]: (63, 51)}
    for seed, (left, right) in positive_counts.items():
        labels = [0]*left + [2]*right + [1]*(PER_SEED_RECORDS-left-right)
        rows = [dict(seed=seed, time_sec=i//10, packet_id=f's{seed}_v{i}', y=y)
                for i, y in enumerate(labels)]
        manifest[str(seed)] = dict(lane_change=rows)
        positive_ordinals = [0] if seed == SEEDS[0] else [left, left+1] if seed == SEEDS[1] else []
        negative_start = left + right
        ordinals = positive_ordinals + list(range(negative_start,
            negative_start+ORIGINAL_SAMPLE_SIZE-len(positive_ordinals)))
        for ordinal in ordinals:
            row = rows[ordinal]
            old.append(dict(seed=seed, time_sec=row['time_sec'], packet_id=row['packet_id'],
                            head='lane_change', split='train', manifest_ordinal=ordinal))
    references, receipt = select_cohort(manifest, dict(references=old))
    assert len(references) == 424 and receipt['selected_class_counts'] == {'0': 165, '1': 93, '2': 166}
    assert receipt['raw_weight_sum'] == 30000. and receipt['objective_weight_sum'] == 1.
    assert all(row['raw_weight'] == 1. for row in references if row['y'] != 1)
    assert all(isclose(row['raw_weight']*row['inclusion_probability'], 1., abs_tol=1e-12)
               for row in references)
    return dict(status='PASS_SYNTHETIC_COHORT_WEIGHT_SMOKE', references=len(references),
                raw_weight_sum=receipt['raw_weight_sum'], objective_weight_sum=receipt['objective_weight_sum'],
                new_full_forward_calls=0, new_native_starts=0, new_fits=0, optimizer_calls=0)


if __name__ == '__main__':
    import json
    print(json.dumps(smoke_check()))
