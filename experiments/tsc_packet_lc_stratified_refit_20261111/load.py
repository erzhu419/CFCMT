"""Load the complete frozen stratified cohort without new physical queries."""

from collections import Counter
import gzip
import json
from math import fsum, isclose, isfinite
from pathlib import Path

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_lc_negative_registration_20261111.sample import (
    POPULATION_CLASS_COUNTS, SEEDS,
)
from experiments.tsc_packet_mixture_confirmation_20261111.evaluate import frozen_candidate, read_json
from experiments.tsc_packet_mixture_confirmation_20261111.readout import (
    MixtureParticleLaw, _proposal_components,
)
from experiments.tsc_packet_real_particle_pilot_20261110.register import reference_key
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


POINTS = ('zero', 'original', 'frequency_bias', 'refitted')
LIMITS = dict(new_full_forward_calls=32384, new_canonical_feature_evaluations=60888,
              exact_leaf_visits=110478)


def _check_receipt(receipt):
    if (receipt['status'] != 'COMPLETE_STRATIFIED_FOUR_POINT_CACHE'
            or not receipt['all_four_points_supported'] or receipt['error'] is not None
            or receipt['completed_case_attempts'] != 1000 or not receipt['consumed_counters_exact']
            or receipt['stopped_after_failure'] or receipt['failed_cases']
            or receipt['incomplete_started_cases'] or receipt['undispatched_case_indices']
            or receipt['reused_case_count'] != 333
            or receipt['calibration_seeds_used']
            or any(receipt[name] != 0 for name in ('new_native_starts', 'new_nuisance_estimations',
                   'new_fits', 'optimizer_invocations', 'new_synthetic_full_forward_calls'))
            or any(receipt[name] is None or not 0 <= receipt[name] <= limit
                   for name, limit in LIMITS.items())):
        raise ValueError('The complete supported 1000-case cache and exact bounded counters are required')
    native = receipt['aggregate']['native']
    if (not receipt['aggregate']['native_support_passed']
            or native['reference_count'] != 1000
            or set(native['points']) != set(POINTS)
            or not isclose(native['objective_weight_sum'], 1., rel_tol=0., abs_tol=1e-12)
            or any(not value[view]['all_references_supported']
                   or value[view]['weighted_nll_contribution']['state'] != 'finite'
                   or not value[view]['weighted_gradient']['defined']
                   for value in native['points'].values() for view in ('primary', '128'))):
        raise ValueError('All four fixed points and both mixed prefixes must retain native support')


def load_training(base):
    """Return all 1000 laws, their unchanged task metadata, and the cache receipt.

    Every mixed law uses the original first-fit candidate for its fixed proposal
    denominator. The caller initializes LCObjective with that same original
    candidate, independently of the returned point from the earlier 424-row fit.
    """
    base = Path(base)
    root = base/'target_packet_lc_negative_cache/evaluation'
    receipt = read_json(root/'result.json')
    _check_receipt(receipt)
    prepared = read_json(root/'prepared_server_only.json')
    tasks = prepared['cases']
    anchor_file = base/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    anchor = frozen_candidate(base)
    if (prepared['status'] != 'PASS_PREPARED_LC_NEGATIVE_COHORT' or len(tasks) != 1000
            or prepared['method_counts'] != {'exact': 921, 'mixture256': 79}
            or prepared['reused_exact_cases'] != 325 or prepared['reused_mixture_cases'] != 8
            or Path(prepared['proposal_anchor_file']) != anchor_file
            or {task['case_index'] for task in tasks} != set(range(1000))
            or len({reference_key(task['reference']) for task in tasks}) != 1000):
        raise ValueError('The frozen stratified tasks and original proposal identity are required')

    laws, references, summaries = {}, [], []
    for task in tasks:
        index, ref = task['case_index'], task['reference']
        row = read_json(root/task['stem']/'summary.json')
        reused = task['proposed_reused_cache']
        if (row['status'] != 'COMPLETE_STRATIFIED_CACHE_CASE' or not row['native_support_passed']
                or index in laws or row['case_index'] != index or row['reference'] != ref
                or ref['seed'] not in SEEDS or reference_key(task) != reference_key(ref)
                or row['reused_cache'] != (reused is not None)
                or any(row[name] != task[name] for name in
                       ('seed', 'y', 'raw_weight', 'objective_weight', 'proposed_method'))
                or task['y'] not in (0, 1, 2)
                or not isfinite(task['raw_weight']) or task['raw_weight'] <= 0.
                or not isclose(task['objective_weight'], task['raw_weight']/30000., rel_tol=0., abs_tol=1e-15)
                or not isclose(task['raw_weight']*task['inclusion_probability'], 1., rel_tol=0., abs_tol=1e-12)
                or (task['y'] != 1 and task['raw_weight'] != 1.)
                or Path(task['proposal_anchor_file']) != anchor_file):
            raise ValueError('A completed cache changed its selected identity, label or natural weight')
        cache_path = Path(row['cache_path'])
        if reused is not None:
            if (cache_path != Path(reused['particle_path'])
                    or reused['reference'] != ref
                    or reused['stochastic_participants'] != task['required_participants']
                    or reused['frame_actor_count'] != task['frame_actor_count']
                    or any(row[name] != 0 for name in ('physical_calls', 'canonical_feature_evaluations', 'leaf_visits'))):
                raise ValueError('A reused cache changed its saved identity or was charged new work')
        else:
            filename = ('exact_cache_server_only.json.gz' if task['proposed_method'] == 'exact'
                        else 'mixture_cache_server_only.json.gz')
            if cache_path != root/task['stem']/filename:
                raise ValueError('A new cache no longer belongs to its frozen case directory')
        if any(not 0 <= row[counter] <= task[limit] for counter, limit in (
                ('physical_calls', 'physical_limit'),
                ('canonical_feature_evaluations', 'feature_limit'), ('leaf_visits', 'leaf_limit'))):
            raise ValueError('A completed cache exceeded its registered case budget')
        with gzip.open(cache_path, 'rt') as stream:
            payload = json.load(stream)
        if (payload['focal_id'] != ref['packet_id']
                or payload['stochastic_participants'] != task['required_participants']
                or len(task['required_participants']) != task['required_actor_count']
                or payload['frame_actor_count'] != task['frame_actor_count']
                or payload['eligibility'] != task['eligibility'] or not payload['eligibility']['lane_change']
                or payload['parameter_names'] != list(PARAMETERS)
                or payload['feature_names'] != list(FEATURE_NAMES)):
            raise ValueError('A cache changed its frozen feature layout or complete geometric domain')
        if task['proposed_method'] == 'exact':
            if reused is not None and reused['kind'] != 'exact':
                raise ValueError('An exact case changed its saved cache kind')
            law = CachedLaw(payload)
        elif task['proposed_method'] == 'mixture256':
            proposal = payload['proposal']
            if (len(payload['particles']) != 256
                    or proposal['design_weights'] != {'0': .5, '1': .5}
                    or proposal['seeds'] != {'0': task['seed_zero'], '1': task['seed_fitted']}
                    or (reused is not None and (reused['kind'] != 'mixture'
                        or reused['particle_count'] != 256 or Path(reused['anchor_file']) != anchor_file))):
                raise ValueError('A mixed case changed its fixed 256-particle proposal or original anchor')
            law = MixtureParticleLaw(payload, anchor)
            _proposal_components(law, None)
        else:
            raise ValueError('The frozen exact/mixture method changed')
        laws[index] = law
        references.append(dict(task, cache_path=str(cache_path)))
        summaries.append(row)

    if (Counter(ref['y'] for ref in references) != {0: 165, 1: 669, 2: 166}
            or Counter(ref['proposed_method'] for ref in references) != {'exact': 921, 'mixture256': 79}
            or sum(row['reused_cache'] and row['proposed_method'] == 'exact' for row in summaries) != 325
            or sum(row['reused_cache'] and row['proposed_method'] == 'mixture256' for row in summaries) != 8
            or not isclose(fsum(ref['raw_weight'] for ref in references), 30000., rel_tol=0., abs_tol=1e-9)
            or not isclose(fsum(ref['objective_weight'] for ref in references), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError('The complete natural-frequency training cohort changed')
    for seed in SEEDS:
        current = [ref for ref in references if ref['seed'] == seed]
        if sum(ref['y'] == 1 for ref in current) != 223:
            raise ValueError('A frozen per-seed negative sample count changed')
        for y, expected in POPULATION_CLASS_COUNTS[seed].items():
            if not isclose(fsum(ref['raw_weight'] for ref in current if ref['y'] == y),
                           expected, rel_tol=0., abs_tol=1e-9):
                raise ValueError('A frozen per-seed natural class weight changed')
    for counter, total in (('physical_calls', 'new_full_forward_calls'),
                           ('canonical_feature_evaluations', 'new_canonical_feature_evaluations'),
                           ('leaf_visits', 'exact_leaf_visits')):
        if sum(row[counter] for row in summaries) != receipt[total]:
            raise ValueError('The complete cache counters did not reproduce their saved receipt')
    return laws, references, receipt
