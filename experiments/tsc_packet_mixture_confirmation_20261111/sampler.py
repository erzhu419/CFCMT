"""Fresh balanced proposal pools using the frozen physical prefix program."""

from copy import copy
from itertools import cycle
from math import log
import time

import numpy as np
from scipy.special import expit, softmax

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_conditional_panel_20261110.stream import _focal_outcome
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order
from experiments.tsc_packet_joint_law_definition_20261110.law import (
    CHOICES, PARAMETERS, _eligibility, _prior,
)
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets
from experiments.tsc_packet_real_particle_pilot_20261110.sampler import LightSampler
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


class MixtureSampler(LightSampler):
    """Use the original numeric memo and physical model at either proposal."""

    def __init__(self, kernel, focal_id, anchor, **kwargs):
        super().__init__(kernel, focal_id, **kwargs)
        self.anchor = _vectors(anchor).copy()

    def draw(self, n, seed):
        """Pure q0 comparison with the archived sampler's payload format."""
        return self._run(n, {0: seed}, record_proposal=False)

    def draw_component(self, n, seed, component):
        """Expose the sampled density of a single proposal for the gate."""
        if component not in (0, 1):
            raise ValueError('Proposal component must be 0 (q0) or 1 (qf)')
        return self._run(n, {component: seed}, record_proposal=True)

    def draw_mixture(self, n, seed_zero, seed_fitted):
        if n % 2:
            raise ValueError('A balanced pool requires an even number of particles')
        return self._run(n, {0: seed_zero, 1: seed_fitted}, record_proposal=True)

    def _run(self, n, seeds, *, record_proposal):
        if n <= 0:
            raise ValueError('The particle pool must be nonempty')
        if self.stats['physical_calls'] + n > self.physical_limit:
            raise self._budget_error('Requested pool exceeds remaining pilot full-forward budget')
        feature_count = self.stats['canonical_feature_evaluations']
        feature_time = self.stats['feature_elapsed_sec']
        context_count = self.stats['physical_context_builds']
        try:
            pool = self._draw_particles(n, seeds, record_proposal=record_proposal)
        except Exception as error:
            error.physical_calls = self.stats['physical_calls']
            error.canonical_feature_evaluations = self.stats['canonical_feature_evaluations']
            error.sampler_stats = dict(self.stats)
            error.current_goal_tuple = self._current_goal_key
            raise
        pool.stats.update(
            canonical_feature_evaluations=self.stats['canonical_feature_evaluations'],
            canonical_feature_evaluations_new=self.stats['canonical_feature_evaluations'] - feature_count,
            feature_elapsed_sec=self.stats['feature_elapsed_sec'],
            feature_elapsed_sec_new=self.stats['feature_elapsed_sec'] - feature_time,
            physical_context_builds=self.stats['physical_context_builds'],
            physical_context_builds_new=self.stats['physical_context_builds'] - context_count,
            base_kernel_deepcopies=1,
            live_goal_views=1,
            peak_live_goal_views=1,
            retained_current_contexts=self.stats['retained_current_contexts'],
            peak_retained_current_contexts=self.stats['peak_retained_current_contexts'],
        )
        self._report(final=True)
        return pool

    def _draw_particles(self, n, seeds, *, record_proposal):
        began = time.monotonic()
        proposal_stream = cycle((component, np.random.default_rng(seed))
                                for component, seed in seeds.items())
        focal = self.kernel.state['packets'][self.focal_id]
        data = dict(focal_id=self.focal_id, old_stopped=focal['speed_mps'] < .1,
                    eligibility=_eligibility(self.kernel, focal),
                    stochastic_participants=list(self.participants), frame_actor_count=len(self.ids),
                    parameter_names=list(PARAMETERS), feature_names=list(FEATURE_NAMES),
                    seed=int(next(iter(seeds.values()))),
                    features=[], factors=[], focal_groups=[], particles=[], stats={})
        if record_proposal:
            data['proposal'] = dict(
                components={str(i): ('q0' if i == 0 else 'qf') for i in seeds},
                seeds={str(i): int(seed) for i, seed in seeds.items()},
                component_counts={str(i): n // len(seeds) for i in seeds},
                design_weights={str(i): 1. / len(seeds) for i in seeds},
                component_order=list(seeds),
            )
        feature_ids, factor_ids, group_ids = {}, {}, {}

        def feature(phi):
            key = tuple(map(float, phi))
            if key not in feature_ids:
                feature_ids[key] = len(data['features'])
                data['features'].append(list(key))
            return feature_ids[key]

        def factor(kind, actor, phi, *, block=None, bit=None, allowed=None, choice=None):
            key = (kind, actor, phi, block, bit, tuple(allowed or ()), choice)
            if key not in factor_ids:
                factor_ids[key] = len(data['factors'])
                row = dict(kind=kind, actor=actor, feature_id=phi)
                row.update(dict(block=block, bit=bit) if kind == 'bernoulli'
                           else dict(allowed=list(allowed), choice=choice))
                data['factors'].append(row)
            return factor_ids[key]

        def group(phi, allowed):
            key = (phi, tuple(allowed))
            if key not in group_ids:
                group_ids[key] = len(data['focal_groups'])
                data['focal_groups'].append(dict(feature_id=phi, allowed=list(allowed)))
            return group_ids[key]

        prepared_before = self.stats['prepared_goal_configurations']
        # The order of goal, Bernoulli, and lateral draws follows the original
        # Sampler.draw. Each component advances only its own generator.
        for _, (component, rng) in zip(range(n), proposal_stream):
            goals = [self._goal(rng, prior) for prior in self.goals]
            k, context, phis, applicable = self._configuration(goals)
            ids_phi = {actor: feature(phis[actor]) for actor in self.participants}
            decisions = {actor: dict(motion_residual_mps=0.) for actor in self.ids}
            permissions = {actor: dict(start=True, **{'yield': True}) for actor in self.ids}
            factors = []
            sampled_log_ratio = 0.
            for actor in self.participants:
                packet = k.state['packets'][actor]
                bits = [1, 1, 1]
                for block, active in enumerate((True, applicable[actor]['start'],
                                                 applicable[actor]['yielding'])):
                    if active:
                        if component == 0:
                            bit = int(rng.random() < .5)
                        else:
                            z = float(self.anchor[block] @ phis[actor])
                            bit = int(rng.random() < expit(z))
                            sampled_log_ratio += log(2.) - float(np.logaddexp(0., -z if bit else z))
                        bits[block] = bit
                        factors.append(factor('bernoulli', actor, ids_phi[actor], block=block, bit=bit))
                brake, start, yielding = bits
                decisions[actor] = dict(motion_residual_mps=-packet['parameters']['decel_mps2'] if brake else 0.)
                permissions[actor] = dict(start=bool(start), **{'yield': bool(yielding)})
            self.stats['physical_calls'] += 1
            resolver = MotionResolver(context, k, decisions, permissions)
            ordered = _priority_order(resolver)
            resolver.priority_index = {actor: i for i, actor in enumerate(ordered)}
            for actor in ordered:
                resolver.resolve(actor)
            results = resolver.results
            projected = {actor: row['projected_packet'] for actor, row in results.items() if not row['ended']}
            outcome = _focal_outcome(k, k.state['packets'][self.focal_id], results[self.focal_id],
                                     projected, {self.focal_id: 'stay'}, self.focal_id)
            focal_group, focal_targets, receiving_goals = -1, {}, []
            if outcome['survived']:
                lc_kernel = copy(k)
                lc_kernel.state = dict(k.state, intents=dict(k.state['intents']))
                for actor in self.participants:
                    if actor in projected:
                        lane = k.topology.lanes[projected[actor]['lane_id']]
                        source_lane = k.topology.lanes[k.state['packets'][actor]['lane_id']]
                        road = lane['edge_id']
                        if (not lane['internal'] and source_lane['edge_id'] != road
                                and len(k.topology.road_lanes(road)) > 1):
                            goal = self._goal(rng, _prior(k, road))
                            receiving_goals.append((actor, goal))
                            lc_kernel.state['intents'][actor] = dict(road_id=road, choice=goal)
                working = projected
                projected_participants = (actor for actor in self.participants if actor in projected)
                for actor in sorted(projected_participants, key=context['packet_order'].__getitem__):
                    packet = working[actor]
                    targets = ({'stay': packet['lane_id']} if lc_kernel.topology.lanes[packet['lane_id']]['internal']
                               else _targets(lc_kernel, packet, working))
                    allowed = [i for i, name in enumerate(CHOICES) if name in targets]
                    if actor == self.focal_id:
                        for choice, target in targets.items():
                            self._move(lc_kernel, working, actor, choice, target)
                        focal_group = group(ids_phi[actor], allowed)
                        focal_targets = targets
                        break
                    if component == 0:
                        chosen_index = int(rng.integers(len(allowed)))
                    else:
                        phi = phis[actor]
                        logits = np.array([self.anchor[3] @ phi, 0., self.anchor[4] @ phi])
                        probabilities = softmax(logits[allowed])
                        chosen_index = int(rng.choice(len(allowed), p=probabilities))
                        sampled_log_ratio += log(float(probabilities[chosen_index])) + log(len(allowed))
                    chosen = allowed[chosen_index]
                    choice = CHOICES[chosen]
                    factors.append(factor('categorical', actor, ids_phi[actor], allowed=allowed, choice=chosen))
                    working = self._move(lc_kernel, working, actor, choice, targets[choice])
            row = dict(factor_ids=factors, focal_group=focal_group, focal_targets=focal_targets,
                       outcome=outcome, current_goals=goals, receiving_goals=receiving_goals)
            if record_proposal:
                row.update(proposal_component=component, proposal_log_ratio=sampled_log_ratio)
            data['particles'].append(row)
        data['stats'].update(physical_calls=n, particle_count=n,
            prepared_goal_configurations_new=self.stats['prepared_goal_configurations'] - prepared_before,
            feature_count=len(data['features']), factor_count=len(data['factors']),
            focal_group_count=len(data['focal_groups']), elapsed_sec=time.monotonic() - began)
        return ParticleLaw(data)
