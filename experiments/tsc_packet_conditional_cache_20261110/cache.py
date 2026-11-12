"""Finite latent branches with reusable weights and exact likelihood gradients.

Construction resolves the frozen conditional law once. Thereafter only cached
Bernoulli/softmax factors and observation endpoints are read; physical motion,
canonical features and sequential legality masks do not depend on theta.
"""
from collections import defaultdict
from copy import copy, deepcopy
from itertools import product
from math import exp, fsum, log
import time

import numpy as np
from scipy.special import expit, logsumexp

from experiments.tsc_packet_conditional_law_20261110.law import _actor_options
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_conditional_panel_20261110.stream import _focal_outcome
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_receiving_context
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order
from experiments.tsc_packet_joint_law_definition_20261110.law import (
    PARAMETERS, CHOICES, OutsidePrototype, _prior, _eligibility,
)
from experiments.tsc_packet_joint_law_definition_20261110.observation import speed_likelihood
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


def _vectors(theta):
    if isinstance(theta, dict):
        theta = np.asarray([theta[name] for name in PARAMETERS], dtype=float)
    else:
        theta = np.asarray(theta, dtype=float)
    if theta.shape != (len(PARAMETERS), len(FEATURE_NAMES)) or not np.isfinite(theta).all():
        raise ValueError('Expected five finite canonical coefficient vectors')
    return theta


class CachedLaw:
    """Shared tables and prefix tree, with no kernel or full background frames."""

    def __init__(self, data):
        self.data = data
        self.focal_id = data['focal_id']
        self.stats = data['stats']
        self.features = np.asarray(data['features'], dtype=float)
        self._last_theta = None
        self._last_evaluation = None

    def to_dict(self):
        return deepcopy(self.data)

    def _evaluate(self, theta):
        theta = _vectors(theta)
        if self._last_theta is not None and np.array_equal(theta, self._last_theta):
            return self._last_evaluation
        nf, shape = len(self.data['factors']), theta.shape
        factor_log = np.empty(nf)
        factor_score = np.zeros((nf,)+shape)
        for i, factor in enumerate(self.data['factors']):
            phi = self.features[factor['feature_id']]
            if factor['kind'] == 'bernoulli':
                block, bit = factor['block'], factor['bit']
                value = float(theta[block] @ phi)
                factor_log[i] = bit*value-np.logaddexp(0., value)
                factor_score[i, block] = (bit-expit(value))*phi
            else:
                scores = np.array([theta[3] @ phi, 0., theta[4] @ phi])
                allowed = np.asarray(factor['allowed'], dtype=int)
                normalizer = float(logsumexp(scores[allowed]))
                chosen = factor['choice']
                factor_log[i] = scores[chosen]-normalizer
                probabilities = np.zeros(3)
                probabilities[allowed] = np.exp(scores[allowed]-normalizer)
                factor_score[i, 3] = (float(chosen == 0)-probabilities[0])*phi
                factor_score[i, 4] = (float(chosen == 2)-probabilities[2])*phi
        prefixes = self.data['prefixes']
        prefix_log = np.zeros(len(prefixes))
        prefix_score = np.zeros((len(prefixes),)+shape)
        for i, node in enumerate(prefixes):
            if node['parent'] >= 0:
                prefix_log[i] = prefix_log[node['parent']]
                prefix_score[i] = prefix_score[node['parent']]
            prefix_log[i] += node['log_prior']
            if node['factor_id'] >= 0:
                prefix_log[i] += factor_log[node['factor_id']]
                prefix_score[i] += factor_score[node['factor_id']]
        indices = [leaf['prefix_id'] for leaf in self.data['leaves']]
        result = (prefix_log[indices], prefix_score[indices])
        self._last_theta = theta.copy()
        self._last_evaluation = result
        return result

    def _mixture(self, theta, values, conditioned):
        log_weights, scores = self._evaluate(theta)
        survived = np.asarray([self.data['outcomes'][leaf['outcome_id']]['survived']
                              for leaf in self.data['leaves']], dtype=bool)
        log_survivor = float(logsumexp(log_weights[survived]))
        survivor = exp(log_survivor)
        selected = survived if conditioned else np.ones(len(survived), dtype=bool)
        log_denominator = log_survivor if conditioned else 0.
        if not np.isfinite(log_denominator):
            return dict(defined=False, likelihood=None, log_likelihood=None, nll=None,
                        gradient=None, survivor_probability=survivor)
        denominator_score = (np.einsum('i,ijk->jk', np.exp(log_weights[selected]-log_denominator), scores[selected])
                             if conditioned else np.zeros_like(scores[0]))
        values = np.asarray(values, dtype=float)
        positive = selected & (values > 0.)
        if not np.any(positive):
            return dict(defined=True, likelihood=0., log_likelihood=-float('inf'), nll=float('inf'),
                        gradient=None, survivor_probability=survivor, supported=False,
                        log_numerator=-float('inf'), log_denominator=log_denominator,
                        numerator_score=None, denominator_score=denominator_score)
        terms = log_weights[positive]+np.log(values[positive])
        log_numerator = float(logsumexp(terms))
        numerator_score = np.einsum('i,ijk->jk', np.exp(terms-log_numerator), scores[positive])
        log_q = log_numerator-log_denominator
        return dict(defined=True, likelihood=exp(log_q), log_likelihood=log_q, nll=-log_q,
                    gradient=denominator_score-numerator_score,
                    survivor_probability=survivor, supported=True,
                    log_numerator=log_numerator, log_denominator=log_denominator,
                    numerator_score=numerator_score, denominator_score=denominator_score)

    def marginal(self, theta, head, observed):
        """Return a selected-head likelihood and its negative-log gradient."""
        if not self.data['eligibility'][head]:
            raise ValueError('The current state is outside this head selector')
        if head == 'motion':
            values = [speed_likelihood(observed, outcome['physical_speed_mps'], self.data['old_stopped'])
                      if outcome['survived'] else 0. for outcome in self._leaf_outcomes()]
        else:
            field = {'start': 'started', 'yield_entry': 'entry_event', 'lane_change': 'lane_change'}[head]
            label = CHOICES[int(observed)] if head == 'lane_change' else observed
            values = [float(outcome[field] == label) for outcome in self._leaf_outcomes()]
        row = self._mixture(theta, values, head in ('motion', 'start', 'lane_change'))
        row.update(reference_measure='delta0_plus_positive_Lebesgue' if head == 'motion' else 'counting',
                   survivor_conditioned=head in ('motion', 'start', 'lane_change'))
        return row

    def class_speed(self, theta, class_label, speed):
        """Joint speed/class likelihood given survival, with its NLL gradient."""
        if not self.data['eligibility']['lane_change']:
            raise ValueError('The current state is outside the LC selector')
        label = CHOICES[int(class_label)]
        values = [speed_likelihood(speed, outcome['physical_speed_mps'], self.data['old_stopped'])
                  if outcome['survived'] and outcome['lane_change'] == label else 0.
                  for outcome in self._leaf_outcomes()]
        row = self._mixture(theta, values, True)
        row.update(reference_measure='counting_times_delta0_plus_positive_Lebesgue', survivor_conditioned=True)
        return row

    def _leaf_outcomes(self):
        return [self.data['outcomes'][leaf['outcome_id']] for leaf in self.data['leaves']]

    def summary(self, theta):
        log_weights, scores = self._evaluate(theta)
        weights = np.exp(log_weights)
        survived = np.asarray([outcome['survived'] for outcome in self._leaf_outcomes()], dtype=bool)
        return dict(total_probability=exp(float(logsumexp(log_weights))),
                    weighted_score=np.einsum('i,ijk->jk', weights, scores),
                    survivor_probability=exp(float(logsumexp(log_weights[survived]))),
                    end_probability=exp(float(logsumexp(log_weights[~survived]))))

    def law(self, theta):
        log_weights, unused_scores = self._evaluate(theta)
        probabilities = defaultdict(list)
        for leaf, weight in zip(self.data['leaves'], np.exp(log_weights)):
            probabilities[leaf['outcome_id']].append(float(weight))
        atoms = [dict(actors={self.focal_id: deepcopy(self.data['outcomes'][key])},
                      probability=fsum(weights), latent_branches=len(weights))
                 for key, weights in probabilities.items()]
        return dict(status='DEFINED_CACHED_CONDITIONAL_LAW', atoms=atoms,
                    total_probability=fsum(atom['probability'] for atom in atoms), focal_id=self.focal_id,
                    old_stopped={self.focal_id: self.data['old_stopped']},
                    eligibility={self.focal_id: dict(self.data['eligibility'])},
                    selection=deepcopy(self.data['selection']),
                    stochastic_participants=list(self.data['stochastic_participants']),
                    stochastic_actor_count=len(self.data['stochastic_participants']),
                    frame_actor_count=self.data['frame_actor_count'],
                    diagnostics=dict(physical_branches=self.stats['physical_calls'],
                                     enumerated_leaf_branches=self.stats['leaf_visits']))


def build_cache(kernel, focal_id, *, max_stochastic=4):
    """Enumerate supported nodes once without evaluating any nonzero theta.

    All bits/classes are enumerated with the frozen physical implementation;
    a failed positive branch rejects construction rather than dropping mass.
    """
    began = time.monotonic()
    selection = select_stochastic(kernel, focal_id, max_stochastic=max_stochastic)
    if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE':
        error = OutsidePrototype(selection['status'])
        error.selection = selection
        raise error
    participants = list(selection['participants'])
    base = deepcopy(kernel)
    old, ids = base.state['packets'], list(base.state['packets'])
    data = dict(focal_id=focal_id, selection=deepcopy(selection),
                stochastic_participants=participants, frame_actor_count=len(ids),
                old_stopped=old[focal_id]['speed_mps'] < .1,
                eligibility=_eligibility(base, old[focal_id]), parameter_names=list(PARAMETERS),
                feature_names=list(FEATURE_NAMES), features=[], factors=[], prefixes=[],
                goal_configurations=[], physical_draws=[], receiving_configurations=[],
                outcomes=[], leaves=[], stats={})
    feature_ids, factor_ids, prefix_ids, outcome_ids = {}, {}, {}, {}

    def feature_id(phi):
        key = tuple(map(float, phi))
        if key not in feature_ids:
            feature_ids[key] = len(data['features'])
            data['features'].append(list(key))
        return feature_ids[key]

    def factor_id(kind, actor, phi_id, *, block=None, bit=None, allowed=None, choice=None):
        key = (kind, actor, phi_id, block, bit, tuple(allowed or ()), choice)
        if key not in factor_ids:
            factor_ids[key] = len(data['factors'])
            row = dict(kind=kind, actor=actor, feature_id=phi_id)
            row.update(dict(block=block, bit=bit) if kind == 'bernoulli'
                       else dict(allowed=list(allowed), choice=choice))
            data['factors'].append(row)
        return factor_ids[key]

    def prefix(parent, factor=-1, log_prior=0.):
        key = (parent, factor, log_prior)
        if key not in prefix_ids:
            prefix_ids[key] = len(data['prefixes'])
            data['prefixes'].append(dict(parent=parent, factor_id=factor, log_prior=log_prior))
        return prefix_ids[key]

    root = prefix(-1)
    goals = []
    for key in participants:
        geometry = base.topology.lanes[old[key]['lane_id']]
        goals.append([(base._connections[key][0]['to_edge'], 1.)] if geometry['internal']
                     else _prior(base, geometry['edge_id']))
    physical_calls = 0
    zero_scores = dict.fromkeys(PARAMETERS, 0.)
    for goal_tuple in product(*goals):
        k = deepcopy(base)
        for key, (goal, unused_probability) in zip(participants, goal_tuple):
            geometry = k.topology.lanes[old[key]['lane_id']]
            if not geometry['internal']:
                k.state['intents'][key] = dict(road_id=geometry['edge_id'], choice=goal)
            k._connections[key] = k._connection(k.state['packets'][key])
        context = prepare_receiving_context(k)
        packets = k.state['packets']
        phis = {key: feature_id(k.model.features(k._context(packets[key], packets, *k._connections[key])))
                for key in participants}
        goal_prefix = prefix(root, log_prior=fsum(log(probability) for unused, probability in goal_tuple))
        goal_id = len(data['goal_configurations'])
        applicability = {}
        for key in participants:
            connection, compatible = k._connections[key]
            applicability[key] = dict(start=old[key]['speed_mps'] < .1,
                yielding=(not k.topology.lanes[old[key]['lane_id']]['internal'] and connection is not None
                          and compatible and k._signal(connection, compatible) == 'g'))
        data['goal_configurations'].append(dict(goals=[dict(actor=key, goal=goal, prior=probability)
            for key, (goal, probability) in zip(participants, goal_tuple)], feature_ids=phis,
            applicability=applicability, prefix_id=goal_prefix))
        options = [_actor_options(k, key, zero_scores) for key in participants]
        for draw in product(*options):
            node = goal_prefix
            draws = []
            decisions = {key: dict(motion_residual_mps=0.) for key in ids}
            permissions = {key: dict(start=True, **{'yield': True}) for key in ids}
            for key, item in zip(participants, draw):
                brake = int(item[0] != 0.)
                node = prefix(node, factor_id('bernoulli', key, phis[key], block=0, bit=brake))
                for flag, block, value in (('start', 1, item[1]), ('yielding', 2, item[2])):
                    if applicability[key][flag]:
                        node = prefix(node, factor_id('bernoulli', key, phis[key], block=block, bit=int(value)))
                draws.append(dict(actor=key, residual_node=brake, residual_mps=item[0],
                                  start_permission=item[1], yield_permission=item[2]))
                decisions[key] = dict(motion_residual_mps=item[0])
                permissions[key] = dict(start=item[1], **{'yield': item[2]})
            physical_calls += 1
            resolver = MotionResolver(context, k, decisions, permissions)
            ordered = _priority_order(resolver)
            resolver.priority_index = {key: index for index, key in enumerate(ordered)}
            for key in ordered:
                resolver.resolve(key)
            results = resolver.results
            draw_id = len(data['physical_draws'])
            data['physical_draws'].append(dict(goal_configuration_id=goal_id, draws=draws,
                                               prefix_id=node))
            projected = {key: result['projected_packet'] for key, result in results.items() if not result['ended']}
            receiving = [key for key in participants if key in projected
                and not k.topology.lanes[projected[key]['lane_id']]['internal']
                and k.topology.lanes[old[key]['lane_id']]['edge_id'] != k.topology.lanes[projected[key]['lane_id']]['edge_id']
                and len(k.topology.road_lanes(k.topology.lanes[projected[key]['lane_id']]['edge_id'])) > 1]
            priors = [_prior(k, k.topology.lanes[projected[key]['lane_id']]['edge_id']) for key in receiving]
            for new_goals in product(*priors):
                lc_kernel = copy(k)
                lc_kernel.state = dict(k.state, intents=dict(k.state['intents']))
                for key, (goal, unused_probability) in zip(receiving, new_goals):
                    lc_kernel.state['intents'][key] = dict(
                        road_id=k.topology.lanes[projected[key]['lane_id']]['edge_id'], choice=goal)
                receiving_prefix = prefix(node, log_prior=fsum(log(probability) for unused, probability in new_goals))
                receiving_id = len(data['receiving_configurations'])
                lc_order = sorted((key for key in participants if key in projected), key=context['packet_order'].__getitem__)
                data['receiving_configurations'].append(dict(physical_draw_id=draw_id, prefix_id=receiving_prefix,
                    goals=[dict(actor=key, goal=goal, prior=probability)
                           for key, (goal, probability) in zip(receiving, new_goals)], lc_order=lc_order))

                def walk(index, working, actions, parent):
                    if index == len(lc_order):
                        outcome = _focal_outcome(k, old[focal_id], results[focal_id], working, actions, focal_id)
                        key = tuple(outcome.items())
                        if key not in outcome_ids:
                            outcome_ids[key] = len(data['outcomes'])
                            data['outcomes'].append(outcome)
                        data['leaves'].append(dict(prefix_id=parent, outcome_id=outcome_ids[key],
                                                   receiving_configuration_id=receiving_id))
                        return
                    actor = lc_order[index]
                    packet = working[actor]
                    targets = ({'stay': packet['lane_id']} if lc_kernel.topology.lanes[packet['lane_id']]['internal']
                               else _targets(lc_kernel, packet, working))
                    allowed = [i for i, name in enumerate(CHOICES) if name in targets]
                    for choice, target in targets.items():
                        child = prefix(parent, factor_id('categorical', actor, phis[actor],
                                                        allowed=allowed, choice=CHOICES.index(choice)))
                        after = working
                        if choice != 'stay':
                            after = dict(working)
                            moved = deepcopy(packet)
                            unresolved = []
                            trail, complete = lc_kernel.sensor_operator._lane_change_trail(
                                packet['body_trail'], target, packet['parameters']['length_m'],
                                packet['position_m'], actor, unresolved)
                            if not complete or unresolved:
                                raise OutsidePrototype('A positive cached LC branch has unresolved body remap')
                            moved['lane_id'] = target
                            lc_kernel._trim_body(moved, trail)
                            after[actor] = moved
                        walk(index+1, after, dict(actions, **{actor: choice}), child)
                walk(0, projected, {}, receiving_prefix)
    data['stats'].update(physical_calls=physical_calls, leaf_visits=len(data['leaves']),
        feature_count=len(data['features']), factor_count=len(data['factors']),
        prefix_count=len(data['prefixes']), leaf_count=len(data['leaves']),
        outcome_count=len(data['outcomes']), elapsed_sec=time.monotonic()-began)
    return CachedLaw(data)
