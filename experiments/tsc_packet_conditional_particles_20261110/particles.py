"""Fixed-proposal prefix particles and a common normalized focal mixture.

All theta readouts reuse frozen physical outcomes and prefix masks. Focal LC
is integrated exactly; its suffix sums to one. This is a finite-sample SNIS
approximation, not an exact population likelihood or population gradient.
"""
from collections import defaultdict
from copy import copy, deepcopy
from math import exp, log
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit, logsumexp

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
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


class ParticleLaw:
    """Parameter reweighting of one fixed particle pool, with no kernel."""

    def __init__(self, data):
        self.data = data
        self.focal_id = data['focal_id']
        self.stats = data['stats']
        self.features = np.asarray(data['features'], dtype=float)
        self.rows = data['particles']
        self.survived = np.asarray([row['outcome']['survived'] for row in self.rows], dtype=bool)
        self.focal_groups = np.asarray([row['focal_group'] for row in self.rows], dtype=int)
        ii, jj = [], []
        for i, row in enumerate(self.rows):
            ii.extend([i]*len(row['factor_ids']))
            jj.extend(row['factor_ids'])
        self.incidence = csr_matrix((np.ones(len(ii)), (ii, jj)),
                                   shape=(len(self.rows), len(data['factors'])))
        self._last_theta = None
        self._last_evaluation = None
        self._values = {}

    def to_dict(self):
        return deepcopy(self.data)

    def _size(self, n):
        n = len(self.rows) if n is None else int(n)
        if not 0 < n <= len(self.rows):
            raise ValueError('The requested prefix must lie inside the fixed particle pool')
        return n

    def _evaluate(self, theta):
        theta = _vectors(theta)
        if self._last_theta is not None and np.array_equal(theta, self._last_theta):
            return self._last_evaluation
        shape = theta.shape
        nf = len(self.data['factors'])
        factor_log_ratio = np.empty(nf)
        factor_score = np.zeros((nf,)+shape)
        for i, factor in enumerate(self.data['factors']):
            phi = self.features[factor['feature_id']]
            if factor['kind'] == 'bernoulli':
                b, bit = factor['block'], factor['bit']
                z = float(theta[b] @ phi)
                factor_log_ratio[i] = bit*z-np.logaddexp(0., z)+log(2.)
                factor_score[i, b] = (bit-expit(z))*phi
            else:
                z = np.array([theta[3] @ phi, 0., theta[4] @ phi])
                allowed = np.asarray(factor['allowed'], dtype=int)
                lz = float(logsumexp(z[allowed]))
                probabilities = np.zeros(3)
                probabilities[allowed] = np.exp(z[allowed]-lz)
                chosen = factor['choice']
                factor_log_ratio[i] = z[chosen]-lz+log(len(allowed))
                factor_score[i, 3] = (float(chosen == 0)-probabilities[0])*phi
                factor_score[i, 4] = (float(chosen == 2)-probabilities[2])*phi
        logw = np.asarray(self.incidence @ factor_log_ratio).ravel()
        scores = np.asarray(self.incidence @ factor_score.reshape(nf, -1)).reshape((len(self.rows),)+shape)
        ng = len(self.data['focal_groups'])
        focal_prob = np.zeros((ng, 3))
        focal_score = np.zeros((ng, 3)+shape)
        for i, group in enumerate(self.data['focal_groups']):
            phi = self.features[group['feature_id']]
            z = np.array([theta[3] @ phi, 0., theta[4] @ phi])
            allowed = np.asarray(group['allowed'], dtype=int)
            lz = float(logsumexp(z[allowed]))
            focal_prob[i, allowed] = np.exp(z[allowed]-lz)
            for chosen in allowed:
                focal_score[i, chosen, 3] = (float(chosen == 0)-focal_prob[i, 0])*phi
                focal_score[i, chosen, 4] = (float(chosen == 2)-focal_prob[i, 2])*phi
        self._last_theta = theta.copy()
        self._last_evaluation = logw, scores, focal_prob, focal_score
        return self._last_evaluation

    @staticmethod
    def _ess(log_values):
        if not len(log_values):
            return 0.
        normalized = np.exp(log_values-logsumexp(log_values))
        return float(1./np.sum(normalized*normalized))

    def _base_values(self, head, observed):
        key = (head, observed)
        if key not in self._values:
            if head == 'motion':
                values = [speed_likelihood(observed, row['outcome']['physical_speed_mps'], self.data['old_stopped'])
                          if row['outcome']['survived'] else 0. for row in self.rows]
            elif head == 'lane_change':
                values = self.survived.astype(float)
            else:
                field = {'start': 'started', 'yield_entry': 'entry_event'}[head]
                values = [float(row['outcome'][field] == observed) for row in self.rows]
            self._values[key] = np.asarray(values, dtype=float)
        return self._values[key]

    def _mixture(self, theta, values, conditioned, *, class_label=None, n=None):
        n = self._size(n)
        logw, scores, focal_prob, focal_score = self._evaluate(theta)
        logw, scores, survived = logw[:n], scores[:n], self.survived[:n]
        log_global = float(logsumexp(logw))
        log_survivor = float(logsumexp(logw[survived]))
        survivor = exp(log_survivor-log_global)
        selected = survived if conditioned else np.ones(n, dtype=bool)
        log_denominator = log_survivor if conditioned else log_global
        common = dict(survivor_probability=survivor, global_ess=self._ess(logw),
                      survivor_ess=self._ess(logw[survived]),
                      max_normalized_weight=float(np.max(np.exp(logw-log_global))))
        if not np.isfinite(log_denominator):
            return dict(defined=False, supported=False, likelihood=None, log_likelihood=None, nll=None,
                        gradient=None, numerator_score=None, denominator_score=None,
                        posterior_ess=0., **common)
        denominator_score = np.einsum('i,ijk->jk',
            np.exp(logw[selected]-log_denominator), scores[selected])
        values = np.array(values[:n], dtype=float, copy=True)
        group_ids = self.focal_groups[:n]
        extra_score = None
        if class_label is not None:
            label = int(class_label)
            values[~survived] = 0.
            values[survived] *= focal_prob[group_ids[survived], label]
            extra_score = np.zeros_like(scores)
            extra_score[survived] = focal_score[group_ids[survived], label]
        positive = selected & (values > 0.)
        if not np.any(positive):
            return dict(defined=True, supported=False, likelihood=0., log_likelihood=-float('inf'),
                        nll=float('inf'), gradient=None, numerator_score=None,
                        denominator_score=denominator_score, log_numerator=-float('inf'),
                        log_denominator=log_denominator, posterior_ess=0., **common)
        terms = logw[positive]+np.log(values[positive])
        log_numerator = float(logsumexp(terms))
        numerator_score = np.einsum('i,ijk->jk', np.exp(terms-log_numerator), scores[positive])
        if extra_score is not None:
            numerator_score += np.einsum('i,ijk->jk', np.exp(terms-log_numerator), extra_score[positive])
        log_q = log_numerator-log_denominator
        return dict(defined=True, supported=True, likelihood=exp(log_q), log_likelihood=log_q,
                    nll=-log_q, gradient=denominator_score-numerator_score,
                    numerator_score=numerator_score, denominator_score=denominator_score,
                    log_numerator=log_numerator, log_denominator=log_denominator,
                    posterior_ess=self._ess(terms), **common)

    def marginal(self, theta, head, observed, n=None):
        if not self.data['eligibility'][head]:
            raise ValueError('The current state is outside this head selector')
        label = int(observed) if head == 'lane_change' else None
        row = self._mixture(theta, self._base_values(head, observed),
            head in ('motion', 'start', 'lane_change'), class_label=label, n=n)
        row.update(reference_measure='delta0_plus_positive_Lebesgue' if head == 'motion' else 'counting',
                   survivor_conditioned=head in ('motion', 'start', 'lane_change'))
        return row

    def class_speed(self, theta, class_label, speed, n=None):
        if not self.data['eligibility']['lane_change']:
            raise ValueError('The current state is outside the LC selector')
        row = self._mixture(theta, self._base_values('motion', speed), True,
                            class_label=class_label, n=n)
        row.update(reference_measure='counting_times_delta0_plus_positive_Lebesgue', survivor_conditioned=True)
        return row

    def summary(self, theta, n=None):
        n = self._size(n)
        logw, scores, unused_p, unused_s = self._evaluate(theta)
        logw, scores, survived = logw[:n], scores[:n], self.survived[:n]
        lz = float(logsumexp(logw))
        probabilities = np.exp(logw-lz)
        average_score = np.einsum('i,ijk->jk', probabilities, scores)
        return dict(total_probability=float(probabilities.sum()),
            survivor_probability=float(probabilities[survived].sum()),
            end_probability=float(probabilities[~survived].sum()),
            weighted_score=np.einsum('i,ijk->jk', probabilities, scores-average_score),
            log_raw_mean_weight=lz-log(n), global_ess=self._ess(logw),
            survivor_ess=self._ess(logw[survived]), max_normalized_weight=float(probabilities.max()))

    def law(self, theta, n=None):
        """Return a normalized observable mixture, with focal LC integrated."""
        n = self._size(n)
        logw, unused_score, focal_prob, unused_focal_score = self._evaluate(theta)
        weights = np.exp(logw[:n]-logsumexp(logw[:n]))
        atoms = []
        for row, weight in zip(self.rows[:n], weights):
            if not row['outcome']['survived']:
                atoms.append(dict(actors={self.focal_id: dict(row['outcome'])}, probability=float(weight)))
            else:
                for label, lane in row['focal_targets'].items():
                    index = CHOICES.index(label)
                    outcome = dict(row['outcome'], lane_change=label, lane_id=lane)
                    atoms.append(dict(actors={self.focal_id: outcome},
                        probability=float(weight*focal_prob[row['focal_group'], index])))
        return dict(atoms=atoms, focal_id=self.focal_id, total_probability=sum(a['probability'] for a in atoms),
            old_stopped={self.focal_id: self.data['old_stopped']},
            eligibility={self.focal_id: dict(self.data['eligibility'])})


class Sampler:
    """Ancestral q0 sampling; only current-goal preparation is memoized."""

    def __init__(self, kernel, focal_id, *, max_stochastic=10):
        selection = select_stochastic(kernel, focal_id, max_stochastic=max_stochastic)
        if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE':
            error = OutsidePrototype(selection['status'])
            error.selection = selection
            raise error
        self.kernel = deepcopy(kernel)
        self.focal_id, self.selection = focal_id, selection
        self.participants = list(selection['participants'])
        self.ids = list(self.kernel.state['packets'])
        self.goals = []
        for key in self.participants:
            packet = self.kernel.state['packets'][key]
            geometry = self.kernel.topology.lanes[packet['lane_id']]
            self.goals.append([(self.kernel._connections[key][0]['to_edge'], 1.)] if geometry['internal']
                              else _prior(self.kernel, geometry['edge_id']))
        self.configurations = {}
        self.stats = dict(physical_calls=0, prepared_goal_configurations=0)

    def _configuration(self, goals):
        key = tuple(goals)
        if key not in self.configurations:
            k = deepcopy(self.kernel)
            for actor, goal in zip(self.participants, goals):
                packet = k.state['packets'][actor]
                geometry = k.topology.lanes[packet['lane_id']]
                if not geometry['internal']:
                    k.state['intents'][actor] = dict(road_id=geometry['edge_id'], choice=goal)
                k._connections[actor] = k._connection(packet)
            context = prepare_receiving_context(k)
            phis = {actor: np.asarray(k.model.features(k._context(k.state['packets'][actor],
                    k.state['packets'], *k._connections[actor])), dtype=float) for actor in self.participants}
            applicable = {}
            for actor in self.participants:
                packet = k.state['packets'][actor]
                connection, compatible = k._connections[actor]
                applicable[actor] = dict(start=packet['speed_mps'] < .1,
                    yielding=(not k.topology.lanes[packet['lane_id']]['internal'] and connection is not None
                              and compatible and k._signal(connection, compatible) == 'g'))
            self.configurations[key] = (k, context, phis, applicable)
            self.stats['prepared_goal_configurations'] += 1
        return self.configurations[key]

    @staticmethod
    def _goal(rng, prior):
        if len(prior) == 1:
            return prior[0][0]
        return prior[int(rng.choice(len(prior), p=[p for unused, p in prior]))][0]

    @staticmethod
    def _move(kernel, working, actor, choice, target):
        if choice == 'stay':
            return working
        packet = working[actor]
        moved, after = deepcopy(packet), dict(working)
        unresolved = []
        trail, complete = kernel.sensor_operator._lane_change_trail(packet['body_trail'], target,
            packet['parameters']['length_m'], packet['position_m'], actor, unresolved)
        if not complete or unresolved:
            raise OutsidePrototype('A positive particle LC branch has unresolved body remap')
        moved['lane_id'] = target
        kernel._trim_body(moved, trail)
        after[actor] = moved
        return after

    def draw(self, n, seed):
        began = time.monotonic()
        rng = np.random.default_rng(seed)
        data = dict(focal_id=self.focal_id, old_stopped=self.kernel.state['packets'][self.focal_id]['speed_mps'] < .1,
            eligibility=_eligibility(self.kernel, self.kernel.state['packets'][self.focal_id]),
            stochastic_participants=list(self.participants), frame_actor_count=len(self.ids),
            parameter_names=list(PARAMETERS), feature_names=list(FEATURE_NAMES), seed=int(seed),
            features=[], factors=[], focal_groups=[], particles=[], stats={})
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
        for index in range(n):
            goals = [self._goal(rng, prior) for prior in self.goals]
            k, context, phis, applicable = self._configuration(goals)
            ids_phi = {actor: feature(phis[actor]) for actor in self.participants}
            decisions = {actor: dict(motion_residual_mps=0.) for actor in self.ids}
            permissions = {actor: dict(start=True, **{'yield': True}) for actor in self.ids}
            factors = []
            for actor in self.participants:
                packet = k.state['packets'][actor]
                brake = int(rng.random() < .5)
                factors.append(factor('bernoulli', actor, ids_phi[actor], block=0, bit=brake))
                start = bool(rng.random() < .5) if applicable[actor]['start'] else True
                yielding = bool(rng.random() < .5) if applicable[actor]['yielding'] else True
                if applicable[actor]['start']:
                    factors.append(factor('bernoulli', actor, ids_phi[actor], block=1, bit=int(start)))
                if applicable[actor]['yielding']:
                    factors.append(factor('bernoulli', actor, ids_phi[actor], block=2, bit=int(yielding)))
                decisions[actor] = dict(motion_residual_mps=-packet['parameters']['decel_mps2'] if brake else 0.)
                permissions[actor] = dict(start=start, **{'yield': yielding})
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
            focal_group, focal_targets = -1, {}
            receiving_goals = []
            if outcome['survived']:
                lc_kernel = copy(k)
                lc_kernel.state = dict(k.state, intents=dict(k.state['intents']))
                for actor in self.participants:
                    if (actor in projected and not k.topology.lanes[projected[actor]['lane_id']]['internal']
                        and k.topology.lanes[k.state['packets'][actor]['lane_id']]['edge_id'] != k.topology.lanes[projected[actor]['lane_id']]['edge_id']
                        and len(k.topology.road_lanes(k.topology.lanes[projected[actor]['lane_id']]['edge_id'])) > 1):
                        road = k.topology.lanes[projected[actor]['lane_id']]['edge_id']
                        goal = self._goal(rng, _prior(k, road))
                        receiving_goals.append((actor, goal))
                        lc_kernel.state['intents'][actor] = dict(road_id=road, choice=goal)
                working = projected
                for actor in sorted((actor for actor in self.participants if actor in projected), key=context['packet_order'].__getitem__):
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
                    choice = CHOICES[allowed[int(rng.integers(len(allowed)))]]
                    factors.append(factor('categorical', actor, ids_phi[actor], allowed=allowed, choice=CHOICES.index(choice)))
                    working = self._move(lc_kernel, working, actor, choice, targets[choice])
            data['particles'].append(dict(factor_ids=factors, focal_group=focal_group,
                focal_targets=focal_targets, outcome=outcome, current_goals=goals, receiving_goals=receiving_goals))
        data['stats'].update(physical_calls=n, particle_count=n,
            prepared_goal_configurations_new=self.stats['prepared_goal_configurations']-prepared_before,
            feature_count=len(data['features']), factor_count=len(data['factors']),
            focal_group_count=len(data['focal_groups']), elapsed_sec=time.monotonic()-began)
        return ParticleLaw(data)


def make_sampler(kernel, focal_id, *, max_stochastic=10):
    return Sampler(kernel, focal_id, max_stochastic=max_stochastic)
