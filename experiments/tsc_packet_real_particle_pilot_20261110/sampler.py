"""The frozen particle law with numeric-only current-goal preparation memo.

One copied base kernel and one shallow physical view are retained. Goal tuples
share only canonical feature vectors and permission applicability; their full
frames and resolver contexts never accumulate in the memo.
"""
from copy import copy
import time

import numpy as np

from experiments.tsc_packet_conditional_particles_20261110.particles import Sampler
from experiments.tsc_packet_discrete_braking_20261110.motion import build_context
from experiments.tsc_packet_joint_law_definition_20261110.law import OutsidePrototype


class LightSampler(Sampler):
    """Preserve proposal draws and likelihoods while bounding prepared views."""

    def __init__(self, kernel, focal_id, *, max_stochastic=10, progress=None,
                 physical_limit=1024, feature_limit=None):
        super().__init__(kernel, focal_id, max_stochastic=max_stochastic)
        self.physical_limit = int(physical_limit)
        self.feature_limit = (self.physical_limit*len(self.participants)
                              if feature_limit is None else int(feature_limit))
        if self.physical_limit < 0 or self.feature_limit < 0:
            raise ValueError('Pilot computation limits must be nonnegative')
        self.progress = progress
        self._started = time.monotonic()
        self._last_reported_calls = -1
        self._physical_view = copy(self.kernel)
        self._physical_view.state = dict(self.kernel.state,
            intents=dict(self.kernel.state['intents']))
        self._physical_view._connections = dict(self.kernel._connections)
        self._current_goal_key = None
        self._current_context = None
        self.stats.update(canonical_feature_evaluations=0, feature_elapsed_sec=0.,
            physical_context_builds=0, physical_context_elapsed_sec=0.,
            base_kernel_deepcopies=1, live_goal_views=1, peak_live_goal_views=1,
            retained_current_contexts=0, peak_retained_current_contexts=0)

    def _budget_error(self, message):
        error = OutsidePrototype(message)
        error.physical_calls = self.stats['physical_calls']
        error.canonical_feature_evaluations = self.stats['canonical_feature_evaluations']
        error.sampler_stats = dict(self.stats)
        return error

    def _report(self, *, final=False):
        completed = self.stats['physical_calls']
        due = completed > 0 and completed % 128 == 0
        if self.progress is not None and (final or due) and completed != self._last_reported_calls:
            self.progress(dict(self.stats, elapsed_sec=time.monotonic()-self._started,
                               complete=final))
            self._last_reported_calls = completed

    def _configuration(self, goals):
        # This hook runs immediately before each inherited particle iteration.
        # The parent increments the physical count before forecasts/resolution.
        self._report()
        if self.stats['physical_calls'] >= self.physical_limit:
            raise self._budget_error('Registered pilot full-forward limit reached')
        key = tuple(goals)
        k = self._physical_view
        if key != self._current_goal_key:
            # Outside J remains the frozen base intervention. Reset current
            # goal dictionaries; the complete observed packet frame is shared.
            k.state = dict(self.kernel.state, intents=dict(self.kernel.state['intents']))
            k._connections = dict(self.kernel._connections)
            for actor, goal in zip(self.participants, goals):
                packet = k.state['packets'][actor]
                geometry = k.topology.lanes[packet['lane_id']]
                if not geometry['internal']:
                    k.state['intents'][actor] = dict(road_id=geometry['edge_id'], choice=goal)
                k._connections[actor] = k._connection(packet)
            context_began = time.monotonic()
            self._current_context = build_context(k)
            self.stats['physical_context_elapsed_sec'] += time.monotonic()-context_began
            self.stats['physical_context_builds'] += 1
            self.stats['retained_current_contexts'] = 1
            self.stats['peak_retained_current_contexts'] = 1
            self._current_goal_key = key
        if key not in self.configurations:
            began = time.monotonic()
            phis, applicable = {}, {}
            try:
                packets = k.state['packets']
                for actor in self.participants:
                    if self.stats['canonical_feature_evaluations'] >= self.feature_limit:
                        raise self._budget_error('Registered pilot canonical-feature limit reached')
                    self.stats['canonical_feature_evaluations'] += 1
                    packet = packets[actor]
                    connection, compatible = k._connections[actor]
                    phis[actor] = np.asarray(k.model.features(
                        k._context(packet, packets, connection, compatible)), dtype=float)
                    applicable[actor] = dict(start=packet['speed_mps'] < .1,
                        yielding=(not k.topology.lanes[packet['lane_id']]['internal']
                                  and connection is not None and compatible
                                  and k._signal(connection, compatible) == 'g'))
                self.configurations[key] = dict(phi=phis, applicability=applicable)
                self.stats['prepared_goal_configurations'] += 1
            finally:
                self.stats['feature_elapsed_sec'] += time.monotonic()-began
        numeric = self.configurations[key]
        return k, self._current_context, numeric['phi'], numeric['applicability']

    def draw(self, n, seed):
        if self.stats['physical_calls']+n > self.physical_limit:
            raise self._budget_error('Requested pool exceeds remaining pilot full-forward budget')
        features_before = self.stats['canonical_feature_evaluations']
        feature_time_before = self.stats['feature_elapsed_sec']
        contexts_before = self.stats['physical_context_builds']
        try:
            pool = super().draw(n, seed)
        except Exception as error:
            error.physical_calls = self.stats['physical_calls']
            error.canonical_feature_evaluations = self.stats['canonical_feature_evaluations']
            error.sampler_stats = dict(self.stats)
            error.current_goal_tuple = self._current_goal_key
            raise
        pool.stats.update(canonical_feature_evaluations=self.stats['canonical_feature_evaluations'],
            canonical_feature_evaluations_new=self.stats['canonical_feature_evaluations']-features_before,
            feature_elapsed_sec=self.stats['feature_elapsed_sec'],
            feature_elapsed_sec_new=self.stats['feature_elapsed_sec']-feature_time_before,
            physical_context_builds=self.stats['physical_context_builds'],
            physical_context_builds_new=self.stats['physical_context_builds']-contexts_before,
            base_kernel_deepcopies=1, live_goal_views=1, peak_live_goal_views=1,
            retained_current_contexts=self.stats['retained_current_contexts'],
            peak_retained_current_contexts=self.stats['peak_retained_current_contexts'])
        self._report(final=True)
        return pool


def make_sampler(kernel, focal_id, *, max_stochastic=10, progress=None,
                 physical_limit=1024, feature_limit=None):
    return LightSampler(kernel, focal_id, max_stochastic=max_stochastic,
                        progress=progress, physical_limit=physical_limit,
                        feature_limit=feature_limit)
