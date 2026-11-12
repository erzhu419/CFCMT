"""Conditional passage probabilities and phase scores using only current flight information."""
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import random

import numpy as np


LEARNED = ("T", "S", "P")
PREDICTORS = (*LEARNED, "analytic")
PERMUTATION_SEED = 11072026


@dataclass
class Curve:
    times: list
    survival: list
    straight_incidence: list

    def at(self, duration):
        i = bisect_right(self.times, duration) - 1
        if i < 0:
            return 1.0, 0.0
        return self.survival[i], self.straight_incidence[i]


def estimate(labels):
    """Aalen–Johansen incidence for first straight-lane detection with competing exits."""
    counts = defaultdict(lambda: [0, 0, 0])
    for duration, status in labels:
        if duration < 0 or status not in ("straight", "other", "terminal", "censored"):
            raise ValueError(f"Unusable release record: {duration}, {status}")
        counts[duration][0] += 1
        counts[duration][1] += status != "censored"
        counts[duration][2] += status == "straight"
    times = sorted(counts)
    alive, incidence, at_risk = 1.0, 0.0, len(labels)
    survivals, incidences = [], []
    for duration in times:
        observed, events, straight = counts[duration]
        # Censoring at the same timestamp occurs after the native passage/terminal events.
        incidence += alive * straight / at_risk
        alive *= (at_risk - events) / at_risk
        survivals.append(alive)
        incidences.append(incidence)
        at_risk -= observed
    return Curve(times, survivals, incidences)


def fit_hours(hours, *, permute=False):
    """hours maps SUMO seed to its released-vehicle records. Keep the full competing label together."""
    groups = [[], []]
    rng = random.Random(PERMUTATION_SEED)
    for seed in sorted(hours):
        by_edge = defaultdict(list)
        for record in hours[seed]:
            by_edge[record["edge"]].append(record)
        for edge in sorted(by_edge):
            records = sorted(by_edge[edge], key=lambda r: (r["entry_s"], r["vid"]))
            labels = [((r["end_s"] - r["entry_s"]) / r["freeflow_s"], r["status"]) for r in records]
            if permute:
                rng.shuffle(labels)
            for record, label in zip(records, labels):
                groups[int(record["queued_at_entry"])].append(label)
    return [estimate(labels) for labels in groups]


class ArrivalPolicy:
    def __init__(self, target, source, placebo):
        self.curves = (target, source, placebo)
        # Hours share integer ages and the repaired networks have few distinct lengths.
        self.cached_learned = lru_cache(maxsize=16384)(self._learned)

    def _learned(self, queued_at_entry, age_s, freeflow_s, tail_s):
        age = age_s / freeflow_s
        end = (age_s + 10) / freeflow_s
        numerators, denominators = [], []
        for table in self.curves:
            curve = table[int(queued_at_entry)]
            if not curve.times or (end > curve.times[-1] and curve.survival[-1] > 0):
                return None
            survival, prior = curve.at(age)
            if survival <= 0:
                return None
            denominators.append(survival)
            # First entry by each signal second, shifted from the detector to the line.
            # Column 11 is detector entry by t+10, used solely for forecast validation.
            bounds = [max(0.0, h-tail_s) for h in range(1, 11)] + [10.0]
            numerators.append(np.array([curve.at((age_s+h)/freeflow_s)[1]-prior for h in bounds]))
        t, h, p = numerators
        st, sh, sp = denominators
        return np.array([t / st, (t+h) / (st+sh), (t+p) / (st+sp)])

    def forecast(self, query):
        """Return edge → [T,S,P,analytic] × [10 cumulative line windows, one detector window]."""
        if query["spillback"] or query["unmatched"]:
            return None
        contributions = {}
        for f in query["flights"]:
            edge = f["edge"]
            learned = self.cached_learned(f["queued_at_entry"], f["age_s"], f["freeflow_s"], f["tail_s"])
            if learned is None:
                return None  # same action support and GP fallback in all arms
            if edge not in contributions:
                contributions[edge] = np.zeros((4, 11))
            contributions[edge][:3] += learned
            if f["on_straight_lane"]:
                eta = f["analytic_eta_s"]
                contributions[edge][3, :10] += np.arange(1, 11) >= eta + f["tail_s"]
                contributions[edge][3, 10] += eta <= 10
        return contributions


def select(base, masks, contributions, gp):
    """masks contains one list of ten-bit protected-green masks per contributing edge, in feasible order."""
    if not contributions:
        return {name: gp for name in PREDICTORS}
    scores = np.broadcast_to(base, (len(PREDICTORS), len(base))).astype(float).copy()
    for edge, masses in contributions.items():
        for index, mask in enumerate(masks[edge]):
            for bit in range(10):
                if int(mask) & (1 << bit):
                    scores[:, index] += masses[:, bit]
    # Ignore machine rounding of the same algebraic score; first feasible index wins ties.
    chosen = np.argmax(np.round(scores, 12), axis=1)
    return {name: int(i) for name, i in zip(PREDICTORS, chosen)}
