"""Physical-ETA anchored conditional arrival calibration from existing PP hours."""
from collections import Counter, defaultdict
import gzip
import json
from math import ceil
from pathlib import Path
import random
import sys

import numpy as np

for sibling in ("tsc_corridor_arrival_residual_20261109", "tsc_corridor_arrival_transfer_20261107"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / sibling))
from history_baseline import historical_speed_mps, release_id  # noqa: E402

ARMS = ("T", "S", "P", "history", "analytic")
PSEUDOCOUNT = 20
PERMUTATION_SEED = 11072026


def rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def feature_context(flight, time_s, history):
    """Current information only; the continuous ETA is never rounded for scoring."""
    speed = historical_speed_mps(flight, time_s, history)
    eta = flight["remaining_m"] / speed if speed is not None else flight["analytic_eta_s"]
    ratio = speed / flight["speed_limit_mps"] if speed is not None else None
    group = ("missing" if ratio is None else "<=0.5" if ratio <= .5 else "(0.5,0.9]" if ratio <= .9
             else "(0.9,1.1]" if ratio <= 1.1 else ">1.1")
    context = bool(flight["on_straight_lane"]), min(11, max(1, ceil(eta))), group, bool(flight["queued_at_entry"])
    return context, eta


def update_history(row, history):
    for flight in row["flights"]:
        history[release_id(flight)] = row["t"], flight["remaining_m"], flight["on_straight_lane"]


def label_category(flight, outcome, time_s):
    """0 means no straight detection in the complete next 10-second window."""
    if release_id(flight) != release_id(outcome) or outcome["end_s"] <= time_s:
        raise ValueError("Label differs from the current in-flight release")
    status, lag = outcome["status"], outcome["end_s"] - time_s
    if status not in ("straight", "other", "terminal", "censored"):
        raise ValueError("Unresolved arrival label")
    if status == "censored" and lag < 10:
        raise ValueError("The ten-second label window is censored")
    if status == "straight" and lag <= 10:
        if lag != int(lag):
            raise ValueError("Expected an integer first-observed detector timestep")
        return int(lag)
    return 0


def physical_cdf(eta, straight, bounds):
    bounds = np.asarray(bounds, dtype=float)
    return ((bounds > 0) & straight & (bounds >= eta)).astype(float)


class HistogramHead:
    """Class 0 is no 10-second arrival; classes 1..10 are observed arrival seconds."""
    def __init__(self, counts=None, metadata=None):
        self.counts = {key: np.asarray(value, dtype=float) for key, value in (counts or {}).items()}
        self.metadata = metadata or {}

    def cdf(self, context, eta, straight, bounds):
        bounds = np.asarray(bounds, dtype=float)
        physical = physical_cdf(eta, straight, bounds)
        counts = self.counts.get(context)
        if counts is None:
            return physical
        cumulative = np.concatenate(([0.0], np.cumsum(counts[1:])))
        indices = np.clip(np.floor(bounds).astype(int), 0, 10)
        empirical = cumulative[indices]
        empirical[bounds <= 0] = 0
        return (PSEUDOCOUNT * physical + empirical) / (PSEUDOCOUNT + counts.sum())


def _new_counts():
    return defaultdict(lambda: [0] * 11)


def _permute_groups(groups, rng, counts):
    changed = 0
    for stratum in sorted(groups):
        examples = groups[stratum]
        labels = [label for context, label in examples]
        rng.shuffle(labels)
        for (context, original), label in zip(examples, labels):
            counts[context][label] += 1
            changed += label != original
    return changed


def _fit(root, part, city, seeds, *, with_placebo):
    real, placebo = _new_counts(), _new_counts()
    metadata = Counter()
    seed_examples = {}
    rng = random.Random(PERMUTATION_SEED)
    for seed in sorted(seeds):
        directory = Path(root) / part / city / str(seed)
        receipt = json.loads((directory / "result.json").read_text())
        if receipt["status"] != "PASS_HOUR" or receipt["city"] != city or receipt["seed"] != seed:
            raise ValueError(f"Invalid training hour: {directory}")
        outcomes = {release_id(record): record for record in rows(directory / "labels.jsonl.gz")}
        history, groups = {}, defaultdict(list)
        before = metadata["snapshot_examples"]
        for row in rows(directory / "decisions.jsonl.gz"):
            metadata["all_decisions"] += 1
            usable = not row["spillback"] and not row["unmatched"] and row["t"] + 10 <= receipt["horizon_sec"]
            if usable:
                for flight in row["flights"]:
                    context, eta = feature_context(flight, row["t"], history)
                    label = label_category(flight, outcomes[release_id(flight)], row["t"])
                    real[context][label] += 1
                    metadata["snapshot_examples"] += 1
                    metadata["positive_arrival_examples"] += label > 0
                    metadata["zero_label_examples"] += label == 0
                    if with_placebo:
                        # Current physical relation is held fixed; only extra conditions are shuffled.
                        stratum = flight["edge"], context[1], context[0]
                        groups[stratum].append((context, label))
            update_history(row, history)
        seed_examples[str(seed)] = metadata["snapshot_examples"] - before
        metadata["hours"] += 1
        if with_placebo:
            metadata["permutation_strata"] += len(groups)
            metadata["permutation_changed_labels"] += _permute_groups(groups, rng, placebo)
    info = dict(city=city, part=part, seeds=sorted(seeds), seed_examples=seed_examples,
                support_cells=len(real), pseudocount=PSEUDOCOUNT,
                feature_names=["current_straight", "history_eta_second_bin", "past_speed_ratio_bin", "entry_queue"], **metadata)
    real_head = HistogramHead(real, dict(info, permuted=False))
    placebo_head = HistogramHead(placebo, dict(info, permuted=True, permutation_seed=PERMUTATION_SEED)) if with_placebo else None
    return real_head, placebo_head


def fit_hours(root, part, city, seeds, *, permute=False):
    real, placebo = _fit(root, part, city, seeds, with_placebo=permute)
    return placebo if permute else real


def fit_source(root, part, city, seeds):
    """One pass through source hours fits both actual and conditional-permutation heads."""
    return _fit(root, part, city, seeds, with_placebo=True)


class ResidualPredictor:
    def __init__(self, target, source, placebo):
        self.target, self.source, self.placebo = target, source, placebo
        self.history = {}
        self.counts = Counter()

    def flight_cdfs(self, flight, time_s, bounds):
        context, eta = feature_context(flight, time_s, self.history)
        straight = context[0]
        local = self.target.cdf(context, eta, straight, bounds)
        if context in self.source.counts:
            source = .5 * (local + self.source.cdf(context, eta, straight, bounds))
            placebo = .5 * (local + self.placebo.cdf(context, eta, straight, bounds))
            self.counts["source_supported_flights"] += 1
        else:
            source = placebo = local
            self.counts["source_missing_flights"] += 1
        self.counts["target_missing_flights"] += context not in self.target.counts
        physical = physical_cdf(eta, straight, bounds)
        analytic = physical_cdf(flight["analytic_eta_s"], straight, bounds)
        return np.asarray([local, source, placebo, physical, analytic])

    def forecast(self, row):
        """Return T/S/P/history/analytic by 10 line-service windows and detector t+10."""
        contributions = {}
        if not row["spillback"] and not row["unmatched"]:
            for flight in row["flights"]:
                edge = flight["edge"]
                bounds = [max(0.0, h - flight["tail_s"]) for h in range(1, 11)] + [10.0]
                contributions.setdefault(edge, np.zeros((len(ARMS), 11)))
                prediction = self.flight_cdfs(flight, row["t"], bounds)
                prediction[:3] = prediction[:3] >= .5
                contributions[edge] += prediction
        update_history(row, self.history)
        return contributions
