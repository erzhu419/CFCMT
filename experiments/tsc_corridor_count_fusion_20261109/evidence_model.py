"""Update target arrival evidence with source counts under one physical prior."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tsc_corridor_residual_transfer_20261109'))
from residual_model import (  # noqa: E402
    HistogramHead, PSEUDOCOUNT, ResidualPredictor, feature_context, physical_cdf,
)


def combine_evidence(target, source):
    """Merge observed categories, leaving the physical pseudocount to cdf()."""
    counts = {context: values.copy() for context, values in target.counts.items()}
    for context, values in source.counts.items():
        if context in counts:
            counts[context] += values
        else:
            counts[context] = values.copy()
    metadata = dict(
        fusion='target_plus_source_counts_one_physical_prior',
        pseudocount=PSEUDOCOUNT,
        target_snapshot_examples=int(sum(values.sum() for values in target.counts.values())),
        source_snapshot_examples=int(sum(values.sum() for values in source.counts.values())),
        support_cells=len(counts),
    )
    return HistogramHead(counts, metadata)


class EvidencePredictor(ResidualPredictor):
    """T unchanged; S/P add actual/permuted source categories before smoothing.

    Each source-supported context uses
    (20*B + C_target + C_source)/(20 + n_target + n_source).
    Continuous ETA, fractional tail, 0.5 decoder and history update are inherited.
    """
    def __init__(self, target, source, placebo):
        super().__init__(target, source, placebo)
        self.pooled_source = combine_evidence(target, source)
        self.pooled_placebo = combine_evidence(target, placebo)

    def flight_cdfs(self, flight, time_s, bounds):
        context, eta = feature_context(flight, time_s, self.history)
        straight = context[0]
        local = self.target.cdf(context, eta, straight, bounds)
        if context in self.source.counts:
            source = self.pooled_source.cdf(context, eta, straight, bounds)
            placebo = self.pooled_placebo.cdf(context, eta, straight, bounds)
            self.counts['source_supported_flights'] += 1
        else:
            source = placebo = local
            self.counts['source_missing_flights'] += 1
        self.counts['target_missing_flights'] += context not in self.target.counts
        physical = physical_cdf(eta, straight, bounds)
        analytic = physical_cdf(flight['analytic_eta_s'], straight, bounds)
        return np.asarray([local, source, placebo, physical, analytic])
