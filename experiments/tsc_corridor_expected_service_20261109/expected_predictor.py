"""Use frozen arrival probabilities for service scores and hard endpoint forecasts."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tsc_corridor_count_fusion_20261109'))
from evidence_model import EvidencePredictor  # noqa: E402
from residual_model import ARMS, update_history  # noqa: E402


class ExpectedServicePredictor(EvidencePredictor):
    """Keep the fitted CDF, timing and endpoint decoder; change service columns only."""
    service_seconds = 13
    detector_seconds = 10

    def forecast(self, row):
        contributions = {}
        self.hard_forecasts = {}
        if not row['spillback'] and not row['unmatched']:
            for flight in row['flights']:
                edge = flight['edge']
                bounds = [min(self.detector_seconds, max(0.0, h-flight['tail_s']))
                          for h in range(1, self.service_seconds+1)] + [float(self.detector_seconds)]
                shape = (len(ARMS), self.service_seconds+1)
                contributions.setdefault(edge, np.zeros(shape))
                self.hard_forecasts.setdefault(edge, np.zeros(shape))
                prediction = self.flight_cdfs(flight, row['t'], bounds)
                hard = prediction.copy()
                hard[:3] = hard[:3] >= .5
                # Decode each vehicle before summing endpoints, never an edge total.
                prediction[:3, -1] = hard[:3, -1]
                contributions[edge] += prediction
                self.hard_forecasts[edge] += hard
        update_history(row, self.history)
        return contributions
