"""Use the frozen ten-second detector forecast in a thirteen-second service view."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tsc_corridor_count_fusion_20261109'))
from evidence_model import EvidencePredictor  # noqa: E402
from residual_model import ARMS, update_history  # noqa: E402


class AlignedPredictor(EvidencePredictor):
    """Extend nominal service queries, without adding detector predictions beyond 10 s."""
    service_seconds = 13
    detector_seconds = 10

    def forecast(self, row):
        contributions = {}
        if not row['spillback'] and not row['unmatched']:
            for flight in row['flights']:
                edge = flight['edge']
                bounds = [min(self.detector_seconds, max(0.0, h-flight['tail_s']))
                          for h in range(1, self.service_seconds+1)] + [float(self.detector_seconds)]
                contributions.setdefault(edge, np.zeros((len(ARMS), self.service_seconds+1)))
                prediction = self.flight_cdfs(flight, row['t'], bounds)
                prediction[:3] = prediction[:3] >= .5
                contributions[edge] += prediction
        update_history(row, self.history)
        return contributions
