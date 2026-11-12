"""Analytic selection witness: call real predict/selector, never fit a model."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cf_h2o.traffic_signal.action_ranker import AntisymmetricPairwiseActionRegressor
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.eval.traffic_signal_v158_native_prefix_ranking_calibration import _select_row


class ExactDifference:
    def predict(self, design):
        return design[:, -1]


cost = np.array([0., .04, -.03, .02, .01, -.01, .03, -.02])
n, reference = len(cost), 3
data = MechanismDataset(
    feature_names=("state", "action", "switch_indicator"),
    features=np.column_stack((np.zeros(n), cost, np.ones(n))),
    context_names=("context",), context=np.zeros((n, 1)), priors={},
    targets={"interval_cost": cost - cost[reference]}, domains=np.array(["toy"] * n),
    metadata={"action_group_ids": ["g"] * n, "is_reference": [i == reference for i in range(n)]},
)
model = AntisymmetricPairwiseActionRegressor(state_feature_names=("state",), action_feature_names=("action",))
model.state_feature_names, model.action_feature_names = ("state",), ("action",)
model.source_contexts = np.zeros((1, 1))
model.context_mean, model.context_scale = np.zeros(1), np.ones(1)
model.model, model.calibration_error = ExactDifference(), 0.0
prediction = model.predict(data)["control_cost"]
differences = cost[:, None] - cost[None, :]
corrected = differences.sum(axis=1) / n
corrected_reference = corrected - corrected[reference]
corrected_cycle = np.abs(differences - (corrected[:, None] - corrected[None, :]))


def select(scores):
    return _select_row(data, np.asarray(scores), np.arange(n), prohibit_stay_override=True)


reference_tie = select(np.zeros(n))
nonreference_tie_scores = np.full(n, .1)
nonreference_tie_scores[reference] = 0.
nonreference_tie_scores[[1, 2]] = -.1
first_nonreference_tie = select(nonreference_tie_scores)
raw_selected, existing_selected = select(cost - cost[reference]), select(prediction["mean"])
data.features[0, 2] = 0.
veto_scores = np.full(n, .1)
veto_scores[reference], veto_scores[0], veto_scores[1] = 0., -1., -.5
veto_selected = select(veto_scores)
checks = {
    "existing_score_is_n_over_n_minus_one": bool(np.allclose(prediction["mean"], n / (n - 1) * (cost - cost[reference]), rtol=0, atol=1e-14)),
    "existing_cycle_nonzero_on_perfect_differences": bool(np.max(prediction["pairwise_cycle_inconsistency"]) > 1e-3),
    "sum_over_n_recovers_reference_cost_and_zero_cycle": bool(np.allclose(corrected_reference, cost - cost[reference], rtol=0, atol=1e-14) and np.max(corrected_cycle) < 1e-14),
    "positive_scaling_preserves_oracle_argmin": raw_selected == existing_selected == 2,
    "tie_prefers_reference": reference_tie == reference,
    "nonreference_tie_uses_first_row": first_nonreference_tie == 1,
    "no_stay_veto_returns_reference_not_second_best": veto_selected == reference and veto_selected != 1,
}
result = {
    "new_fits": 0, "simulator_calls": 0,
    "fixture": "Eight actions with exact analytic d_ij=c_i-c_j (scale S=1); zero analytic predictor error. No production code changed.",
    "reference_index": reference, "raw_reference_delta": (cost - cost[reference]).tolist(),
    "existing_mean": prediction["mean"].tolist(),
    "existing_cycle_inconsistency": prediction["pairwise_cycle_inconsistency"].tolist(),
    "corrected_mean_sum_over_n": corrected_reference.tolist(),
    "corrected_max_cycle_residual": float(np.max(corrected_cycle)),
    "selected": {"raw_oracle": raw_selected, "existing": existing_selected, "reference_tie": reference_tie,
                 "nonreference_tie": first_nonreference_tie, "no_stay_veto": veto_selected},
    "checks": checks,
}
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"checks": checks, "new_fits": 0, "simulator_calls": 0}))
if not all(checks.values()):
    raise AssertionError("analytic witness differs from this audit's observed implementation")
