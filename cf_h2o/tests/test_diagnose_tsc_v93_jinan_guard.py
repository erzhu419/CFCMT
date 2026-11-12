import numpy as np
import json
import pickle
from types import SimpleNamespace

from scripts.cluster import diagnose_tsc_v93_jinan_guard as worker
from scripts.cluster.diagnose_tsc_v93_jinan_guard import pair_error_scores, guard_reason
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def test_exact_three_action_pairs_have_no_corrected_error_and_pressure_gate_still_applies():
    # Exact pairwise predictions still produce the frozen K/(K-1) scale error.
    truth = np.array([0.0, -0.3, 0.2])
    left, right = np.array([0, 0, 1]), np.array([1, 2, 2])
    differences = truth[left] - truth[right]
    pair_score = 1.5 * truth
    old_v, corrected_v = pair_error_scores(differences, left, right, pair_score, np.array([True, False, False]))
    np.testing.assert_allclose(old_v, [0, .2, .175])
    np.testing.assert_allclose(corrected_v, 0, atol=1e-15)
    # Means, alpha and beta stay fixed. Removing the geometric component can
    # admit the beneficial action, but cannot bypass the pressure condition.
    score = .1 * truth + .9 * pair_score
    old_u = .8 + .9 * old_v
    new_u = .8 + .9 * corrected_v
    gap = np.array([0., .5, .5])
    assert guard_reason(1, 0, gap, score, old_u) == "confidence"
    assert guard_reason(1, 0, gap, score, new_u) == "eligible"
    assert truth[1] < truth[0]
    gap[1] = 1.1
    assert guard_reason(1, 0, gap, score, new_u) == "service_safety"
    assert guard_reason(0, 0, gap, score, new_u) == "agreement"


def test_main_uses_serialized_gp_spec_for_action_reference(tmp_path, monkeypatch):
    prior = PressurePolicySpec(downstream_occupancy_weight=.02)
    bundle = tmp_path / "bundle.pkl"
    bundle.write_bytes(pickle.dumps({
        "city": "jinan", "selected_candidate": "constant_alpha_0p9",
        "prior_policy": prior.key, "selected_group_ids": ["g"],
        "scenarios": ["jinan_3x4_real"],
        "component_models": {"atlanta": SimpleNamespace(prior_spec=prior)},
    }))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "selected_weights_by_city": {"jinan": {"atlanta": 1.0}},
        "selected_guard_by_city": {"jinan": {"margin": 0., "max_relative_rule_gap": 1.,
            "min_context_trust": 0., "risk_multiplier": .5}},
    }))
    names = tuple(worker.runtime.CONTRAST_FEATURES_V3)
    features = np.zeros((2, len(names)))
    features[:, names.index("green_q")] = 3.
    features[:, names.index("green_down_occ")] = [.8, .1]
    absolute = MechanismDataset(
        feature_names=names, features=features, context_names=(), context=np.zeros((2, 0)),
        priors={"interval_cost": np.zeros(2)}, targets={"interval_cost": np.array([2., 1.])},
        domains=np.array(["jinan"] * 2), metadata={"action_group_ids": ["g", "g"]},
    )
    monkeypatch.setattr(worker, "load_frozen_counterfactual_bank",
                        lambda *_args, **_kwargs: ({"jinan_3x4_real": absolute}, {}))
    def inspect_contrast(_component, contrast):
        assert contrast.metadata["reference_rows"] == [1, 1]
        assert contrast.metadata["reference_policy_spec"] == prior.to_dict()
        return {"e_A": .7, "e_P": .8}
    monkeypatch.setattr(worker, "decompose", inspect_contrast)
    monkeypatch.setattr(worker, "summarize_groups", lambda *_args: {})
    out = tmp_path / "audit.json"
    assert worker.main(["--bundle", str(bundle), "--confirmation-spec", str(spec),
                        "--adaptation-cache", str(tmp_path), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["reference_policy_spec"] == prior.to_dict()
