import json
import pickle
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.cluster import run_tsc_v93_selected_action_calibration as worker
from cf_h2o.traffic_signal.action_ranker import ActionAdvantageConfig, PairwisePreferenceConfig
from cf_h2o.traffic_signal.generalized_pressure import PressurePolicySpec
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


class FixedRegressor:
    def __init__(self, config):
        self.config = config

    def predict(self, data):
        queue = data.features[:, data.feature_names.index("green_q")]
        score = np.where(queue == 2., -.1, np.where(queue == 1., .1, 0.))
        return {"control_cost": dict(mean=score, uncertainty=np.where(queue == 3., 0., 1.),
                                     context_trust=np.ones(data.size))}


def component(marker, training_groups=()):
    return SimpleNamespace(
        marker=marker, prior_spec=PressurePolicySpec(downstream_occupancy_weight=.02),
        family_models={worker.FAMILIES[0]: FixedRegressor(ActionAdvantageConfig()),
                       worker.FAMILIES[1]: FixedRegressor(PairwisePreferenceConfig())},
        objective_modes={family: "control_only" for family in worker.FAMILIES},
        diagnostics={"target_adaptation_group_ids": list(training_groups)},
    )


def dataset(scenario, counts, level=0., costs=(1., .9, 0.)):
    group_ids = [f"{scenario}:seed{seed}:{index}:tls"
                 for seed, count in counts.items() for index in range(count)]
    groups = np.repeat(group_ids, 3)
    names = tuple(worker.runtime.CONTRAST_FEATURES_V3)
    features = np.zeros((len(groups), len(names)))
    features[:, names.index("green_q")] = np.tile([3., 2., 1.], len(group_ids))
    return MechanismDataset(
        feature_names=names, features=features, context_names=("context",),
        context=np.full((len(groups), 1), level),
        priors={"interval_cost": np.zeros(len(groups))},
        targets={"interval_cost": np.tile(costs, len(group_ids))},
        domains=np.array([scenario] * len(groups)),
        metadata={"action_group_ids": groups.tolist(), "counterfactual_horizon_sec": 60,
                  "local_mechanism_horizon_sec": 10, "rollout_value_horizon_sec": 60,
                  # Settings deliberately live in nested shard metadata, as
                  # they do after the original multi-seed cache merge.
                  "source_datasets": [{"control_interval_sec": 10, "counterfactual_horizon_intervals": 6}]},
    )


def merge_test_datasets(items):
    return replace(items[0], features=np.vstack([x.features for x in items]),
                   context=np.vstack([x.context for x in items]),
                   priors={"interval_cost": np.concatenate([x.priors["interval_cost"] for x in items])},
                   targets={"interval_cost": np.concatenate([x.targets["interval_cost"] for x in items])},
                   domains=np.concatenate([x.domains for x in items]),
                   metadata={**items[0].metadata,
                             "action_group_ids": sum((x.metadata["action_group_ids"] for x in items), [])})


def test_records_use_original_means_and_contrast_range_not_absolute_label_floor():
    data = dataset(worker.SCENARIOS[0], {5057: 1}, costs=(1e6, 1e6 - .1, 1e6 + .1))
    record = worker.selected_action_records(data, component("original"), worker.SCENARIOS[0], 5057)[0]
    assert record["normalization_range"] == pytest.approx(.2)
    assert record["score"] == pytest.approx(-.1)
    assert record["selected_cost"] == 1e6 - .1
    assert record["gp_cost"] == 1e6
    assert record["is_reference"] is False and record["pressure_eligible"] is True


def test_runner_excludes_seed_freezes_q_before_validation_and_uses_original_full_model(tmp_path, monkeypatch):
    out_dir = tmp_path / "output"
    adaptation = {scenario: dataset(scenario, counts, level=i)
                  for i, (scenario, counts) in enumerate(zip(worker.SCENARIOS, (
                      {5057: 150, 6067: 154}, {5057: 148, 6067: 147}, {5057: 154, 6067: 158})))}
    validation = {scenario: dataset(scenario, {8171: count}, level=i)
                  for i, (scenario, count) in enumerate(zip(worker.SCENARIOS, (158, 130, 169)))}
    source = dataset(worker.SOURCE, {2027: 203, 3037: 203, 4047: 203})
    fitted_groups = set().union(*(worker.group_set(x) for x in adaptation.values()))
    bundle_path = tmp_path / "bundle.pkl"
    full = component("original-full")
    bundle_path.write_bytes(pickle.dumps(dict(
        city="jinan", selected_candidate="constant_alpha_0p9", scenarios=worker.SCENARIOS,
        component_models={"atlanta": full}, selected_group_ids=sorted(fitted_groups),
        prior_policy=full.prior_spec.key,
    )))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(dict(
        alpha=.9, beta=.5, source_group="atlanta", source_seeds=[2027, 3037, 4047],
        adaptation_seeds=[5057, 6067], validation_seeds=[8171],
        expected_adaptation_groups=911, expected_validation_groups=457,
    )))
    fits, predictions, repairs = [], [], []
    def load(path, scenarios, seeds):
        if path.name == "source":
            return {worker.SOURCE: source}, {}
        if path.name == "adaptation":
            return adaptation, {}
        assert path.name == "validation" and seeds == (8171,)
        assert len(fits) == 2
        assert (out_dir / "calibration.json").exists()
        assert json.loads((out_dir / "calibration.json").read_text())["calibration_seeds"] == [5057, 6067]
        return validation, {}
    def fit(bank, target, **kwargs):
        groups = worker.group_set(bank[target])
        seeds = {worker._group_seed_v3(group) for group in groups}
        assert len(seeds) == 1 and seeds <= {5057, 6067}
        assert groups == set(kwargs["target_adaptation_group_ids"])
        assert kwargs["model_families"] == worker.FAMILIES
        assert kwargs["prior_policy_override"] == full.prior_spec.key
        assert kwargs["source_rule_specs"] == {full.prior_spec.key: full.prior_spec}
        # Frozen V93 reads this before honoring the fixed prior override.
        # Source GP selects green_q=3, whose observed cost is 1 (not 0 or
        # the mean of all action costs).
        assert kwargs["source_rule_costs"] == {worker.SOURCE: {full.prior_spec.key: 1.}}
        fits.append(seeds)
        return component(f"fold-{next(iter(seeds))}", groups)
    original_records = worker.selected_action_records
    def records(absolute, model, scenario, seed):
        predictions.append((model.marker, seed))
        return original_records(absolute, model, scenario, seed)
    def repair(data, context):
        # The runner must recover these before calling the original repair.
        assert data.metadata["control_interval_sec"] == 10
        assert data.metadata["counterfactual_horizon_intervals"] == 6
        repairs.append(context.tolist())
        return replace(data, metadata={**data.metadata, "static_topology_context_repair": {
            "maximum_absolute_context_change": 0., "label_free": True}})
    monkeypatch.setattr(worker, "load_bank", load)
    monkeypatch.setattr(worker, "fit_target_screening_models", fit)
    monkeypatch.setattr(worker, "selected_action_records", records)
    monkeypatch.setattr(worker, "repair_dataset_topology_context", repair)
    monkeypatch.setattr(worker.runtime, "merge_counterfactual_datasets_v3", merge_test_datasets)
    assert worker.main([
        "--bundle", str(bundle_path), "--source-cache", str(tmp_path / "source"),
        "--adaptation-cache", str(tmp_path / "adaptation"), "--validation-cache", str(tmp_path / "validation"),
        "--protocol", str(protocol), "--out-dir", str(out_dir),
    ]) == 0
    assert fits == [{6067}, {5057}]
    assert predictions == [("fold-6067", 5057)] * 3 + [("fold-5057", 6067)] * 3 + [("original-full", 8171)] * 3
    assert repairs == [[0.], [1.], [2.]]
    receipts = json.loads((out_dir / "fit_receipts.json").read_text())
    for receipt in receipts:
        source_cost = receipt["source_rule_cost_input"]
        assert source_cost["cost"] == 1. and source_cost["group_count"] == 609
        assert source_cost["contextual_selector_bypassed"] is True
        assert "source-cache interval_cost" in source_cost["provenance"]
    result = json.loads((out_dir / "result.json").read_text())
    assert result["execution"]["component_fit_calls"] == 2
    assert result["execution"]["full_model_refit_calls"] == 0
    assert result["historical_validation"]["group_count"] == 457
    assert result["closed_loop_expansion_authorized"] is False
