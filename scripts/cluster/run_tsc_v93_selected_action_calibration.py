#!/usr/bin/env python3
"""Fit two V93 seed-OOF components, freeze final-action q, then read 8171 labels.

Use the original V93 source on PYTHONPATH. All input caches/models remain
unchanged; outputs contain records and analysis, never a deployment model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as runtime
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_seed_v3, _group_subset_v3, _static_context_from_dataset,
)
from cf_h2o.eval.traffic_signal_topology_context_repair import repair_dataset_topology_context
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models, load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids, build_action_contrast_dataset, rule_reference_indices,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig

if __package__:
    from .tsc_v93_selected_action_calibration import calibrate, analyze_with_calibration
else:
    from tsc_v93_selected_action_calibration import calibrate, analyze_with_calibration

SCENARIOS = ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500")
TARGET = "external_city_jinan"
SOURCE = "atlanta_1x5"
FAMILIES = ("causal_group_normalized_rigid_advantage", "causal_antisymmetric_pairwise_advantage")
FIELDS = ("group_id", "seed", "scenario", "score", "original_uncertainty", "is_reference",
          "pressure_eligible", "gp_cost", "selected_cost", "normalization_range")
LABEL_METADATA = ("counterfactual_horizon_sec", "local_mechanism_horizon_sec", "rollout_value_horizon_sec",
                  "counterfactual_cost_population", "counterfactual_cost_scope", "counterfactual_cost_normalization",
                  "counterfactual_cost_mode", "control_interval_sec", "counterfactual_horizon_intervals")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def write_records(path, records):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def load_bank(path, scenarios, seeds):
    # The cache reader only uses manifest.sumocfgs keys; no simulator inputs
    # need parsing for these already labelled shards.
    return load_frozen_counterfactual_bank(
        path, SimpleNamespace(sumocfgs=dict.fromkeys(scenarios)),
        seeds=seeds, collection_shards=16, workers=1,
    )


def group_set(dataset):
    return set(map(str, action_group_ids(dataset)))


def metadata_value(metadata, key):
    """Recover collection settings retained in the merged cache's shard metadata."""
    values = [metadata[key]] if metadata.get(key) is not None else []
    for child in metadata.get("source_datasets", ()):
        value = metadata_value(child, key)
        if value is not None:
            values.append(value)
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"cache shards disagree on {key}")
    return values[0] if values else None


def check_schema(dataset, reference):
    if (dataset.feature_names != reference.feature_names or dataset.context_names != reference.context_names
        or set(dataset.targets) != set(reference.targets) or set(dataset.priors) != set(reference.priors)):
        raise ValueError("cached feature/context/label/prior schemas differ")
    for key in LABEL_METADATA:
        if metadata_value(dataset.metadata, key) != metadata_value(reference.metadata, key):
            raise ValueError(f"cached label contract differs: {key}")


def selected_action_records(absolute, component, scenario, expected_seed):
    contrast = build_action_contrast_dataset(
        absolute, reference_policy=component.prior_spec, contrast_features=runtime.CONTRAST_FEATURES_V3,
    )
    wrapper = FrozenAnchoredBlendModel(
        anchor_model=component.family_models[FAMILIES[0]], correction_model=component.family_models[FAMILIES[1]],
        candidate="constant_alpha_0p9", anchor_objective_mode=component.objective_modes[FAMILIES[0]],
        correction_objective_mode=component.objective_modes[FAMILIES[1]],
    )
    score, uncertainty, _, refs = runtime._group_adjusted_scores(contrast, wrapper.predict(contrast))
    pressure_gap = runtime._relative_contrast_rule_gap(contrast)
    groups = action_group_ids(contrast).astype(str)
    costs = np.asarray(absolute.targets["interval_cost"])
    label_contrasts = np.asarray(contrast.targets["interval_cost"])
    records = []
    for group in dict.fromkeys(groups):
        seed = _group_seed_v3(group)
        if seed != expected_seed:
            raise ValueError("prediction subset contains the wrong simulator seed")
        rows = np.flatnonzero(groups == group)
        reference = int(refs[rows[0]])
        selected = int(rows[np.argmin(score[rows])])
        records.append(dict(
            group_id=group, seed=int(seed), scenario=scenario,
            score=float(score[selected]), original_uncertainty=float(uncertainty[selected]),
            is_reference=selected == reference, pressure_eligible=bool(pressure_gap[selected] <= 1 + 1e-12),
            gp_cost=float(costs[reference]), selected_cost=float(costs[selected]),
            normalization_range=float(action_group_range(label_contrasts[rows])),
        ))
    return records


def fit_seed_oof(source, adaptation, prior_spec):
    target_all = runtime.merge_counterfactual_datasets_v3([adaptation[name] for name in SCENARIOS])
    all_groups = group_set(target_all)
    static_context = np.mean(np.vstack([_static_context_from_dataset(adaptation[name]) for name in SCENARIOS]), axis=0)
    # Frozen V93 materializes source selector inputs even with a fixed prior.
    # Supply the actual matched-label mean for that one GP; no selector runs.
    reference_rows = list(rule_reference_indices(source, policy=prior_spec).values())
    source_gp_cost = float(np.mean(np.asarray(source.targets["interval_cost"])[reference_rows]))
    source_rule_costs = {SOURCE: {prior_spec.key: source_gp_cost}}
    records, receipts = [], []
    for heldout_seed in (5057, 6067):
        training_ids = sorted(group for group in all_groups if _group_seed_v3(group) != heldout_seed)
        validation_ids = all_groups - set(training_ids)
        training = _group_subset_v3(target_all, selected_groups=set(training_ids))
        # The held-out seed is physically absent from fit_bank, including all
        # its scenarios, actions, normalization labels and fitted contexts.
        component = fit_target_screening_models(
            {SOURCE: source, TARGET: training}, TARGET,
            fit_config=MechanismFitConfig(), source_rule_costs=source_rule_costs,
            source_rule_specs={prior_spec.key: prior_spec}, model_families=FAMILIES,
            target_group_budget=len(training_ids), target_adaptation_selection_seed=20260803,
            target_calibration_seeds=(), target_calibration_fraction=0., source_selector_min_context_domains=5,
            scenario_city_groups={SOURCE: "atlanta", TARGET: "jinan"},
            target_adaptation_group_ids=training_ids, target_static_context=static_context,
            prior_policy_override=prior_spec.key,
        )
        fitted_ids = set(map(str, component.diagnostics["target_adaptation_group_ids"]))
        if fitted_ids != set(training_ids) or fitted_ids & validation_ids:
            raise ValueError("fitted target groups violate the seed holdout")
        receipts.append(dict(
            heldout_seed=heldout_seed, target_training_groups=len(training_ids),
            target_training_seed_counts=dict(Counter(_group_seed_v3(group) for group in training_ids)),
            heldout_groups=len(validation_ids), training_validation_overlap=0,
            source_groups=len(group_set(source)), source_seeds=[2027, 3037, 4047],
            source_rule_cost_input=dict(
                scenario=SOURCE, policy=prior_spec.key, cost=source_gp_cost,
                provenance="source-cache interval_cost at fixed-GP reference row in each action group",
                aggregation="unweighted mean over complete source action groups",
                group_count=len(reference_rows), contextual_selector_bypassed=True,
            ),
            families=list(FAMILIES), family_configs={name: asdict(component.family_models[name].config) for name in FAMILIES},
        ))
        for scenario in SCENARIOS:
            subset = _group_subset_v3(adaptation[scenario], selected_groups=validation_ids)
            records.extend(selected_action_records(subset, component, scenario, heldout_seed))
    return records, receipts


def repair_validation(dataset, adaptation):
    check_schema(dataset, adaptation)
    settings = {key: metadata_value(dataset.metadata, key)
                for key in ("control_interval_sec", "counterfactual_horizon_intervals")}
    if settings != {"control_interval_sec": 10, "counterfactual_horizon_intervals": 6}:
        raise ValueError("historical validation does not use the original 10s/60s label contract")
    # Seed/shard merging retains these fields in source_datasets rather than
    # consistently at the outer level. Restore them only in this memory view.
    repair_input = replace(dataset, metadata={**dataset.metadata, **settings})
    repaired = repair_dataset_topology_context(repair_input, _static_context_from_dataset(adaptation))
    if not np.array_equal(repaired.features, dataset.features) or any(
        not np.array_equal(repaired.targets[key], dataset.targets[key]) for key in dataset.targets
    ):
        raise ValueError("in-memory topology repair changed action features or labels")
    return repaired, dict(
        repair=repaired.metadata["static_topology_context_repair"], features_equal=True, labels_equal=True,
        schema_equal=True, feature_count=len(dataset.feature_names), context_count=len(dataset.context_names),
        label_metadata={key: metadata_value(dataset.metadata, key) for key in LABEL_METADATA},
        maximum_absolute_prior_change=max(float(np.max(np.abs(repaired.priors[key] - dataset.priors[key])))
                                          for key in dataset.priors),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "source-cache", "adaptation-cache", "validation-cache", "protocol", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text())
    expected = dict(alpha=.9, beta=.5, source_group="atlanta", adaptation_seeds=[5057, 6067],
                    source_seeds=[2027, 3037, 4047], validation_seeds=[8171],
                    expected_adaptation_groups=911, expected_validation_groups=457)
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol does not match fixed calibration design: {key}")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    bundle = pickle.loads(args.bundle.read_bytes())
    if (bundle["city"] != "jinan" or bundle["selected_candidate"] != "constant_alpha_0p9"
        or set(bundle["scenarios"]) != set(SCENARIOS)):
        raise ValueError("bundle is not the frozen V93 Jinan component bundle")
    full_component = bundle["component_models"]["atlanta"]
    prior_spec = full_component.prior_spec
    if prior_spec.key != bundle["prior_policy"] or prior_spec.key != "gp_d0_o0p02_s0":
        raise ValueError("bundle reference differs from frozen V93 GP")
    fitted_ids = set(map(str, bundle["selected_group_ids"]))
    with threadpool_limits(limits=1):
        source_bank, source_audit = load_bank(args.source_cache, (SOURCE,), (2027, 3037, 4047))
        adaptation, adaptation_audit = load_bank(args.adaptation_cache, SCENARIOS, (5057, 6067))
        source = source_bank[SOURCE]
        for dataset in (source, *adaptation.values()):
            check_schema(dataset, adaptation[SCENARIOS[0]])
        target_ids = set().union(*(group_set(dataset) for dataset in adaptation.values()))
        seed_counts = dict(Counter(_group_seed_v3(group) for group in target_ids))
        if target_ids != fitted_ids or len(target_ids) != 911 or seed_counts != {5057: 452, 6067: 459}:
            raise ValueError("adaptation cache does not match all 911 frozen target groups and seeds")
        source_ids = group_set(source)
        if len(source_ids) != 609 or {_group_seed_v3(group) for group in source_ids} != {2027, 3037, 4047}:
            raise ValueError("source cache does not match the 609 Atlanta groups")
        oof_records, fit_receipts = fit_seed_oof(source, adaptation, prior_spec)
        if len(oof_records) != 911 or {row["group_id"] for row in oof_records} != fitted_ids:
            raise ValueError("OOF predictions did not cover each frozen target group exactly once")
        write_records(args.out_dir / "oof_records.csv", oof_records)
        calibration = calibrate(oof_records)
        write_json(args.out_dir / "calibration.json", calibration)
        write_json(args.out_dir / "fit_receipts.json", fit_receipts)
        print("Calibration frozen before historical validation cache access", flush=True)

        validation, validation_audit = load_bank(args.validation_cache, SCENARIOS, (8171,))
        validation_ids = set().union(*(group_set(dataset) for dataset in validation.values()))
        if len(validation_ids) != 457 or validation_ids & fitted_ids:
            raise ValueError("historical validation is incomplete or overlaps fitted target groups")
        validation_records, repair_audits = [], {}
        for scenario in SCENARIOS:
            repaired, repair_audits[scenario] = repair_validation(validation[scenario], adaptation[scenario])
            validation_records.extend(selected_action_records(repaired, full_component, scenario, 8171))
        write_records(args.out_dir / "validation_records.csv", validation_records)
        result = analyze_with_calibration(calibration, oof_records, validation_records)
    result["execution"] = dict(
        protocol_file=str(args.protocol.resolve()), protocol=protocol, runtime_module=runtime.__file__,
        bundle=str(args.bundle.resolve()), reference_policy_spec=prior_spec.to_dict(),
        source_cache_audit=source_audit, adaptation_cache_audit=adaptation_audit,
        validation_cache_audit=validation_audit, in_memory_validation_repairs=repair_audits,
        component_fit_calls=2, tree_fit_calls=4, full_model_refit_calls=0,
        validation_predictor="original frozen full Atlanta component; unchanged means, alpha and GP",
        calibration_saved_before_validation_load=True, simulator_runs=0, deployment_model_written=False,
    )
    write_json(args.out_dir / "result.json", result)
    print(f"Eval complete: {args.out_dir / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
