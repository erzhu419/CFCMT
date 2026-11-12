#!/usr/bin/env python3
"""Read frozen V93 models and existing Jinan labels; write a small guard audit.

Run with PYTHONPATH pointing at the original V93 snapshot. This script never
fits a model, starts SUMO, or changes a frozen artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as runtime
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import load_frozen_counterfactual_bank
from cf_h2o.traffic_signal.action_contrast import action_group_ids, build_action_contrast_dataset
from cf_h2o.traffic_signal.action_scaling import action_group_range

ANCHOR = "causal_group_normalized_rigid_advantage"
PAIRWISE = "causal_antisymmetric_pairwise_advantage"
ALPHA = 0.9
BETA = 0.5


def pair_error_scores(pair_difference, left, right, score, reference):
    """Compare pair differences with original and scale-consistent projections."""
    counts = np.zeros(len(score))
    for rows in (left, right):
        np.add.at(counts, rows, 1)
    implied = score[left] - score[right]
    # The frozen score divides by K-1; the consistent complete-graph
    # projection divides by K. Only the error diagnostic changes here.
    corrected = implied * counts[left] / (counts[left] + 1)
    output = []
    for residual in (np.abs(pair_difference - implied), np.abs(pair_difference - corrected)):
        values = np.zeros(len(score))
        for rows in (left, right):
            np.add.at(values, rows, residual)
        values /= np.maximum(counts, 1)
        values[reference] = 0
        output.append(values)
    return output


def decompose(component, contrast):
    anchor = component.family_models[ANCHOR]
    pairwise = component.family_models[PAIRWISE]
    a = anchor.predict(contrast)["control_cost"]
    p = pairwise.predict(contrast)["control_cost"]
    reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    design, left, right, _ = pairwise._pair_design(contrast)
    if pairwise.model is None:
        differences = np.zeros(len(left))
    else:
        width = len(pairwise.state_feature_names)
        reverse = np.column_stack([design[:, :width], -design[:, width:]])
        differences = 0.5 * (pairwise.model.predict(design) - pairwise.model.predict(reverse))
    old_v, corrected_v = pair_error_scores(differences, left, right, p["mean"], reference)
    # A mismatch means these formulas do not reconstruct this frozen model;
    # do not publish a counterfactual based on the wrong implementation.
    np.testing.assert_allclose(old_v, p["pairwise_cycle_inconsistency"], rtol=1e-10, atol=1e-12)
    e_a, e_p = float(anchor.calibration_error), float(pairwise.calibration_error)
    d_a, d_p = np.asarray(a["context_distance"]), np.asarray(p["context_distance"])
    score = (1 - ALPHA) * a["mean"] + ALPHA * p["mean"]
    base_a = np.full(contrast.size, (1 - ALPHA) * e_a)
    base_p = np.full(contrast.size, ALPHA * e_p)
    cycle = ALPHA * old_v
    distance = 0.25 * ((1 - ALPHA) * e_a * d_a + ALPHA * (e_p + old_v) * d_p)
    for values in (base_a, base_p, cycle, distance):
        values[reference] = 0
    uncertainty = base_a + base_p + cycle + distance
    corrected_u = (1 - ALPHA) * a["uncertainty"] + ALPHA * (e_p + corrected_v) * (1 + 0.25 * d_p)
    corrected_u[reference] = 0
    wrapper = FrozenAnchoredBlendModel(
        anchor_model=anchor, correction_model=pairwise, candidate="constant_alpha_0p9",
        anchor_objective_mode=component.objective_modes[ANCHOR],
        correction_objective_mode=component.objective_modes[PAIRWISE],
    )
    frozen = wrapper.predict(contrast)["control_cost"]
    np.testing.assert_allclose(score, frozen["mean"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(uncertainty, frozen["uncertainty"], rtol=1e-10, atol=1e-12)
    no_v = (1 - ALPHA) * a["uncertainty"] + ALPHA * e_p * (1 + 0.25 * d_p)
    no_v[reference] = 0
    return {
        "score": score, "uncertainty": uncertainty, "corrected_uncertainty": corrected_u,
        "no_v_uncertainty": no_v, "no_distance_uncertainty": base_a + base_p + cycle,
        "anchor_base_penalty": BETA * base_a, "pairwise_base_penalty": BETA * base_p,
        "v_penalty": BETA * cycle, "distance_penalty": BETA * distance,
        "anchor_distance": d_a, "pairwise_distance": d_p,
        "v": old_v, "corrected_v": corrected_v,
        "e_A": e_a, "e_P": e_p,
    }


def guard_reason(learned, reference, gap, score, uncertainty):
    if learned == reference:
        return "agreement"
    if gap[learned] > 1 + 1e-12:
        return "service_safety"
    if score[learned] + BETA * uncertainty[learned] >= 0:
        return "confidence"
    return "eligible"


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    return dict(zip(("mean", "min", "p50", "p90", "max"), map(float, (
        np.mean(values), np.min(values), *np.quantile(values, [.5, .9]), np.max(values),
    ))))


def summarize_groups(absolute, contrast, parts, fitted_groups):
    groups = action_group_ids(contrast).astype(str)
    refs = np.asarray(contrast.metadata["reference_rows"], dtype=int)
    gap = runtime._relative_contrast_rule_gap(contrast)
    y = np.asarray(absolute.targets["interval_cost"])
    records = {"adaptation_fitted": [], "evaluation_not_fitted": []}
    for group in dict.fromkeys(groups):
        rows = np.flatnonzero(groups == group)
        reference = int(refs[rows[0]])
        learned = int(rows[np.argmin(parts["score"][rows])])
        reasons = {
            name: guard_reason(learned, reference, gap, parts["score"], parts[key])
            for name, key in (("original", "uncertainty"), ("scale_consistent", "corrected_uncertainty"),
                              ("without_v", "no_v_uncertainty"), ("without_distance", "no_distance_uncertainty"))
        }
        choices = {"gp": reference, "raw": learned,
                   **{name: learned if reason == "eligible" else reference for name, reason in reasons.items()}}
        scale, optimum = action_group_range(y[rows]), float(np.min(y[rows]))
        record = {"group": group, "reference_row": reference, "learned_row": learned,
                  "reasons": reasons, "disagreement": learned != reference,
                  "changed": choices["original"] != choices["scale_consistent"],
                  "pressure_gap": float(gap[learned]),
                  "policy_cost": {name: float(y[row]) for name, row in choices.items()},
                  "regret": {name: float((y[row] - optimum) / scale) for name, row in choices.items()}}
        record["terms"] = {name: float(parts[name][learned]) for name in (
            "score", "anchor_base_penalty", "pairwise_base_penalty", "v_penalty", "distance_penalty",
            "anchor_distance", "pairwise_distance", "v", "corrected_v",
        )}
        role = "adaptation_fitted" if group in fitted_groups else "evaluation_not_fitted"
        records[role].append(record)
    output = {}
    for role, entries in records.items():
        disagreements = [row for row in entries if row["disagreement"]]
        changed = [row for row in entries if row["changed"]]
        output[role] = {
            "group_count": len(entries), "model_reference_disagreements": len(disagreements),
            "gate_counts": {mode: dict(Counter(row["reasons"][mode] for row in entries))
                            for mode in ("original", "scale_consistent", "without_v", "without_distance")},
            "predicted_terms_at_disagreements": {
                name: describe([row["terms"][name] for row in disagreements])
                for name in (entries[0]["terms"] if entries else [])},
            "mean_label_cost": {mode: float(np.mean([row["policy_cost"][mode] for row in entries]))
                                for mode in (entries[0]["policy_cost"] if entries else [])},
            "mean_group_normalized_regret": {
                mode: float(np.mean([row["regret"][mode] for row in entries]))
                for mode in (entries[0]["regret"] if entries else [])},
            "scale_consistent_changed_actions": len(changed),
            "changed_action_label_effects": dict(Counter(
                "better" if row["policy_cost"]["scale_consistent"] < row["policy_cost"]["original"]
                else "worse" if row["policy_cost"]["scale_consistent"] > row["policy_cost"]["original"]
                else "tie" for row in changed)),
            "changed_examples": changed[:6],
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--confirmation-spec", type=Path, required=True)
    parser.add_argument("--adaptation-cache", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    if bool(args.evaluation_cache) != bool(args.evaluation_seeds):
        parser.error("evaluation cache and seeds must be provided together")
    bundle = pickle.loads(args.bundle.read_bytes())
    spec = json.loads(args.confirmation_spec.read_text())
    if (bundle["city"] != "jinan" or bundle["selected_candidate"] != "constant_alpha_0p9"
        or spec["selected_weights_by_city"]["jinan"] != {"atlanta": 1.0}
        or spec["selected_guard_by_city"]["jinan"] != {
            "margin": 0.0, "max_relative_rule_gap": 1.0, "min_context_trust": 0.0, "risk_multiplier": BETA}):
        raise ValueError("inputs are not the frozen V93 Jinan selector")
    component = bundle["component_models"]["atlanta"]
    # Generalized-pressure keys identify a spec but are not accepted legacy
    # names. Reuse the serialized spec carried by the fitted component.
    prior_spec = component.prior_spec
    if prior_spec.key != bundle["prior_policy"]:
        raise ValueError("component reference differs from the bundle reference")
    fitted_groups = set(map(str, bundle["selected_group_ids"]))
    # This existing reader consumes only manifest.sumocfgs keys. No network or
    # demand files need parsing to read already labelled NPZ shards.
    manifest = SimpleNamespace(sumocfgs=dict.fromkeys(bundle["scenarios"]))
    inputs = [("adaptation", args.adaptation_cache, [5057, 6067])]
    if args.evaluation_cache:
        if set(args.evaluation_seeds) & {5057, 6067}:
            raise ValueError("evaluation seeds overlap adaptation seeds")
        inputs.append(("evaluation", args.evaluation_cache, args.evaluation_seeds))
    result = {
        "protocol": "v93-jinan-frozen-label-guard-decomposition-v1",
        "bundle": str(args.bundle.resolve()), "runtime_module": runtime.__file__,
        "alpha": ALPHA, "beta": BETA, "source": "atlanta", "source_mass": 1.0,
        "reference_policy_spec": prior_spec.to_dict(),
        "fitted_target_groups": len(fitted_groups), "banks": {},
        "scope": "Existing matched offline labels; equal action-group means within each scenario. No fitting or simulation.",
        "limitations": [
            "These are offline gate decisions, before sparse coordination and physical execution; not the 713664 closed-loop decisions.",
            "Stored interval_cost labels are 60-second outcomes with PP continuation, not one-hour waiting or native-prefix recollection.",
            "Adaptation groups were fitted; evaluation_not_fitted refers only to group membership and does not imply a new untouched evaluation.",
            "Changing v alone is a diagnostic counterfactual; alpha, beta, means and pressure gate remain frozen.",
        ],
    }
    seen_fitted = set()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for role, cache, seeds in inputs:
            bank, audit = load_frozen_counterfactual_bank(cache, manifest, seeds=seeds, collection_shards=16, workers=1)
            scenarios = {}
            for scenario, absolute in bank.items():
                groups = set(map(str, action_group_ids(absolute)))
                if role == "evaluation" and groups & fitted_groups:
                    raise ValueError("evaluation group IDs overlap fitted target groups")
                seen_fitted.update(groups & fitted_groups)
                contrast = build_action_contrast_dataset(absolute, reference_policy=prior_spec,
                                                        contrast_features=runtime.CONTRAST_FEATURES_V3)
                parts = decompose(component, contrast)
                scenarios[scenario] = summarize_groups(absolute, contrast, parts, fitted_groups)
                result["residual_error_constants"] = {"e_A": parts["e_A"], "e_P": parts["e_P"],
                    "minimum_error_penalty": BETA * ((1 - ALPHA) * parts["e_A"] + ALPHA * parts["e_P"])}
            result["banks"][role] = {"cache_audit": audit, "scenarios": scenarios}
    if seen_fitted != fitted_groups:
        raise ValueError(f"adaptation cache omitted {len(fitted_groups - seen_fitted)} fitted groups")
    result["evaluation_bank_available"] = bool(args.evaluation_cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Eval complete: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
