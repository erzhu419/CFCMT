"""Parallel leave-one-network-out suite for the CFCMT V2 benchmark."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import (
    EXTENDED_SCENARIOS,
    FittedTransferModelsV2,
    GuardConfigV2,
    _runtime_metadata,
    collect_mechanism_transitions,
    evaluate_policy_v2,
    fit_transfer_models_v2,
    merge_mechanism_datasets,
    transition_diagnostics,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import DEFAULT_ENV_ROOT, _scenario_sumocfg
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset, MechanismFitConfig


DEFAULT_POLICIES_V2 = (
    "fixed_program",
    "max_pressure",
    "phase_pressure",
    "spillback_pressure",
    "simulator_mpc",
    "dense_mpc",
    "cfcmt_mpc",
    "simulator_guard",
    "dense_guard",
    "cfcmt_guard",
)

DEFAULT_METRICS = (
    "mean_queue",
    "p90_queue",
    "mean_tripinfo_duration",
    "mean_tripinfo_waiting_time",
    "mean_tripinfo_time_loss",
    "throughput_ratio",
)


def _collect_worker(payload: Mapping[str, Any]) -> tuple[str, MechanismDataset]:
    sumo_api = _load_libsumo()
    scenario = str(payload["scenario"])
    env_root = Path(payload["env_root"])
    dataset = collect_mechanism_transitions(
        sumo_api=sumo_api,
        sumocfg=_scenario_sumocfg(env_root, scenario),
        scenario=scenario,
        duration_sec=float(payload["duration_sec"]),
        control_interval_sec=int(payload["control_interval_sec"]),
        warmup_sec=float(payload["warmup_sec"]),
        seed=int(payload["seed"]),
    )
    return scenario, dataset


def collect_transition_bank_v2(
    *,
    env_root: Path,
    scenarios: Sequence[str],
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
    workers: int,
) -> dict[str, MechanismDataset]:
    payloads = [
        {
            "env_root": str(env_root),
            "scenario": scenario,
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed + idx * 1009),
        }
        for idx, scenario in enumerate(scenarios)
    ]
    if int(workers) <= 1:
        return dict(_collect_worker(payload) for payload in payloads)
    context = mp.get_context("spawn")
    bank: dict[str, MechanismDataset] = {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(payloads)), mp_context=context) as pool:
        futures = [pool.submit(_collect_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            scenario, dataset = future.result()
            bank[scenario] = dataset
    return {scenario: bank[scenario] for scenario in scenarios}


def _policy_family(policy: str) -> str | None:
    if policy.startswith("simulator_"):
        return "simulator"
    if policy.startswith("dense_"):
        return "dense"
    if policy.startswith("cfcmt_"):
        return "cfcmt"
    return None


def _evaluate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    sumo_api = _load_libsumo()
    scenario = str(payload["scenario"])
    policy = str(payload["policy"])
    models: FittedTransferModelsV2 = payload["models"]
    family = _policy_family(policy)
    guard = models.guard_configs[family] if family is not None else GuardConfigV2(
        prior_policy=str(payload["prior_policy"])
    )
    if family is not None and guard.prior_policy != str(payload["prior_policy"]):
        guard = GuardConfigV2(
            prior_policy=str(payload["prior_policy"]),
            margin=guard.margin,
            risk_multiplier=guard.risk_multiplier,
            min_context_trust=guard.min_context_trust,
        )
    metrics = evaluate_policy_v2(
        sumo_api=sumo_api,
        sumocfg=_scenario_sumocfg(Path(payload["env_root"]), scenario),
        scenario=scenario,
        policy=policy,
        models=models,
        guard=guard,
        duration_sec=float(payload["duration_sec"]),
        control_interval_sec=int(payload["control_interval_sec"]),
        warmup_sec=float(payload["warmup_sec"]),
        seed=int(payload["seed"]),
        tripinfo_output=Path(payload["tripinfo_output"]),
    )
    return {
        "target": scenario,
        "policy": policy,
        "seed": int(payload["seed"]),
        "metrics": metrics,
    }


def _fit_target_models(
    bank: Mapping[str, MechanismDataset],
    target: str,
    *,
    fit_config: MechanismFitConfig,
) -> FittedTransferModelsV2:
    source_names = [name for name in bank if name != target]
    if target in source_names:
        raise AssertionError("target data entered its source set")
    train = merge_mechanism_datasets([bank[name] for name in source_names])
    model = fit_transfer_models_v2(train, cfcmt_config=fit_config, calibrate_guards=True)
    if target in model.diagnostics["source_domains"]:
        raise AssertionError("target domain is present in fitted model diagnostics")
    return model


def _aggregate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    policies: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets = sorted({str(row["target"]) for row in rows})
    target_rows = []
    for target in targets:
        policy_metrics = {}
        for policy in policies:
            items = [
                row["metrics"]
                for row in rows
                if row["target"] == target and row["policy"] == policy and row["metrics"].get("ok")
            ]
            policy_metrics[policy] = {
                metric: float(np.mean([item[metric] for item in items])) if items else float("nan")
                for metric in metrics
            }
            policy_metrics[policy]["seed_count"] = len(items)
            if items and policy.endswith("_guard"):
                policy_metrics[policy]["guard_audit"] = _sum_audit_rows([item.get("guard_audit", {}) for item in items])
            if items and policy != "fixed_program":
                policy_metrics[policy]["phase_execution_audit"] = _sum_audit_rows(
                    [item.get("phase_execution_audit", {}) for item in items]
                )
        target_rows.append({"target": target, "policy_metrics": policy_metrics})

    aggregate = {}
    for policy in policies:
        aggregate[policy] = {
            metric: float(np.mean([target["policy_metrics"][policy][metric] for target in target_rows]))
            for metric in metrics
        }
        aggregate[policy]["target_count"] = len(target_rows)
    return target_rows, aggregate


def _sum_audit_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    count_keys = {
        "decisions",
        "prior_agreements",
        "proposed_overrides",
        "accepted_overrides",
        "rejected_margin",
        "rejected_context",
        "requests",
        "accepted_requests",
        "same_phase_requests",
        "rejected_busy",
        "rejected_min_green",
        "switches",
    }
    second_keys = {"green_seconds", "yellow_seconds", "all_red_seconds"}
    result: dict[str, float | int] = {}
    for key in count_keys:
        result[key] = int(sum(int(row.get(key, 0)) for row in rows))
    for key in second_keys:
        result[key] = float(sum(float(row.get(key, 0.0)) for row in rows))
    if result.get("decisions", 0):
        decisions = max(int(result["decisions"]), 1)
        proposed = max(int(result["proposed_overrides"]), 1)
        result["agreement_rate"] = int(result["prior_agreements"]) / decisions
        result["proposal_rate"] = int(result["proposed_overrides"]) / decisions
        result["acceptance_rate"] = int(result["accepted_overrides"]) / proposed
    return result


def paired_hierarchical_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference: str,
    baseline: str,
    metric: str = "mean_queue",
    effect: str = "absolute",
    samples: int = 5000,
    seed: int = 20260609,
    clusters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bootstrap independent networks, then matched seeds within each network.

    ``relative`` effects make heterogeneous network sizes comparable by using
    ``(reference - baseline) / abs(baseline)`` within every matched seed.
    """

    if effect not in {"absolute", "relative"}:
        raise ValueError("bootstrap effect must be 'absolute' or 'relative'")

    def paired_effect(reference_value: float, baseline_value: float) -> float:
        delta = float(reference_value) - float(baseline_value)
        if effect == "absolute":
            return delta
        return delta / max(abs(float(baseline_value)), 1e-8)

    values: dict[str, list[tuple[float, float]]] = {}
    lookup = {(row["target"], row["policy"], row["seed"]): row["metrics"] for row in rows}
    targets = sorted({str(row["target"]) for row in rows})
    for target in targets:
        seeds = sorted(
            {
                int(row["seed"])
                for row in rows
                if row["target"] == target and row["policy"] == reference
            }
        )
        pairs = []
        for item_seed in seeds:
            ref = lookup.get((target, reference, item_seed), {})
            base = lookup.get((target, baseline, item_seed), {})
            if ref.get("ok") and base.get("ok"):
                pairs.append((float(ref[metric]), float(base[metric])))
        if pairs:
            values[target] = pairs
    if not values:
        return {"reference": reference, "baseline": baseline, "metric": metric, "network_count": 0}

    network_delta_map = {
        target: float(
            np.mean(
                [paired_effect(reference_value, baseline_value) for reference_value, baseline_value in pairs]
            )
        )
        for target, pairs in values.items()
    }
    network_deltas = np.asarray(list(network_delta_map.values()), dtype=float)
    cluster_for_target = {
        target: str((clusters or {}).get(target, target)) for target in values
    }
    targets_by_cluster = {
        cluster: tuple(
            target for target in values if cluster_for_target[target] == cluster
        )
        for cluster in sorted(set(cluster_for_target.values()))
    }
    cluster_delta_map = {
        cluster: float(np.mean([network_delta_map[target] for target in cluster_targets]))
        for cluster, cluster_targets in targets_by_cluster.items()
    }
    rng = np.random.default_rng(seed)
    cluster_names = tuple(targets_by_cluster)
    draws = np.empty(int(samples), dtype=float)
    for draw_idx in range(int(samples)):
        sampled_clusters = rng.integers(0, len(cluster_names), size=len(cluster_names))
        cluster_effects = []
        for cluster_idx in sampled_clusters:
            cluster_targets = targets_by_cluster[cluster_names[int(cluster_idx)]]
            sampled_networks = rng.integers(0, len(cluster_targets), size=len(cluster_targets))
            network_effects = []
            for network_idx in sampled_networks:
                pairs = values[cluster_targets[int(network_idx)]]
                sampled_seeds = rng.integers(0, len(pairs), size=len(pairs))
                network_effects.append(
                    float(
                        np.mean(
                            [paired_effect(*pairs[int(seed_idx)]) for seed_idx in sampled_seeds]
                        )
                    )
                )
            cluster_effects.append(float(np.mean(network_effects)))
        draws[draw_idx] = float(np.mean(cluster_effects))
    return {
        "reference": reference,
        "baseline": baseline,
        "metric": metric,
        "effect": effect,
        "network_count": len(values),
        "cluster_count": len(cluster_names),
        "mean_delta": float(np.mean(list(cluster_delta_map.values()))),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "win_networks": int(np.sum(network_deltas < 0.0)),
        "win_clusters": int(sum(value < 0.0 for value in cluster_delta_map.values())),
        "cluster_mean_delta": cluster_delta_map,
    }


def run_leave_one_network_out_v2(
    *,
    env_root: Path = DEFAULT_ENV_ROOT,
    scenarios: Sequence[str] = EXTENDED_SCENARIOS,
    policies: Sequence[str] = DEFAULT_POLICIES_V2,
    seeds: Sequence[int] = (131, 337, 911),
    source_duration_sec: float = 600.0,
    duration_sec: float = 600.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    source_seed: int = 2027,
    workers: int = 8,
    prior_policy: str = "max_pressure",
    tripinfo_root: Path = Path("cf_h2o/results/tripinfo/traffic_signal_resco_cfcmt_v2"),
    fit_config: MechanismFitConfig | None = None,
) -> dict[str, Any]:
    scenarios = tuple(scenarios)
    policies = tuple(policies)
    seeds = tuple(int(seed) for seed in seeds)
    fit_config = fit_config or MechanismFitConfig()
    bank = collect_transition_bank_v2(
        env_root=env_root,
        scenarios=scenarios,
        duration_sec=source_duration_sec,
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seed=source_seed,
        workers=workers,
    )
    models = {target: _fit_target_models(bank, target, fit_config=fit_config) for target in scenarios}

    jobs = []
    for target in scenarios:
        for seed in seeds:
            for policy in policies:
                jobs.append(
                    {
                        "env_root": str(env_root),
                        "scenario": target,
                        "policy": policy,
                        "models": models[target],
                        "prior_policy": prior_policy,
                        "duration_sec": float(duration_sec),
                        "control_interval_sec": int(control_interval_sec),
                        "warmup_sec": float(warmup_sec),
                        "seed": int(seed),
                        "tripinfo_output": str(tripinfo_root / f"{target}__{policy}__seed{seed}.xml"),
                    }
                )
    rows = []
    if int(workers) <= 1:
        rows = [_evaluate_worker(job) for job in jobs]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as pool:
            futures = [pool.submit(_evaluate_worker, job) for job in jobs]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: (row["target"], row["seed"], row["policy"]))
    target_rows, aggregate = _aggregate_seed_rows(rows, policies=policies)
    bootstrap = [
        paired_hierarchical_bootstrap(
            rows,
            reference="cfcmt_guard",
            baseline=baseline,
            metric="mean_queue",
        )
        for baseline in policies
        if baseline != "cfcmt_guard"
    ]
    model_diagnostics = {
        target: {
            "source_domains": model.diagnostics["source_domains"],
            "source_rows": model.diagnostics["source_rows"],
            "cfcmt": model.diagnostics["cfcmt"],
            "guard_calibration": model.diagnostics["guard_calibration"],
        }
        for target, model in models.items()
    }
    return {
        "experiment": "traffic_signal_resco_cfcmt_v2_leave_one_network_out",
        "runtime": _runtime_metadata(),
        "setting": {
            "env_root": str(env_root),
            "scenarios": list(scenarios),
            "policies": list(policies),
            "seeds": list(seeds),
            "source_duration_sec": float(source_duration_sec),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "source_seed": int(source_seed),
            "workers": int(workers),
            "prior_policy": prior_policy,
            "target_information_budget": "static_network_summary_plus_online_state_no_target_transition_labels",
            "signal_protocol": "safe_min_green_yellow_all_red_v2",
            "fit_config": asdict(fit_config),
        },
        "source_transition_diagnostics": {
            name: transition_diagnostics(dataset) for name, dataset in bank.items()
        },
        "model_diagnostics": model_diagnostics,
        "aggregate": aggregate,
        "targets": target_rows,
        "hierarchical_bootstrap": bootstrap,
        "seed_results": rows,
    }


def write_markdown_v2(result: Mapping[str, Any], path: Path) -> None:
    policies = result["setting"]["policies"]
    lines = [
        "# CFCMT V2 RESCO Leave-One-Network-Out Benchmark",
        "",
        "All externally controlled policies use explicit minimum-green, yellow, and all-red execution.",
        "The target contributes static topology summaries and online state only; no target transition labels are collected.",
        "",
        "| Policy | Mean queue | P90 queue | Trip duration | Trip waiting | Throughput |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in sorted(policies, key=lambda name: result["aggregate"][name]["mean_queue"]):
        metrics = result["aggregate"][policy]
        lines.append(
            f"| {policy} | {metrics['mean_queue']:.4f} | {metrics['p90_queue']:.4f} | "
            f"{metrics['mean_tripinfo_duration']:.4f} | {metrics['mean_tripinfo_waiting_time']:.4f} | "
            f"{metrics['throughput_ratio']:.4f} |"
        )
    lines.extend(["", "## Hierarchical Bootstrap", "", "| Baseline | Delta | 95% CI | Wins |", "|---|---:|---:|---:|"])
    for row in result["hierarchical_bootstrap"]:
        lines.append(
            f"| {row['baseline']} | {row.get('mean_delta', float('nan')):.4f} | "
            f"[{row.get('ci_low', float('nan')):.4f}, {row.get('ci_high', float('nan')):.4f}] | "
            f"{row.get('win_networks', 0)}/{row.get('network_count', 0)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scenarios", nargs="+", default=list(EXTENDED_SCENARIOS))
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES_V2))
    parser.add_argument("--seeds", nargs="+", type=int, default=[131, 337, 911])
    parser.add_argument("--source-duration-sec", type=float, default=600.0)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--source-seed", type=int, default=2027)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prior-policy", choices=["max_pressure", "phase_pressure", "spillback_pressure"], default="max_pressure")
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_cfcmt_v2_validation.json"))
    parser.add_argument("--md-out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_cfcmt_v2_validation.md"))
    args = parser.parse_args()
    result = run_leave_one_network_out_v2(
        env_root=args.env_root,
        scenarios=args.scenarios,
        policies=args.policies,
        seeds=args.seeds,
        source_duration_sec=args.source_duration_sec,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        source_seed=args.source_seed,
        workers=args.workers,
        prior_policy=args.prior_policy,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_v2(result, args.md_out)
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
