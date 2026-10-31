#!/usr/bin/env python3
"""Audit the pre-efficacy RoadnetSZ Shenzhen network admission matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402


AUDIT_PROTOCOL = "tsc-v94-roadnetsz-shenzhen-network-admission-audit-v1"
RESULT_PROTOCOL = "full-horizon-libsumo-demand-conserving-diagnostic-admission-v4"
PROTOCOL_PROTOCOLS = frozenset(
    {
        "tsc-v94-roadnetsz-shenzhen-pre-efficacy-network-admission-v1",
        "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v1",
        "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v2",
        "tsc-v94-roadnetsz-shenzhen-pcl-pre-efficacy-network-admission-v3",
    }
)
CITYFLOW_TLS_PROTOCOL = (
    "cityflow-roadlink-to-netconvert-linkindex-remap-"
    "permissive-intergreen-merge-conflict-yield-v4"
)
NATIVE_TLS_PROTOCOL = "roadnetsz-native-sumo-tls-preserved-v1"
NATIVE_TLS_REBUILD_PROTOCOL = "roadnetsz-source-tll-sumo122-rebuild-v2"
VIRTUAL_LANE_PROTOCOL = "cityflow-virtual-cartesian-lane-link-monotone-rank-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_matrix(*, protocol_path: Path, results_root: Path) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") not in PROTOCOL_PROTOCOLS:
        raise ValueError("Shenzhen admission protocol identity changed")
    scenarios = tuple(str(value) for value in protocol["scenarios"])
    seeds = tuple(int(value) for value in protocol["seeds"])
    if len(set(scenarios)) != len(scenarios) or len(set(seeds)) != len(seeds):
        raise ValueError("Shenzhen admission matrix contains duplicate identities")
    conversion_spec = dict(protocol["conversion"])
    runtime_spec = dict(protocol["runtime"])
    admission_spec = dict(protocol["admission"])

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for scenario in scenarios:
        for seed in seeds:
            path = Path(results_root) / f"{scenario}_seed{seed}" / "admission.json"
            if not path.is_file():
                errors.append(f"missing result: {scenario}/{seed}")
                continue
            result = _read_json(path)
            conversion = dict(result.get("conversion", {}))
            network = dict(conversion.get("network", {}))
            tls = dict(network.get("tls_semantic_audit", {}))
            virtual = dict(network.get("virtual_lane_link_reduction", {}))
            observed = dict(result.get("observed", {}))
            runtime = dict(result.get("runtime", {}))
            checks = dict(result.get("checks", {}))
            is_cityflow = conversion_spec["protocol"] == (
                "cfcmt-cityflow-full-external-sumo-conversion-v10"
            )
            is_native_v1 = conversion_spec["protocol"] == (
                "cfcmt-roadnetsz-native-sumo-window-package-v1"
            )
            is_native_v2 = conversion_spec["protocol"] == (
                "cfcmt-roadnetsz-native-sumo122-window-package-v2"
            )
            is_native_v3 = conversion_spec["protocol"] == (
                "cfcmt-roadnetsz-native-sumo122-window-package-v3"
            )
            is_native_v4 = conversion_spec["protocol"] == (
                "cfcmt-roadnetsz-native-sumo122-window-package-v4"
            )
            is_native = (
                is_native_v1 or is_native_v2 or is_native_v3 or is_native_v4
            )
            if not (is_cityflow or is_native):
                raise ValueError(
                    f"unsupported Shenzhen conversion protocol: {conversion_spec['protocol']}"
                )
            identity_gate = {
                "result_protocol": result.get("protocol") == RESULT_PROTOCOL,
                "scenario_seed": (
                    str(result.get("scenario")),
                    int(result.get("seed", -1)),
                )
                == (scenario, seed),
                "conversion_protocol": conversion.get("protocol")
                == conversion_spec["protocol"],
                "conversion_manifest": conversion.get("manifest_sha256")
                == conversion_spec["manifest_sha256"],
                "conversion_tree": conversion.get("tree_sha256")
                == conversion_spec["tree_sha256"],
                "sumo_version": runtime.get("libsumo_version")
                == runtime_spec["sumo_version"],
                "full_horizon": float(observed.get("final_time_sec", -1.0))
                >= float(runtime_spec["horizon_sec"]),
                "controllable_tls": int(
                    observed.get("controllable_tls_count", -1)
                )
                >= int(runtime_spec["minimum_controllable_tls"]),
            }
            if is_cityflow:
                identity_gate.update(
                    {
                        "tls_protocol": tls.get("protocol")
                        == CITYFLOW_TLS_PROTOCOL,
                        "positive_conflict_yield_count": int(
                            tls.get("conflict_yield_phase_link_count", 0)
                        )
                        > 0,
                        "virtual_lane_protocol": virtual.get("protocol")
                        == VIRTUAL_LANE_PROTOCOL,
                        "virtual_lane_audit": int(
                            virtual.get(
                                "post_reduction_crossing_road_link_count", -1
                            )
                        )
                        == 0
                        and int(
                            virtual.get("source_lane_coverage_failures", -1)
                        )
                        == 0
                        and int(
                            virtual.get("target_lane_coverage_failures", -1)
                        )
                        == 0,
                    }
                )
            elif is_native_v1:
                source_window = dict(network.get("source_window", {}))
                network_build = dict(network.get("network_build", {}))
                identity_gate.update(
                    {
                        "tls_protocol": tls.get("protocol")
                        == NATIVE_TLS_PROTOCOL,
                        "native_tls_identity": tls.get(
                            "source_tls_programs_preserved_byte_for_byte"
                        )
                        is True
                        and tls.get("source_net_sha256")
                        == tls.get("packaged_net_sha256"),
                        "native_network_build": network_build.get("protocol")
                        == "roadnetsz-native-sumo-window-duarouter-v1"
                        and int(
                            dict(network_build.get("duarouter", {})).get(
                                "returncode", -1
                            )
                        )
                        == 0,
                        "routing_retention": float(
                            source_window.get("routing_retention", -1.0)
                        )
                        >= float(admission_spec["minimum_routing_retention"]),
                    }
                )
            else:
                source_window = dict(network.get("source_window", {}))
                network_build = dict(network.get("network_build", {}))
                declared_count = int(tls.get("declared_tls_node_count", -1))
                explicit_id_count = int(
                    tls.get("explicit_source_tls_id_count", -1)
                )
                rebuilt_count = int(tls.get("rebuilt_tls_id_count", -1))
                generated_count = int(
                    tls.get("auto_generated_tls_id_count", -1)
                )
                identity_gate.update(
                    {
                        "tls_protocol": tls.get("protocol")
                        == NATIVE_TLS_REBUILD_PROTOCOL,
                        "native_tls_reconstruction": declared_count > 0
                        and explicit_id_count > 0
                        and rebuilt_count == declared_count
                        and rebuilt_count
                        == int(network.get("controlled_intersection_count", -2))
                        and generated_count == rebuilt_count - explicit_id_count
                        and int(tls.get("rebuilt_phase_count", 0)) > 0
                        and int(
                            tls.get("missing_declared_tls_id_count", -1)
                        )
                        == 0
                        and int(
                            tls.get("unexpected_rebuilt_tls_id_count", -1)
                        )
                        == 0
                        and int(
                            tls.get("missing_explicit_tls_program_count", -1)
                        )
                        == 0
                        and int(
                            tls.get("explicit_phase_count_mismatch_count", -1)
                        )
                        == 0
                        and tls.get("all_declared_tls_nodes_rebuilt") is True
                        and tls.get(
                            "explicit_source_tls_program_phase_counts_preserved"
                        )
                        is True
                        and (
                            tls.get("auto_generated_phase_layout") == "incoming"
                            and int(
                                tls.get(
                                    "auto_generated_minor_green_signal_count",
                                    -1,
                                )
                            )
                            == 0
                            if is_native_v4
                            else True
                        ),
                        "native_network_build": network_build.get("protocol")
                        == (
                            (
                                "roadnetsz-source-components-sumo122-"
                                "protected-incoming-window-duarouter-v4"
                            )
                            if is_native_v4
                            else (
                                (
                                "roadnetsz-source-components-sumo122-"
                                "turnspeed55-window-duarouter-v3"
                                )
                                if is_native_v3
                                else (
                                    "roadnetsz-source-components-sumo122-"
                                    "window-duarouter-v2"
                                )
                            )
                        )
                        and (
                            float(
                                network_build.get(
                                    "junction_turn_speed_limit_mps", -1.0
                                )
                            )
                            == 5.5
                            if is_native_v3
                            else True
                        )
                        and (
                            network_build.get(
                                "auto_generated_tls_phase_layout"
                            )
                            == "incoming"
                            if is_native_v4
                            else True
                        )
                        and int(
                            dict(network_build.get("netconvert", {})).get(
                                "returncode", -1
                            )
                        )
                        == 0
                        and int(
                            dict(network_build.get("duarouter", {})).get(
                                "returncode", -1
                            )
                        )
                        == 0,
                        "routing_retention": float(
                            source_window.get("routing_retention", -1.0)
                        )
                        >= float(admission_spec["minimum_routing_retention"]),
                    }
                )
            identity_gate["passed"] = all(identity_gate.values())
            if not identity_gate["passed"]:
                errors.append(f"identity gate failed: {scenario}/{seed}")
            failed_checks = sorted(name for name, passed in checks.items() if not passed)
            admitted = result.get("passed") is True and not failed_checks
            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": _sha256(path),
                    "hostname": runtime.get("hostname"),
                    "identity_gate": identity_gate,
                    "admitted": admitted,
                    "failed_checks": failed_checks,
                    "collision_incidents": observed.get(
                        "unique_collision_incidents"
                    ),
                    "teleports": observed.get("starting_teleports"),
                    "departed": observed.get("departed"),
                    "arrived": observed.get("arrived"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                }
            )

    expected = {(scenario, seed) for scenario in scenarios for seed in seeds}
    observed_files = set(Path(results_root).glob("*/admission.json"))
    scenario_rows = {
        scenario: [row for row in rows if row["scenario"] == scenario]
        for scenario in scenarios
    }
    scenario_decisions = {
        scenario: {
            "eligible": len(group) == len(seeds)
            and all(row["admitted"] for row in group),
            "passed_seed_count": sum(row["admitted"] for row in group),
            "seed_count": len(seeds),
            "total_collision_incidents": sum(
                int(row["collision_incidents"] or 0) for row in group
            ),
            "failed_seeds": [
                int(row["seed"]) for row in group if not row["admitted"]
            ],
        }
        for scenario, group in scenario_rows.items()
    }
    eligible = [
        scenario
        for scenario in scenarios
        if scenario_decisions[scenario]["eligible"]
    ]
    integrity_gate = {
        "exact_matrix": len(rows) == len(expected),
        "no_extra_result_files": len(observed_files) == len(expected),
        "all_identity_gates_pass": bool(rows)
        and all(row["identity_gate"]["passed"] for row in rows),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    selection_gate = {
        "minimum_eligible_scenarios": len(eligible)
        >= int(admission_spec["minimum_eligible_scenarios"]),
        "only_all_seed_pass_scenarios_selected": all(
            scenario_decisions[scenario]["eligible"]
            for scenario in eligible
        ),
    }
    selection_gate["passed"] = all(selection_gate.values())
    authorized = integrity_gate["passed"] and selection_gate["passed"]
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if authorized else "FAIL",
        "decision": (
            "authorize_selected_shenzhen_scenarios_for_pre_registered_efficacy"
            if authorized
            else "block_shenzhen_efficacy_experiments"
        ),
        "claim_boundary": (
            "Pre-efficacy network and simulator-safety admission only. Rejected "
            "scenarios remain evidence and no controller outcome is used here."
        ),
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "conversion": conversion_spec,
        "matrix_size": len(expected),
        "rows": rows,
        "scenario_decisions": scenario_decisions,
        "eligible_scenarios": eligible,
        "rejected_scenarios": [
            scenario for scenario in scenarios if scenario not in eligible
        ],
        "integrity_gate": integrity_gate,
        "selection_gate": selection_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite admission audit: {args.out}")
    payload = audit_matrix(
        protocol_path=args.protocol,
        results_root=args.results_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "eligible_scenarios": payload["eligible_scenarios"],
                "rejected_scenarios": payload["rejected_scenarios"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
