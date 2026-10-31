#!/usr/bin/env python3
"""Submit the seven V156A feature-aligned V150O recalculations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility import (
    UTILITY_GATE_PROTOCOL,
    V150N_AGGREGATE_PROTOCOL,
    V150N_REJECT_DECISION,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_feature_aligned_dense_source_utility import (
    ACTION_RANKER_PARENT_SHA256,
    ACTION_RANKER_PATCHED_SHA256,
    DERIVED_MARKER_SHA256,
    DERIVED_SNAPSHOT_ROOT,
    DERIVED_SNAPSHOT_SHA256,
    DERIVED_SOURCE_TREE_SHA256,
    EXPECTED_INPUT_SHA256,
    EXPECTED_LEGACY_RESULT_SHA256,
    FIT_PROTOCOL_SHA256,
    PARENT_SNAPSHOT_SHA256,
    SOURCE_MANIFEST_SHA256,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MINIMUM_OOF_GAIN,
    _read_json,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    GATE_CONFIG,
    MINIMUM_INTERVENTION_GROUPS,
    MINIMUM_NONDEGRADING_FOLDS,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
    V150J_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    SOURCE_AUDIT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import DEFAULT_SCHEDULER
from scripts.cluster.launch_tsc_state_conditioned_source_utility import (
    build_specs as _build_specs,
    select_target_specs,
)


LAUNCH_PROTOCOL = "tsc-v156a-feature-aligned-dense-source-utility-launch-v2"
CONFIG_PROTOCOL = "tsc-v156a-feature-aligned-dense-source-utility-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v156a_feature_aligned_dense_source_utility.json"
)
MODULE = "cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility"
SIGNATURE_PREFIX = "CFCMT/v156a/feature-aligned-dense-source-utility-replay-v1"
RESOURCE_FAMILY = "CFCMT-v156a-feature-aligned-dense-source-utility"
EXPECTED_SOURCE_CACHE_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v114_pure_waiting_rebuild_20260831/source_pure_waiting_cache_v1"
)
EXPECTED_CONVERSION_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v53r49_external_network_repair_20260810/"
    "conversion_v9_task/full_networks_v9"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
ACTION_RANKER_RELATIVE = Path("cf_h2o/traffic_signal/action_ranker.py")


def _validate_config(config: Mapping[str, Any]) -> None:
    gate = dict(config.get("development_gate", {}))
    city_gate = dict(config.get("city_specific_followup_gate", {}))
    identities = dict(config.get("frozen_input_identities", {}))
    utility = dict(config.get("utility_gate", {}))
    checks = (
        config.get("protocol") == CONFIG_PROTOCOL,
        config.get("recomputes")
        == "tsc-v150o-dense-pressure-pairwise-source-utility-v1",
        config.get("only_method_change")
        == "resolve the frozen rigid model inputs by their stored feature names",
        tuple(config.get("target_city_groups", ())) == EXPECTED_CITY_GROUPS,
        int(config.get("target_budget", -1)) == TARGET_BUDGET,
        int(config.get("common_evaluation_reserve_budget", -1)) == TARGET_BUDGET,
        float(config.get("source_prior_strength", -1.0)) == SOURCE_PRIOR_STRENGTH,
        int(config.get("outer_fold_count", -1)) == FOLD_COUNT,
        int(config.get("inner_fold_count", -1)) == FOLD_COUNT - 1,
        int(config.get("candidate_count_per_target", -1)) == 36,
        config.get("candidate_representation") == "single-state-conditioner-v1",
        config.get("action_pair") == "phase_pressure_reference_vs_rigid_target_only",
        config.get("utility_record_protocol") == DENSE_PRESSURE_PAIRWISE_RECORD_PROTOCOL,
        config.get("selector") == "nested_dense_fixed_pair_source_mechanism_utility",
        config.get("feature_binding") == "stored_feature_names",
        identities
        == {
            "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "derived_snapshot_sha256": DERIVED_SNAPSHOT_SHA256,
            "derived_source_tree_sha256": DERIVED_SOURCE_TREE_SHA256,
            "derived_snapshot_root": DERIVED_SNAPSHOT_ROOT,
            "derived_marker_sha256": DERIVED_MARKER_SHA256,
            "action_ranker_parent_sha256": ACTION_RANKER_PARENT_SHA256,
            "action_ranker_patched_sha256": ACTION_RANKER_PATCHED_SHA256,
            "source_cache_root": str(EXPECTED_SOURCE_CACHE_ROOT),
            "conversion_root": str(EXPECTED_CONVERSION_ROOT),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "fit_protocol_sha256": FIT_PROTOCOL_SHA256,
            "authorization_result_sha256": EXPECTED_INPUT_SHA256["authorization"],
            "v150j_rejection_sha256": EXPECTED_INPUT_SHA256["v150j_rejection"],
            "v150n_rejection_sha256": EXPECTED_INPUT_SHA256["v150n_rejection"],
            "fit_result_sha256": EXPECTED_INPUT_SHA256["fit_result"],
            "source_cache_audit_sha256": EXPECTED_INPUT_SHA256[
                "source_cache_audit"
            ],
            "legacy_result_sha256": EXPECTED_LEGACY_RESULT_SHA256,
        },
        utility.get("protocol") == UTILITY_GATE_PROTOCOL,
        float(utility.get("ridge_alpha", -1.0)) == GATE_CONFIG.ridge_alpha,
        float(utility.get("error_quantile", -1.0)) == GATE_CONFIG.error_quantile,
        int(utility.get("minimum_records", -1)) == GATE_CONFIG.minimum_records,
        float(utility.get("minimum_predicted_gain", -1.0))
        == GATE_CONFIG.minimum_predicted_gain,
        utility.get("source_identity_feature") is False,
        float(config.get("minimum_target_oof_gain", -1.0)) == MINIMUM_OOF_GAIN,
        float(config.get("minimum_identity_oof_gain", -1.0)) == MINIMUM_OOF_GAIN,
        float(config.get("minimum_source_blind_oof_gain", -1.0))
        == MINIMUM_OOF_GAIN,
        int(config.get("minimum_nondegrading_outer_folds", -1))
        == MINIMUM_NONDEGRADING_FOLDS,
        int(config.get("minimum_crossfitted_intervention_groups", -1))
        == MINIMUM_INTERVENTION_GROUPS,
        config.get("source_null_fallback") == "bitwise_exact_rigid_cfcmt",
        config.get("matched_placebo_pipeline")
        == "separately_nested_same_capacity_dense_pairwise_utility_gate",
        config.get("source_blind_pipeline")
        == "same_target_labels_state_features_and_gate_without_source_evidence",
        config.get("inference_unit") == "city",
        int(gate.get("minimum_identity_verified_source_cities", -1)) == 2,
        gate.get("require_no_city_regression") is True,
        gate.get("require_every_admitted_city_improve_rigid") is True,
        gate.get("require_every_admitted_city_beat_matched_placebo") is True,
        gate.get("require_every_admitted_city_beat_source_blind") is True,
        gate.get("require_exact_fallback_for_rejected_cities") is True,
        city_gate.get("require_crossfitted_selector_admission") is True,
        city_gate.get("require_improvement_over_rigid") is True,
        city_gate.get("require_improvement_over_matched_placebo") is True,
        city_gate.get("require_improvement_over_source_blind") is True,
        config.get("next_step_if_any_city_passes_city_specific_gate")
        == "freeze untouched-seed city-specific closed-loop confirmation before using a new external city",
    )
    if not all(checks):
        raise ValueError("V156A frozen configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    authorization_result_remote: Path,
    authorization_sha256: str,
    v150j_rejection_remote: Path,
    v150j_rejection_sha256: str,
    v150n_rejection_remote: Path,
    v150n_rejection_sha256: str,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    source_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
    cache_workers: int,
) -> list[dict[str, Any]]:
    return _build_specs(
        snapshot_root=snapshot_root,
        authorization_result_remote=authorization_result_remote,
        authorization_sha256=authorization_sha256,
        v150j_rejection_remote=v150j_rejection_remote,
        v150j_rejection_sha256=v150j_rejection_sha256,
        conversion_root=conversion_root,
        fit_result_remote=fit_result_remote,
        fit_result_sha256=fit_result_sha256,
        source_cache_root=source_cache_root,
        source_cache_audit_remote=source_cache_audit_remote,
        source_cache_audit_sha256=source_cache_audit_sha256,
        remote_output_root=remote_output_root,
        local_output_root=local_output_root,
        cache_workers=cache_workers,
        runtime_model_root=None,
        module=MODULE,
        signature_prefix=SIGNATURE_PREFIX,
        version_label="V156A",
        resource_family=RESOURCE_FAMILY,
        extra_target_args=(
            "--v150n-rejection",
            str(v150n_rejection_remote),
            "--v150n-rejection-sha256",
            str(v150n_rejection_sha256),
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--v150j-rejection-local", type=Path, required=True)
    parser.add_argument("--v150j-rejection-remote", type=Path, required=True)
    parser.add_argument("--v150n-rejection-local", type=Path, required=True)
    parser.add_argument("--v150n-rejection-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--target-city", action="append", choices=EXPECTED_CITY_GROUPS)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V156A launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V156A cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    if args.source_cache_root != EXPECTED_SOURCE_CACHE_ROOT:
        raise ValueError("V156A source cache root changed")
    if args.conversion_root != EXPECTED_CONVERSION_ROOT:
        raise ValueError("V156A conversion root changed")

    authorization = _read_json(args.authorization_result_local)
    v150j = _read_json(args.v150j_rejection_local)
    v150n = _read_json(args.v150n_rejection_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V156A requires the frozen V150C authorization")
    if (
        v150j.get("protocol") != V150J_AGGREGATE_PROTOCOL
        or v150j.get("development_gate", {}).get("passed") is not False
    ):
        raise ValueError("V156A requires the frozen V150J rejection")
    if (
        v150n.get("protocol") != V150N_AGGREGATE_PROTOCOL
        or v150n.get("development_gate", {}).get("passed") is not False
        or v150n.get("development_gate", {}).get("decision")
        != V150N_REJECT_DECISION
    ):
        raise ValueError("V156A requires the frozen V150N rejection")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V156A requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V156A source cache audit changed")

    stage = _read_json(args.stage_manifest)
    if (
        _sha256(args.stage_manifest) != DERIVED_MARKER_SHA256
        or
        stage.get("derived_from_snapshot_sha256")
        != PARENT_SNAPSHOT_SHA256
        or stage.get("snapshot_sha256") != DERIVED_SNAPSHOT_SHA256
        or stage.get("source_tree_sha256") != DERIVED_SOURCE_TREE_SHA256
        or stage.get("snapshot_root") != DERIVED_SNAPSHOT_ROOT
        or stage.get("patch", {}).get("path")
        != ACTION_RANKER_RELATIVE.as_posix()
        or stage.get("patch", {}).get("parent_sha256")
        != ACTION_RANKER_PARENT_SHA256
        or stage.get("patch", {}).get("patched_sha256")
        != ACTION_RANKER_PATCHED_SHA256
    ):
        raise ValueError("V156A requires the correction-only derived V150O snapshot")
    snapshot_root = Path(stage["snapshot_root"])
    targets = tuple(args.target_city or EXPECTED_CITY_GROUPS)
    local_inputs = (
        args.authorization_result_local,
        args.v150j_rejection_local,
        args.v150n_rejection_local,
        args.fit_result_local,
        args.source_cache_audit_local,
    )
    remote_inputs = (
        args.authorization_result_remote,
        args.v150j_rejection_remote,
        args.v150n_rejection_remote,
        args.fit_result_remote,
        args.source_cache_audit_remote,
    )
    hashes = [_sha256(path) for path in local_inputs]
    input_hashes = dict(zip(EXPECTED_INPUT_SHA256, hashes, strict=True))
    if input_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError(f"V156A frozen local input identities changed: {input_hashes}")
    scheduler = _load_scheduler(args.scheduler)
    marker_code, marker_stdout, marker_stderr = scheduler.run_on(
        "node001",
        shlex.join(["cat", str(snapshot_root / ".cfcmt_snapshot.json")]),
        timeout=120,
        check=False,
    )
    if int(marker_code) != 0 or json.loads(str(marker_stdout)) != stage:
        raise RuntimeError(f"remote V156A snapshot marker changed: {marker_stderr}")
    snapshot_files = {
        "source_manifest": snapshot_root / SOURCE_MANIFEST_RELATIVE,
        "fit_protocol": snapshot_root / FIT_PROTOCOL_RELATIVE,
        "action_ranker": snapshot_root / ACTION_RANKER_RELATIVE,
        "snapshot_marker": snapshot_root / ".cfcmt_snapshot.json",
    }
    checks = [
        shlex.join(["test", "-d", str(args.source_cache_root)]),
        shlex.join(["test", "-d", str(args.conversion_root)]),
        *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        *(
            shlex.join(["test", "-f", str(path)])
            for path in snapshot_files.values()
        ),
        *(
            shlex.join(["test", "!", "-e", str(args.remote_output_root / city)])
            for city in targets
        ),
    ]
    code, _, stderr = scheduler.run_on(
        "node001", " && ".join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V156A preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in str(stdout).splitlines() if line.split()]
    if int(hash_code) != 0 or remote_hashes != hashes:
        raise RuntimeError(f"remote V156A input identities changed: {remote_hashes}; {hash_stderr}")
    snapshot_hash_code, snapshot_stdout, snapshot_hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in snapshot_files.values())]),
        timeout=120,
        check=False,
    )
    snapshot_hashes = dict(
        zip(
            snapshot_files,
            [
                line.split()[0]
                for line in str(snapshot_stdout).splitlines()
                if line.split()
            ],
            strict=True,
        )
    )
    expected_snapshot_hashes = {
        "source_manifest": SOURCE_MANIFEST_SHA256,
        "fit_protocol": FIT_PROTOCOL_SHA256,
        "action_ranker": ACTION_RANKER_PATCHED_SHA256,
        "snapshot_marker": DERIVED_MARKER_SHA256,
    }
    if (
        int(snapshot_hash_code) != 0
        or snapshot_hashes != expected_snapshot_hashes
    ):
        raise RuntimeError(
            "remote V156A snapshot file identities changed: "
            f"{snapshot_hashes}; {snapshot_hash_stderr}"
        )

    specs = select_target_specs(
        build_specs(
            snapshot_root=snapshot_root,
            authorization_result_remote=args.authorization_result_remote,
            authorization_sha256=hashes[0],
            v150j_rejection_remote=args.v150j_rejection_remote,
            v150j_rejection_sha256=hashes[1],
            v150n_rejection_remote=args.v150n_rejection_remote,
            v150n_rejection_sha256=hashes[2],
            conversion_root=args.conversion_root,
            fit_result_remote=args.fit_result_remote,
            fit_result_sha256=hashes[3],
            source_cache_root=args.source_cache_root,
            source_cache_audit_remote=args.source_cache_audit_remote,
            source_cache_audit_sha256=hashes[4],
            remote_output_root=args.remote_output_root,
            local_output_root=args.local_output_root,
            cache_workers=args.cache_workers,
        ),
        targets,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v156a-feature-aligned-dense-source-utility",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    submitted_tasks: list[dict[str, Any]] = []
    submission_parsed = False
    if completed.returncode == 0:
        try:
            scheduler_payload = json.loads(completed.stdout)
            submitted_tasks = list(scheduler_payload.get("submitted", ()))
            submission_parsed = len(submitted_tasks) == len(specs)
        except (TypeError, ValueError):
            submission_parsed = False
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0 and submission_parsed,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "derived_from_snapshot_sha256": stage["derived_from_snapshot_sha256"],
        "snapshot_marker_verified": True,
        "snapshot_patch": stage["patch"],
        "execution_module": MODULE,
        "frozen_config": str((PROJECT_ROOT / CONFIG_RELATIVE).resolve()),
        "frozen_config_sha256": _sha256(PROJECT_ROOT / CONFIG_RELATIVE),
        "input_sha256": {
            "authorization": hashes[0],
            "v150j_rejection": hashes[1],
            "v150n_rejection": hashes[2],
            "fit_result": hashes[3],
            "source_cache_audit": hashes[4],
        },
        "snapshot_file_sha256": snapshot_hashes,
        "source_cache_root": str(args.source_cache_root),
        "conversion_root": str(args.conversion_root),
        "target_cities": list(targets),
        "task_count": len(specs),
        "submitted_tasks": submitted_tasks,
        "remote_output_root": str(args.remote_output_root),
        "task_specs": specs,
    }
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "submitted": payload["submitted"],
                "task_count": len(specs),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    if not payload["submitted"]:
        raise RuntimeError("V156A scheduler submission failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
