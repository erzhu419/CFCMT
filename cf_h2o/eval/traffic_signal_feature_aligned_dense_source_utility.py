"""Aggregate the correction-only V150O replay with feature-name binding."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility import (
    AGGREGATE_PROTOCOL as V150O_AGGREGATE_PROTOCOL,
    RESULT_PROTOCOL as V150O_RESULT_PROTOCOL,
    aggregate_results as _aggregate_v150o_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    _read_json,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v156a-feature-aligned-dense-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v156a-feature-aligned-dense-source-utility-aggregate-v1"
FEATURE_BINDING = "stored_feature_names"
LAUNCH_PROTOCOL = "tsc-v156a-feature-aligned-dense-source-utility-launch-v2"
PARENT_SNAPSHOT_SHA256 = (
    "0eb4466ee3617a0cda755e1b2229ecccafbee76cf06acaad0d24250eab9228e0"
)
DERIVED_SNAPSHOT_SHA256 = (
    "9ba65e5006b01245ef439aed509f279b14b35b9372aa9d22e127d95ae38ff173"
)
DERIVED_SOURCE_TREE_SHA256 = (
    "a78eb925c1165330fe3468eda7924a464a29cba1d93d637a0b554056eca71650"
)
DERIVED_SNAPSHOT_ROOT = (
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/9ba65e5006b01245ef43"
)
DERIVED_MARKER_SHA256 = (
    "c89438d0775caab8e2bcd63128def43d5ca53ef58f61ceea2cc988437464fbdf"
)
ACTION_RANKER_PARENT_SHA256 = (
    "b503a4e004ee44e6c630b0e2e7ccfb07eff418e6fe2dc7faf0cdc67a0d73253f"
)
ACTION_RANKER_PATCHED_SHA256 = (
    "aa70a323feff130dfbbfb750149d913c913d7ea7d8c20939212e1dd1aeef4c6d"
)
SOURCE_MANIFEST_SHA256 = (
    "f61ae8deb6f3bf00fee05779a8b5c54dad84263c5077b5ae7ec810c344f4c6cb"
)
FIT_PROTOCOL_SHA256 = (
    "1d64b4e8711475a9f4c18a066d2d8a80e5cf7d55bbad302dd0a4d9e1886ab5e3"
)
EXPECTED_INPUT_SHA256 = {
    "authorization": "76e7ef3559e2ebbf68869970ab4ebec0740037368175ef5feff310d01ca0ad0b",
    "v150j_rejection": "a2498680f233285dd28b1c11b51a654f7c0a9d8b8da0e25ac5c8c5cb69a366db",
    "v150n_rejection": "e1c73a8f8c49dd5a633efc5563066e72ee7db07605d5084b999d8f40375dab3b",
    "fit_result": "b88943de80d19871a9babe817155391c34aec87087017c62502f045b2f68f683",
    "source_cache_audit": "0114c009a17a261ce240c27a8557a43f0d9eed0e25faead53eb149d13d9daa1a",
}
EXPECTED_SOURCE_CACHE_ROOT = (
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v114_pure_waiting_rebuild_20260831/source_pure_waiting_cache_v1"
)
EXPECTED_CONVERSION_ROOT = (
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v53r49_external_network_repair_20260810/"
    "conversion_v9_task/full_networks_v9"
)
EXPECTED_LEGACY_RESULT_SHA256 = {
    "atlanta": "1d499dd21b49aca45ab1787ad1319129008c4a9f86898c337e6f16a12bc60a3d",
    "cologne": "03666664fea96c0f7cacf6dc592822e9cd2551daf320f422df6d13bd63e69eab",
    "hangzhou": "92b3ec8eb545b811d786639afd78b3fe82e0050e6d747193eac65f21e984e04b",
    "ingolstadt": "21382fa7abef6c1ac69e8099ad4c87aa05330203cc10273744506ec3aa834645",
    "new_york": "ca97aa189a44f4e4399c2a0de806c223468bd9d16766a0eff0daff7fa48b1a9d",
    "resco_synthetic": "0f04c12cde7380edd640e7fac459cefbe1739ce41349c0ae47230d17e185dbc2",
    "salt_lake_city": "258c334120777accf8f1b7b37573391eac8d290653078e76ea923eaee08fed55",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_correction_contract() -> dict[str, Any]:
    return {
        "parent_protocol": V150O_RESULT_PROTOCOL,
        "feature_binding": FEATURE_BINDING,
        "parent_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
        "derived_snapshot_sha256": DERIVED_SNAPSHOT_SHA256,
        "derived_source_tree_sha256": DERIVED_SOURCE_TREE_SHA256,
        "action_ranker_parent_sha256": ACTION_RANKER_PARENT_SHA256,
        "action_ranker_patched_sha256": ACTION_RANKER_PATCHED_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "fit_protocol_sha256": FIT_PROTOCOL_SHA256,
        "target_counterfactual_costs_reused": True,
        "source_counterfactual_costs_reused": True,
        "utility_records_recomputed": True,
        "target_budget_unchanged": True,
        "folds_and_thresholds_unchanged": True,
        "new_simulations": 0,
    }


def annotate_target_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach V156A provenance to one raw result from the derived snapshot."""

    result = deepcopy(dict(result))
    if result.get("protocol") != V150O_RESULT_PROTOCOL:
        raise ValueError("V156A parent result protocol changed")
    result["protocol"] = RESULT_PROTOCOL
    result["scientific_status"] = "feature-aligned-v150o-development-recalculation"
    result["correction_only_contract"] = _target_correction_contract()
    result["claim_boundary"] = (
        "V156A recomputes V150O on reused seven-city development data after "
        "the stored-feature-name binding correction. It is not closed-loop or "
        "fresh-city confirmation."
    )
    return result


def _arm_effect(row: Mapping[str, Any], candidate: str, reference: str) -> float:
    arms = row.get("arm_summaries", {})
    return float(arms[candidate]["mean"]) - float(arms[reference]["mean"])


def _validate_replay_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    patch = dict(provenance.get("snapshot_patch", {}))
    snapshot_files = dict(provenance.get("snapshot_file_sha256", {}))
    expected_signatures = {
        f"CFCMT/v156a/feature-aligned-dense-source-utility-replay-v1/{city}"
        for city in EXPECTED_CITY_GROUPS
    }
    task_specs = tuple(provenance.get("task_specs", ()))
    submitted_tasks = tuple(provenance.get("submitted_tasks", ()))
    checks = {
        "launch_protocol": provenance.get("protocol") == LAUNCH_PROTOCOL,
        "submitted": provenance.get("submitted") is True,
        "seven_targets": tuple(provenance.get("target_cities", ()))
        == EXPECTED_CITY_GROUPS,
        "snapshot": provenance.get("snapshot_sha256") == DERIVED_SNAPSHOT_SHA256,
        "snapshot_root": provenance.get("snapshot_root") == DERIVED_SNAPSHOT_ROOT,
        "source_tree": provenance.get("source_tree_sha256")
        == DERIVED_SOURCE_TREE_SHA256,
        "snapshot_marker": provenance.get("snapshot_marker_verified") is True,
        "parent_snapshot": provenance.get("derived_from_snapshot_sha256")
        == PARENT_SNAPSHOT_SHA256,
        "patch_path": patch.get("path") == "cf_h2o/traffic_signal/action_ranker.py",
        "patch_parent": patch.get("parent_sha256") == ACTION_RANKER_PARENT_SHA256,
        "patch_result": patch.get("patched_sha256") == ACTION_RANKER_PATCHED_SHA256,
        "input_hashes": provenance.get("input_sha256") == EXPECTED_INPUT_SHA256,
        "source_manifest": snapshot_files.get("source_manifest")
        == SOURCE_MANIFEST_SHA256,
        "fit_protocol": snapshot_files.get("fit_protocol") == FIT_PROTOCOL_SHA256,
        "action_ranker": snapshot_files.get("action_ranker")
        == ACTION_RANKER_PATCHED_SHA256,
        "snapshot_marker_file": snapshot_files.get("snapshot_marker")
        == DERIVED_MARKER_SHA256,
        "execution_module": provenance.get("execution_module")
        == "cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility",
        "source_cache_root": provenance.get("source_cache_root")
        == EXPECTED_SOURCE_CACHE_ROOT,
        "conversion_root": provenance.get("conversion_root")
        == EXPECTED_CONVERSION_ROOT,
        "task_specs": len(task_specs) == len(EXPECTED_CITY_GROUPS)
        and {str(spec.get("signature")) for spec in task_specs}
        == expected_signatures,
        "submitted_tasks": len(submitted_tasks) == len(EXPECTED_CITY_GROUPS)
        and {str(task.get("signature")) for task in submitted_tasks}
        == expected_signatures,
    }
    if not all(checks.values()):
        raise ValueError(f"V156A replay provenance changed: {checks}")
    return checks


def _validate_result_file_bindings(
    result_paths: Sequence[Path],
    result_rows: Sequence[Mapping[str, Any]],
    legacy_paths: Sequence[Path],
    legacy_rows: Sequence[Mapping[str, Any]],
    replay_provenance: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    specs = tuple(replay_provenance.get("task_specs", ()))
    expected_corrected = {
        Path(str(spec["local_result_dir"])).name: (
            Path(str(spec["local_result_dir"])) / "result.json"
        ).resolve()
        for spec in specs
    }
    actual_corrected = {
        str(row.get("target_city")): path.resolve()
        for path, row in zip(result_paths, result_rows, strict=True)
    }
    if (
        set(expected_corrected) != set(EXPECTED_CITY_GROUPS)
        or actual_corrected != expected_corrected
    ):
        raise ValueError("V156A corrected result paths do not match launch manifest")

    actual_legacy_hashes = {
        str(row.get("target_city")): _sha256(path)
        for path, row in zip(legacy_paths, legacy_rows, strict=True)
    }
    if actual_legacy_hashes != EXPECTED_LEGACY_RESULT_SHA256:
        raise ValueError("V156A legacy result identities changed")
    return {
        "corrected_result_sha256": {
            str(row.get("target_city")): _sha256(path)
            for path, row in zip(result_paths, result_rows, strict=True)
        },
        "legacy_result_sha256": actual_legacy_hashes,
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    legacy_results: Sequence[Mapping[str, Any]],
    replay_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate corrected results and prove the data split stayed fixed."""

    provenance_checks = _validate_replay_provenance(replay_provenance)
    corrected = {
        str(row.get("target_city")): (
            annotate_target_result(row)
            if row.get("protocol") == V150O_RESULT_PROTOCOL
            else dict(row)
        )
        for row in results
    }
    legacy = {str(row.get("target_city")): dict(row) for row in legacy_results}
    expected = set(EXPECTED_CITY_GROUPS)
    if (
        len(results) != len(expected)
        or len(legacy_results) != len(expected)
        or set(corrected) != expected
        or set(legacy) != expected
        or any(row.get("protocol") != RESULT_PROTOCOL for row in corrected.values())
        or any(row.get("protocol") != V150O_RESULT_PROTOCOL for row in legacy.values())
    ):
        raise ValueError("V156A requires seven corrected and seven legacy targets")
    if any(
        row.get("correction_only_contract") != _target_correction_contract()
        for row in corrected.values()
    ):
        raise ValueError("V156A correction-only contract changed")

    invariants: dict[str, dict[str, bool]] = {}
    comparison: dict[str, dict[str, Any]] = {}
    parent_rows = []
    for city in EXPECTED_CITY_GROUPS:
        new = corrected[city]
        old = legacy[city]
        checks = {
            "target_scenarios_equal": new.get("target_scenarios")
            == old.get("target_scenarios"),
            "source_city_groups_equal": new.get("source_city_groups")
            == old.get("source_city_groups"),
            "target_budget_equal": new.get("target_budget") == old.get("target_budget"),
            "estimand_equal": new.get("estimand") == old.get("estimand"),
            "selection_audit_equal": new.get("selection_audit")
            == old.get("selection_audit"),
            "evaluation_reserve_audit_equal": new.get("evaluation_reserve_audit")
            == old.get("evaluation_reserve_audit"),
            "information_budget_equal": new.get("information_budget")
            == old.get("information_budget"),
            "input_evidence_equal": {
                key: value
                for key, value in dict(new.get("inputs", {})).items()
                if key != "source_manifest"
            }
            == {
                key: value
                for key, value in dict(old.get("inputs", {})).items()
                if key != "source_manifest"
            },
            "source_manifest_filename_equal": Path(
                str(new.get("inputs", {}).get("source_manifest", ""))
            ).name
            == Path(str(old.get("inputs", {}).get("source_manifest", ""))).name,
            "source_manifest_filename_frozen": Path(
                str(new.get("inputs", {}).get("source_manifest", ""))
            ).name
            == "traffic_signal_cross_city_v2_saltlake18.json",
            "source_bank_audit_equal": bool(
                new.get("input_audits", {}).get("source_bank")
            )
            and new.get("input_audits", {}).get("source_bank")
            == old.get("input_audits", {}).get("source_bank"),
            "source_cache_root_frozen": new.get("input_audits", {})
            .get("source_bank", {})
            .get("cache_root")
            == EXPECTED_SOURCE_CACHE_ROOT,
        }
        if not all(checks.values()):
            raise ValueError(f"V156A changed more than feature binding for {city}: {checks}")
        invariants[city] = checks
        comparison[city] = {
            "legacy_admitted": bool(old["method"]["selector"]["admitted"]),
            "corrected_admitted": bool(new["method"]["selector"]["admitted"]),
            "legacy_selected_effect_vs_rigid": _arm_effect(
                old, "selected_dense_source_pairwise", "rigid_target_only"
            ),
            "corrected_selected_effect_vs_rigid": _arm_effect(
                new, "selected_dense_source_pairwise", "rigid_target_only"
            ),
            "legacy_selected_effect_vs_placebo": _arm_effect(
                old,
                "selected_dense_source_pairwise",
                "selected_dense_matched_placebo",
            ),
            "corrected_selected_effect_vs_placebo": _arm_effect(
                new,
                "selected_dense_source_pairwise",
                "selected_dense_matched_placebo",
            ),
            "legacy_selected_effect_vs_source_blind": _arm_effect(
                old,
                "selected_dense_source_pairwise",
                "selected_dense_source_blind",
            ),
            "corrected_selected_effect_vs_source_blind": _arm_effect(
                new,
                "selected_dense_source_pairwise",
                "selected_dense_source_blind",
            ),
        }
        parent = deepcopy(new)
        parent["protocol"] = V150O_RESULT_PROTOCOL
        parent_rows.append(parent)

    aggregate = _aggregate_v150o_results(parent_rows)
    if aggregate.get("protocol") != V150O_AGGREGATE_PROTOCOL:
        raise ValueError("V156A parent aggregate protocol changed")
    aggregate["protocol"] = AGGREGATE_PROTOCOL
    aggregate["development_gate"]["decision"] = (
        "authorize_v156a_global_development_followup"
        if aggregate["development_gate"]["passed"]
        else "reject_v156a_global_source_selector"
    )
    aggregate["scientific_status"] = (
        "feature-aligned-development-gate-pass"
        if aggregate["development_gate"]["passed"]
        else "feature-aligned-development-gate-reject"
    )
    aggregate["correction_only_contract"] = {
        "parent_protocol": V150O_AGGREGATE_PROTOCOL,
        "feature_binding": FEATURE_BINDING,
        "all_city_input_and_split_invariants_passed": True,
        "per_city_checks": invariants,
        "replay_provenance_checks": provenance_checks,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "fit_protocol_sha256": FIT_PROTOCOL_SHA256,
        "new_simulations": 0,
    }
    candidates = [
        city
        for city in EXPECTED_CITY_GROUPS
        if comparison[city]["corrected_admitted"]
        and comparison[city]["corrected_selected_effect_vs_rigid"] < 0.0
        and comparison[city]["corrected_selected_effect_vs_placebo"] < 0.0
        and comparison[city]["corrected_selected_effect_vs_source_blind"] < 0.0
    ]
    aggregate["city_specific_followup_gate"] = {
        "requirements": {
            "crossfitted_selector_admission": True,
            "improves_rigid_target_only": True,
            "beats_matched_placebo": True,
            "beats_source_blind": True,
        },
        "candidates": candidates,
        "passed": bool(candidates),
        "decision": (
            "authorize_untouched_seed_city_specific_closed_loop"
            if candidates
            else "stop_dense_source_selector_family"
        ),
        "does_not_override_global_gate": True,
    }
    aggregate["legacy_v150o_comparison"] = comparison
    aggregate["claim_boundary"] = (
        "Correction-only replay on the reused seven-city development cache; "
        "not closed-loop or fresh-city confirmation."
    )
    return aggregate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs=7, type=Path, required=True)
    parser.add_argument("--legacy-results", nargs=7, type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V156A result: {args.out}")
    corrected_rows = [_read_json(path) for path in args.results]
    legacy_rows = [_read_json(path) for path in args.legacy_results]
    launch = _read_json(args.launch_manifest)
    file_identities = _validate_result_file_bindings(
        args.results,
        corrected_rows,
        args.legacy_results,
        legacy_rows,
        launch,
    )
    result = aggregate_results(corrected_rows, legacy_rows, launch)
    result["artifact_provenance"] = {
        "launch_manifest": str(args.launch_manifest.resolve()),
        "launch_manifest_sha256": _sha256(args.launch_manifest),
        "snapshot_root": launch["snapshot_root"],
        "snapshot_sha256": launch["snapshot_sha256"],
        "submitted_task_ids": [
            str(task["id"]) for task in launch.get("submitted_tasks", ())
        ],
        **file_identities,
    }
    status = "PASS" if result["development_gate"]["passed"] else "REJECT"
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": status,
                "protocol": result["protocol"],
                "target_city": result.get("target_city"),
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
