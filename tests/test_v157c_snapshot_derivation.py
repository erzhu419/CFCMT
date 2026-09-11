from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

from scripts.cluster import derive_tsc_v157c_jinan_native_action_branches_snapshot as derive


def test_v157c_snapshot_binds_reviewed_parent_and_runtime_dependencies():
    assert derive.PARENT_ROOT == Path(
        "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/"
        "645262f59ee4e0487f0f"
    )
    assert derive.PARENT_SNAPSHOT_SHA256 == (
        "645262f59ee4e0487f0fd480e831fa3b09e1b6a66604f236a1531e207199235a"
    )
    assert derive.PARENT_SOURCE_TREE_SHA256 == (
        "0dfbeb334f072431598c47dd13f4b40a36349ceb69c71d0c7b443abd0d02dc64"
    )
    assert derive.PARENT_DERIVATION_PROTOCOL == (
        "tsc-v157b-runtime-refit-freeze-snapshot-derivation-v2"
    )
    assert derive.DERIVATION_PROTOCOL == (
        "tsc-v157c-jinan-native-one-action-snapshot-derivation-v3"
    )
    assert derive.OVERLAYS == (
        Path("cf_h2o/eval/traffic_signal_v157c_jinan_native_action_branches.py"),
        Path("cf_h2o/config/traffic_signal_tsc_v157c_jinan_native_action_branches.json"),
        Path("scripts/cluster/launch_tsc_v157c_jinan_native_action_branches.py"),
        derive.EVALUATOR_PATCH_RELATIVE,
        derive.V2_EVALUATOR_RELATIVE,
        derive.OCCUPANCY_EQUATIONS_RELATIVE,
        derive.SAFE_PHASE_CONTROLLER_RELATIVE,
    )
    assert derive.PARENT_ABSENT_OVERLAYS == derive.OVERLAYS[:4] + (
        derive.OCCUPANCY_EQUATIONS_RELATIVE,
    )

    runtime = derive.runtime_dependency_contract()
    assert runtime["protocol"] == derive.RUNTIME_DEPENDENCY_PROTOCOL
    assert runtime["patched_evaluator"] == {
        "path": derive.EVALUATOR_RELATIVE.as_posix(),
        "parent_sha256": derive.PARENT_EVALUATOR_SHA256,
        "patch_path": derive.EVALUATOR_PATCH_RELATIVE.as_posix(),
        "patch_sha256": derive.OVERLAY_SHA256[derive.EVALUATOR_PATCH_RELATIVE],
        "sha256": derive.V157C_EVALUATOR_SHA256,
        "semantics": [
            "action_originator_prepare_interval_before_tls_scoring",
            "action_originator_selection_layer_provenance",
            "step_observer_before_step_after_step_before_executor_and_after_executor",
        ],
    }
    assert [row["path"] for row in runtime["replaced_dependencies"]] == [
        derive.V2_EVALUATOR_RELATIVE.as_posix(),
        derive.SAFE_PHASE_CONTROLLER_RELATIVE.as_posix(),
    ]
    assert runtime["added_dependency"]["path"] == (
        derive.OCCUPANCY_EQUATIONS_RELATIVE.as_posix()
    )


def test_v157c_snapshot_overlay_hashes_match_local_execution_files():
    observed = derive.validate_local_overlays()
    assert observed == {
        relative.as_posix(): expected
        for relative, expected in derive.OVERLAY_SHA256.items()
    }
    for relative, expected in derive.OVERLAY_SHA256.items():
        assert hashlib.sha256(
            (derive.PROJECT_ROOT / relative).read_bytes()
        ).hexdigest() == expected


def test_v157c_evaluator_patch_applies_only_required_runtime_hooks(tmp_path: Path):
    parent_blob = subprocess.run(
        ["git", "cat-file", "blob", "98c586b0fecb39c0cfd17a94457d736c6f20862f"],
        cwd=derive.PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    assert hashlib.sha256(parent_blob).hexdigest() == derive.PARENT_EVALUATOR_SHA256

    evaluator = tmp_path / derive.EVALUATOR_RELATIVE
    evaluator.parent.mkdir(parents=True)
    evaluator.write_bytes(parent_blob)
    completed = subprocess.run(
        [
            "patch",
            "--batch",
            "--forward",
            "-d",
            str(tmp_path),
            "-p1",
            "-i",
            str((derive.PROJECT_ROOT / derive.EVALUATOR_PATCH_RELATIVE).resolve()),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert hashlib.sha256(evaluator.read_bytes()).hexdigest() == (
        derive.V157C_EVALUATOR_SHA256
    )
    source = evaluator.read_text(encoding="utf-8")
    compile(source, str(evaluator), "exec")
    assert "step_observer: Callable[..., None] | None = None" in source
    assert source.count("step_observer(stage=") == 3
    assert "prepare_interval = getattr(" in source
    assert "action_originator,\n                    \"selection_layer\"" in source
    assert "selected_action_group_ids" not in source
    assert "OCCUPANCY_EQUATION_PROTOCOL" not in source
