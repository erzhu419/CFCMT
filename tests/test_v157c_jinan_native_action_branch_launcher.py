from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex

import pytest

from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze import (
    ARM_MANIFEST_PROTOCOL,
    ARM_ORDER,
    RESULT_PROTOCOL as V157B_RESULT_PROTOCOL,
    RUNTIME_BUNDLE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import SEEDS
from scripts.cluster import (
    derive_tsc_v157c_jinan_native_action_branches_snapshot as derive_snapshot,
)
from scripts.cluster import launch_tsc_v157c_jinan_native_action_branches as launcher


def test_build_specs_freezes_three_seed_shards_and_v157b_inputs():
    specs = launcher.build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        v157a_authorization_remote=Path("/evidence/v157a.json"),
        runtime_models_remote=Path("/v157b/runtime_models.pkl"),
        runtime_models_sha256="runtime-sha",
        arm_manifest_remote=Path("/v157b/arm_manifest.json"),
        arm_manifest_sha256="manifest-sha",
        remote_output_root=Path("/results/v157c"),
        remote_scratch_root=Path("/scratch/v157c"),
        nodes=("node003", "node004", "node005"),
    )
    assert len(specs) == 3
    assert [spec["require_node"] for spec in specs] == [
        "node003",
        "node004",
        "node005",
    ]
    assert [spec["signature"] for spec in specs] == [
        f"{launcher.SIGNATURE_PREFIX}/seed{seed}" for seed in SEEDS
    ]
    for seed, spec in zip(SEEDS, specs, strict=True):
        tokens = shlex.split(spec["cmd"].removesuffix(" && printf 'TASK_DONE\\n'"))
        assert tokens[tokens.index("--seed") + 1] == str(seed)
        assert tokens[tokens.index("--runtime-bundle") + 1] == "/v157b/runtime_models.pkl"
        assert tokens[tokens.index("--runtime-bundle-sha256") + 1] == "runtime-sha"
        assert tokens[tokens.index("--arm-manifest") + 1] == "/v157b/arm_manifest.json"
        assert tokens[tokens.index("--arm-manifest-sha256") + 1] == "manifest-sha"
        assert tokens[tokens.index("--authorization") + 1] == "/evidence/v157a.json"
        assert tokens[tokens.index("--output") + 1] == f"/results/v157c/seed_{seed}"
        assert tokens[tokens.index("--scratch-root") + 1] == f"/scratch/v157c/seed_{seed}"
        assert spec["cpu"] == 2 and spec["ram_mb"] == 32768


def test_build_specs_rejects_nonunique_or_incomplete_node_roster():
    kwargs = dict(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        v157a_authorization_remote=Path("/evidence/v157a.json"),
        runtime_models_remote=Path("/v157b/runtime_models.pkl"),
        runtime_models_sha256="runtime-sha",
        arm_manifest_remote=Path("/v157b/arm_manifest.json"),
        arm_manifest_sha256="manifest-sha",
        remote_output_root=Path("/results/v157c"),
        remote_scratch_root=Path("/scratch/v157c"),
    )
    with pytest.raises(ValueError, match="three distinct"):
        launcher.build_specs(**kwargs, nodes=("node003", "node003", "node005"))
    with pytest.raises(ValueError, match="three distinct"):
        launcher.build_specs(**kwargs, nodes=("node003", "node004"))


def test_v157b_summary_must_bind_both_runtime_artifacts():
    runtime = Path("/v157b/runtime_models.pkl")
    manifest = Path("/v157b/arm_manifest.json")
    summary = {
        "protocol": V157B_RESULT_PROTOCOL,
        "scientific_status": "runtime_arms_frozen_branch_not_run",
        "city": "jinan",
        "arm_order": list(ARM_ORDER),
        "target_action_groups": 100,
        "v157a_exact_refit_identity_passed": True,
        "domain_aligned_source_gate_passed": True,
        "placebo_capacity_passed": True,
        "runtime_models": {
            "path": str(runtime),
            "sha256": "runtime-sha",
            "protocol": RUNTIME_BUNDLE_PROTOCOL,
        },
        "arm_manifest": {
            "path": str(manifest),
            "sha256": "manifest-sha",
            "protocol": ARM_MANIFEST_PROTOCOL,
        },
    }
    launcher.validate_v157b_summary(
        summary,
        runtime_models=runtime,
        runtime_models_sha256="runtime-sha",
        arm_manifest=manifest,
        arm_manifest_sha256="manifest-sha",
    )
    summary["domain_aligned_source_gate_passed"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        launcher.validate_v157b_summary(
            summary,
            runtime_models=runtime,
            runtime_models_sha256="runtime-sha",
            arm_manifest=manifest,
            arm_manifest_sha256="manifest-sha",
        )
    summary["domain_aligned_source_gate_passed"] = True
    summary["v157a_exact_refit_identity_passed"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        launcher.validate_v157b_summary(
            summary,
            runtime_models=runtime,
            runtime_models_sha256="runtime-sha",
            arm_manifest=manifest,
            arm_manifest_sha256="manifest-sha",
        )


def test_launcher_binds_v157c_stage_parent_overlays_and_content_address():
    root = Path(__file__).resolve().parents[1]
    digest = "a" * 64
    stage = {
        "protocol": derive_snapshot.DERIVATION_PROTOCOL,
        "snapshot_sha256": digest,
        "source_tree_sha256": "b" * 64,
        "snapshot_root": str(derive_snapshot.REMOTE_BASE / digest[:20]),
        "derived_from_snapshot_root": str(derive_snapshot.PARENT_ROOT),
        "derived_from_snapshot_sha256": derive_snapshot.PARENT_SNAPSHOT_SHA256,
        "derived_from_source_tree_sha256": (
            derive_snapshot.PARENT_SOURCE_TREE_SHA256
        ),
        "overlays": [
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            }
            for relative in derive_snapshot.OVERLAYS
        ],
        "runtime_dependencies": derive_snapshot.runtime_dependency_contract(),
    }
    assert launcher._validate_stage_manifest(stage) == Path(stage["snapshot_root"])
    assert launcher.SNAPSHOT_IDENTITY_ORDER == (
        "stage_marker",
        "runner",
        "config",
        "launcher",
        "evaluator",
        "v2_evaluator",
        "occupancy_equations",
        "safe_phase_controller",
        "evaluator_patch",
        "v157b_module",
        "network_manifest",
    )

    mutations = (
        ("protocol", "wrong"),
        ("derived_from_snapshot_sha256", "0" * 64),
        ("snapshot_root", str(derive_snapshot.REMOTE_BASE / ("c" * 20))),
    )
    for key, value in mutations:
        changed = json.loads(json.dumps(stage))
        changed[key] = value
        with pytest.raises(ValueError, match="derived snapshot manifest"):
            launcher._validate_stage_manifest(changed)

    changed = json.loads(json.dumps(stage))
    changed["overlays"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="derived snapshot manifest"):
        launcher._validate_stage_manifest(changed)

    changed = json.loads(json.dumps(stage))
    changed["overlays"].append(dict(changed["overlays"][0]))
    with pytest.raises(ValueError, match="derived snapshot manifest"):
        launcher._validate_stage_manifest(changed)

    changed = json.loads(json.dumps(stage))
    changed["runtime_dependencies"]["patched_evaluator"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="derived snapshot manifest"):
        launcher._validate_stage_manifest(changed)


def test_remote_hash_probe_requires_the_complete_ordered_identity_vector():
    class FakeScheduler:
        def __init__(self, hashes):
            self.hashes = list(hashes)

        def run_on(self, node, command, *, timeout, check):
            assert node == "node003"
            assert shlex.split(command) == ["sha256sum", "/marker", "/runner"]
            assert timeout == 300 and check is False
            stdout = "\n".join(
                f"{value}  {path}"
                for value, path in zip(
                    self.hashes, ("/marker", "/runner"), strict=False
                )
            )
            return 0, stdout, ""

    paths = (Path("/marker"), Path("/runner"))
    expected = ("a" * 64, "b" * 64)
    assert launcher._validate_remote_hashes(
        FakeScheduler(expected),
        node="node003",
        paths=paths,
        expected=expected,
    ) == list(expected)

    with pytest.raises(RuntimeError, match="remote V157C identities changed"):
        launcher._validate_remote_hashes(
            FakeScheduler(("0" * 64, expected[1])),
            node="node003",
            paths=paths,
            expected=expected,
        )


def test_result_summary_uses_frozen_absolute_python_not_bare_python3():
    tokens = shlex.split(launcher._result_summary_command(Path("/v157b/result.json")))
    assert tokens[0] == str(derive_snapshot.REMOTE_PYTHON)
    assert Path(tokens[0]).is_absolute()
    assert tokens[0] != "python3"


def test_remote_input_validation_uses_snapshot_wrapper_and_real_input_paths():
    snapshot = Path("/snapshots/abc")
    conversion = Path("/conversion")
    command = launcher._input_validation_command(
        snapshot_root=snapshot,
        conversion_root=conversion,
        authorization_path=Path("/evidence/v157a.json"),
        runtime_bundle_path=Path("/v157b/runtime_models.pkl"),
        runtime_bundle_sha256="runtime-sha",
        arm_manifest_path=Path("/v157b/arm_manifest.json"),
        arm_manifest_sha256="manifest-sha",
    )
    tokens = shlex.split(command)
    assert tokens[:4] == [
        "env",
        "CFCMT_SOURCE_ROOT=/snapshots/abc",
        "CFCMT_EXTERNAL_CONVERSION_ROOT=/conversion",
        f"CFCMT_PYTHON={derive_snapshot.REMOTE_PYTHON}",
    ]
    assert tokens[4] == "/snapshots/abc/scripts/cluster/run_cfcmt_sumo122.sh"
    assert tokens[5:8] == ["-m", launcher.MODULE, "validate-inputs"]
    assert tokens[tokens.index("--protocol") + 1] == (
        "/snapshots/abc/cf_h2o/config/"
        "traffic_signal_tsc_v157c_jinan_native_action_branches.json"
    )
    assert tokens[tokens.index("--authorization") + 1] == "/evidence/v157a.json"
    assert tokens[tokens.index("--runtime-bundle") + 1] == (
        "/v157b/runtime_models.pkl"
    )
    assert tokens[tokens.index("--runtime-bundle-sha256") + 1] == "runtime-sha"
    assert tokens[tokens.index("--arm-manifest") + 1] == "/v157b/arm_manifest.json"
    assert tokens[tokens.index("--arm-manifest-sha256") + 1] == "manifest-sha"
    assert "python3" not in tokens[:6]

    class FakeScheduler:
        def __init__(self, code):
            self.code = code
            self.command = None

        def run_on(self, node, remote_command, *, timeout, check):
            assert node == "node003"
            assert timeout == 300 and check is False
            self.command = remote_command
            result = {
                "protocol": launcher.INPUT_VALIDATION_PROTOCOL,
                "status": "PASS",
                "learned_arms": list(launcher.LEARNED_ARMS),
                "sumo_started": False,
                "artifacts_written": False,
            }
            return (
                self.code,
                json.dumps(result) if not self.code else "",
                "invalid input" if self.code else "",
            )

    scheduler = FakeScheduler(0)
    observed = launcher._validate_remote_inputs(
        scheduler,
        node="node003",
        snapshot_root=snapshot,
        conversion_root=conversion,
        authorization_path=Path("/evidence/v157a.json"),
        runtime_bundle_path=Path("/v157b/runtime_models.pkl"),
        runtime_bundle_sha256="runtime-sha",
        arm_manifest_path=Path("/v157b/arm_manifest.json"),
        arm_manifest_sha256="manifest-sha",
    )
    assert scheduler.command == command
    assert observed["status"] == "PASS"

    with pytest.raises(RuntimeError, match="input validation failed"):
        launcher._validate_remote_inputs(
            FakeScheduler(1),
            node="node003",
            snapshot_root=snapshot,
            conversion_root=conversion,
            authorization_path=Path("/evidence/v157a.json"),
            runtime_bundle_path=Path("/v157b/runtime_models.pkl"),
            runtime_bundle_sha256="runtime-sha",
            arm_manifest_path=Path("/v157b/arm_manifest.json"),
            arm_manifest_sha256="manifest-sha",
        )


def test_launcher_and_seed_commands_do_not_use_sumo_state_restore():
    source = Path(launcher.__file__).read_text(encoding="utf-8")
    forbidden = ("simulation." + "saveState", "simulation." + "loadState")
    assert all(value not in source for value in forbidden)
