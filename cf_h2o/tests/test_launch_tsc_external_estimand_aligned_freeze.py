from pathlib import Path

from scripts.cluster.launch_tsc_external_estimand_aligned_freeze import build_specs


def test_v9_aligned_freeze_specs_bind_repaired_inputs(tmp_path: Path) -> None:
    parent_audit = tmp_path / "joint.json"
    parent_audit.write_text("{}\n", encoding="utf-8")
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        source_tree_sha256="a" * 64,
        source_cache_root=Path("/remote/source-cache"),
        source_baseline_result=Path("/remote/source-baseline.json"),
        development_audit=Path("/remote/development.json"),
        diagnostic_failure_result=Path("/remote/diagnostic.json"),
        parent_joint_audit_local=parent_audit,
        parent_joint_audit_remote=Path("/remote/joint.json"),
        external_cache_root=Path("/remote/external-cache-v9"),
        conversion_root=Path("/remote/conversion-v9"),
        expected_external_cache_sha256="b" * 64,
        remote_results_root=Path("/remote/results-v9"),
        local_results_root=tmp_path / "results-v9",
        workers=24,
        fit_workers=3,
        cpu_cores=32,
        ram_mb=65536,
        generation="v9",
        failed_v52_audit=Path("/remote/v52.json"),
        network_admission_audit=Path("/remote/admission.json"),
        counterfactual_cache_audit=Path("/remote/cache-audit.json"),
    )

    assert len(specs) == 2
    assert all("v55r51" in spec["signature"] for spec in specs)
    assert all(
        "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        in spec["cmd"]
        for spec in specs
    )
    assert all("--failed-v52-audit /remote/v52.json" in spec["cmd"] for spec in specs)
    assert all(spec["cpu"] == 32 for spec in specs)
