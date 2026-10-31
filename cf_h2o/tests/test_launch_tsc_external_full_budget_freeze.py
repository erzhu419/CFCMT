from pathlib import Path

from scripts.cluster.launch_tsc_external_full_budget_freeze import (
    CITY_NODES,
    build_specs,
)


def test_external_full_budget_specs_jointly_freeze_method_and_baselines(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        source_tree_sha256="a" * 64,
        source_cache_root=Path("/remote/source-cache"),
        source_baseline_result=Path("/remote/baseline.json"),
        development_audit=Path("/remote/v41-audit.json"),
        diagnostic_failure_result=Path("/remote/v42-failure.json"),
        external_cache_root=Path("/remote/external-cache"),
        conversion_root=Path("/remote/conversion"),
        expected_external_cache_sha256="b" * 64,
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        workers=24,
        fit_workers=3,
        cpu_cores=8,
        ram_mb=65536,
    )

    assert len(specs) == 2
    assert {spec["require_node"] for spec in specs} == set(CITY_NODES.values())
    for spec in specs:
        assert "traffic_signal_external_full_budget_freeze" in spec["cmd"]
        assert "--fit-workers 3" in spec["cmd"]
        assert "--diagnostic-failure-result /remote/v42-failure.json" in spec["cmd"]
        assert "method_model.pkl" in spec["cmd"]
        assert "baseline_model.pkl" in spec["cmd"]
        assert "method_freeze.json" in spec["cmd"]
        assert "baseline_freeze.json" in spec["cmd"]
        assert spec["cpu"] == 8
        assert spec["extra_env"]["OPENBLAS_NUM_THREADS"] == "1"


def test_v9_full_budget_specs_bind_repair_evidence_and_manifest(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot-v9"),
        source_tree_sha256="c" * 64,
        source_cache_root=Path("/remote/source-cache"),
        source_baseline_result=Path("/remote/baseline.json"),
        development_audit=Path("/remote/v41-audit.json"),
        diagnostic_failure_result=Path("/remote/v42-failure.json"),
        external_cache_root=Path("/remote/external-cache-v9"),
        conversion_root=Path("/remote/conversion-v9"),
        expected_external_cache_sha256="d" * 64,
        remote_results_root=Path("/remote/results-v9"),
        local_results_root=tmp_path / "results-v9",
        workers=24,
        fit_workers=3,
        cpu_cores=8,
        ram_mb=65536,
        generation="v9",
        failed_v52_audit=Path("/remote/v52-audit.json"),
        network_admission_audit=Path("/remote/admission-v9.json"),
        counterfactual_cache_audit=Path("/remote/cache-v9.json"),
    )

    assert all("v54r50" in spec["signature"] for spec in specs)
    assert all(
        "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        in spec["cmd"]
        for spec in specs
    )
    assert all("--failed-v52-audit /remote/v52-audit.json" in spec["cmd"] for spec in specs)
    assert all(
        "--counterfactual-cache-audit /remote/cache-v9.json" in spec["cmd"]
        for spec in specs
    )
