from pathlib import Path

from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
)
from scripts.cluster.launch_tsc_external_city_baseline_freeze import (
    CITY_NODES,
    build_specs,
)


def test_external_baseline_freeze_specs_cover_both_cities(tmp_path: Path) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        source_tree_sha256="a" * 64,
        source_cache_root=Path("/remote/source-cache"),
        source_baseline_result=Path("/remote/baseline.json"),
        development_audit=Path("/remote/v41-audit.json"),
        external_cache_root=Path("/remote/external-cache"),
        conversion_root=Path("/remote/conversion"),
        expected_external_cache_sha256="b" * 64,
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        workers=24,
        fold_workers=5,
        cpu_cores=24,
        ram_mb=65536,
    )

    assert len(specs) == 2
    assert {spec["require_node"] for spec in specs} == set(CITY_NODES.values())
    assert len(BENCHMARK_FAMILIES) == 6
    for spec in specs:
        assert "traffic_signal_external_city_baseline_freeze" in spec["cmd"]
        assert "--expected-external-cache-sha256" in spec["cmd"]
        assert "--fold-workers 5" in spec["cmd"]
        assert "baseline_model_ensemble.pkl" in spec["cmd"]
        assert spec["extra_env"]["OPENBLAS_NUM_THREADS"] == "1"
