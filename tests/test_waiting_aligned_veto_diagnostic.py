from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cf_h2o.traffic_signal.benchmark_manifest import (
    TrafficSignalBenchmarkManifest,
    TrafficSignalScenarioSpec,
)


def test_filtered_manifest_keeps_only_the_declared_scenario() -> None:
    specs = tuple(
        TrafficSignalScenarioSpec(
            scenario=name,
            city_group="jinan",
            suite="test",
            sumocfg=Path(f"/{name}.sumocfg"),
            provenance="test",
            tls_count=1,
            demand_count=1,
        )
        for name in ("jinan_3x4_real", "jinan_3x4_real_2000")
    )
    manifest = TrafficSignalBenchmarkManifest(
        path=Path("/manifest.json"), version=2, scenarios=specs, protocol="test"
    )
    selected = replace(
        manifest,
        scenarios=tuple(
            item for item in manifest.scenarios if item.scenario == "jinan_3x4_real"
        ),
    )
    assert list(selected.sumocfgs) == ["jinan_3x4_real"]
