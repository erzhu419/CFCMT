import json
from pathlib import Path

import pytest

from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)


def _minimal_sumo(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "mini.net.xml").write_text(
        '<net><tlLogic id="tls" type="static" programID="0" offset="0">'
        '<phase duration="10" state="G"/></tlLogic></net>\n',
        encoding="utf-8",
    )
    (root / "mini.rou.xml").write_text(
        '<routes><vehicle id="v0" depart="0"/></routes>\n',
        encoding="utf-8",
    )
    (root / "mini.sumocfg").write_text(
        '<configuration><input><net-file value="mini.net.xml"/>'
        '<route-files value="mini.rou.xml"/></input></configuration>\n',
        encoding="utf-8",
    )


def test_manifest_expands_an_explicit_sumo_root_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sumo_root = tmp_path / "sumo"
    _minimal_sumo(sumo_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "protocol": "environment-root-test-v1",
                "scenarios": [
                    {
                        "scenario": "mini",
                        "city_group": "test_city",
                        "suite": "test",
                        "sumocfg": "${TEST_SUMO_ROOT}/mini.sumocfg",
                        "provenance": "test fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_SUMO_ROOT", str(sumo_root))

    manifest = load_traffic_signal_manifest(manifest_path)

    assert manifest.sumocfgs["mini"] == (sumo_root / "mini.sumocfg").resolve()
    assert manifest.scenarios[0].tls_count == 1
    assert manifest.scenarios[0].demand_count == 1


def test_manifest_rejects_unresolved_sumo_root_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("MISSING_SUMO_ROOT", raising=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 2,
                "protocol": "environment-root-test-v1",
                "scenarios": [
                    {
                        "scenario": "mini",
                        "city_group": "test_city",
                        "suite": "test",
                        "sumocfg": "${MISSING_SUMO_ROOT}/mini.sumocfg",
                        "provenance": "test fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unresolved environment variable"):
        load_traffic_signal_manifest(manifest_path)
