from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.package_github_intas_sumo import (
    CAR_ROUTE_NAMES,
    END_SEC,
    ROUTE_NAMES,
    SEED,
    STEP_LENGTH_SEC,
    WORKERS,
    _flow_count,
    _write_strict_config,
)


def test_complete_route_inventory_is_frozen() -> None:
    assert len(CAR_ROUTE_NAMES) == 22
    assert len(ROUTE_NAMES) == 24
    assert CAR_ROUTE_NAMES[0] == "routes/InTAS_001.rou.xml"
    assert CAR_ROUTE_NAMES[-1] == "routes/InTAS_022.rou.xml"
    assert WORKERS == 8


def test_flow_expansion_matches_sumo_period_semantics() -> None:
    assert _flow_count({"begin": "17640", "end": "69840", "period": "900"}) == 58
    assert _flow_count({"begin": "17940", "end": "48540", "period": "30600"}) == 1
    assert _flow_count({"begin": "0", "end": "10", "number": "3"}) == 3
    with pytest.raises(ValueError, match="lacks"):
        _flow_count({"begin": "0", "end": "10"})


def test_strict_config_preserves_full_demand_and_disables_loss(tmp_path: Path) -> None:
    path = tmp_path / "strict.sumocfg"
    _write_strict_config(path)
    root = ET.parse(path).getroot()

    assert root.find("./input/net-file").attrib["value"] == (
        "source/ingolstadt.net.xml"
    )
    route_files = root.find("./input/route-files").attrib["value"].split(",")
    assert route_files == [f"source/{name}" for name in ROUTE_NAMES]
    assert root.find("./time/step-length").attrib["value"] == str(STEP_LENGTH_SEC)
    assert root.find("./time/end").attrib["value"] == str(END_SEC)
    assert root.find("./processing/max-depart-delay").attrib["value"] == "-1"
    assert root.find("./processing/time-to-teleport").attrib["value"] == "-1"
    assert root.find("./processing/scale").attrib["value"] == "1"
    assert root.find("./processing/ignore-route-errors").attrib["value"] == "false"
    assert root.find("./random_number/seed").attrib["value"] == str(SEED)
