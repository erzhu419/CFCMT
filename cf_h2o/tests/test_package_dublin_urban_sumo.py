from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.package_dublin_urban_sumo import (
    PACKAGED_ADDITIONAL_FILES,
    _write_config,
)


SOURCE_CONFIG = """<?xml version="1.0"?>
<configuration>
  <input>
    <net-file value="DCC.net.xml"/>
    <additional-files value="DCC_trafficlights.add.xml,vtypes.add.xml,DCC_routes.rou.xml,DCC_emitters.emi.xml,DCC_detectors.poi.xml"/>
  </input>
  <time><begin value="0"/><end value="86400"/><step-length value="0.5"/></time>
  <output><device.rerouting.probability value="1"/></output>
</configuration>
"""


def test_write_config_preserves_source_additional_loading_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sumocfg"
    destination = tmp_path / "packaged.sumocfg"
    source.write_text(SOURCE_CONFIG, encoding="utf-8")

    _write_config(source_config=source, destination=destination)

    root = ET.parse(destination).getroot()
    assert root.find(".//route-files") is None
    additional = root.find(".//additional-files")
    assert additional is not None
    assert tuple(additional.attrib["value"].split(",")) == PACKAGED_ADDITIONAL_FILES
    assert root.find(".//end").attrib["value"] == "86400"


def test_write_config_rejects_source_route_reordering(tmp_path: Path) -> None:
    source = tmp_path / "source.sumocfg"
    destination = tmp_path / "packaged.sumocfg"
    source.write_text(
        SOURCE_CONFIG.replace(
            '<additional-files value="',
            '<route-files value="DCC_routes.rou.xml"/>'
            '<additional-files value="',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source input ordering changed"):
        _write_config(source_config=source, destination=destination)
