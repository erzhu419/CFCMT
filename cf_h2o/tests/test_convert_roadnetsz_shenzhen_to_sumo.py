from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.data import convert_roadnetsz_shenzhen_to_sumo as converter


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_acquisition(root: Path, *, nonexplicit: bool = False) -> None:
    roadnet = root / converter.ROADNET_RELATIVE
    roadnet.parent.mkdir(parents=True, exist_ok=True)
    roadnet.write_text('{"intersections": [], "roads": []}\n', encoding="utf-8")
    files = {}
    for relative in [
        converter.ROADNET_RELATIVE,
        *(relative for _, relative in converter.VARIANTS),
    ]:
        path = root / relative
        if relative != converter.ROADNET_RELATIVE:
            path.write_text(
                json.dumps(
                    [
                        {
                            "startTime": 0,
                            "endTime": 1 if nonexplicit else 0,
                            "interval": 1.0,
                            "route": ["edge"],
                            "vehicle": {},
                        }
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest = {
        "protocol": converter.ACQUISITION_PROTOCOL,
        "commit": converter.COMMIT,
        "files": files,
        "candidate_networks": {
            converter.SOURCE_NETWORK: {
                "roadnet": converter.ROADNET_RELATIVE,
                "flows": [relative for _, relative in converter.VARIANTS],
            }
        },
    }
    (root / "acquisition_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_shenzhen_conversion_freezes_three_pre_efficacy_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    acquisition = tmp_path / "acquisition"
    _write_acquisition(acquisition)
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("runtime\n", encoding="utf-8")

    def fake_convert_cityflow(**kwargs):
        assert kwargs["yield_to_netconvert_priority"] is True
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir()
        sumocfg = output_dir / f"{kwargs['prefix']}.sumocfg"
        sumocfg.write_text("<configuration/>\n", encoding="utf-8")
        return {
            "city": kwargs["city"],
            "network": kwargs["network"],
            "vehicle_count": 1,
            "sumocfg": f"{kwargs['network']}/{sumocfg.name}",
        }

    monkeypatch.setattr(converter, "convert_cityflow", fake_convert_cityflow)
    output = tmp_path / "converted"
    payload = converter.convert_shenzhen(
        acquisition_root=acquisition,
        output_root=output,
        netconvert=netconvert,
    )

    assert payload["protocol"] == converter.PROTOCOL
    assert payload["city_groups"] == {
        "shenzhen": [scenario for scenario, _ in converter.VARIANTS]
    }
    assert set(payload["networks"]) == {
        scenario for scenario, _ in converter.VARIANTS
    }
    assert payload["selection"]["source_network"] == "fuhua_metavim"
    assert "no controller outcomes inspected" in payload["selection"]["rule"]
    assert all(
        row["source_flow_audit"]["all_rows_encode_exactly_one_vehicle"]
        for row in payload["networks"].values()
    )
    assert json.loads((output / "conversion_manifest.json").read_text()) == payload


def test_shenzhen_conversion_rejects_nonexplicit_flow_rows(tmp_path: Path) -> None:
    acquisition = tmp_path / "acquisition"
    _write_acquisition(acquisition, nonexplicit=True)
    netconvert = tmp_path / "netconvert"
    netconvert.write_text("runtime\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explicit one-vehicle rows"):
        converter.convert_shenzhen(
            acquisition_root=acquisition,
            output_root=tmp_path / "converted",
            netconvert=netconvert,
        )
