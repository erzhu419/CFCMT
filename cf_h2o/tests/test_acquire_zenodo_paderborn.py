from __future__ import annotations

from pathlib import PurePosixPath

from scripts.data.acquire_zenodo_paderborn import (
    FILES,
    RECORD_ID,
    _content_url,
    _manifest,
    _receive_command,
)


def test_fixed_record_inventory_is_complete_and_unique() -> None:
    names = [name for name, _, _ in FILES]
    payload = _manifest()

    assert len(FILES) == 11
    assert len(names) == len(set(names))
    assert payload["file_count"] == 11
    assert set(payload["files"]) == set(names)
    assert payload["total_size_bytes"] == sum(size for _, size, _ in FILES)
    assert payload["coverage"] == "complete fixed Zenodo record"


def test_receive_command_checks_size_without_local_raw_file() -> None:
    command = _receive_command(
        staging=PurePosixPath("/remote/staging"),
        name="allroutes.rou.xml",
        expected_size=123,
    )

    assert "cat > /remote/staging/.allroutes.rou.xml.part" in command
    assert "-eq 123" in command
    assert "mv " in command
    assert _content_url("allroutes.rou.xml") == (
        f"https://zenodo.org/api/records/{RECORD_ID}/files/"
        "allroutes.rou.xml/content"
    )
