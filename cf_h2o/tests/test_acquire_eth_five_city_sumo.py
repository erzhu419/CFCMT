from __future__ import annotations

from pathlib import PurePosixPath

from scripts.data.acquire_eth_five_city_sumo import (
    ARCHIVE_NAME,
    BITSTREAM_URL,
    EXPECTED_MD5,
    EXPECTED_SIZE_BYTES,
    USER_AGENT,
    _manifest,
    _remote_receive_command,
)


def test_remote_receive_command_checks_size_and_lists_archive() -> None:
    command = _remote_receive_command(PurePosixPath("/remote/staging"))

    assert str(EXPECTED_SIZE_BYTES) in command
    assert f"/remote/staging/{ARCHIVE_NAME}" in command
    assert "cat >" in command
    assert "unzip -Z1" in command


def test_manifest_records_complete_remote_only_acquisition() -> None:
    payload = _manifest(archive_entry_count=30)

    assert payload["archive_entry_count"] == 30
    assert payload["coverage"] == "complete published five-city archive"
    assert payload["transfer_mode"] == (
        "https_stream_through_local_to_shared_cluster_no_local_raw_file"
    )
    assert payload["size_bytes"] == EXPECTED_SIZE_BYTES
    assert payload["md5"] == EXPECTED_MD5
    assert payload["bitstream_url"] == BITSTREAM_URL
    assert USER_AGENT.startswith("Mozilla/5.0 ")
