from __future__ import annotations

from dataclasses import asdict
from pathlib import PurePosixPath

import pytest

from scripts.data.acquire_toronto_ptc_unseen import (
    FRAMEWORK_COMMIT,
    FRAMEWORK_NAME,
    FRAMEWORK_URL,
    RESOURCES,
    _manifest,
    _receive_command,
    _validate_resource_metadata,
)


def test_frozen_resource_inventory_is_complete_and_unique() -> None:
    keys = [resource.key for resource in RESOURCES]
    resource_ids = [resource.resource_id for resource in RESOURCES]
    output_names = [resource.output_name for resource in RESOURCES]

    assert len(RESOURCES) == 11
    assert len(keys) == len(set(keys))
    assert len(resource_ids) == len(set(resource_ids))
    assert len(output_names) == len(set(output_names))
    assert sum(resource.size_bytes for resource in RESOURCES) == 279_478_393
    assert FRAMEWORK_COMMIT in FRAMEWORK_URL
    assert FRAMEWORK_COMMIT in FRAMEWORK_NAME


def test_metadata_validation_is_fail_closed() -> None:
    resource = RESOURCES[0]
    observed = {
        "id": resource.resource_id,
        "name": resource.source_name,
        "format": resource.format,
        "size": resource.size_bytes,
        "last_modified": resource.last_modified,
        "url": resource.url,
    }

    _validate_resource_metadata(resource, observed)
    observed["size"] = resource.size_bytes + 1
    with pytest.raises(ValueError, match="source metadata changed"):
        _validate_resource_metadata(resource, observed)


def test_receive_command_checks_remote_size_and_digest() -> None:
    command = _receive_command(
        staging=PurePosixPath("/remote/staging"),
        name="trips_2025.zip",
        expected_size=123,
    )

    assert "cat > /remote/staging/.trips_2025.zip.part" in command
    assert "-eq 123" in command
    assert "sha256sum /remote/staging/trips_2025.zip" in command


def test_manifest_records_remote_only_source_identity() -> None:
    observations = {
        resource.key: {
            "size_bytes": resource.size_bytes,
            "sha256": f"sha-{index}",
            "ckan_created": "created",
        }
        for index, resource in enumerate(RESOURCES)
    }
    payload = _manifest(
        observations=observations,
        framework={"size_bytes": 42, "sha256": "framework-sha"},
        framework_entry_count=17,
    )

    assert payload["resource_count"] == len(RESOURCES)
    assert payload["total_frozen_resource_bytes"] == sum(
        resource.size_bytes for resource in RESOURCES
    )
    assert payload["coverage"] == "all frozen Toronto v105 source resources"
    assert payload["transfer_mode"].endswith("no_local_raw_file")
    assert payload["framework"]["commit"] == FRAMEWORK_COMMIT
    assert payload["framework"]["entry_count"] == 17
    assert payload["resources"][RESOURCES[0].key]["resource_id"] == (
        asdict(RESOURCES[0])["resource_id"]
    )
