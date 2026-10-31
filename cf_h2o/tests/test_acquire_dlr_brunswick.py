import pytest

from scripts.data import acquire_dlr_brunswick as acquisition


def _tree_payload() -> dict:
    return {
        "sha": acquisition.SUBTREE_SHA,
        "truncated": False,
        "tree": [
            {
                "path": "miv/input.xml",
                "type": "blob",
                "size": 11,
                "sha": "a" * 40,
            },
            {
                "path": "osm/source.xml",
                "type": "blob",
                "size": 13,
                "sha": "b" * 40,
            },
            {
                "path": "miv",
                "type": "tree",
                "sha": "c" * 40,
            },
        ],
    }


def test_brunswick_acquisition_freezes_complete_official_subtree() -> None:
    assert len(acquisition.COMMIT) == 40
    assert acquisition.SUBTREE == "brunswick"
    assert acquisition.SUBTREE_SHA == "04bc93e619e60faa95dc5b18432fb07a5434634c"
    assert acquisition.EXPECTED_FILE_COUNT == 66
    assert acquisition.EXPECTED_TOTAL_SIZE_BYTES == 234_720_146


def test_validate_tree_rejects_identity_and_path_drift(monkeypatch) -> None:
    monkeypatch.setattr(acquisition, "EXPECTED_FILE_COUNT", 2)
    monkeypatch.setattr(acquisition, "EXPECTED_TOTAL_SIZE_BYTES", 24)
    assert [
        row["path"] for row in acquisition._validate_tree(_tree_payload())
    ] == ["miv/input.xml", "osm/source.xml"]

    wrong_tree = _tree_payload()
    wrong_tree["sha"] = "d" * 40
    with pytest.raises(ValueError, match="tree identity"):
        acquisition._validate_tree(wrong_tree)

    unsafe_path = _tree_payload()
    unsafe_path["tree"][0]["path"] = "../escape.xml"
    with pytest.raises(ValueError, match="invalid Brunswick source path"):
        acquisition._validate_tree(unsafe_path)
