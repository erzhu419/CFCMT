from __future__ import annotations

from scripts.data.acquire_roadnetsz_shenzhen import COMMIT, FILES


def test_roadnetsz_acquisition_contract_is_small_and_commit_frozen() -> None:
    assert len(COMMIT) == 40
    assert len(FILES) == 7
    assert sum(FILES.values()) == 9_441_058
    assert all(path.startswith("data_cityflow/") for path in FILES)
