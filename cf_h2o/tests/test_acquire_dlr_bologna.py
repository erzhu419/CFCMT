from scripts.data.acquire_dlr_bologna import COMMIT, FILES, GIT_BLOBS, SCENARIOS


def test_dlr_bologna_acquisition_covers_the_frozen_repository_tree() -> None:
    assert len(COMMIT) == 40
    assert SCENARIOS == ("acosta", "acosta_persontrips", "joined", "pasubio")
    assert len(FILES) == 39
    assert set(GIT_BLOBS) == set(FILES)
    assert all(len(value) == 40 for value in GIT_BLOBS.values())
    assert sum(FILES.values()) == 11_583_748
    assert {path.split("/")[1] for path in FILES} == set(SCENARIOS)
    assert {
        path for path in FILES if path.endswith("/run.sumocfg")
    } == {f"bologna/{scenario}/run.sumocfg" for scenario in SCENARIOS}
