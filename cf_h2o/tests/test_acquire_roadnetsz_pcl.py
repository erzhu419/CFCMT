from scripts.data.acquire_roadnetsz_pcl import COMMIT, FILES


def test_roadnetsz_pcl_acquisition_contract_is_small_and_commit_frozen() -> None:
    assert len(COMMIT) == 40
    assert len(FILES) == 8
    assert sum(FILES.values()) == 5_225_179
    assert set(FILES) == {
        "data_sumo/pcl.con.xml",
        "data_sumo/pcl.edg.xml",
        "data_sumo/pcl.net.xml",
        "data_sumo/pcl.nod.xml",
        "data_sumo/pcl.sumocfg",
        "data_sumo/pcl.tll.xml",
        "data_sumo/pcl.trips.xml",
        "data_sumo/pcl.typ.xml",
    }
