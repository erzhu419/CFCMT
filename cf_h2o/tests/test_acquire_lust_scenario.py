from scripts.data import acquire_lust_scenario as acquisition


def test_lust_due_static_inventory_is_complete() -> None:
    expected_demand = {
        "scenario/buslines.rou.xml",
        "scenario/DUERoutes/local.static.0.rou.xml",
        "scenario/DUERoutes/local.static.1.rou.xml",
        "scenario/DUERoutes/local.static.2.rou.xml",
        "scenario/transit.rou.xml",
    }

    assert set(acquisition.DEMAND_FILES) == expected_demand
    assert expected_demand.issubset(acquisition.FILES)
    assert "scenario/lust.net.xml" in acquisition.FILES
    assert "scenario/tll.static.xml" in acquisition.FILES
    assert sum(size for size, _ in acquisition.FILES.values()) == 179_381_797


def test_lust_source_identity_is_fixed() -> None:
    assert acquisition.COMMIT == "5edb7ecb9ad196c39172b6eb95d19ed789f4b1a6"
    assert acquisition.PROTOCOL.endswith("full-acquisition-v1")
    assert all(len(blob) == 40 for _, blob in acquisition.FILES.values())
