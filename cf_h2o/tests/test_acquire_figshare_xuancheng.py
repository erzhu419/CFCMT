from scripts.data import acquire_figshare_xuancheng as acquisition


def _metadata() -> dict:
    files = [
        {
            "id": 100 + day,
            "name": f"data_2023_04_{day:02d}_type_filtered.json",
            "size": 1_000 + day,
            "computed_md5": f"{day:032x}",
            "supplied_md5": f"{day:032x}",
            "download_url": f"https://example.test/{day}",
        }
        for day in range(1, 31)
    ]
    for index, name in enumerate(acquisition.STATIC_FILES):
        files.append(
            {
                "id": 200 + index,
                "name": name,
                "size": 2_000 + index,
                "computed_md5": f"{100 + index:032x}",
                "supplied_md5": f"{100 + index:032x}",
                "download_url": f"https://example.test/{name}",
            }
        )
    return {
        "id": acquisition.ARTICLE_ID,
        "version": acquisition.ARTICLE_VERSION,
        "files": files,
    }


def test_candidate_selection_is_fixed_before_safety_outcome() -> None:
    selected = acquisition.select_files(_metadata(), coverage="candidate_screen")

    assert [row["name"] for row in selected] == [
        *acquisition.STATIC_FILES,
        acquisition.CANDIDATE_DAY,
    ]


def test_full_month_requires_all_thirty_days() -> None:
    selected = acquisition.select_files(_metadata(), coverage="full_month")

    assert len(selected) == 33
    assert selected[3]["name"] == "data_2023_04_01_type_filtered.json"
    assert selected[-1]["name"] == "data_2023_04_30_type_filtered.json"


def test_full_month_rejects_missing_day() -> None:
    metadata = _metadata()
    metadata["files"] = [
        row
        for row in metadata["files"]
        if row["name"] != "data_2023_04_17_type_filtered.json"
    ]

    try:
        acquisition.select_files(metadata, coverage="full_month")
    except ValueError as exc:
        assert "daily coverage changed" in str(exc)
    else:
        raise AssertionError("incomplete month was accepted")


def test_selection_rejects_source_md5_disagreement() -> None:
    metadata = _metadata()
    candidate = next(
        row
        for row in metadata["files"]
        if row["name"] == acquisition.CANDIDATE_DAY
    )
    candidate["supplied_md5"] = "0" * 32

    try:
        acquisition.select_files(metadata, coverage="candidate_screen")
    except ValueError as exc:
        assert "MD5 metadata disagrees" in str(exc)
    else:
        raise AssertionError("disagreeing Figshare MD5 metadata was accepted")
