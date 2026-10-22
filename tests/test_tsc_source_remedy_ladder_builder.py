from paper_tsc.figures.build_tsc_source_remedy_ladder import (
    V158_RESULT,
    V159_RESULT,
    build_rows,
    render_table,
)


def test_source_remedy_ladder_preserves_negative_adjudications() -> None:
    assert V158_RESULT.parent.name == "paper_artifacts"
    assert V158_RESULT.name == "tsc_v158_native_prefix_ranking_calibration_v1.json"
    assert V159_RESULT.parent.name == "paper_artifacts"
    assert V159_RESULT.name == "tsc_v159_native_endpoint_ablation_v1.json"
    rows = build_rows()
    assert len(rows) == 13
    assert rows[0]["source_admission"] == "Oracle only"
    assert rows[4]["adjudication"] == "Historical; reject after closed-loop smoke"
    assert rows[5]["source_admission"] == "1/7 inner; 0 follow-up"
    assert rows[5]["adjudication"] == "Reject globally and city-specifically"
    assert rows[-5]["source_admission"] == "0/7"
    assert rows[-5]["adjudication"] == "Reject; exact rigid fallback"
    assert rows[-4]["source_admission"] == "One-action branch only"
    assert rows[-4]["adjudication"] == "B100 gate pass; V157C incomplete"
    assert rows[-3]["source_admission"] == "2/3 valid seeds"
    assert rows[-3]["adjudication"] == "Incomplete; no three-seed inference"
    assert rows[-2]["stage"] == "Native-prefix ranking calibration"
    assert rows[-2]["source_admission"] == "OOF 0/3; no reserve"
    assert rows[-2]["adjudication"] == "Reject; OOF gate failed; reserve withheld"
    assert rows[-1]["stage"] == "Native endpoint ablation"
    assert rows[-1]["source_admission"] == "Target only; no source"
    assert rows[-1]["adjudication"] == "Reject; direct-endpoint hypothesis closed"
    table = render_table(rows)
    assert "Cologne +37.9\\%" in table
    assert "macro +0.00135 vs rigid" in table
    assert "forced mean -0.01213" in table
    assert "V123/V157A confounded; reused 22-seed Jinan selector" in table
    assert "source-target -0.01357" in table
    assert "95\\% [-0.01784, -0.00935]" in table
    assert "20/22 improve" in table
    assert "+0.02534 vs PhasePressure" in table
    assert "seed 27178 checkpoint at 480 s invalid" in table
    assert "source-target +0.00236 (1/2 improve)" in table
    assert "source-placebo -0.00232" in table
    assert "target-PhasePressure +0.00125" in table
    assert "source-PhasePressure +0.00362 (0/2 improve)" in table
    assert "collisions 0" in table
    assert "PhasePressure is the reference heuristic" in table
    assert "source-target -0.00113, 95\\% [-0.00240, +0.00019] (6/10 improve)" in table
    assert "source-placebo -0.00072, 95\\% [-0.00219, +0.00091] (7/10 improve)" in table
    assert "source-PhasePressure +0.00014, 95\\% [-0.00239, +0.00261] (5/10 improve)" in table
    assert "100 groups/800 rows/700 branches; 0 restores" in table
    assert "Reject; OOF gate failed; reserve withheld" in table
    assert r"endpoint-delta -0.00061, 95\% [-0.00224, +0.00081] (5/10 improve)" in table
    assert "base/2000/2500 -0.00201/+0.00157/-0.00091" in table
    assert "endpoint-PhasePressure +0.00065 (3/10 improve)" in table
    assert "placebo pass" not in table.lower()
    assert "closed-loop pass" not in table.lower()
    assert "branch not run" not in table.lower()
    assert "v153a" not in table
