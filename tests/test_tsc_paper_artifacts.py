from __future__ import annotations

from copy import deepcopy
import re
from pathlib import Path

import pytest

from paper_tsc.figures.build_tsc_external_confirmation import (
    DEFAULT_EXTERNAL_ROOT,
    DEFAULT_HELDOUT,
    DEFAULT_SOURCE_ABLATION,
    DEFAULT_SOURCE_CONFIRMATION,
    DEFAULT_V98_REMOTE_INTEGRITY,
    DEFAULT_V98_SOURCE_CONFIRMATION,
    jinan_conditional_source_transfer_rows,
    load_json,
    method_freeze_rows,
    offline_rows,
    source_contribution_rows,
    source_contribution_seed_rows,
    validate_jinan_conditional_source_transfer,
    validate_source_contribution,
    validate_v98_remote_integrity,
)
from paper_tsc.figures.build_tsc_postfreeze_stress_test import build_rows


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_tsc"


def test_submission_references_only_canonical_figures() -> None:
    results = (PAPER / "sections/2_results.tex").read_text(encoding="utf-8")
    figures = set(
        re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", results)
    )

    assert figures == {
        "traffic_signal_transfer_networks.pdf",
        "tsc_anchored_transfer_schematic.pdf",
        "tsc_external_confirmation.pdf",
        "tsc_source_contribution_diagnostic.pdf",
    }
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert r"\graphicspath{{figures/}}" in main


def test_current_method_schematic_has_no_pressure_guard() -> None:
    schematic = (
        PAPER / "figures/tsc_anchored_transfer_schematic.svg"
    ).read_text(encoding="utf-8").lower()

    assert "pressure guard" not in schematic
    assert "maxpressure" in schematic
    assert ">pressure</text>" in schematic
    assert "not zero-shot or field causal identification" in schematic


def test_postfreeze_claim_is_rebuilt_from_frozen_audits() -> None:
    rows = build_rows()
    assert len(rows) == 4
    confirmation = rows[-1]

    assert confirmation["stage"] == "v89 frozen legacy"
    assert confirmation["evidence_class"] == "fresh confirmation"
    assert confirmation["seed_count"] == 64
    assert confirmation["mean_relative_delta"] == pytest.approx(
        -0.0004255945, abs=5e-11
    )
    assert confirmation["ci95_low"] == pytest.approx(-0.0044875, abs=5e-8)
    assert confirmation["ci95_high"] == pytest.approx(0.0036962, abs=5e-8)
    assert confirmation["improved_seed_fraction"] == pytest.approx(34 / 64)
    assert confirmation["one_sided_sign_p"] == pytest.approx(
        0.3539903771, abs=5e-11
    )
    assert confirmation["joint_gate_passed"] is False
    assert confirmation["interpretation"] == "confirmation rejected"

    results = (PAPER / "sections/2_results.tex").read_text(encoding="utf-8")
    for claim in (
        r"$-0.043\%$",
        r"$[-0.449,0.370]\%$",
        "34/64 seeds",
        "probability was 0.354",
        r"regressed by 5.21\%",
        "joint confirmation was therefore rejected",
    ):
        assert claim in results


def test_source_contribution_claim_is_rebuilt_from_frozen_audits() -> None:
    heldout = load_json(DEFAULT_HELDOUT)
    v90 = load_json(DEFAULT_SOURCE_ABLATION)
    v91 = load_json(DEFAULT_SOURCE_CONFIRMATION)
    validate_source_contribution(v90, v91)

    offline = offline_rows(heldout, method_freeze_rows(DEFAULT_EXTERNAL_ROOT))
    offline_by_city = {row["city"]: row for row in offline}
    assert offline_by_city["los_angeles"]["target_only_regret"] == pytest.approx(
        0.2171953499, abs=5e-11
    )
    assert offline_by_city["jinan"]["target_only_regret"] == pytest.approx(
        0.4035714143, abs=5e-11
    )
    target_macro = sum(row["target_only_regret"] for row in offline) / 2
    method_macro = sum(row["selected_regret"] for row in offline) / 2
    assert (target_macro - method_macro) / target_macro == pytest.approx(
        0.134806, abs=5e-7
    )

    rows = source_contribution_rows(v90, v91)
    assert len(rows) == 2
    confirmation = rows[-1]
    assert confirmation["stage"] == "v91 fresh"
    assert confirmation["seed_count"] == 64
    assert confirmation["paired_cell_count"] == 256
    assert confirmation["target_only_wait_sec"] == pytest.approx(149.0920788)
    assert confirmation["cfcmt_wait_sec"] == pytest.approx(155.1445622)
    assert confirmation["mean_relative_delta"] == pytest.approx(0.04254473785)
    assert confirmation["los_angeles_relative_delta"] == pytest.approx(
        0.07865341918
    )
    assert confirmation["jinan_relative_delta"] == pytest.approx(-0.03044646977)
    assert confirmation["decision"] == "confirmation rejected"

    seed_rows = source_contribution_seed_rows(v91)
    assert len(seed_rows) == 64
    assert len({row["seed"] for row in seed_rows}) == 64
    assert sum(row["macro_relative_delta"] < 0.0 for row in seed_rows) == 29
    assert sum(row["los_angeles_relative_delta"] < 0.0 for row in seed_rows) == 22
    assert all(row["jinan_relative_delta"] < 0.0 for row in seed_rows)
    assert max(row["macro_relative_delta"] for row in seed_rows) == pytest.approx(
        2.6755648985, abs=5e-10
    )

    results = (PAPER / "sections/2_results.tex").read_text(encoding="utf-8")
    for claim in (
        "0.3104 to 0.2685 (13.48\\%)",
        r"$+4.254\%$",
        r"$[-0.775,13.211]\%$",
        "29/64 seeds",
        "7.865\\%",
        "confirmation was rejected",
    ):
        assert claim in results


def test_jinan_conditional_transfer_rows_keep_v91_and_v98_separate() -> None:
    v91 = load_json(DEFAULT_SOURCE_CONFIRMATION)
    v98 = load_json(DEFAULT_V98_SOURCE_CONFIRMATION)
    integrity = load_json(DEFAULT_V98_REMOTE_INTEGRITY)
    validate_jinan_conditional_source_transfer(v91, v98)
    validate_v98_remote_integrity(v98, integrity)
    assert integrity["observed"]["unique_identity_count"] == 336

    rows = jinan_conditional_source_transfer_rows(v91, v98)
    assert len(rows) == 2
    assert all(row["pooled_with_other_row"] is False for row in rows)

    v91_row, v98_row = rows
    assert v91_row["experiment"] == "V91 Jinan city diagnostic"
    assert (
        v91_row["selection_and_evaluation"]
        == "frozen two-city protocol; Jinan diagnostic"
    )
    assert v91_row["seed_count"] == 64
    assert v91_row["mean_relative_waiting_delta"] == pytest.approx(
        -0.030446469768574826, abs=5e-13
    )
    assert v91_row["improved_seed_count"] == 64
    assert v91_row["ci95_low"] == ""
    assert v91_row["comparator"] == "final V43 legacy near-target-only"

    assert v98_row["seed_count"] == 56
    assert v98_row["mean_relative_waiting_delta"] == pytest.approx(
        -0.045274013085975714, abs=5e-13
    )
    assert v98_row["ci95_low"] == pytest.approx(-0.04786736703256771)
    assert v98_row["ci95_high"] == pytest.approx(-0.04278512539219397)
    assert v98_row["improved_seed_count"] == 56
    assert v98_row["experiment"] == "V98 Jinan controller-pair confirmation"
    assert "strict zero-source-row lower-capacity" in v98_row["comparator"]
    assert "post-V91 shared inherited" in v98_row["guard"]
    assert "controller-pair contrast" in v98_row["comparison_scope"]
    assert "source-row attribution" in v98_row["comparison_scope"]

    source_csv = (
        PAPER / "source_data/jinan_conditional_source_transfer.csv"
    ).read_text(encoding="utf-8")
    assert "final V43 legacy near-target-only" in source_csv
    assert "strict zero-source-row lower-capacity target model" in source_csv
    assert "controller-pair contrast; lower-capacity comparator" in source_csv
    assert source_csv.count(",False") == 2

    table = (PAPER / "tables/jinan_conditional_source_transfer.tex").read_text(
        encoding="utf-8"
    )
    assert "64/64 & -3.045\\%" in table
    assert "56/56 & -4.527\\% & [-4.787, -4.279]\\%" in table
    assert "the two rows are not pooled" in table
    assert "V98 does not isolate source-row contribution" in table
    legacy_table = (
        PAPER / "tables/external_source_contribution_closed_loop.tex"
    ).read_text(encoding="utf-8")
    assert "Legacy near-target-only wait" in legacy_table

    manifest = load_json(
        PAPER / "source_data/external_confirmation_artifact_manifest.json"
    )
    assert manifest["protocol"] == "paper-tsc-external-confirmation-artifacts-v2"
    assert manifest["hard_gates"]["conditional_transfer_rows_pooled"] is False
    assert manifest["hard_gates"]["v98_remote_result_identity_passed"] is True
    assert manifest["hard_gates"]["v98_remote_unique_identities"] == 336
    assert (
        manifest["provenance"]["v98_preselected_controller-pair_confirmation"]
        == "bb00da6d3ee0f9420bdd7827eeada7c0b4755fadae4a871e3de6c42ded1dc65d"
    )


def test_jinan_conditional_transfer_rejects_nonzero_source_comparator() -> None:
    v91 = load_json(DEFAULT_SOURCE_CONFIRMATION)
    v98 = deepcopy(load_json(DEFAULT_V98_SOURCE_CONFIRMATION))
    v98["target_only_anchor"]["source_rows_consumed"] = 1

    with pytest.raises(ValueError, match="strict zero-source-row"):
        validate_jinan_conditional_source_transfer(v91, v98)
