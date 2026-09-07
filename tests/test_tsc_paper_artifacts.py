from __future__ import annotations

import re
from pathlib import Path

import pytest

from paper_tsc.figures.build_tsc_external_confirmation import (
    DEFAULT_EXTERNAL_ROOT,
    DEFAULT_HELDOUT,
    DEFAULT_SOURCE_ABLATION,
    DEFAULT_SOURCE_CONFIRMATION,
    load_json,
    method_freeze_rows,
    offline_rows,
    source_contribution_rows,
    validate_source_contribution,
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
