from pathlib import Path
import subprocess
import sys

from paper_tsc.figures.build_tsc_source_gate_development import build


ROOT = Path(__file__).resolve().parents[1]


def test_source_gate_builder_preserves_frozen_negative_decisions() -> None:
    rows = build()
    assert [row["version"] for row in rows] == ["v145", "v146", "v148"]
    assert [row["selected_source_cities"] for row in rows] == [6, 7, 0]
    assert all(row["decision"] == "REJECT" for row in rows)
    assert all(row["dense_mean"] < 0.0 for row in rows)
    assert rows[0]["target_maximum_city_regression"] > 0.02
    assert rows[1]["target_maximum_city_regression"] > 0.02
    assert rows[2]["target_mean"] == 0.0

    table = ROOT / "paper_tsc/tables/source_gate_development.tex"
    source = ROOT / "paper_tsc/source_data/source_gate_development.csv"
    assert table.is_file()
    assert source.is_file()
    assert "Mechanism compatibility" in table.read_text(encoding="utf-8")


def test_source_gate_builder_is_directly_executable() -> None:
    completed = subprocess.run(
        [sys.executable, "paper_tsc/figures/build_tsc_source_gate_development.py"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
