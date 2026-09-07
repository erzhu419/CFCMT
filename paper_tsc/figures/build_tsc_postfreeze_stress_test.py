from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAPER = REPO / "paper_tsc"
SOURCE_DATA = PAPER / "source_data"
TABLES = PAPER / "tables"

V86 = (
    REPO
    / "cf_h2o/results/cluster/tsc_v86r82_external_v9_interference_aware_redevelopment_20260830"
    / "interference_aware_redevelopment_audit_v1.json"
)
V87 = (
    REPO
    / "cf_h2o/results/cluster/tsc_v87_external_action_aware_redevelopment_20260830"
    / "phase_transition_aware_redevelopment_audit_v1.json"
)
V88 = (
    REPO
    / "cf_h2o/results/cluster/tsc_v88_external_bounded_stay_redevelopment_20260830"
    / "bounded_stay_redevelopment_audit_v1.json"
)
V89 = (
    REPO
    / "cf_h2o/results/cluster/tsc_v89_external_fresh_confirmation_20260830"
    / "fresh_confirmation_audit_v1.json"
)

EXPECTED_PROTOCOLS = {
    "v86": "tsc-v86r82-interference-aware-redevelopment-audit-v1",
    "v87": "tsc-v87r83-phase-transition-aware-redevelopment-audit-v1",
    "v88": "tsc-v88r84-bounded-stay-redevelopment-audit-v1",
    "v89": "tsc-v89r85-fresh-independent-confirmation-audit-v1",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _candidate(audit: Mapping[str, Any], policy: str) -> Mapping[str, Any]:
    rows = [row for row in audit["candidate_summaries"] if row["policy"] == policy]
    _require(len(rows) == 1, f"expected one {policy} row")
    return rows[0]


def _development_row(
    *,
    stage: str,
    evidence_class: str,
    audit: Mapping[str, Any],
    policy: str,
    interpretation: str,
) -> dict[str, Any]:
    row = _candidate(audit, policy)
    ci = row["bootstrap"]["ci95"]
    return {
        "stage": stage,
        "evidence_class": evidence_class,
        "policy": policy,
        "seed_count": int(row["seed_count"]),
        "mean_relative_delta": float(row["mean_relative_delta"]),
        "ci95_low": float(ci[0]),
        "ci95_high": float(ci[1]),
        "improved_seed_fraction": float(row["improved_seed_fraction"]),
        "worst_seed_relative_delta": float(row["worst_seed_relative_delta"]),
        "one_sided_sign_p": "",
        "joint_gate_passed": False,
        "interpretation": interpretation,
    }


def build_rows() -> list[dict[str, Any]]:
    audits = {name: _load(path) for name, path in {
        "v86": V86,
        "v87": V87,
        "v88": V88,
        "v89": V89,
    }.items()}
    for name, audit in audits.items():
        _require(audit.get("protocol") == EXPECTED_PROTOCOLS[name], f"{name} protocol changed")
        _require(audit.get("status") == "PASS", f"{name} evidence audit failed")

    v89 = audits["v89"]
    _require(v89["integrity_gate"]["passed"] is True, "v89 integrity gate failed")
    _require(v89["confirmation_gate"]["passed"] is False, "v89 decision changed")
    _require(
        v89["decision"] == "reject_preregistered_same_network_confirmation",
        "v89 rejection decision changed",
    )
    _require(
        v89["scientific_status"]["v85_prospective_evidence_used"] is False,
        "sealed v85 evidence entered the post-freeze result",
    )

    rows = [
        _development_row(
            stage="v86 legacy global",
            evidence_class="adaptive screen",
            audit=audits["v86"],
            policy="uh_global_h120_m1",
            interpretation="development only",
        ),
        _development_row(
            stage="v87 unbounded stay",
            evidence_class="adaptive ablation",
            audit=audits["v87"],
            policy="uh_global_h120_stay_aware",
            interpretation="variant rejected",
        ),
        _development_row(
            stage="v88 one-stay budget",
            evidence_class="adaptive ablation",
            audit=audits["v88"],
            policy="uh_global_h120_stay_budget1",
            interpretation="variant rejected",
        ),
    ]
    summary = v89["confirmation_summary"]
    ci = summary["bootstrap"]["ci95"]
    rows.append(
        {
            "stage": "v89 frozen legacy",
            "evidence_class": "fresh confirmation",
            "policy": v89["confirmation_gate"]["policy"],
            "seed_count": int(summary["seed_count"]),
            "mean_relative_delta": float(summary["mean_relative_delta"]),
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "improved_seed_fraction": float(summary["improved_seed_fraction"]),
            "worst_seed_relative_delta": float(summary["worst_seed_relative_delta"]),
            "one_sided_sign_p": float(summary["exact_sign_test"]["one_sided_p"]),
            "joint_gate_passed": bool(v89["confirmation_gate"]["passed"]),
            "interpretation": "confirmation rejected",
        }
    )
    return rows


def _latex_escape(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_outputs(rows: list[dict[str, Any]]) -> None:
    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = SOURCE_DATA / "postfreeze_execution_stress_test.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Stage & Evidence & Seeds & Mean $\Delta$ & 95\% CI & Improved & Decision \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(row['stage'])} & {_latex_escape(row['evidence_class'])} & "
            f"{row['seed_count']} & {100.0 * row['mean_relative_delta']:.3f}\\% & "
            f"[{100.0 * row['ci95_low']:.3f}, {100.0 * row['ci95_high']:.3f}]\\% & "
            f"{100.0 * row['improved_seed_fraction']:.1f}\\% & "
            f"{_latex_escape(row['interpretation'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "postfreeze_execution_stress_test.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    rows = build_rows()
    write_outputs(rows)
    confirmation = rows[-1]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "v89_mean_relative_delta": confirmation["mean_relative_delta"],
                "v89_joint_gate_passed": confirmation["joint_gate_passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
