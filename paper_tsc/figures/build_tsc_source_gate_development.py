#!/usr/bin/env python3
"""Build the frozen seven-city source-gate development table and source data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper_tsc"
TABLE_OUT = PAPER / "tables" / "source_gate_development.tex"
CSV_OUT = PAPER / "source_data" / "source_gate_development.csv"
EXPECTED_CITIES = {
    "atlanta",
    "cologne",
    "hangzhou",
    "ingolstadt",
    "new_york",
    "resco_synthetic",
    "salt_lake_city",
}
CONFIGS = (
    {
        "name": "Robust history",
        "version": "v145",
        "path": REPO
        / "cf_h2o/results/cluster/tsc_v145_multicity_robust_source_gate_20260901/development_v1/aggregate.json",
        "protocol": "tsc-v145-multicity-robust-source-gate-aggregate-v1",
        "effect_prefix": "robust_gate_minus_",
    },
    {
        "name": "OOF action agreement",
        "version": "v146",
        "path": REPO
        / "cf_h2o/results/cluster/tsc_v146_multicity_oof_compatibility_gate_20260901/development_v1/aggregate.json",
        "protocol": "tsc-v146-multicity-oof-compatibility-gate-aggregate-v1",
        "effect_prefix": "oof_gate_minus_",
    },
    {
        "name": "Mechanism compatibility",
        "version": "v148",
        "path": REPO
        / "cf_h2o/results/cluster/tsc_v148_multicity_mechanism_compatibility_gate_20260901/development_v2_corrected/aggregate.json",
        "protocol": "tsc-v148-multicity-mechanism-compatibility-aggregate-v1",
        "effect_prefix": "mechanism_gate_minus_",
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _summary(
    result: Mapping[str, Any], effect_prefix: str, comparator: str
) -> Mapping[str, Any]:
    key = f"{effect_prefix}{comparator}"
    summaries = result.get("paired_city_effects", {})
    if key not in summaries:
        raise ValueError(f"missing frozen source-gate comparison: {key}")
    return summaries[key]


def _row(config: Mapping[str, Any]) -> dict[str, Any]:
    result = _read_json(Path(config["path"]))
    gate = dict(result.get("development_gate", {}))
    cities = tuple(str(value) for value in result.get("target_city_groups", ()))
    decisions = dict(result.get("city_decisions", {}))
    if (
        result.get("protocol") != config["protocol"]
        or set(cities) != EXPECTED_CITIES
        or set(decisions) != EXPECTED_CITIES
        or gate.get("passed") is not False
        or not str(gate.get("decision", "")).startswith("close_v")
    ):
        raise ValueError(f"{config['version']} frozen source-gate contract changed")
    target = _summary(result, str(config["effect_prefix"]), "target_only_causal")
    placebo = _summary(
        result, str(config["effect_prefix"]), "matched_source_placebo"
    )
    dense = _summary(result, str(config["effect_prefix"]), "pooled_h2oplus")
    return {
        "version": config["version"],
        "selector": config["name"],
        "selected_source_cities": sum(
            value.get("selected_source") is not None for value in decisions.values()
        ),
        "target_mean": float(target["mean"]),
        "target_upper_95_one_sided": float(target["upper_95_one_sided"]),
        "target_improving_cities": int(target["improving_city_count"]),
        "target_maximum_city_regression": float(target["maximum_city_regression"]),
        "placebo_mean": float(placebo["mean"]),
        "placebo_upper_95_one_sided": float(placebo["upper_95_one_sided"]),
        "dense_mean": float(dense["mean"]),
        "dense_upper_95_one_sided": float(dense["upper_95_one_sided"]),
        "decision": "REJECT",
    }


def build() -> list[dict[str, Any]]:
    rows = [_row(config) for config in CONFIGS]
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        (
            r"Version & Source gate & Source & $\Delta$ target & Upper & "
            r"Cities & Max. reg. & $\Delta$ dense \\"
        ),
        r" &  & selected & mean & 95\% & improved &  & mean \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['version']} & {row['selector']} & "
            f"{row['selected_source_cities']}/7 & {row['target_mean']:+.4f} & "
            f"{row['target_upper_95_one_sided']:+.4f} & "
            f"{row['target_improving_cities']}/7 & "
            f"{row['target_maximum_city_regression']:+.4f} & "
            f"{row['dense_mean']:+.4f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    print(json.dumps({"rows": build(), "table": str(TABLE_OUT), "csv": str(CSV_OUT)}))
