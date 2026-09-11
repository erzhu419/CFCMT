#!/usr/bin/env python3
"""Audit the 336 V98 result identities remotely without copying raw files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)


PROTOCOL = "tsc-v98-strict-target-anchor-remote-integrity-audit-v1"
RESULT_PROTOCOL = "tsc-v93-source-city-causal-component-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v93-source-city-causal-selection-development-v1"
REMOTE_ROOT = (
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v98_target_offline_source_selection_20260831/strict_fresh_confirmation_v2"
)
REMOTE_PYTHON = (
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "freqduet-cpu-py310/bin/python3.10"
)
SPEC = ROOT / "cf_h2o/config/traffic_signal_tsc_v98_strict_target_fresh_confirmation_v2.json"
LAUNCH = ROOT / (
    "cf_h2o/results/cluster/tsc_v98_target_offline_source_selection_20260831/"
    "strict_fresh_confirmation_v2/launch.json"
)
NUMERICAL_AUDIT = LAUNCH.with_name("audit_v2.json")
OUT = ROOT / (
    "cf_h2o/results/paper_artifacts/"
    "tsc_v98_strict_target_anchor_remote_integrity_audit_v1.json"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    spec, launch, numerical = map(read_json, (SPEC, LAUNCH, NUMERICAL_AUDIT))
    artifacts = launch["artifact_contract"]
    scenarios = launch["scenarios"]
    seeds = [int(value) for value in spec["confirmation_seeds"]]
    expected = {
        "remote_root": REMOTE_ROOT,
        "scenarios": scenarios,
        "seeds": seeds,
        "arms": {
            "target_only": [{}, 0.0, "cfcmt_source_component_target_only_guard"],
            "offline_selector": [
                {"hangzhou": 1.0},
                1.0,
                "cfcmt_source_component_offline_selector_guard",
            ],
        },
        "guard": {
            "enabled": True,
            **spec["candidates"][0]["guard_by_city"]["jinan"],
        },
        "result_protocol": RESULT_PROTOCOL,
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "target_hash": spec["strict_target_only_contract"]["jinan_model_sha256"],
        "main_hash": artifacts["jinan_main_model"]["sha256"],
        "component_hash": artifacts["jinan_component_bundle"]["sha256"],
    }
    if not (
        len(seeds) == 56
        and scenarios
        == ["jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"]
        and numerical["confirmation_passed"] is True
        and numerical["matrix_size"] == 336
    ):
        raise ValueError("local frozen V98 contract changed")

    remote_program = r'''
import glob, json, sys
from collections import Counter
e=json.loads(sys.argv[1]); files=sorted(glob.glob(e["remote_root"]+"/*/seed_*/*/result.json"))
expected={(s,int(seed),arm) for s in e["scenarios"] for seed in e["seeds"] for arm in e["arms"]}
seen=set(); bad=Counter(); counts={k:Counter() for k in ("arms","scenarios","result_protocols","analysis_protocols","target_model_hashes","main_model_hashes","component_bundle_hashes")}
for path in files:
 try: row=json.load(open(path,encoding="utf-8"))
 except Exception: bad["unreadable_result"]+=1; continue
 arm=str(row.get("candidate_key")); scenario=str(row.get("scenario")); seed=int(row.get("seed")); identity=(scenario,seed,arm)
 if identity in seen: bad["duplicate_identity"]+=1
 seen.add(identity); counts["arms"][arm]+=1; counts["scenarios"][scenario]+=1
 counts["result_protocols"][str(row.get("protocol"))]+=1; counts["analysis_protocols"][str(row.get("analysis_protocol"))]+=1
 if row.get("protocol")!=e["result_protocol"]: bad["result_protocol_mismatch"]+=1
 if row.get("analysis_protocol")!=e["analysis_protocol"]: bad["analysis_protocol_mismatch"]+=1
 if row.get("city")!="jinan" or arm not in e["arms"]: bad["city_or_arm_mismatch"]+=1; continue
 if [row.get("source_city_weights"),row.get("source_mass"),row.get("policy")]!=e["arms"][arm]: bad["arm_contract_mismatch"]+=1
 if row.get("source_guard_config")!=e["guard"]: bad["guard_contract_mismatch"]+=1
 anchor=row.get("target_only_anchor") or {}; prov=(row.get("method_runtime_override") or {}).get("provenance") or {}; runtime_anchor=prov.get("target_only_anchor") or {}
 anchor_expected={"family":"causal_target_only_v2","protocol":"cfcmt-strict-target-only-action-model-v1","sha256":e["target_hash"],"source_rows_consumed":0}
 counts["target_model_hashes"][str(anchor.get("sha256"))]+=1; counts["main_model_hashes"][str(prov.get("main_model_sha256"))]+=1; counts["component_bundle_hashes"][str(prov.get("component_bundle_sha256"))]+=1
 if any(anchor.get(k)!=v for k,v in anchor_expected.items()) or any(runtime_anchor.get(k)!=v for k,v in anchor_expected.items()): bad["target_anchor_mismatch"]+=1
 if prov.get("main_model_sha256")!=e["main_hash"]: bad["main_model_hash_mismatch"]+=1
 if prov.get("component_bundle_sha256")!=e["component_hash"]: bad["component_bundle_hash_mismatch"]+=1
bad["missing_identity"]+=len(expected-seen); bad["unexpected_identity"]+=len(seen-expected)
print(json.dumps({"result_count":len(files),"unique_identity_count":len(seen),"counts":{k:dict(v) for k,v in counts.items()},"bad_counts":{k:v for k,v in bad.items() if v}},sort_keys=True))
'''
    scheduler = _load_scheduler(DEFAULT_SCHEDULER)
    command = (
        f"{REMOTE_PYTHON} -c {shlex.quote(remote_program)} "
        f"{shlex.quote(json.dumps(expected, sort_keys=True))}"
    )
    code, stdout, stderr = scheduler.run_on("node001", command, timeout=120, check=False)
    if code != 0:
        raise RuntimeError(f"remote scan failed ({code}): {stderr[-2000:]}")
    observed = json.loads([line for line in stdout.splitlines() if line.strip()][-1])
    expected_arm_count = len(seeds) * len(scenarios)
    expected_scenario_count = len(seeds) * len(expected["arms"])
    expected_summary = {
        "result_count": 336,
        "unique_identity_count": 336,
        "seed_count": 56,
        "scenario_count": 3,
        "arm_counts": {key: expected_arm_count for key in expected["arms"]},
        "scenario_result_counts": {key: expected_scenario_count for key in scenarios},
        "result_protocol": RESULT_PROTOCOL,
        "analysis_protocol": ANALYSIS_PROTOCOL,
        "model_hashes": {
            "strict_target_only": expected["target_hash"],
            "jinan_main_model": expected["main_hash"],
            "jinan_component_bundle": expected["component_hash"],
        },
        "target_model_family": "causal_target_only_v2",
        "target_model_protocol": "cfcmt-strict-target-only-action-model-v1",
        "target_source_rows_consumed": 0,
    }
    bad = observed["bad_counts"]
    summary_checks = {
        "result_count": observed["result_count"] == 336,
        "unique_identity_count": observed["unique_identity_count"] == 336,
        "arm_counts": observed["counts"]["arms"] == expected_summary["arm_counts"],
        "scenario_counts": observed["counts"]["scenarios"]
        == expected_summary["scenario_result_counts"],
    }
    bad.update({f"summary_{key}_mismatch": 1 for key, ok in summary_checks.items() if not ok})
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "retrospective_remote_provenance_integrity_audit",
        "passed": not bad,
        "scan": {
            "execution_node": "node001",
            "remote_root": REMOTE_ROOT,
            "file_pattern": "*/seed_*/*/result.json",
            "mode": "remote_in_place_json_metadata_only",
            "raw_results_copied_to_local": False,
        },
        "inputs": {
            "candidate_spec": str(SPEC),
            "candidate_spec_protocol": spec["protocol"],
            "launch": str(LAUNCH),
            "numerical_audit": str(NUMERICAL_AUDIT),
            "numerical_audit_protocol": numerical["protocol"],
        },
        "expected": expected_summary,
        "observed": {
            "result_count": observed["result_count"],
            "unique_identity_count": observed["unique_identity_count"],
            **observed["counts"],
        },
        "bad_counts": bad,
        "limitations": [
            "This retrospective audit checks result identity and frozen model provenance; it does not recompute rollout metrics or the V98 statistical analysis.",
            "Freshness is limited to simulator seeds on the same Jinan network and three demand scenarios; this is not unseen-city evidence.",
            "The shared pressure guard inherited its risk multiplier from V93 target closed-loop development.",
            "The V98 strict target-only estimator consumed zero source rows but had lower capacity than the source-augmented model, so V98 alone does not isolate source-row value from architecture.",
            "This audit does not compare either V98 arm directly with PhasePressure.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT), "passed": payload["passed"]}, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
