#!/usr/bin/env python3
"""Redistribute missing V114 Manhattan seed/shards across all CPU nodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
NODES = tuple(f"node00{index}" for index in range(1, 7))
SCENARIO = "manhattan_28x7"
MANIFEST_RELATIVE = Path("cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json")
LAUNCH_PROTOCOL = "tsc-v114-manhattan-explicit-shard-recovery-launch-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def _load_scheduler(scheduler_path: Path):
    sys.path.insert(0, str(Path(scheduler_path).parent))
    import scheduler  # type: ignore

    return scheduler


def assign_pairs(
    pairs: Sequence[tuple[int, int]], nodes: Sequence[str] = NODES
) -> dict[str, tuple[tuple[int, int], ...]]:
    assignments = {str(node): [] for node in nodes}
    for index, pair in enumerate(sorted((int(seed), int(shard)) for seed, shard in pairs)):
        assignments[str(nodes[index % len(nodes)])].append(pair)
    return {node: tuple(values) for node, values in assignments.items() if values}


def _remote_inventory(
    *,
    scheduler: Any,
    snapshot_root: Path,
    protocol: Mapping[str, Any],
    remote_cache_root: Path,
) -> dict[str, Any]:
    collection = dict(protocol["collection"])
    seeds = [int(value) for value in collection["seeds"]]
    shards = int(
        dict(collection.get("scenario_collection_shards_per_seed", {})).get(
            SCENARIO, collection["collection_shards_per_seed"]
        )
    )
    script = f"""
import json
from pathlib import Path
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _counterfactual_cache_path
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
manifest = load_traffic_signal_manifest(Path({str(snapshot_root / MANIFEST_RELATIVE)!r}))
sumocfg = manifest.sumocfgs[{SCENARIO!r}]
rows = []
for seed in {seeds!r}:
    for shard in range({shards}):
        path = _counterfactual_cache_path(
            cache_root=Path({str(remote_cache_root)!r}),
            sumocfg=sumocfg,
            scenario={SCENARIO!r},
            duration_sec={float(collection['duration_sec'])!r},
            control_interval_sec={int(collection['control_interval_sec'])!r},
            warmup_sec={float(collection['warmup_sec'])!r},
            seed=seed,
            max_focal_tls={int(collection['max_focal_tls'])!r},
            counterfactual_horizon_intervals={int(collection['counterfactual_horizon_intervals'])!r},
            behavior_policy={str(collection['behavior_policy'])!r},
            counterfactual_cost_mode={str(collection['counterfactual_cost_mode'])!r},
            collection_shard_index=shard,
            collection_shard_count={shards},
            rollout_prefix_horizons_sec={list(collection['rollout_prefix_horizons_sec'])!r},
        )
        rows.append({{'seed': seed, 'shard': shard, 'path': str(path), 'exists': path.is_file()}})
print(json.dumps({{'rows': rows}}, sort_keys=True))
"""
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = (
        f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))} "
        + shlex.join([str(wrapper), "-c", script])
    )
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=300, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote Manhattan inventory failed: {stderr}")
    lines = [line for line in str(stdout).splitlines() if line.strip()]
    return json.loads(lines[-1])


def build_specs(
    *,
    snapshot_root: Path,
    protocol: Mapping[str, Any],
    assignments: Mapping[str, Sequence[tuple[int, int]]],
    remote_cache_root: Path,
    remote_task_root: Path,
) -> list[dict[str, Any]]:
    collection = dict(protocol["collection"])
    specs = []
    for node in NODES:
        pairs = tuple(assignments.get(node, ()))
        if not pairs:
            continue
        workers = min(len(pairs), 20)
        result_root = remote_task_root / node
        command = shlex.join(
            [
                str(snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"),
                "-m",
                "cf_h2o.eval.traffic_signal_counterfactual_shard_subset",
                "--manifest",
                str(snapshot_root / MANIFEST_RELATIVE),
                "--scenario",
                SCENARIO,
                "--seed-shards",
                *(f"{seed}:{shard}" for seed, shard in pairs),
                "--duration-sec",
                str(collection["duration_sec"]),
                "--control-interval-sec",
                str(collection["control_interval_sec"]),
                "--warmup-sec",
                str(collection["warmup_sec"]),
                "--max-focal-tls",
                str(collection["max_focal_tls"]),
                "--counterfactual-horizon-intervals",
                str(collection["counterfactual_horizon_intervals"]),
                "--collection-shards",
                str(
                    dict(
                        collection.get(
                            "scenario_collection_shards_per_seed", {}
                        )
                    ).get(SCENARIO, collection["collection_shards_per_seed"])
                ),
                "--workers",
                str(workers),
                "--cache-root",
                str(remote_cache_root),
                "--behavior-policy",
                str(collection["behavior_policy"]),
                "--counterfactual-cost-mode",
                str(collection["counterfactual_cost_mode"]),
                "--rollout-prefix-horizons-sec",
                *(str(value) for value in collection["rollout_prefix_horizons_sec"]),
                "--expected-sumo-version",
                str(collection["sumo_version"]),
                "--out",
                str(result_root / "summary.json"),
            ]
        )
        specs.append(
            {
                "description": (
                    f"CFCMT V114 Manhattan explicit shard recovery {node}: "
                    f"pairs={len(pairs)}"
                ),
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v114/manhattan-shard-recovery-v2/{node}",
                "resource_family": "CFCMT-v114-manhattan-shard-recovery-v2",
                "vram": 0,
                "ram_mb": 49152,
                "cpu": workers,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {
                    "CFCMT_SOURCE_ROOT": str(snapshot_root),
                    "PYTHONUNBUFFERED": "1",
                },
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-task-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite Manhattan recovery launch artifact")
    stage = _read_json(args.stage_manifest)
    protocol = _read_json(args.protocol)
    snapshot_root = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    for command in (
        shlex.join(["test", "-d", str(args.remote_cache_root)]),
        shlex.join(["test", "!", "-e", str(args.remote_task_root)]),
    ):
        code, _, stderr = scheduler.run_on(
            "node001", command, timeout=120, check=False
        )
        if int(code) != 0:
            raise RuntimeError(f"Manhattan recovery preflight failed: {stderr}")
    inventory = _remote_inventory(
        scheduler=scheduler,
        snapshot_root=snapshot_root,
        protocol=protocol,
        remote_cache_root=args.remote_cache_root,
    )
    existing = [
        (int(item["seed"]), int(item["shard"]))
        for item in inventory["rows"]
        if bool(item["exists"])
    ]
    missing = [
        (int(item["seed"]), int(item["shard"]))
        for item in inventory["rows"]
        if not bool(item["exists"])
    ]
    assignments = assign_pairs(missing)
    specs = build_specs(
        snapshot_root=snapshot_root,
        protocol=protocol,
        assignments=assignments,
        remote_cache_root=args.remote_cache_root,
        remote_task_root=args.remote_task_root,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v114-manhattan-shard-recovery-v2",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_task_root": str(args.remote_task_root),
        "existing_pairs": [f"{seed}:{shard}" for seed, shard in sorted(existing)],
        "missing_pairs": [f"{seed}:{shard}" for seed, shard in sorted(missing)],
        "assignments": {
            node: [f"{seed}:{shard}" for seed, shard in pairs]
            for node, pairs in assignments.items()
        },
        "specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
