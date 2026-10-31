#!/usr/bin/env python3
"""Submit the frozen V158 collection, fit, reserve, and aggregate stages."""

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
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_v158_native_prefix_ranking_calibration import (  # noqa: E402
    CALIBRATOR_BUNDLE_PROTOCOL,
    FIT_RESULT_PROTOCOL,
    PROTOCOL,
    RESERVE_AGGREGATE_PROTOCOL,
    RESERVE_SEED_PROTOCOL,
    SEED_RESULT_PROTOCOL,
    validate_protocol,
)


DERIVATION_PROTOCOL = "tsc-v158-native-prefix-ranking-snapshot-derivation-v1"
LAUNCH_PROTOCOL = "tsc-v158-native-prefix-ranking-launch-v1"
MODULE = "cf_h2o.eval.traffic_signal_v158_native_prefix_ranking_calibration"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v158_native_prefix_ranking_calibration.json"
)
RUNNER_RELATIVE = Path(
    "cf_h2o/eval/traffic_signal_v158_native_prefix_ranking_calibration.py"
)
ACTION_RANKER_RELATIVE = Path("cf_h2o/traffic_signal/action_ranker.py")
LAUNCHER_RELATIVE = Path(
    "scripts/cluster/launch_tsc_v158_native_prefix_ranking_calibration.py"
)
MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
OVERLAYS = (
    RUNNER_RELATIVE,
    CONFIG_RELATIVE,
    ACTION_RANKER_RELATIVE,
    LAUNCHER_RELATIVE,
)
PARENT_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/102b77f2df56f9400f85"
)
PARENT_SNAPSHOT_SHA256 = (
    "102b77f2df56f9400f85702f548fe849e0524ca0b7063f93ebc648c1e9f1be4c"
)
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)
DEFAULT_SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
DEFAULT_SCHEDULER_DIR = DEFAULT_SCHEDULER.parent
DEFAULT_CONVERSION_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v53r49_external_network_repair_20260810/"
    "conversion_v9_task/full_networks_v9"
)
DEFAULT_RUNTIME_BUNDLE = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v157b_feature_aligned_b100_runtime_freeze_20260911/"
    "freeze_v2/runtime_models.pkl"
)
RUNTIME_BUNDLE_SHA256 = (
    "a6bdea34175f06a399dc00c1c1df06781e578035a2806c434e0f4eb9be61ad93"
)
DEFAULT_ARM_MANIFEST = DEFAULT_RUNTIME_BUNDLE.with_name("arm_manifest.json")
ARM_MANIFEST_SHA256 = (
    "02419e1b9b2370978ed0fe4525c1d4600fd13470c7a76edd625d9c1d126ce52f"
)
DEFAULT_V157B_RESULT = DEFAULT_RUNTIME_BUNDLE.with_name("result.json")
V157B_RESULT_SHA256 = (
    "fc731f16e9b2ffc4cacdfdd474f67ef91d71f4a7e64f86f10a92ff560d65d7b5"
)
DEFAULT_RESULT_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/"
    "tsc_v158_native_prefix_ranking_calibration_20260911"
)
DEFAULT_SCRATCH_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_SCRATCH/"
    "tsc_v158_native_prefix_ranking_calibration_20260911"
)
DEFAULT_NODES = ("node003", "node004", "node005", "node006", "node007")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_scheduler(directory: Path) -> Any:
    sys.path.insert(0, str(directory))
    import scheduler  # type: ignore

    return scheduler


def _stage_root(stage_path: Path) -> tuple[Path, dict[str, Any]]:
    stage = _read_json(stage_path)
    root = Path(str(stage.get("snapshot_root", "")))
    digest = str(stage.get("snapshot_sha256", ""))
    expected_overlays = [
        {"path": path.as_posix(), "sha256": _sha256(PROJECT_ROOT / path)}
        for path in OVERLAYS
    ]
    if (
        stage.get("protocol") != DERIVATION_PROTOCOL
        or stage.get("derived_from_snapshot_root") != str(PARENT_ROOT)
        or stage.get("derived_from_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or stage.get("overlays") != expected_overlays
        or len(digest) != 64
        or root.name != digest[:20]
    ):
        raise ValueError("V158 staged snapshot contract changed")
    return root, stage


def _run_on(scheduler: Any, node: str, command: str, *, timeout: int = 180) -> str:
    code, stdout, stderr = scheduler.run_on(
        str(node), command, timeout=timeout, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"V158 remote preflight failed: {stderr[-4000:]}")
    return str(stdout)


def _remote_preflight(
    scheduler: Any,
    *,
    node: str,
    snapshot_root: Path,
    conversion_root: Path,
    runtime_bundle: Path,
    arm_manifest: Path,
    v157b_result: Path,
) -> dict[str, Any]:
    paths = (
        snapshot_root / ".cfcmt_snapshot.json",
        snapshot_root / RUNNER_RELATIVE,
        snapshot_root / CONFIG_RELATIVE,
        snapshot_root / ACTION_RANKER_RELATIVE,
        snapshot_root / LAUNCHER_RELATIVE,
        snapshot_root / MANIFEST_RELATIVE,
        snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh",
        runtime_bundle,
        arm_manifest,
        v157b_result,
    )
    _run_on(
        scheduler,
        node,
        " && ".join(
            [shlex.join(["test", "-d", str(conversion_root)])]
            + [shlex.join(["test", "-f", str(path)]) for path in paths]
        ),
    )
    observed = [
        line.split()[0]
        for line in _run_on(
            scheduler,
            node,
            shlex.join(["sha256sum", *(str(path) for path in paths)]),
        ).splitlines()
        if line.split()
    ]
    expected = [
        _sha256(PROJECT_ROOT / path) for path in OVERLAYS[:2]
    ]
    if (
        observed[1] != expected[0]
        or observed[2] != expected[1]
        or observed[3] != _sha256(PROJECT_ROOT / ACTION_RANKER_RELATIVE)
        or observed[4] != _sha256(PROJECT_ROOT / LAUNCHER_RELATIVE)
        or observed[5]
        != _read_json(PROJECT_ROOT / CONFIG_RELATIVE)["network_manifest"]["sha256"]
        or observed[7:] != [
            RUNTIME_BUNDLE_SHA256,
            ARM_MANIFEST_SHA256,
            V157B_RESULT_SHA256,
        ]
    ):
        raise RuntimeError("V158 remote input identity changed")
    summary_program = """
import json,sys
arm=json.load(open(sys.argv[1],encoding='utf-8'))
result=json.load(open(sys.argv[2],encoding='utf-8'))
assert arm['protocol']=='tsc-v157b-feature-aligned-b100-arm-manifest-v2'
assert result['protocol']=='tsc-v157b-feature-aligned-b100-runtime-refit-freeze-result-v2'
assert result['scientific_status']=='runtime_arms_frozen_branch_not_run'
print(json.dumps({'arm_manifest_protocol':arm['protocol'],'v157b_protocol':result['protocol'],'v157b_scientific_status':result['scientific_status']},sort_keys=True,separators=(',',':')))
""".strip()
    summary = json.loads(
        _run_on(
            scheduler,
            node,
            shlex.join(
                [str(REMOTE_PYTHON), "-c", summary_program, str(arm_manifest), str(v157b_result)]
            ),
        ).splitlines()[-1]
    )
    import_program = (
        "import json; from pathlib import Path; "
        f"from {MODULE} import _read_json,validate_protocol; "
        f"validate_protocol(_read_json(Path({str(snapshot_root / CONFIG_RELATIVE)!r}))); "
        "print(json.dumps({'status':'PASS'}))"
    )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    smoke = _run_on(
        scheduler,
        node,
        shlex.join(
            [
                "env",
                f"CFCMT_SOURCE_ROOT={snapshot_root}",
                f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
                str(wrapper),
                "-c",
                import_program,
            ]
        ),
        timeout=300,
    )
    if json.loads(smoke.splitlines()[-1]).get("status") != "PASS":
        raise RuntimeError("V158 remote module import smoke changed")
    return {"sha256": observed, "v157b": summary, "module_import": "PASS"}


def _runner_prefix(snapshot_root: Path, conversion_root: Path) -> list[str]:
    return [
        "env",
        f"CFCMT_SOURCE_ROOT={snapshot_root}",
        f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
        "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "NUMEXPR_NUM_THREADS=1",
        "PYTHONUNBUFFERED=1",
        str(snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"),
        "-m",
        MODULE,
    ]


def _short_task_command(run: Sequence[str], scratch: Path | None = None) -> str:
    steps = [shlex.join([*run, "--reuse-complete"])]
    if scratch is not None:
        steps.append(shlex.join(["rm", "-rf", str(scratch)]))
    steps.append("printf 'TASK_DONE\\n'")
    return " && ".join(steps)


def _spec(
    *, description: str, command: str, signature: str, node: str,
    snapshot_root: Path, result_dir: Path, cpu: int, ram_mb: int,
    training: bool = False,
) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "description": description,
        "project": "CFCMT",
        "cmd": command,
        "cwd": str(snapshot_root),
        "signature": signature,
        "resource_family": "CFCMT-v158-native-prefix-ranking-v1",
        "ram_resource_family": "CFCMT-v158-native-prefix-ranking-v1",
        "vram": 0,
        "ram_mb": int(ram_mb),
        "cpu": int(cpu),
        "priority": "high",
        "require_node": str(node),
        "skip_launch_staging": True,
        "skip_resume_scan": True,
        "env_spec": "none",
        "extra_env": {},
    }
    if training:
        spec.update(
            allow_cpu_training=True,
            cpu_training_justification=(
                "V158 fits small deterministic sklearn calibrators on 800 rows; "
                "GPU execution is unsupported and would not accelerate this fit."
            ),
        )
    return spec


def _submit(scheduler_path: Path, specs: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            str(scheduler_path),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            label,
        ],
        input=json.dumps(list(specs)),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"scheduler rejected V158 tasks: {completed.stderr[-4000:]}")
    return json.loads(completed.stdout)


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "fit", "reserve", "aggregate"))
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, default=DEFAULT_CONVERSION_ROOT)
    parser.add_argument("--runtime-bundle", type=Path, default=DEFAULT_RUNTIME_BUNDLE)
    parser.add_argument("--arm-manifest", type=Path, default=DEFAULT_ARM_MANIFEST)
    parser.add_argument("--v157b-result", type=Path, default=DEFAULT_V157B_RESULT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--nodes", nargs=5, default=DEFAULT_NODES)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _base_parser().parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V158 launch record")
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    validate_protocol(config)
    snapshot_root, stage = _stage_root(args.stage_manifest)
    scheduler = _load_scheduler(args.scheduler.parent)
    preflight = _remote_preflight(
        scheduler,
        node=args.nodes[0],
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        runtime_bundle=args.runtime_bundle,
        arm_manifest=args.arm_manifest,
        v157b_result=args.v157b_result,
    )
    prefix = _runner_prefix(snapshot_root, args.conversion_root)
    protocol_path = snapshot_root / CONFIG_RELATIVE
    manifest_path = snapshot_root / MANIFEST_RELATIVE
    specs: list[dict[str, Any]] = []
    if args.stage == "collect":
        for index, seed in enumerate(config["development"]["seeds"]):
            output = args.result_root / "development_v1" / f"seed_{seed}"
            scratch = args.scratch_root / "development_v1" / f"seed_{seed}"
            run = [
                *prefix,
                "collect-seed",
                "--protocol", str(protocol_path),
                "--manifest", str(manifest_path),
                "--conversion-root", str(args.conversion_root),
                "--runtime-bundle", str(args.runtime_bundle),
                "--runtime-bundle-sha256", RUNTIME_BUNDLE_SHA256,
                "--seed", str(seed),
                "--scratch-root", str(scratch),
                "--output", str(output),
            ]
            specs.append(
                _spec(
                    description=f"CFCMT V158 native-prefix development seed {seed}",
                    command=_short_task_command(run, scratch),
                    signature=f"CFCMT/v158/native-prefix/collect/seed{seed}",
                    node=args.nodes[index % len(args.nodes)], snapshot_root=snapshot_root,
                    result_dir=output, cpu=2, ram_mb=4096,
                )
            )
    elif args.stage == "fit":
        output = args.result_root / "fit_v1"
        run = [
            *prefix,
            "fit",
            "--protocol", str(protocol_path),
            "--collection-root", str(args.result_root / "development_v1"),
            "--output", str(output),
        ]
        specs.append(
            _spec(
                description="CFCMT V158 five-fold native-prefix calibration",
                command=_short_task_command(run),
                signature="CFCMT/v158/native-prefix/fit-v1", node=args.nodes[0],
                snapshot_root=snapshot_root, result_dir=output, cpu=2, ram_mb=4096,
                training=True,
            )
        )
    elif args.stage == "reserve":
        fit_result = args.result_root / "fit_v1/result.json"
        calibrator = args.result_root / "fit_v1/calibrators.pkl"
        fit_summary_program = """
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
assert p.get('protocol')==sys.argv[2] and p.get('status')=='PASS' and p.get('reserve_authorized') is True
assert p['artifacts']['calibrators']['path']==sys.argv[3]
print(p['artifacts']['calibrators']['sha256'])
""".strip()
        calibrator_sha = _run_on(
            scheduler,
            args.nodes[0],
            shlex.join(
                [str(REMOTE_PYTHON), "-c", fit_summary_program, str(fit_result), FIT_RESULT_PROTOCOL, str(calibrator)]
            ),
        ).splitlines()[-1]
        for index, seed in enumerate(config["reserve"]["seeds"]):
            output = args.result_root / "reserve_v1" / f"seed_{seed}"
            scratch = args.scratch_root / "reserve_v1" / f"seed_{seed}"
            run = [
                *prefix,
                "run-reserve-seed",
                "--protocol", str(protocol_path),
                "--fit-result", str(fit_result),
                "--manifest", str(manifest_path),
                "--conversion-root", str(args.conversion_root),
                "--runtime-bundle", str(args.runtime_bundle),
                "--runtime-bundle-sha256", RUNTIME_BUNDLE_SHA256,
                "--calibrator-bundle", str(calibrator),
                "--calibrator-bundle-sha256", calibrator_sha,
                "--seed", str(seed),
                "--scratch-root", str(scratch),
                "--output", str(output),
            ]
            specs.append(
                _spec(
                    description=f"CFCMT V158 closed-loop reserve seed {seed}",
                    command=_short_task_command(run, scratch),
                    signature=f"CFCMT/v158/native-prefix/reserve/seed{seed}",
                    node=args.nodes[index % len(args.nodes)], snapshot_root=snapshot_root,
                    result_dir=output, cpu=2, ram_mb=4096,
                )
            )
    else:
        output = args.result_root / "reserve_aggregate_v1.json"
        run = [
            *prefix,
            "aggregate-reserve",
            "--protocol", str(protocol_path),
            "--reserve-root", str(args.result_root / "reserve_v1"),
            "--output", str(output),
        ]
        specs.append(
            _spec(
                description="CFCMT V158 closed-loop reserve aggregate",
                command=_short_task_command(run),
                signature="CFCMT/v158/native-prefix/reserve-aggregate-v1",
                node=args.nodes[0], snapshot_root=snapshot_root,
                result_dir=output.parent, cpu=1, ram_mb=4096,
            )
        )
    submission = _submit(
        args.scheduler,
        specs,
        f"CFCMT-v158-native-prefix-{args.stage}-v1",
    )
    record = {
        "protocol": LAUNCH_PROTOCOL,
        "stage": args.stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": True,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "remote_preflight": preflight,
        "result_root": str(args.result_root),
        "scratch_root": str(args.scratch_root),
        "specs": specs,
        "submission": submission,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(submission, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
