#!/usr/bin/env python3
"""Acquire the complete DLR Bad Hersfeld scenario at one fixed commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import ssl
import subprocess
import sys
import time
from typing import Sequence
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import parse_xml


PROTOCOL = "dlr-badhersfeld-fixed-commit-complete-acquisition-v1"
REPOSITORY = "https://github.com/DLR-TS/sumo-scenarios"
RAW_BASE = "https://raw.githubusercontent.com/DLR-TS/sumo-scenarios"
COMMIT = "00f6eb479a9dc0fbaeb731495c11d48d7a9661d3"
SCENARIOS = ("present", "future_no_prt")
FILES = {
    "BadHersfeld/README.md": (
        1_834,
        "d1c0cd0e94f08072717b1d3ccb694291f30a79f9",
    ),
    "BadHersfeld/demand/osm_activitygen.lkw.rou.xml": (
        5_622,
        "f0b25a86c2b0d2c588973d80edf31813bec76252",
    ),
    "BadHersfeld/demand/osm_activitygen_future_no_prt.rou.xml.gz": (
        8_180_049,
        "0b2a5d121d738e934b9086f84b980c9394a1bd0c",
    ),
    "BadHersfeld/demand/osm_activitygen_present_no_prt.rou.xml.gz": (
        8_228_786,
        "cc4e9a9496d84d356c6aec7e652d2b8dead8397b",
    ),
    "BadHersfeld/evaluation/detail.xml": (
        59_762,
        "581efad715c910d54988a2e5a523bf6a2e2c1380",
    ),
    "BadHersfeld/evaluation/evaluate.sh": (
        95,
        "9bc9d06f3a3113bcfe18fe638c2fb1e755ae9b70",
    ),
    "BadHersfeld/evaluation/future_no_prt.sumocfg": (
        2_919,
        "11095ae254215ab59f9511d9b1446f6d8aa6b778",
    ),
    "BadHersfeld/evaluation/future_with_prt.sumocfg": (
        2_935,
        "d8a99422e55647cf6dddc943435edbe4b83163c5",
    ),
    "BadHersfeld/evaluation/present.sumocfg": (
        2_921,
        "ffeb2abf0bdac00185ea846b2015624eb2dbe354",
    ),
    "BadHersfeld/evaluation/tripinfoEmissions.py": (
        6_483,
        "86417abfa3f1e6cfe997af8e15d0694fb8892d75",
    ),
    "BadHersfeld/evaluation/view.xml": (
        59_754,
        "3147b53d88aab58afe68c3704ca30bf6b80045dd",
    ),
    "BadHersfeld/osm/build.bat": (
        427,
        "a3b51a0d83da5c28951a70009546690bc9256967",
    ),
    "BadHersfeld/osm/build.sh": (
        942,
        "6d1bb0d19fbf0465356e5c327e310a79415b8b98",
    ),
    "BadHersfeld/osm/buildings/osm_buildings.add.csv": (
        336_572,
        "209a95c7d0164527e8582bd0ca8cc8e407907869",
    ),
    "BadHersfeld/osm/buildings/osm_buildings.all.csv": (
        707_086,
        "99df349ed0af6553dfd3577415776f654e6dad44",
    ),
    "BadHersfeld/osm/calib.add.xml": (
        1_327,
        "d047963448deecab4b808d9ec4ca57e518d65bec",
    ),
    "BadHersfeld/osm/defaults/basic.vType.xml": (
        3_702,
        "4787016deda730a492900dffcedfd4b9f1205958",
    ),
    "BadHersfeld/osm/defaults/default-gui.xml": (
        248,
        "b688d0cb22ce0b4d21bf3f1f6b7fd3b47011499b",
    ),
    "BadHersfeld/osm/defaults/osm.netccfg": (
        3_654,
        "ce5148e90f4de22746584f7d4016636e9ad2d5f7",
    ),
    "BadHersfeld/osm/defaults/osm.polycfg": (
        615,
        "38cf10d54714232d63c1ab61a5ffa11c6f675691",
    ),
    "BadHersfeld/osm/defaults/osmNetconvert.typ.xml": (
        5_025,
        "e92374704db69e28c5933a1fb3cf3b40d7eda4e7",
    ),
    "BadHersfeld/osm/defaults/osmNetconvertPedestrians.typ.xml": (
        1_181,
        "db015f068b4e16a678b6cc3c21097492bef882ae",
    ),
    "BadHersfeld/osm/defaults/osmNetconvertUrbanDe.typ.xml": (
        735,
        "bf9888103786fc2b7510e7f3b4438c92461bb004",
    ),
    "BadHersfeld/osm/duarouter.sumocfg": (
        2_960,
        "9f2ada2ba82735a4f67e3313887243d4515abc27",
    ),
    "BadHersfeld/osm/edge_lane_data.add.xml": (
        202,
        "bdf447603033d81f50de21aad4cadb52c49f7d0f",
    ),
    "BadHersfeld/osm/gtfs_publictransport.add.xml": (
        236_697,
        "6616ac5c962fe2c0a73fb5f4f178153618bbaaea",
    ),
    "BadHersfeld/osm/gtfs_publictransport.rou.xml": (
        61_038,
        "0b65f904c9e1eb55225fc8c3436f2a58e1dfd039",
    ),
    "BadHersfeld/osm/induction_loops.add.xml": (
        1_273,
        "9fbc87977aed254704601e6bccf48ea381fc30b3",
    ),
    "BadHersfeld/osm/netdiff/diff.con.xml": (
        22_468,
        "ee566596ebe82a7ee423f62b90568414861587bc",
    ),
    "BadHersfeld/osm/netdiff/diff.edg.xml": (
        16_430,
        "7ff0164ae2383c5980135fe4cec795116bb1439c",
    ),
    "BadHersfeld/osm/netdiff/diff.nod.xml": (
        3_133,
        "c87d81fd5ba4460812c1bbc4dc6d90ef787b23ca",
    ),
    "BadHersfeld/osm/netdiff/diff.tll.xml": (
        8_820,
        "df0c77ad3ba8f3bb3d273d703f3e9976969a3ecd",
    ),
    "BadHersfeld/osm/obstacle_seilerweg.add.xml": (
        1_840,
        "897071fc8dbbb66894ad0e86420216881fd92096",
    ),
    "BadHersfeld/osm/osm.sumocfg": (
        3_683,
        "8f2280df80757186d8dcc7a74c78baad00ba7800",
    ),
    "BadHersfeld/osm/osm_activitygen.json": (
        4_506,
        "8114dc4f4875b714c7c67fadcb496caf42ec1962",
    ),
    "BadHersfeld/osm/osm_bbox.osm.xml.gz": (
        2_224_642,
        "9a0633102fbadbd14b6ade962f4c11bd2ae8f95a",
    ),
    "BadHersfeld/osm/osm_complete_parking_areas.add.xml": (
        59_921,
        "130c5a45350d2210fc369d73e093433854255f74",
    ),
    "BadHersfeld/osm/osm_edited.net.xml.gz": (
        3_618_753,
        "fb041f1f90977fb45bf33a5eb0a5366ba3ff3942",
    ),
    "BadHersfeld/osm/osm_parking_rerouters.add.xml": (
        403_750,
        "c81d09ad5cbcae92177b8e158fc658d00140188e",
    ),
    "BadHersfeld/osm/osm_polygons.add.xml": (
        3_857_756,
        "d37f38fd4605a47e678fe48b5ca9cbf0c8a71062",
    ),
    "BadHersfeld/osm/osm_taz.xml": (
        115_826,
        "9961900d3dcef27745d5a25155abf61e2f47aae5",
    ),
    "BadHersfeld/osm/osm_taz_weight.csv": (
        55,
        "752ff1f5c5bd1b7018dc4e2af285636ac97b0fa5",
    ),
    "BadHersfeld/osm/outputs/.gitkeep": (
        0,
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391",
    ),
    "BadHersfeld/osm/patched.netccfg": (
        991,
        "7feca335474aeb1905337e6e980803bfa638eddc",
    ),
    "BadHersfeld/osm/pt_vtypes.xml": (
        636,
        "a2269aab04324ee0c8a6fa8cda963f42a4814a48",
    ),
    "BadHersfeld/osm/run_activitygen.bat": (
        64,
        "602a30b00044d8b3a796d6d9985cc57bac7f2567",
    ),
    "BadHersfeld/osm/run_activitygen.sh": (
        77,
        "8bf357630371035ce35326b861fcf8a6f6fa2d27",
    ),
}


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> bool:
    insecure_retry = False
    try:
        response = urlopen(url, timeout=300)
    except URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        insecure_retry = True
        response = urlopen(
            url,
            timeout=300,
            context=ssl._create_unverified_context(),
        )
    with response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    return insecure_retry


def _is_xml(path: Path) -> bool:
    name = path.name
    return name.endswith((".xml", ".xml.gz", ".sumocfg", ".netccfg", ".polycfg"))


def _manifest(files: dict[str, dict[str, object]], *, transfer_mode: str) -> dict:
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "commit": COMMIT,
        "license": "EPL-2.0",
        "city": "Bad Hersfeld",
        "coverage": "complete tracked BadHersfeld directory",
        "scenarios": list(SCENARIOS),
        "file_count": len(files),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in files.values()),
        "transfer_mode": transfer_mode,
        "files": files,
    }


def acquire(*, output_root: Path) -> dict:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite Bad Hersfeld acquisition: {output_root}"
        )
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        files: dict[str, dict[str, object]] = {}
        insecure_retry_count = 0
        for relative, (expected_size, expected_blob) in FILES.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = "/".join(quote(part) for part in Path(relative).parts)
            url = f"{RAW_BASE}/{COMMIT}/{encoded}"
            insecure_retry_count += int(_download(url, destination))
            observed_size = destination.stat().st_size
            observed_blob = _git_blob_sha1(destination)
            if observed_size != expected_size or observed_blob != expected_blob:
                raise ValueError(
                    f"Bad Hersfeld source identity changed for {relative}: "
                    f"size={observed_size} blob={observed_blob}"
                )
            if _is_xml(destination):
                parse_xml(destination)
            files[relative] = {
                "url": url,
                "size_bytes": observed_size,
                "git_blob_sha1": observed_blob,
            }
        payload = _manifest(files, transfer_mode="https_to_local")
        payload["insecure_tls_retry_count"] = insecure_retry_count
        (staging / "acquisition_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _cluster_connection(
    *, node: str, scheduler_skill_dir: Path
) -> tuple[list[str], str]:
    sys.path.insert(0, str(scheduler_skill_dir.resolve()))
    import scheduler  # type: ignore[import-not-found]

    return (
        shlex.split(scheduler._ssh_rsync_shell_for_node(node)),
        str(scheduler._ssh_target_for_node(node)),
    )


def _remote_run(
    ssh: Sequence[str], target: str, command: str, *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*ssh, target, command],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _check_remote(result: subprocess.CompletedProcess[bytes], context: str) -> None:
    if result.returncode == 0:
        return
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    raise RuntimeError(f"{context} failed ({result.returncode}): {stderr}")


def _stream_remote_file(
    *,
    relative: str,
    expected_size: int,
    expected_blob: str,
    staging: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    destination = staging / relative
    partial = destination.parent / f".{destination.name}.part"
    encoded = "/".join(quote(part) for part in Path(relative).parts)
    url = f"{RAW_BASE}/{COMMIT}/{encoded}"
    command = (
        "set -e; "
        f"mkdir -p {shlex.quote(str(destination.parent))}; "
        f"test ! -e {shlex.quote(str(destination))}; "
        f"cat > {shlex.quote(str(partial))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    response = urlopen(url, timeout=300)
    transfer = subprocess.Popen(
        [*ssh, target, command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    observed_size = 0
    with response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            observed_size += len(block)
            digest.update(block)
            transfer.stdin.write(block)
    transfer.stdin.close()
    stdout = transfer.stdout.read() if transfer.stdout is not None else b""
    stderr = transfer.stderr.read() if transfer.stderr is not None else b""
    result = subprocess.CompletedProcess(
        transfer.args, transfer.wait(), stdout=stdout, stderr=stderr
    )
    _check_remote(result, f"Bad Hersfeld remote transfer for {relative}")
    if observed_size != expected_size or digest.hexdigest() != expected_blob:
        raise ValueError(
            f"Bad Hersfeld source identity changed for {relative}: "
            f"size={observed_size} blob={digest.hexdigest()}"
        )


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict:
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    initialization = _remote_run(
        ssh,
        target,
        (
            "set -e; "
            f"test ! -e {shlex.quote(str(output_root))}; "
            f"test ! -e {shlex.quote(str(staging))}; "
            f"mkdir -p {shlex.quote(str(staging))}"
        ),
    )
    _check_remote(initialization, "Bad Hersfeld remote acquisition initialization")
    try:
        for relative, (size, blob) in FILES.items():
            print(f"streaming {relative} ({size} bytes)", flush=True)
            for attempt in range(1, 5):
                try:
                    _stream_remote_file(
                        relative=relative,
                        expected_size=size,
                        expected_blob=blob,
                        staging=staging,
                        ssh=ssh,
                        target=target,
                    )
                    break
                except RuntimeError:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        files = {
            relative: {
                "url": (
                    f"{RAW_BASE}/{COMMIT}/"
                    + "/".join(quote(part) for part in Path(relative).parts)
                ),
                "size_bytes": size,
                "git_blob_sha1": blob,
            }
            for relative, (size, blob) in FILES.items()
        }
        payload = _manifest(
            files,
            transfer_mode="https_stream_to_shared_cluster_no_local_file",
        )
        payload["insecure_tls_retry_count"] = 0
        upload = _remote_run(
            ssh,
            target,
            "set -e; cat > "
            + shlex.quote(str(staging / "acquisition_manifest.json")),
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Bad Hersfeld acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Bad Hersfeld acquisition finalization")
        return payload
    except Exception:
        _remote_run(ssh, target, f"rm -rf -- {shlex.quote(str(staging))}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--remote-output-root")
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    if (args.output_root is None) == (args.remote_output_root is None):
        raise ValueError(
            "select exactly one local or remote Bad Hersfeld output root"
        )
    payload = (
        acquire(output_root=args.output_root)
        if args.output_root is not None
        else acquire_remote(
            output_root=PurePosixPath(args.remote_output_root),
            node=args.remote_node,
            scheduler_skill_dir=args.scheduler_skill_dir,
        )
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "commit": payload["commit"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "scenarios": payload["scenarios"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
