"""Download and probe RESCO SUMO traffic-signal benchmark scenarios.

This is a data-preparation/probe stage, not a CFCMT policy evaluation.  It
creates a reproducible local RESCO checkout, copies selected environments into
``H2Oplus/downloads``, unpacks route files referenced by ``.sumocfg`` when
needed, and validates that each scenario can be loaded by SUMO/libsumo.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_transfer_feasibility import _format_table


DEFAULT_REPO_URL = "https://github.com/Pi-Star-Lab/RESCO.git"
DEFAULT_OUT_ROOT = Path("H2Oplus/downloads/traffic_signal_resco")
DEFAULT_REPORT = Path("cf_h2o/results/traffic_signal_resco_probe.json")
DEFAULT_MD_REPORT = Path("cf_h2o/results/traffic_signal_resco_probe.md")
DEFAULT_SCENARIOS = ("cologne1", "cologne3", "cologne8", "ingolstadt1", "ingolstadt7")


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _ensure_resco_checkout(repo_url: str, repo_dir: Path) -> None:
    if (repo_dir / ".git").exists():
        _run(["git", "fetch", "--depth", "1", "origin"], cwd=repo_dir)
        _run(["git", "checkout", "HEAD", "--", "resco_benchmark/environments"], cwd=repo_dir)
        return
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", repo_url, str(repo_dir)])
    _run(["git", "sparse-checkout", "set", "resco_benchmark/environments"], cwd=repo_dir)


def _sumocfg_files(scenario_dir: Path) -> list[Path]:
    return sorted(scenario_dir.glob("*.sumocfg"))


def _sumocfg_input_files(sumocfg: Path) -> tuple[list[str], list[str]]:
    root = ET.parse(sumocfg).getroot()
    net_files: list[str] = []
    route_files: list[str] = []
    for item in root.findall(".//net-file"):
        value = item.attrib.get("value")
        if value:
            net_files.extend(part.strip() for part in value.split(",") if part.strip())
    for item in root.findall(".//route-files"):
        value = item.attrib.get("value")
        if value:
            route_files.extend(part.strip() for part in value.split(",") if part.strip())
    return net_files, route_files


def _extract_referenced_routes(scenario_dir: Path) -> list[str]:
    extracted: list[str] = []
    for sumocfg in _sumocfg_files(scenario_dir):
        _, route_files = _sumocfg_input_files(sumocfg)
        for route_name in route_files:
            route_path = scenario_dir / route_name
            if route_path.exists():
                continue
            for archive in scenario_dir.glob("*.zip"):
                with zipfile.ZipFile(archive) as handle:
                    if route_name in handle.namelist():
                        handle.extract(route_name, scenario_dir)
                        extracted.append(route_name)
                        break
    return extracted


def prepare_scenarios(
    *,
    repo_url: str,
    out_root: Path,
    scenarios: tuple[str, ...],
) -> dict[str, Any]:
    repo_dir = out_root / "repo"
    env_src_root = repo_dir / "resco_benchmark" / "environments"
    env_out_root = out_root / "environments"
    _ensure_resco_checkout(repo_url, repo_dir)
    env_out_root.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, Any]] = []
    for scenario in scenarios:
        src = env_src_root / scenario
        dst = env_out_root / scenario
        if not src.exists():
            copied.append({"scenario": scenario, "ok": False, "error": "missing in RESCO checkout"})
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        extracted = _extract_referenced_routes(dst)
        copied.append({"scenario": scenario, "ok": True, "path": str(dst), "extracted_routes": extracted})
    return {
        "repo_url": repo_url,
        "repo_dir": str(repo_dir),
        "environments_dir": str(env_out_root),
        "copied": copied,
    }


def _count_route_rows(route_path: Path) -> dict[str, int]:
    if not route_path.exists():
        return {"vehicles": 0, "trips": 0, "flows": 0, "routes": 0}
    root = ET.parse(route_path).getroot()
    return {
        "vehicles": len(root.findall("vehicle")),
        "trips": len(root.findall("trip")),
        "flows": len(root.findall("flow")),
        "routes": len(root.findall("route")),
    }


def _net_stats(net_path: Path) -> dict[str, Any]:
    root = ET.parse(net_path).getroot()
    tls = [item.attrib.get("id", "") for item in root.findall(".//tlLogic")]
    edges = [item for item in root.findall("edge") if "function" not in item.attrib]
    phases = [phase for phase in root.findall(".//tlLogic/phase")]
    return {
        "tls_count": len(tls),
        "tls_ids": tls,
        "edge_count": len(edges),
        "phase_count": len(phases),
    }


def _probe_with_libsumo(sumocfg: Path, *, steps: int, seed: int) -> dict[str, Any]:
    sumo_api = _load_libsumo()
    try:
        try:
            sumo_api.close()
        except Exception:
            pass
        sumo_api.start(
            [
                "sumo",
                "-c",
                str(sumocfg),
                "--seed",
                str(seed),
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
                "--duration-log.disable",
                "true",
            ]
        )
        begin_time = float(sumo_api.simulation.getTime())
        tls_ids = list(sumo_api.trafficlight.getIDList())
        min_expected_start = int(sumo_api.simulation.getMinExpectedNumber())
        departed = 0
        arrived = 0
        for _ in range(max(0, int(steps))):
            sumo_api.simulationStep()
            departed += int(sumo_api.simulation.getDepartedNumber())
            arrived += int(sumo_api.simulation.getArrivedNumber())
        return {
            "ok": True,
            "begin_time": begin_time,
            "end_time": float(sumo_api.simulation.getTime()),
            "tls_count_libsumo": len(tls_ids),
            "tls_ids_libsumo": tls_ids,
            "min_expected_start": min_expected_start,
            "vehicle_count_after_steps": int(sumo_api.vehicle.getIDCount()),
            "departed_in_probe": int(departed),
            "arrived_in_probe": int(arrived),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass


def probe_scenarios(
    *,
    env_root: Path,
    scenarios: tuple[str, ...],
    steps: int,
    seed: int,
    skip_sumo: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        scenario_dir = env_root / scenario
        sumocfgs = _sumocfg_files(scenario_dir)
        if not scenario_dir.exists() or not sumocfgs:
            rows.append({"scenario": scenario, "ok": False, "error": "missing scenario directory or sumocfg"})
            continue
        sumocfg = sumocfgs[0]
        net_names, route_names = _sumocfg_input_files(sumocfg)
        net_stats = _net_stats(scenario_dir / net_names[0]) if net_names else {}
        route_counts: dict[str, int] = {"vehicles": 0, "trips": 0, "flows": 0, "routes": 0}
        for route_name in route_names:
            counts = _count_route_rows(scenario_dir / route_name)
            route_counts = {key: route_counts[key] + counts[key] for key in route_counts}
        libsumo_probe = (
            {"ok": None, "skipped": True}
            if skip_sumo
            else _probe_with_libsumo(sumocfg, steps=steps, seed=seed + idx)
        )
        rows.append(
            {
                "scenario": scenario,
                "ok": bool(libsumo_probe.get("ok", False)) if not skip_sumo else True,
                "scenario_dir": str(scenario_dir),
                "sumocfg": str(sumocfg),
                "net_files": net_names,
                "route_files": route_names,
                **net_stats,
                **route_counts,
                "libsumo_probe": libsumo_probe,
            }
        )
    return rows


def write_markdown(result: dict[str, Any], path: Path) -> None:
    rows = [
        {
            "scenario": row["scenario"],
            "ok": row["ok"],
            "tls_count": row.get("tls_count", 0),
            "trips": row.get("trips", 0),
            "vehicles": row.get("vehicles", 0),
            "flows": row.get("flows", 0),
            "min_expected_start": row.get("libsumo_probe", {}).get("min_expected_start", ""),
            "departed_probe": row.get("libsumo_probe", {}).get("departed_in_probe", ""),
        }
        for row in result["probes"]
    ]
    lines = [
        "# RESCO SUMO Benchmark Probe",
        "",
        f"Source: {result['prepare']['repo_url']}",
        "",
        "This probe downloads selected RESCO scenarios, unpacks referenced route files, and verifies SUMO/libsumo startup. It does not run CFCMT policy evaluation yet.",
        "",
        _format_table(rows, ["scenario", "ok", "tls_count", "trips", "vehicles", "flows", "min_expected_start", "departed_probe"]),
        "",
        "## Next Integration Step",
        "",
        "Use the probed TLS phase programs directly instead of the synthetic NS/EW phase assumption. For arbitrary RESCO intersections, actions should select among existing green phases from the SUMO program logic.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_probe(
    *,
    repo_url: str = DEFAULT_REPO_URL,
    out_root: Path = DEFAULT_OUT_ROOT,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    steps: int = 30,
    seed: int = 73,
    skip_sumo: bool = False,
) -> dict[str, Any]:
    prepare = prepare_scenarios(repo_url=repo_url, out_root=out_root, scenarios=scenarios)
    probes = probe_scenarios(
        env_root=Path(prepare["environments_dir"]),
        scenarios=scenarios,
        steps=steps,
        seed=seed,
        skip_sumo=skip_sumo,
    )
    return {
        "experiment": "traffic_signal_resco_probe",
        "prepare": prepare,
        "setting": {
            "steps": int(steps),
            "seed": int(seed),
            "skip_sumo": bool(skip_sumo),
            "scenarios": list(scenarios),
        },
        "probes": probes,
    }


def _parse_scenarios(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--scenarios", type=_parse_scenarios, default=DEFAULT_SCENARIOS)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--skip-sumo", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_REPORT)
    args = parser.parse_args(argv)

    result = run_probe(
        repo_url=args.repo_url,
        out_root=args.out_root,
        scenarios=args.scenarios,
        steps=args.steps,
        seed=args.seed,
        skip_sumo=args.skip_sumo,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    ok = sum(1 for row in result["probes"] if row.get("ok"))
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    print(f"ok={ok}/{len(result['probes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
