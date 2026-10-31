"""Cross-city compatibility validation for generated open-transit city bundles.

This script deliberately checks the data/config/interface layer only. It does
not train or evaluate H2O+/CFCMT policies, so its result should not be reported
as cross-city performance validation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_LINE_FILES = {
    "stop_news.xlsx",
    "route_news.xlsx",
    "time_table.xlsx",
    "passenger_OD.xlsx",
}


@dataclass(frozen=True)
class CityValidation:
    key: str
    city: str
    env_path: Path
    ok: bool
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _line_dirs(env_path: Path) -> list[Path]:
    data_dir = env_path / "data"
    if not data_dir.exists():
        return []
    return [
        child
        for child in sorted(data_dir.iterdir())
        if child.is_dir() and REQUIRED_LINE_FILES.issubset({item.name for item in child.iterdir()})
    ]


def _city_manifest(env_path: Path) -> tuple[Path | None, dict[str, Any]]:
    for name in ("gtfs_city_manifest.json", "lta_city_manifest.json"):
        path = env_path / name
        if path.exists():
            return path, _read_json(path)
    return None, {}


def _sum_manifest_lines(manifest: dict[str, Any], field: str) -> int | None:
    lines = manifest.get("lines")
    if not isinstance(lines, list):
        return None
    return sum(int(item.get(field, 0)) for item in lines)


def validate_city(root: Path, key: str, spec: dict[str, Any]) -> CityValidation:
    errors: list[str] = []
    warnings: list[str] = []
    env_path = _resolve_path(root, str(spec["env_path"]))
    metrics: dict[str, Any] = {
        "city": spec.get("city", key),
        "env_path": str(env_path),
    }

    if not env_path.exists():
        return CityValidation(
            key=key,
            city=str(spec.get("city", key)),
            env_path=env_path,
            ok=False,
            errors=[f"env_path does not exist: {env_path}"],
            warnings=warnings,
            metrics=metrics,
        )

    for required in ("config.json", "data"):
        if not (env_path / required).exists():
            errors.append(f"missing required {required}")

    line_dirs = _line_dirs(env_path)
    expected_lines = int(spec.get("line_count", 0))
    metrics["line_dirs"] = len(line_dirs)
    metrics["expected_lines"] = expected_lines
    if expected_lines and len(line_dirs) != expected_lines:
        errors.append(f"line directory count {len(line_dirs)} != configured line_count {expected_lines}")

    manifest_path, manifest = _city_manifest(env_path)
    metrics["manifest_path"] = str(manifest_path) if manifest_path else None
    if not manifest:
        errors.append("missing gtfs_city_manifest.json/lta_city_manifest.json")
    else:
        manifest_line_count = int(manifest.get("line_count", -1))
        manifest_failure_count = int(manifest.get("failure_count", 0) or 0)
        metrics["manifest_line_count"] = manifest_line_count
        metrics["manifest_failure_count"] = manifest_failure_count
        if manifest_failure_count:
            warnings.append(f"manifest reports {manifest_failure_count} route/direction conversion failures")
        if expected_lines and manifest_line_count != expected_lines:
            errors.append(f"manifest line_count {manifest_line_count} != configured line_count {expected_lines}")
        for manifest_field, spec_field in (
            ("stops", "stations"),
            ("segments", "segments"),
            ("timetable_rows", "timetables"),
        ):
            manifest_total = _sum_manifest_lines(manifest, manifest_field)
            configured_total = int(spec.get(spec_field, 0))
            metrics[f"manifest_{spec_field}"] = manifest_total
            metrics[f"expected_{spec_field}"] = configured_total
            if manifest_total is not None and configured_total and manifest_total != configured_total:
                errors.append(
                    f"manifest {spec_field} total {manifest_total} != configured {spec_field} {configured_total}"
                )

    smoke_path = env_path / str(spec.get("validation_smoke", "validation_smoke.json"))
    metrics["validation_smoke_path"] = str(smoke_path)
    if not smoke_path.exists():
        errors.append(f"missing validation smoke result: {smoke_path}")
    else:
        smoke = _read_json(smoke_path)
        metrics.update(
            {
                "smoke_state_dim": smoke.get("state_dim"),
                "smoke_first_obs_dim": smoke.get("first_obs_dim"),
                "smoke_feature_nodes": smoke.get("feature_nodes"),
                "smoke_hard_mask_edges": smoke.get("hard_mask_edges"),
                "smoke_lines": smoke.get("lines"),
                "smoke_lines_loaded": smoke.get("lines_loaded", smoke.get("lines")),
                "smoke_route_graph_stations": smoke.get("route_graph_stations"),
                "smoke_route_graph_segments": smoke.get("route_graph_segments"),
                "smoke_first_decision_time": smoke.get("first_decision_time"),
            }
        )
        if int(smoke.get("state_dim", -1)) != 15:
            errors.append(f"smoke state_dim {smoke.get('state_dim')} != 15")
        if int(smoke.get("first_obs_dim", -1)) != 15:
            errors.append(f"smoke first_obs_dim {smoke.get('first_obs_dim')} != 15")
        if int(smoke.get("feature_nodes", 0)) <= 0:
            errors.append("smoke feature_nodes is empty")
        if int(smoke.get("hard_mask_edges", 0)) <= 0:
            errors.append("smoke hard_mask_edges is empty")
        if int(smoke.get("route_graph_stations", 0)) <= 0:
            errors.append("smoke route_graph_stations is empty")
        if int(smoke.get("route_graph_segments", 0)) <= 0:
            errors.append("smoke route_graph_segments is empty")

        loaded_lines = int(metrics["smoke_lines_loaded"] or 0)
        smoke_lines = int(metrics["smoke_lines"] or 0)
        if smoke_lines and loaded_lines < smoke_lines:
            warnings.append(f"dynamic smoke loaded {loaded_lines}/{smoke_lines} lines")

    return CityValidation(
        key=key,
        city=str(spec.get("city", key)),
        env_path=env_path,
        ok=not errors,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def expand_splits(config: dict[str, Any]) -> list[dict[str, Any]]:
    env_keys = list(config["generated_envs"].keys())
    expanded: list[dict[str, Any]] = []
    for split in config.get("cross_city_splits", []):
        source_envs = split.get("source_envs")
        if source_envs == "all_except_target":
            for target in split.get("target_envs", []):
                expanded.append(
                    {
                        "name": f"{split['name']}::{target}",
                        "source_envs": [key for key in env_keys if key != target],
                        "target_env": target,
                    }
                )
        else:
            expanded.append(
                {
                    "name": split["name"],
                    "source_envs": list(source_envs or []),
                    "target_env": split["target_env"],
                }
            )
    return expanded


def validate_split(split: dict[str, Any], cities: dict[str, CityValidation]) -> dict[str, Any]:
    errors: list[str] = []
    source_envs = list(split["source_envs"])
    target_env = str(split["target_env"])
    missing = [key for key in source_envs + [target_env] if key not in cities]
    if missing:
        errors.append(f"unknown env keys: {missing}")
        return {**split, "ok": False, "errors": errors}
    if target_env in source_envs:
        errors.append(f"target_env {target_env} is also in source_envs")

    participants = [cities[key] for key in source_envs + [target_env]]
    for city in participants:
        if not city.ok:
            errors.append(f"{city.key} failed city validation")

    compatibility_fields = [
        "smoke_state_dim",
        "smoke_first_obs_dim",
        "smoke_feature_nodes",
        "smoke_hard_mask_edges",
    ]
    compatibility: dict[str, list[Any]] = {}
    for field in compatibility_fields:
        values = [city.metrics.get(field) for city in participants]
        compatibility[field] = values
        if len(set(values)) != 1:
            errors.append(f"incompatible {field}: {values}")

    source_totals = {
        "lines": sum(int(cities[key].metrics.get("expected_lines") or 0) for key in source_envs),
        "stations": sum(int(cities[key].metrics.get("expected_stations") or 0) for key in source_envs),
        "segments": sum(int(cities[key].metrics.get("expected_segments") or 0) for key in source_envs),
        "timetables": sum(int(cities[key].metrics.get("expected_timetables") or 0) for key in source_envs),
    }
    target_metrics = cities[target_env].metrics
    target_totals = {
        "lines": int(target_metrics.get("expected_lines") or 0),
        "stations": int(target_metrics.get("expected_stations") or 0),
        "segments": int(target_metrics.get("expected_segments") or 0),
        "timetables": int(target_metrics.get("expected_timetables") or 0),
    }

    return {
        **split,
        "ok": not errors,
        "errors": errors,
        "compatibility": compatibility,
        "source_totals": source_totals,
        "target_totals": target_totals,
    }


def run(config_path: Path) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, str(config_path)))
    city_results = {
        key: validate_city(root, key, spec)
        for key, spec in config.get("generated_envs", {}).items()
    }
    split_results = [validate_split(split, city_results) for split in expand_splits(config)]
    ok = all(city.ok for city in city_results.values()) and all(split["ok"] for split in split_results)
    return {
        "ok": ok,
        "validation_level": "cross_city_manifest_and_interface_compatibility",
        "performance_validation": False,
        "config": str(_resolve_path(root, str(config_path))),
        "cities": {
            key: {
                "ok": city.ok,
                "city": city.city,
                "errors": city.errors,
                "warnings": city.warnings,
                "metrics": city.metrics,
            }
            for key, city in city_results.items()
        },
        "splits": split_results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("cf_h2o/config/cross_city_open_transit.json"),
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args.config)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
