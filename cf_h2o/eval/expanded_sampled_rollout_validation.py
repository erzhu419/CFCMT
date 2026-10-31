"""Expanded BusSimEnv sampled rollout for control-facing diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.paper_experiment_suite import (
    POLICY_ACTIONS,
    _parse_float_list,
    _read_json,
    _repo_root,
    _resolve_path,
    build_all_city_stats,
    run_sampled_rollout,
)


def run(args: argparse.Namespace) -> dict:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    t0 = time.time()
    city_stats, _, sanity = build_all_city_stats(config, root, args)
    rollout = run_sampled_rollout(
        city_stats,
        config,
        sanity,
        root,
        args.ridge,
        args.policy_actions,
        args.rollout_lines_per_city,
        args.rollout_max_decisions,
        args.seed,
        source_weight_temperature=args.source_weight_temperature,
        source_weight_floor=args.source_weight_floor,
    )
    rollout["elapsed_total_sec"] = time.time() - t0
    rollout["expanded_rollout"] = True
    return rollout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/expanded_sampled_rollout_validation.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--actions", type=_parse_float_list, default=[0.0, 30.0])
    parser.add_argument("--policy-actions", type=_parse_float_list, default=list(POLICY_ACTIONS))
    parser.add_argument("--max-lines-per-city", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout-lines-per-city", type=int, default=20)
    parser.add_argument("--rollout-max-decisions", type=int, default=240)
    parser.add_argument("--source-weight-temperature", type=float, default=1.0)
    parser.add_argument("--source-weight-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = SimpleNamespace(**vars(parse_args(argv)))
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({
            "ok": result["ok"],
            "elapsed_total_sec": result["elapsed_total_sec"],
            "summary_by_policy": result["summary_by_policy"],
        }, indent=2))
    else:
        print(text)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
