#!/usr/bin/env python3
"""Audit that preregistered seeds have no prior simulator-seed token."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "cfcmt-simulator-seed-token-novelty-audit-v1"


def audit_seed_novelty(
    *, seeds: Sequence[int], roots: Sequence[Path]
) -> dict[str, Any]:
    normalized = tuple(int(value) for value in seeds)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("seed novelty audit requires unique seeds")
    alternatives = "|".join(re.escape(str(seed)) for seed in normalized)
    pattern = rf"seed[_:]?({alternatives})([^0-9]|$)"
    command = [
        "rg",
        "-n",
        "--no-heading",
        "--color=never",
        pattern,
        *(str(Path(root)) for root in roots),
    ]
    completed = subprocess.run(
        command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(f"seed novelty rg failed: {completed.stderr}")
    hits = [line for line in completed.stdout.splitlines() if line.strip()]
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not hits else "FAIL",
        "claim_boundary": (
            "This proves absence of literal simulator-seed tokens in the scanned "
            "workspace roots before protocol generation. It does not claim that the "
            "same integer never appeared as a metric, timestamp, or vehicle ID."
        ),
        "seeds": list(normalized),
        "roots": [str(Path(root)) for root in roots],
        "pattern": pattern,
        "returncode": int(completed.returncode),
        "hit_count": len(hits),
        "hits": hits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--roots",
        type=Path,
        nargs="+",
        default=(Path("cf_h2o"), Path("scripts"), Path("paper"), Path("paper_tsc")),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite seed novelty audit: {args.out}")
    payload = audit_seed_novelty(seeds=args.seeds, roots=args.roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "hit_count": payload["hit_count"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
