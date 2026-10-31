"""Own the two completed MC-WM mathematical dependencies before training."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
SOURCES = {
    "qdelta_bellman.py": Path("/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_bus/runtime/MC-WM/mc_wm/policy/qdelta_bellman.py"),
    "mechanisms.py": Path("/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_usage_v3/mechanisms.py"),
}


def main():
    destination = HERE / "runtime_train_snapshot"
    marker = destination / "snapshot_ready.json"
    if marker.exists():
        print(marker.read_text(), flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        shutil.copy2(source, destination / name)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {name: str(path) for name, path in SOURCES.items()},
        "scope": "Independent mathematical definitions; no mutable Thesis code imported by training",
    }
    marker.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
