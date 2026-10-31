"""Copy the seven asset-side simulator modules once, without touching Thesis."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent


def main():
    marker = HERE / "runtime_supplement" / "snapshot_ready.json"
    if marker.exists():
        print(marker.read_text())
        return
    runtime = json.loads((HERE / "runtime.json").read_text())
    source = Path(runtime["frozen_snapshot"]) / "assets/SUMO_ruiguang/online_control/sim_obj"
    destination = HERE / "runtime_supplement/sim_obj"
    modules = sorted(source.glob("*.py"))
    if not modules:
        raise FileNotFoundError(source)
    destination.mkdir(parents=True, exist_ok=True)
    for path in modules:
        shutil.copy2(path, destination / path.name)
    record = dict(source=str(source), copied_modules=[p.name for p in modules],
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  scope="Owned copies of asset-side simulator code; no mutable Thesis code imported")
    marker.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
