"""Frozen rule cases and small JSON output for the fixed-rule supplement."""

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_text())
SNAPSHOT = HERE / P["execution_code_snapshot"]
REFERENCE = HERE / P["reference_outputs"]


def module(name, path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def case_spec(case):
    return dict(method=P["rule"], execution_layer=P["execution_layer"],
        demand_scale=P["evaluation_demand_scales"][case // 5],
        **P["evaluation_blocks"][case % 5], case_id=f"case{case:02d}")


def result_path(case, smoke=False, root=HERE):
    return root / ("rule_smoke" if smoke else "rules") / f"case{case:02d}" / "results" / f"case{case:02d}.json"
