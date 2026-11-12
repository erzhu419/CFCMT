import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "recovery", HERE.parents[1] / "scripts/cluster/recover_tsc_v93_fresh_confirmation_shard.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def inputs():
    manifest = json.loads((HERE / "recovery_manifest.json").read_text())
    first = json.loads((HERE / "evaluation/interrupted_shard_4_first_attempt.json").read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    shard = [(s, seed, method) for scenarios in protocol["city_scenarios"].values()
             for s in scenarios for seed in protocol["confirmation_seeds"]
             for method in protocol["methods"]][4::5]
    return manifest, first, shard


def test_only_four_interrupted_and_59_unstarted_are_executed():
    manifest, first, shard = inputs()
    selected = recovery.recovery_identities(manifest, first, shard)
    assert len(selected) == len(set(selected)) == 63
    assert set(selected).isdisjoint(map(recovery.identity, first["completed"]))
    assert sum(row["execution_kind"] == "technical_retry" for row in manifest["execute"]) == 4


@pytest.mark.parametrize("replacement", ["completed", "substitute_seed"])
def test_completed_or_substitute_identity_is_rejected(replacement):
    manifest, first, shard = inputs()
    if replacement == "completed":
        manifest["execute"][0].update({k: first["completed"][0][k] for k in ("scenario", "seed", "method")})
    else:
        manifest["execute"][0]["seed"] = 1
    with pytest.raises(ValueError, match="exactly the 4 interrupted"):
        recovery.recovery_identities(manifest, first, shard)
