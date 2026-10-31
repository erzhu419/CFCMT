"""Check registered identities, original evaluator arguments and raw-result reuse."""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

SPEC = importlib.util.spec_from_file_location("fresh_dense_run", Path(__file__).with_name("run.py"))
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "HERE", tmp_path)
    (tmp_path / "protocol.json").write_text(json.dumps(PROTOCOL))
    (tmp_path / "evaluation_protocol.json").write_text("{}")
    module = ModuleType("cf_h2o.eval.traffic_signal_external_closed_loop_confirmation")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_registered_eighty_identities_keep_all_ten_seed_blocks():
    roster = driver.identities(PROTOCOL)
    assert len(roster) == len(set(roster)) == 80
    assert roster[:2] == [("los_angeles", "la_1x4", PROTOCOL["seeds"][0], policy)
        for policy in PROTOCOL["policies"]]
    assert [row[2] for row in roster[::8]] == PROTOCOL["seeds"]
    assert all(sum(row[2] == seed for row in roster) == 8 for seed in PROTOCOL["seeds"])


def test_original_parameters_pass_unchanged_and_complete_raw_is_reused(tmp_path, monkeypatch):
    module = fixture(tmp_path, monkeypatch)
    calls = []

    def evaluate(**kwargs):
        calls.append(kwargs)
        kwargs["tripinfo_out"].write_text("<tripinfos/>")
        return dict(protocol="tsc-v43r39-external-closed-loop-rollout-v4",
            city="los_angeles", scenario=kwargs["scenario"], seed=kwargs["seed"], policy=kwargs["policy"],
            runtime_policy="causal_anchored_blend_contrast_raw",
            metrics=dict(ok=True, mean_tripinfo_waiting_time=123.), sumo_version="1.22.0",
            evaluation_horizon_sec=3600, prediction_horizon_sec=60, control_interval_sec=10,
            warmup_sec=60, selected_candidate="local_cap_1_conf_0p25",
            model_artifacts={key: dict(sha256=key + "_existing") for key in ("method_model", "baseline_model")})

    module.run_external_closed_loop_rollout = evaluate
    result = driver.run(0)
    assert len(calls) == 1 and not result["reused_complete_raw_result"]
    kwargs = calls[0]
    assert set(kwargs) == {*driver.ARGUMENTS.values(), "protocol_spec_path", "scenario", "seed", "policy", "tripinfo_out"}
    for flag, name in driver.ARGUMENTS.items():
        expected = PROTOCOL["frozen_runtime_arguments"][flag]
        assert kwargs[name] == (Path(expected) if name in driver.PATH_ARGUMENTS else expected)
    assert kwargs["protocol_spec_path"] == tmp_path / "evaluation_protocol.json"
    assert result["analysis_protocol"] == PROTOCOL["name"]
    assert result["model_artifacts"]["method_model"]["sha256"] == "method_model_existing"
    assert driver.run(0)["reused_complete_raw_result"] and len(calls) == 1
    assert json.loads((tmp_path / "summaries/run_000/summary.json").read_text())["complete"]


def test_partial_raw_does_not_call_simulator_or_overwrite_failure(tmp_path, monkeypatch):
    module = fixture(tmp_path, monkeypatch)
    module.run_external_closed_loop_rollout = lambda **kwargs: pytest.fail("Partial raw evidence must not be rerun")
    raw = tmp_path / "raw/la_1x4" / f"seed_{PROTOCOL['seeds'][0]}" / PROTOCOL["policies"][0]
    raw.mkdir(parents=True)
    (raw / "tripinfo.xml").write_text("<tripinfos/>")
    with pytest.raises(RuntimeError, match="Partial"):
        driver.run(0)
    error = tmp_path / "summaries/run_000/error.json"
    first = error.read_text()
    with pytest.raises(RuntimeError, match="Partial"):
        driver.run(0)
    assert error.read_text() == first
    assert (raw / "tripinfo.xml").read_text() == "<tripinfos/>"
