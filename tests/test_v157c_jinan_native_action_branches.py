from __future__ import annotations

import inspect
import json
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_feature_aligned_b100_runtime_freeze as v157b
from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as resco_v3
from cf_h2o.eval import traffic_signal_v157c_jinan_native_action_branches as v157c


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "cf_h2o/config/traffic_signal_tsc_v157c_jinan_native_action_branches.json"
)
AUTHORIZATION = (
    PROJECT_ROOT
    / "cf_h2o/results/paper_artifacts/"
    "tsc_v157a_feature_aligned_target_budget_source_value_curve.json"
)


class FakeModel:
    def predict(self, dataset):
        return object()


class FakeExecutor:
    def __init__(self, *states: str):
        self.states = tuple(states)

    def feasible_states_now(self):
        return self.states


def dataset(tls_id: str, time_sec: float):
    return SimpleNamespace(
        size=3,
        metadata={
            "row_tls": [tls_id] * 3,
            "row_times": [time_sec] * 3,
            "candidate_states": ["PP", "A", "B"],
        },
    )


def decision(
    *, arm: str, tls_id: str, selected: str, priority: float, time_sec: int = 300
):
    states = {"PP": 0, "A": 1, "B": 2}
    selected_index = states[selected]
    return v157c.V157CActionDecision(
        reference_index=0,
        proposed_index=selected_index,
        selected_index=selected_index,
        learned_differs=selected != "PP",
        eligible=selected != "PP",
        priority=priority,
        rejection=None,
        arm=arm,
        tls_id=tls_id,
        time_sec=float(time_sec),
        proposed_state=selected,
        selected_state=selected,
        reference_state="PP",
        predicted_score=-priority,
        reference_score=0.0,
    ).to_dict()


def checkpoint_rows(tls_id: str, source: str, target: str, placebo: str, priority=1.0):
    return {
        "target_only": decision(
            arm="target_only", tls_id=tls_id, selected=target, priority=priority / 2
        ),
        "uniform_source": decision(
            arm="uniform_source", tls_id=tls_id, selected=source, priority=priority
        ),
        "source_label_placebo": decision(
            arm="source_label_placebo",
            tls_id=tls_id,
            selected=placebo,
            priority=priority / 3,
        ),
    }


def test_protocol_freezes_direct_v123_estimand_roster():
    protocol = json.loads(CONFIG.read_text())
    v157c.validate_protocol(protocol)
    assert protocol["scenario"] == "jinan_3x4_real"
    assert protocol["seeds"] == [80314, 88625, 27178]
    assert protocol["checkpoints_sec"] == [300, 480, 660]
    assert protocol["execution"]["total_native_runs"] == 30
    assert protocol["branch_estimand"].startswith("one_focal_tls_action")
    assert protocol["same_state_contract"]["snapshot_restore_used"] is False
    assert protocol["focal_tls_rule"] == v157c.FOCAL_TLS_RULE
    assert protocol["same_state_contract"]["action_roster"] == v157c.ACTION_ROSTER
    assert protocol["same_state_contract"]["action_eligibility"] == (
        "executor_feasible_states_now_count_greater_than_1_at_checkpoint"
    )
    assert protocol["same_state_contract"]["zero_action_eligible_checkpoint"] == (
        "invalid_without_checkpoint_shift_or_drop"
    )
    assert protocol["runtime_arm_semantics"] == {
        "target_only_training_domain_handling": "relabel_all_b100_rows_to_jinan",
        "v157a_exact_target_refit_used_at_runtime": False,
        "uniform_source_training_path": "exact_v157a_source_component_refits",
    }


def test_v157c_runtime_protocols_match_v157b_v2():
    assert v157c.RUNTIME_BUNDLE_PROTOCOL == v157b.RUNTIME_BUNDLE_PROTOCOL
    assert v157c.ARM_MANIFEST_PROTOCOL == v157b.ARM_MANIFEST_PROTOCOL


def test_runtime_model_shell_is_defined_locally_with_phase_pressure_prior():
    family = v157c.CFCMT_CORE_FAMILY_V3
    assert v157c.RUNTIME_POLICY == f"{family}_contrast_source_utility"
    originators = (
        v157c.RosterProbeOriginator(
            {arm: FakeModel() for arm in v157c.LEARNED_ARMS}
        ),
        v157c.OneActionBranchOriginator(
            model=FakeModel(),
            arm="target_only",
            checkpoint_sec=300,
            focal_tls="tls",
            expected_roster={},
        ),
    )
    for originator in originators:
        models = v157c._runtime_models(originator, prediction_horizon_sec=450)
        assert models.family_models == {family: None}
        assert models.prior_policy == "phase_pressure"
        assert models.prior_spec.key == "phase_pressure"
        assert models.prediction_horizon_sec == 450
        assert models.objective_modes == {family: "control_only"}
        assert models.action_originator is originator
        assert models.diagnostics == {"runtime": originator.selection_layer}


def test_recursive_runner_import_excludes_legacy_closed_loop_module():
    legacy = "cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop"
    program = (
        "import sys; "
        "import cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches; "
        f"assert {legacy!r} not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_focal_selection_uses_prediction_disagreement_without_outcomes():
    roster = {
        "tls_b": checkpoint_rows("tls_b", source="B", target="A", placebo="PP", priority=2.0),
        "tls_a": checkpoint_rows("tls_a", source="B", target="A", placebo="A", priority=2.0),
        "tls_c": checkpoint_rows("tls_c", source="A", target="A", placebo="B", priority=9.0),
    }
    selected = v157c.select_focal_tls(roster)
    assert selected["focal_tls"] == "tls_a"
    assert selected["source_target_disagreement_tls_count"] == 2
    assert selected["source_target_selected_actions_differ"] is True
    assert selected["outcomes_used_for_selection"] is False


def test_no_disagreement_is_retained_as_zero_action_contrast():
    roster = {
        "tls_a": checkpoint_rows("tls_a", source="A", target="A", placebo="B", priority=1.0),
        "tls_b": checkpoint_rows("tls_b", source="B", target="B", placebo="PP", priority=3.0),
    }
    selected = v157c.select_focal_tls(roster)
    assert selected["focal_tls"] == "tls_b"
    assert selected["selection_basis"] == "no_source_target_action_disagreement_zero_primary_contrast"
    assert selected["selected_actions"]["uniform_source"] == selected["selected_actions"]["target_only"]


def test_checkpoint_roster_excludes_tls_that_contrast_proposal_cannot_score(
    monkeypatch,
):
    states = {
        "tls_a": SimpleNamespace(info=object(), sim_time=300.0),
        "tls_b": SimpleNamespace(info=object(), sim_time=300.0),
        "tls_single": SimpleNamespace(info=object(), sim_time=300.0),
    }
    executors = {
        "tls_a": FakeExecutor("PP", "A"),
        "tls_b": FakeExecutor("PP", "B"),
        "tls_single": FakeExecutor("PP"),
    }
    assert v157c._checkpoint_scorable_tls_ids(states, executors) == (
        "tls_a",
        "tls_b",
    )

    originator = SimpleNamespace(
        selection_layer="test",
        select=lambda *_args, **_kwargs: pytest.fail(
            "single-candidate TLS must bypass the action originator"
        ),
    )
    models = v157c._runtime_models(originator, prediction_horizon_sec=450)
    only_candidate = SimpleNamespace(state="PP")
    monkeypatch.setattr(
        resco_v3,
        "_absolute_target_candidates",
        lambda *_args, **_kwargs: ((only_candidate,), object()),
    )
    proposal = resco_v3._contrast_proposal(
        model=None,
        family=v157c.CFCMT_CORE_FAMILY_V3,
        models=models,
        state=states["tls_single"],
        executor=executors["tls_single"],
        control_interval_sec=10,
        guarded=False,
        regularized=False,
    )
    assert proposal.prior_candidate is only_candidate
    assert proposal.selected_candidate is only_candidate
    assert proposal.originator_diagnostics is None


def test_probe_finalizes_only_checkpoint_scorable_tls(monkeypatch):
    monkeypatch.setattr(v157c, "_controlled_lane_set", lambda _infos: {"lane"})
    monkeypatch.setattr(
        v157c,
        "_group_adjusted_scores",
        lambda *_args, **_kwargs: (
            np.asarray([0.0, -2.0, -1.0]),
            None,
            None,
            None,
        ),
    )
    probe = v157c.RosterProbeOriginator(
        {arm: FakeModel() for arm in v157c.LEARNED_ARMS}
    )
    executors = {
        "tls_a": FakeExecutor("PP", "A"),
        "tls_b": FakeExecutor("PP", "B"),
        "tls_single": FakeExecutor("PP"),
    }
    for checkpoint in v157c.CHECKPOINTS_SEC:
        states = {
            tls_id: SimpleNamespace(info=object(), sim_time=float(checkpoint))
            for tls_id in executors
        }
        probe.prepare_interval(states=states, executors=executors)
        probe.select(dataset("tls_a", checkpoint), reference_index=0)
        probe.select(dataset("tls_b", checkpoint), reference_index=0)

    roster = probe.finalized_roster()
    assert probe.tls_ids == ("tls_a", "tls_b", "tls_single")
    assert probe.scorable_tls_ids == {
        checkpoint: ("tls_a", "tls_b")
        for checkpoint in v157c.CHECKPOINTS_SEC
    }
    assert all(tuple(rows) == ("tls_a", "tls_b") for rows in roster.values())


def test_branch_validates_same_scorable_roster_while_ignoring_single_candidate_tls(
    monkeypatch,
):
    monkeypatch.setattr(v157c, "_controlled_lane_set", lambda _infos: {"lane"})
    monkeypatch.setattr(
        v157c,
        "_group_adjusted_scores",
        lambda *_args, **_kwargs: (
            np.asarray([0.0, -2.0, -1.0]),
            None,
            None,
            None,
        ),
    )
    expected = {
        tls_id: {
            arm: decision(
                arm=arm,
                tls_id=tls_id,
                selected="A",
                priority=2.0,
            )
            for arm in v157c.LEARNED_ARMS
        }
        for tls_id in ("tls_a", "tls_b")
    }
    branch = v157c.OneActionBranchOriginator(
        model=FakeModel(),
        arm="uniform_source",
        checkpoint_sec=300,
        focal_tls="tls_a",
        expected_roster=expected,
    )
    states = {
        "tls_a": SimpleNamespace(info=object(), sim_time=300.0),
        "tls_b": SimpleNamespace(info=object(), sim_time=300.0),
        "tls_single": SimpleNamespace(info=object(), sim_time=300.0),
    }
    executors = {
        "tls_a": FakeExecutor("PP", "A"),
        "tls_b": FakeExecutor("PP", "B"),
        "tls_single": FakeExecutor("PP"),
    }
    branch.prepare_interval(states=states, executors=executors)
    branch.select(dataset("tls_a", 300), reference_index=0)
    branch.select(dataset("tls_b", 300), reference_index=0)
    branch.validate_checkpoint_roster()
    assert branch.scorable_tls_ids == ("tls_a", "tls_b")
    assert "tls_single" not in branch.checkpoint_decisions


def test_branch_rejects_checkpoint_scorable_roster_change(monkeypatch):
    monkeypatch.setattr(v157c, "_controlled_lane_set", lambda _infos: {"lane"})
    expected = {
        tls_id: checkpoint_rows(tls_id, source="A", target="A", placebo="A")
        for tls_id in ("tls_a", "tls_b")
    }
    branch = v157c.OneActionBranchOriginator(
        model=FakeModel(),
        arm="uniform_source",
        checkpoint_sec=300,
        focal_tls="tls_a",
        expected_roster=expected,
    )
    states = {
        tls_id: SimpleNamespace(info=object(), sim_time=300.0)
        for tls_id in ("tls_a", "tls_b")
    }
    branch.prepare_interval(
        states=states,
        executors={
            "tls_a": FakeExecutor("PP"),
            "tls_b": FakeExecutor("PP", "B"),
        },
    )
    with pytest.raises(ValueError, match="scorable TLS roster changed"):
        branch.validate_checkpoint_roster()


def test_probe_rejects_checkpoint_with_zero_scorable_tls(monkeypatch):
    monkeypatch.setattr(v157c, "_controlled_lane_set", lambda _infos: {"lane"})
    probe = v157c.RosterProbeOriginator(
        {arm: FakeModel() for arm in v157c.LEARNED_ARMS}
    )
    executors = {"tls_single": FakeExecutor("PP")}
    for checkpoint in v157c.CHECKPOINTS_SEC:
        probe.prepare_interval(
            states={
                "tls_single": SimpleNamespace(
                    info=object(), sim_time=float(checkpoint)
                )
            },
            executors=executors,
        )
    with pytest.raises(ValueError, match="checkpoint 300 has no scorable TLS"):
        probe.finalized_roster()


def test_branch_originator_exposes_only_focal_checkpoint_action(monkeypatch):
    def adjusted(_dataset, _prediction, *, objective_mode):
        assert objective_mode == "control_only"
        return np.asarray([0.0, -2.0, -1.0]), None, None, None

    monkeypatch.setattr(v157c, "_group_adjusted_scores", adjusted)
    expected = {
        tls_id: {
            arm: decision(
                arm=arm,
                tls_id=tls_id,
                selected="A",
                priority=2.0,
            )
            for arm in v157c.LEARNED_ARMS
        }
        for tls_id in ("tls_a", "tls_b")
    }
    originator = v157c.OneActionBranchOriginator(
        model=FakeModel(),
        arm="uniform_source",
        checkpoint_sec=300,
        focal_tls="tls_a",
        expected_roster=expected,
    )

    before = originator.select(dataset("tls_a", 290), reference_index=0)
    focal = originator.select(dataset("tls_a", 300), reference_index=0)
    nonfocal = originator.select(dataset("tls_b", 300), reference_index=0)
    after = originator.select(dataset("tls_a", 310), reference_index=0)

    assert before.selected_state == "PP" and before.forced_reference
    assert focal.selected_state == "A" and not focal.forced_reference
    assert nonfocal.selected_state == "PP" and nonfocal.forced_reference
    assert after.selected_state == "PP" and after.forced_reference


def test_exact_physical_comparison_includes_pending_vehicle_ids():
    state = {
        "time_sec": 300.0,
        "vehicles": {
            "veh": {
                "lane_id": "lane",
                "route_index": 0,
                "route": ("edge",),
                "lane_position": 1.0,
                "speed": 2.0,
            }
        },
        "executors": {
            "tls": {key: 0 for key in v157c.EXECUTOR_FIELDS}
        },
        "pending_ids": ("pending",),
    }
    assert v157c.compare_physical(state, state, tolerance=0.0)["passed"]
    changed = {**state, "pending_ids": ("different",)}
    comparison = v157c.compare_physical(state, changed, tolerance=0.0)
    assert comparison["passed"] is False
    assert comparison["mismatches"][0]["path"] == "pending_ids"


def test_rejected_learned_request_invalidates_intervention_contract():
    chosen = decision(
        arm="uniform_source", tls_id="tls_a", selected="A", priority=2.0
    )
    rejected_trace = [
        {
            "time_sec": 300.0,
            "tls_id": "tls_a",
            "execution_effective": False,
        }
    ]
    checks = v157c._intervention_trace_checks(
        rejected_trace,
        decision=chosen,
        checkpoint=300,
        focal_tls="tls_a",
    )
    assert checks["at_most_one_learned_action"] is True
    assert checks["learned_action_only_at_focal_checkpoint"] is True
    assert checks["learned_action_execution_effective"] is False


def test_reference_action_requires_no_intervention_trace():
    reference = decision(
        arm="target_only", tls_id="tls_a", selected="PP", priority=0.0
    )
    assert all(
        v157c._intervention_trace_checks(
            [],
            decision=reference,
            checkpoint=300,
            focal_tls="tls_a",
        ).values()
    )


def test_arm_manifest_matches_v157b_runtime_interface(tmp_path: Path):
    bundle = tmp_path / "runtime_models.pkl"
    bundle.write_bytes(b"frozen-runtime-fixture")
    bundle_sha = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    manifest = {
        "protocol": v157c.ARM_MANIFEST_PROTOCOL,
        "city": "jinan",
        "arm_order": list(v157c.ALL_ARMS),
        "runtime_bundle": {
            "path": str(bundle.resolve()),
            "sha256": bundle_sha,
            "size_bytes": bundle.stat().st_size,
            "protocol": v157c.RUNTIME_BUNDLE_PROTOCOL,
        },
        "arms": {
            "target_only": {
                "kind": "learned_model",
                "bundle_key": "target_only",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "kind": "learned_model",
                "bundle_key": "uniform_source",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_path": "exact_v157a_source_component_refits",
                "source_component_count": 7,
            },
            "source_label_placebo": {
                "kind": "learned_model",
                "bundle_key": "source_label_placebo",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "source_component_count": 7,
                "source_labels_permuted": True,
                "target_city_labels_permuted": False,
            },
            "phase_pressure": {
                "kind": "reference_policy",
                "policy": "phase_pressure",
                "bundle_key": "phase_pressure",
            },
        },
        "scope": "one_frozen_focal_tls_action_then_phase_pressure_continuation",
    }
    path = tmp_path / "arm_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True))
    assert tuple(json.loads(path.read_text())["arms"]) != v157c.ALL_ARMS
    manifest_sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    assert v157c._validate_arm_manifest(
        path,
        expected_sha256=manifest_sha,
        bundle_path=bundle,
        bundle_sha256=bundle_sha,
    ) == manifest

    manifest["arms"]["unexpected"] = dict(manifest["arms"]["phase_pressure"])
    path.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="arm manifest contract"):
        v157c._validate_arm_manifest(
            path,
            expected_sha256=manifest_sha,
            bundle_path=bundle,
            bundle_sha256=bundle_sha,
        )


def test_runtime_bundle_requires_domain_aligned_target_contract(tmp_path: Path):
    bundle = tmp_path / "runtime_models.pkl"
    payload = {
        "protocol": v157b.RUNTIME_BUNDLE_PROTOCOL,
        "arm_order": v157b.ARM_ORDER,
        "runtime_target_comparator": "domain_aligned_b100_target_only_v2",
        "arms": {
            "target_only": {
                "model": FakeModel(),
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "model": FakeModel(),
                "training_path": "exact_v157a_source_component_refits",
                "source_labels_permuted": False,
            },
            "source_label_placebo": {"model": FakeModel()},
            "phase_pressure": {"model": None},
        },
    }
    bundle.write_bytes(pickle.dumps(payload))
    digest = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    assert tuple(v157c._load_arm_models(bundle, digest)) == v157c.LEARNED_ARMS

    payload["arms"]["target_only"]["training_domain_handling"] = (
        "preserve_scenario_domains"
    )
    bundle.write_bytes(pickle.dumps(payload))
    digest = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="runtime arm semantics"):
        v157c._load_arm_models(bundle, digest)


def test_evaluate_policy_v3_preflight_rejects_frozen_signature_without_observer():
    def frozen_evaluator_without_observer():
        raise AssertionError("signature validation must not call the evaluator")

    current = inspect.signature(resco_v3.evaluate_policy_v3)
    frozen_evaluator_without_observer.__signature__ = current.replace(
        parameters=[
            parameter
            for parameter in current.parameters.values()
            if parameter.name != "step_observer"
        ]
    )
    with pytest.raises(ValueError, match="step_observer"):
        v157c.validate_evaluate_policy_v3_interface(
            frozen_evaluator_without_observer
        )


def test_evaluate_policy_v3_preflight_accepts_current_keyword_contract():
    result = v157c.validate_evaluate_policy_v3_interface(
        resco_v3.evaluate_policy_v3
    )
    assert result == {
        "passed": True,
        "required_keyword_arguments": list(
            v157c.EVALUATE_POLICY_V3_REQUIRED_KEYWORDS
        ),
        "accepts_var_keyword": False,
    }
    assert "step_observer" in result["required_keyword_arguments"]


def test_validate_inputs_cli_checks_real_contract_without_starting_sumo(
    tmp_path: Path, monkeypatch, capsys
):
    bundle = tmp_path / "runtime_models.pkl"
    payload = {
        "protocol": v157c.RUNTIME_BUNDLE_PROTOCOL,
        "arm_order": v157b.ARM_ORDER,
        "runtime_target_comparator": "domain_aligned_b100_target_only_v2",
        "arms": {
            "target_only": {
                "model": FakeModel(),
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "model": FakeModel(),
                "training_path": "exact_v157a_source_component_refits",
                "source_labels_permuted": False,
            },
            "source_label_placebo": {"model": FakeModel()},
            "phase_pressure": {"model": None},
        },
    }
    bundle.write_bytes(pickle.dumps(payload))
    bundle_sha = __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    manifest = {
        "protocol": v157c.ARM_MANIFEST_PROTOCOL,
        "city": "jinan",
        "arm_order": list(v157c.ALL_ARMS),
        "runtime_bundle": {
            "path": str(bundle.resolve()),
            "sha256": bundle_sha,
            "size_bytes": bundle.stat().st_size,
            "protocol": v157c.RUNTIME_BUNDLE_PROTOCOL,
        },
        "arms": {
            "target_only": {
                "kind": "learned_model",
                "bundle_key": "target_only",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_domain_handling": "relabel_all_b100_rows_to_jinan",
                "v157a_exact_target_refit_used_at_runtime": False,
            },
            "uniform_source": {
                "kind": "learned_model",
                "bundle_key": "uniform_source",
                "predict_interface": "model.predict(action_contrast_dataset)",
                "training_path": "exact_v157a_source_component_refits",
            },
            "source_label_placebo": {
                "kind": "learned_model",
                "bundle_key": "source_label_placebo",
                "predict_interface": "model.predict(action_contrast_dataset)",
            },
            "phase_pressure": {
                "kind": "reference_policy",
                "policy": "phase_pressure",
                "bundle_key": "phase_pressure",
            },
        },
        "scope": "one_frozen_focal_tls_action_then_phase_pressure_continuation",
    }
    arm_manifest = tmp_path / "arm_manifest.json"
    arm_manifest.write_text(json.dumps(manifest, sort_keys=True))
    manifest_sha = __import__("hashlib").sha256(
        arm_manifest.read_bytes()
    ).hexdigest()

    def unexpected_sumo_load():
        raise AssertionError("validate-inputs must not load SUMO")

    monkeypatch.setattr(v157c, "load_libsumo", unexpected_sumo_load)
    assert v157c._main(
        [
            "validate-inputs",
            "--protocol",
            str(CONFIG),
            "--authorization",
            str(AUTHORIZATION),
            "--runtime-bundle",
            str(bundle),
            "--runtime-bundle-sha256",
            bundle_sha,
            "--arm-manifest",
            str(arm_manifest),
            "--arm-manifest-sha256",
            manifest_sha,
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "arm_manifest_protocol": v157c.ARM_MANIFEST_PROTOCOL,
        "artifacts_written": False,
        "authorization_protocol": v157c.V157A_PROTOCOL,
        "evaluate_policy_v3_interface": {
            "accepts_var_keyword": False,
            "passed": True,
            "required_keyword_arguments": list(
                v157c.EVALUATE_POLICY_V3_REQUIRED_KEYWORDS
            ),
        },
        "learned_arms": list(v157c.LEARNED_ARMS),
        "parent_protocol": v157c.PROTOCOL,
        "protocol": v157c.INPUT_VALIDATION_PROTOCOL,
        "runtime_bundle_protocol": v157c.RUNTIME_BUNDLE_PROTOCOL,
        "status": "PASS",
        "sumo_started": False,
    }
    assert set(path.name for path in tmp_path.iterdir()) == {
        "arm_manifest.json",
        "runtime_models.pkl",
    }


def test_aggregate_uses_seed_means_and_retains_zero_windows():
    protocol = json.loads(CONFIG.read_text())
    results = []
    for index, seed in enumerate(v157c.SEEDS):
        windows = {}
        for checkpoint in v157c.CHECKPOINTS_SEC:
            costs = {
                "target_only": 2.0 + index,
                "uniform_source": 1.5 + index,
                "source_label_placebo": 1.75 + index,
                "phase_pressure": 1.0 + index,
            }
            windows[str(checkpoint)] = {
                "costs": costs,
                "contrasts": {
                    "uniform_source_minus_target_only": -0.5,
                    "uniform_source_minus_source_label_placebo": -0.25,
                    "target_only_minus_phase_pressure": 1.0,
                    "uniform_source_minus_phase_pressure": 0.5,
                    "source_label_placebo_minus_phase_pressure": 0.75,
                },
                "focal": {"source_target_selected_actions_differ": checkpoint != 480},
            }
        results.append(
            {
                "protocol": v157c.SEED_RESULT_PROTOCOL,
                "parent_protocol": v157c.PROTOCOL,
                "status": "PASS",
                "scenario": v157c.SCENARIO,
                "seed": seed,
                "windows": windows,
                "native_simulation_count": 10,
            }
        )
    aggregate = v157c.aggregate_seed_results(protocol, results)
    assert aggregate["native_simulation_count"] == 30
    assert aggregate["source_target_disagreement_window_count"] == 6
    assert aggregate["comparisons"]["uniform_source_minus_target_only"]["mean_of_seed_means"] == -0.5
    assert aggregate["identical_action_windows_retained_as_zero"] is True
    assert aggregate["outcomes_used_for_checkpoint_or_focal_selection"] is False


def test_runner_has_no_sumo_state_checkpoint_api_calls():
    source = Path(v157c.__file__).read_text(encoding="utf-8")
    forbidden = ("simulation." + "saveState", "simulation." + "loadState")
    assert all(value not in source for value in forbidden)
