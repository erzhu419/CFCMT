from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_feature_aligned_b100_runtime_freeze as subject
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.launch_tsc_feature_aligned_b100_runtime_freeze import (
    SIGNATURE,
    _validate_stage_manifest,
    build_spec,
)
from scripts.cluster import derive_tsc_v157b_runtime_freeze_snapshot as derive_snapshot


class _NamedFeatureModel:
    def __init__(self, name: str, scale: float) -> None:
        self.name = name
        self.scale = float(scale)

    def predict(self, dataset: MechanismDataset):
        index = dataset.feature_names.index(self.name)
        value = np.asarray(dataset.features[:, index], dtype=float) * self.scale
        return {
            "control_cost": {
                "mean": value,
                "uncertainty": np.abs(value) * 0.1,
                "context_trust": np.full(value.shape, 0.8),
            }
        }


def _absolute_dataset() -> MechanismDataset:
    groups = np.asarray([name for name in ("a", "b", "c", "d") for _ in range(3)])
    green_q = np.asarray(
        [3.0, 1.0, 2.0, 1.0, 4.0, 2.0, 5.0, 4.0, 3.0, 1.0, 2.0, 6.0]
    )
    interval = np.asarray(
        [10.0, 12.0, 9.0, 20.0, 21.0, 18.0, 30.0, 25.0, 35.0, 40.0, 44.0, 39.0]
    )
    return MechanismDataset(
        feature_names=("green_q",),
        features=green_q[:, None],
        context_names=(),
        context=np.zeros((groups.size, 0)),
        priors={"interval_cost": np.zeros(groups.size), "prefix_mean_cost_450s": np.zeros(groups.size)},
        targets={"interval_cost": interval, "prefix_mean_cost_450s": interval.copy()},
        domains=np.asarray(["source"] * groups.size),
        metadata={"action_group_ids": groups.tolist()},
    )


def _contrast_dataset() -> MechanismDataset:
    return build_action_contrast_dataset(
        _absolute_dataset(),
        reference_policy="phase_pressure",
        contrast_features=("green_q",),
    )


def test_placebo_mapping_is_deterministic_deranged_and_label_free() -> None:
    contrast = _contrast_dataset()
    groups = np.asarray(contrast.metadata["action_group_ids"], dtype=str)
    references = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    first, audit = subject.derive_placebo_donor_mapping(
        groups, references, scenario="fixture", seed=17
    )
    second, _ = subject.derive_placebo_donor_mapping(
        groups, references, scenario="fixture", seed=17
    )
    assert first == second
    assert set(first) == set(first.values()) == {"a", "b", "c", "d"}
    assert all(destination != donor for destination, donor in first.items())
    assert audit["outcome_labels_used_to_derive_mapping"] == 0
    assert audit["mapping_sha256"] == subject._canonical_sha256(audit["mapping_rows"])


def test_placebo_moves_whole_nonreference_delta_blocks_and_keeps_reference() -> None:
    original = _absolute_dataset()
    placebo, audit = subject.source_label_block_placebo(
        original,
        scenario="fixture",
        seed=17,
        target_names=("interval_cost", "prefix_mean_cost_450s"),
    )
    original_contrast = build_action_contrast_dataset(
        original, reference_policy="phase_pressure", contrast_features=("green_q",)
    )
    placebo_contrast = build_action_contrast_dataset(
        placebo, reference_policy="phase_pressure", contrast_features=("green_q",)
    )
    groups = np.asarray(original_contrast.metadata["action_group_ids"], dtype=str)
    references = np.asarray(original_contrast.metadata["is_reference"], dtype=bool)
    mapping = {
        row["destination_group"]: row["donor_group"]
        for row in audit["mapping_rows"]
    }
    for destination, donor in mapping.items():
        destination_rows = np.flatnonzero(groups == destination)
        donor_rows = np.flatnonzero(groups == donor)
        observed = placebo_contrast.targets["interval_cost"][
            destination_rows[~references[destination_rows]]
        ]
        expected = original_contrast.targets["interval_cost"][
            donor_rows[~references[donor_rows]]
        ]
        assert np.array_equal(observed, expected)
    assert np.all(placebo_contrast.targets["interval_cost"][references] == 0.0)
    assert np.array_equal(
        placebo.targets["interval_cost"], placebo.targets["prefix_mean_cost_450s"]
    )
    assert np.array_equal(placebo.features, original.features)


def test_reused_identity_checks_all_arrays_and_name_binding() -> None:
    contrast = _contrast_dataset()
    target = _NamedFeatureModel("green_q", 1.0)
    sources = {
        "atlanta": _NamedFeatureModel("green_q", 2.0),
        "cologne": _NamedFeatureModel("green_q", 3.0),
    }
    artifact = {
        "protocol": subject.V123_ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": subject.WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "source_group_order": ("atlanta", "cologne"),
        "target_predictions": {"100": subject._prediction_arrays(target, contrast)},
        "source_predictions": {
            "100": {
                name: subject._prediction_arrays(model, contrast)
                for name, model in sources.items()
            }
        },
    }
    audit = subject.validate_reused_prediction_identity(
        target_model=target,
        source_components=sources,
        selector_contrast=contrast,
        v157a_artifact=artifact,
    )
    assert audit["passed"] is True
    assert audit["uniform_component_mean_score_uncertainty_trust_equal"] is True
    assert audit["uniform_runtime_score_uncertainty_trust_equal"] is True
    assert all(audit["feature_order_invariance_on_one_group"].values())
    assert audit["runtime_reuse_authorized"] is False
    assert audit["decision"] == "exact_refit_identity_validated_for_audit_only"


def test_reused_identity_logs_lightweight_exact_mismatch_diagnostics(capsys) -> None:
    contrast = _contrast_dataset()
    target = _NamedFeatureModel("green_q", 1.0)
    sources = {
        "atlanta": _NamedFeatureModel("green_q", 2.0),
        "cologne": _NamedFeatureModel("green_q", 3.0),
    }
    expected_target = list(subject._prediction_arrays(target, contrast))
    expected_target[0] = expected_target[0].copy()
    expected_target[0][1] = np.nextafter(expected_target[0][1], np.inf)
    artifact = {
        "protocol": subject.V123_ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": subject.WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "source_group_order": ("atlanta", "cologne"),
        "target_predictions": {"100": tuple(expected_target)},
        "source_predictions": {
            "100": {
                name: subject._prediction_arrays(model, contrast)
                for name, model in sources.items()
            }
        },
    }

    try:
        subject.validate_reused_prediction_identity(
            target_model=target,
            source_components=sources,
            selector_contrast=contrast,
            v157a_artifact=artifact,
        )
    except ValueError as error:
        assert str(error) == (
            "V115 runtime pickles do not reproduce corrected V157A B100 arrays"
        )
    else:
        raise AssertionError("one-bit prediction mismatch passed exact identity")

    marker = "V157B_PREDICTION_IDENTITY_DIAGNOSTIC="
    line = capsys.readouterr().out.strip()
    assert line.startswith(marker)
    diagnostic = json.loads(line.removeprefix(marker))
    assert diagnostic["array_equality"] == "numpy_array_equal"
    assert diagnostic["failed_sections"] == ["target_only"]
    score = diagnostic["target_only"]["arrays"]["score"]
    assert score["array_equal"] is False
    assert score["observed_shape"] == score["expected_shape"] == [contrast.size]
    assert score["mismatch_count"] == 1
    assert score["max_abs_difference"] > 0.0
    assert diagnostic["target_only"]["arrays"]["uncertainty"][
        "array_equal"
    ] is True
    assert all(
        row["all_array_equal"] for row in diagnostic["per_source"].values()
    )
    assert diagnostic["uniform_component_mean"]["all_array_equal"] is True
    assert diagnostic["uniform_runtime"]["all_array_equal"] is True
    assert diagnostic["uniform_runtime_vs_component_mean"]["all_array_equal"] is True


def test_legacy_v115_mismatch_is_diagnostic_and_never_authorizes_reuse() -> None:
    contrast = _contrast_dataset()
    target = _NamedFeatureModel("green_q", 1.0)
    sources = {"atlanta": _NamedFeatureModel("green_q", 2.0)}
    expected_target = list(subject._prediction_arrays(target, contrast))
    expected_target[0] = expected_target[0] + 1.0
    artifact = {
        "protocol": subject.V123_ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": subject.WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "source_group_order": ("atlanta",),
        "target_predictions": {"100": tuple(expected_target)},
        "source_predictions": {
            "100": {
                "atlanta": subject._prediction_arrays(sources["atlanta"], contrast)
            }
        },
    }
    audit = subject.validate_reused_prediction_identity(
        target_model=target,
        source_components=sources,
        selector_contrast=contrast,
        v157a_artifact=artifact,
        comparison="legacy_v115_pickles_vs_v157a_b100_predictions",
        require_exact=False,
    )
    assert audit["passed"] is False
    assert audit["runtime_reuse_authorized"] is False
    assert audit["decision"] == "diagnostic_only_do_not_reuse"


def test_v157b_refit_roster_parallelizes_two_targets_and_two_source_families() -> None:
    roster = subject._fit_task_roster(("cologne", "atlanta"))
    assert roster == (
        ("target_v157a_audit", None),
        ("target_domain_aligned", None),
        ("source_aligned", "atlanta"),
        ("source_aligned", "cologne"),
        ("source_label_placebo", "atlanta"),
        ("source_label_placebo", "cologne"),
    )


def test_selector_prediction_cache_is_reused_by_both_identity_audits() -> None:
    contrast = _contrast_dataset()

    class CountingModel(_NamedFeatureModel):
        def __init__(self, name: str, scale: float) -> None:
            super().__init__(name, scale)
            self.predicted_sizes: list[int] = []

        def predict(self, dataset: MechanismDataset):
            self.predicted_sizes.append(int(dataset.size))
            return super().predict(dataset)

    audit_target = CountingModel("green_q", 1.0)
    aligned_target = CountingModel("green_q", 1.0)
    sources = {
        "atlanta": CountingModel("green_q", 2.0),
        "cologne": CountingModel("green_q", 3.0),
    }
    uniform = subject.UniformAnchoredRuntimeModel(sources)
    cache = subject._build_selector_prediction_cache(
        v157a_audit_target_model=audit_target,
        domain_aligned_target_model=aligned_target,
        source_components=sources,
        uniform_runtime_model=uniform,
        selector_contrast=contrast,
    )
    full_size = int(contrast.size)
    assert audit_target.predicted_sizes.count(full_size) == 1
    assert aligned_target.predicted_sizes.count(full_size) == 1
    assert all(model.predicted_sizes.count(full_size) == 2 for model in sources.values())

    artifact = {
        "protocol": subject.V123_ARTIFACT_PROTOCOL,
        "city": "jinan",
        "estimand": subject.WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "source_group_order": tuple(sources),
        "target_predictions": {"100": cache.v157a_audit_target},
        "source_predictions": {"100": cache.source_components},
    }
    calls_after_cache = {
        "audit": tuple(audit_target.predicted_sizes),
        "aligned": tuple(aligned_target.predicted_sizes),
        **{name: tuple(model.predicted_sizes) for name, model in sources.items()},
    }
    common = {
        "source_components": sources,
        "selector_contrast": contrast,
        "v157a_artifact": artifact,
        "source_prediction_arrays": cache.source_components,
        "uniform_runtime_prediction_arrays": cache.uniform_runtime,
    }
    subject.validate_reused_prediction_identity(
        target_model=audit_target,
        target_prediction_arrays=cache.v157a_audit_target,
        feature_order_checks=cache.v157a_audit_feature_order,
        **common,
    )
    subject.validate_reused_prediction_identity(
        target_model=aligned_target,
        target_prediction_arrays=cache.domain_aligned_target,
        feature_order_checks=cache.domain_aligned_feature_order,
        comparison="cached_domain_aligned_diagnostic",
        require_exact=False,
        **common,
    )
    assert tuple(audit_target.predicted_sizes) == calls_after_cache["audit"]
    assert tuple(aligned_target.predicted_sizes) == calls_after_cache["aligned"]
    assert all(
        tuple(model.predicted_sizes) == calls_after_cache[name]
        for name, model in sources.items()
    )


def test_v157b_target_refits_keep_audit_path_and_align_runtime_domain(monkeypatch) -> None:
    import contextlib
    import threadpoolctl

    adaptation = _absolute_dataset()

    def fake_target_model(dataset, *, group_ids, candidate):
        return {
            "domains": tuple(sorted(set(np.asarray(dataset.domains, dtype=str)))),
            "group_ids": tuple(group_ids),
            "candidate": candidate,
        }

    monkeypatch.setattr(subject, "_target_model", fake_target_model)
    monkeypatch.setattr(
        threadpoolctl,
        "threadpool_limits",
        lambda **_: contextlib.nullcontext(),
    )
    rows = subject._fit_runtime_components(
        adaptation_dataset=adaptation,
        selected_group_ids=("a", "b"),
        candidate="constant_alpha_0",
        aligned_states={},
        placebo_states={},
        workers=1,
    )
    assert rows == [
        {
            "arm": "target_v157a_audit",
            "source_group": None,
            "model": {
                "domains": ("source",),
                "group_ids": ("a", "b"),
                "candidate": "constant_alpha_0",
            },
        },
        {
            "arm": "target_domain_aligned",
            "source_group": None,
            "model": {
                "domains": ("jinan",),
                "group_ids": ("a", "b"),
                "candidate": "constant_alpha_0",
            },
        },
    ]


def test_domain_aligned_source_gate_uses_fixed_seed_level_cost_difference(
    monkeypatch,
) -> None:
    actual = np.asarray([0.0, -0.002, 0.0, -0.002, 0.0, -0.001, 0.0, -0.001])
    groups = tuple(np.asarray([2 * index, 2 * index + 1]) for index in range(4))
    references = np.asarray([True, False] * 4)
    group_seeds = np.asarray([1, 1, 2, 2])
    target_score = np.asarray([0.0, 1.0] * 4)
    source_score = np.asarray([1.0, 0.0] * 4)
    target = object()
    source = object()
    monkeypatch.setattr(
        subject,
        "_normalized_pressure_targets",
        lambda *_: (actual, np.arange(4), groups, references, group_seeds),
    )
    monkeypatch.setattr(
        subject,
        "_prediction_arrays",
        lambda *_: (_ for _ in ()).throw(
            AssertionError("cached gate repeated selector inference")
        ),
    )
    gate = subject.domain_aligned_source_gate(
        selector=object(),
        selector_contrast=object(),
        target_model=target,
        source_model=source,
        selector_seeds=(1, 2),
        mean_maximum=-0.0005,
        upper_95_maximum=0.0,
        target_prediction_score=target_score,
        source_prediction_score=source_score,
    )
    assert gate["selector_seed_order"] == [1, 2]
    assert gate["source_minus_target_by_seed"] == [-0.002, -0.001]
    assert gate["paired_bootstrap"]["mean"] == -0.0015
    assert gate["paired_bootstrap"]["upper_95"] < 0.0
    assert gate["passed"] is True


def test_uniform_runtime_model_and_bundle_loader(tmp_path: Path) -> None:
    contrast = _contrast_dataset()
    model = subject.UniformAnchoredRuntimeModel(
        {
            "left": _NamedFeatureModel("green_q", 1.0),
            "right": _NamedFeatureModel("green_q", 3.0),
        }
    )
    prediction = model.predict(contrast)["control_cost"]
    expected = tuple(
        np.mean(
            np.vstack(
                [
                    subject._prediction_arrays(
                        _NamedFeatureModel("green_q", scale), contrast
                    )[index]
                    for scale in (1.0, 3.0)
                ]
            ),
            axis=0,
        )
        for index in range(3)
    )
    assert np.array_equal(prediction["mean"], expected[0])
    assert np.array_equal(prediction["uncertainty"], expected[1])
    assert np.array_equal(prediction["context_trust"], expected[2])

    payload = {
        "protocol": subject.RUNTIME_BUNDLE_PROTOCOL,
        "arm_order": subject.ARM_ORDER,
        "arms": {
            "target_only": {"model": model},
            "uniform_source": {"model": model},
            "source_label_placebo": {"model": model},
            "phase_pressure": {"model": None},
        },
    }
    bundle = tmp_path / "runtime.pkl"
    bundle.write_bytes(__import__("pickle").dumps(payload))
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    restored = subject.load_runtime_arms(bundle, expected_sha256=digest)
    assert subject.runtime_arm_model(restored, "uniform_source") is not None
    assert subject.runtime_arm_model(restored, "phase_pressure") is None


def test_uniform_runtime_wrapper_unpickles_in_fresh_process(tmp_path: Path) -> None:
    model = subject.UniformAnchoredRuntimeModel.__new__(
        subject.UniformAnchoredRuntimeModel
    )
    model.component_names = ()
    model.component_models = ()
    bundle = tmp_path / "runtime.pkl"
    bundle.write_bytes(
        pickle.dumps(
            {
                "protocol": subject.RUNTIME_BUNDLE_PROTOCOL,
                "arm_order": subject.ARM_ORDER,
                "arms": {
                    arm: {"model": None if arm == "phase_pressure" else model}
                    for arm in subject.ARM_ORDER
                },
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze "
                "import load_runtime_arms; "
                f"load_runtime_arms(Path({str(bundle)!r}))"
            ),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_v157b_config_and_launcher_freeze_exact_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "cf_h2o/config/traffic_signal_tsc_v157b_feature_aligned_b100_runtime_freeze.json"
        ).read_text()
    )
    assert len(config["b100_group_ids"]) == len(set(config["b100_group_ids"])) == 100
    assert subject._canonical_sha256(config["b100_group_ids"]) == config[
        "b100_group_ids_sha256"
    ]
    assert len(config["contrast_features"]) == 40
    assert len(config["rigid_model_feature_names"]) == 29
    assert config["source_label_placebo"]["evaluation_outcomes_consumed"] == 0
    assert config["runtime_refit_gate"]["fit_task_roster"] == {
        "target_v157a_audit_only": 1,
        "target_domain_aligned_runtime": 1,
        "source_aligned_components": 7,
        "source_label_placebo_components": 7,
        "total": 16,
    }
    assert config["runtime_refit_gate"]["v157a_exact_target_refit_is_audit_only"] is True
    assert config["domain_aligned_source_gate"] == {
        "protocol": "v157b-domain-aligned-source-minus-target-gate-v2",
        "candidate": "uniform_all_sources__source_weight_1",
        "source_weight": 1.0,
        "selector_seed_count": 22,
        "comparison": "uniform_source_minus_domain_aligned_target_only",
        "mean_maximum": -0.0005,
        "upper_95_strict_maximum": 0.0,
        "failure_action": (
            "write_result_only_without_runtime_bundle_or_v157c_authorization"
        ),
    }
    assert config["v1_failure_provenance"]["task_id"] == "t92500"
    assert config["v1_failure_provenance"]["legacy_runtime_reuse_authorized"] is False
    assert not any(
        key.startswith("legacy_v115") for key in config["frozen_identities"]
    )
    assert derive_snapshot.CONFIG_PROTOCOL == subject.CONFIG_PROTOCOL
    assert set(derive_snapshot.OVERLAYS) == {
        Path("cf_h2o/eval/traffic_signal_feature_aligned_b100_runtime_freeze.py"),
        Path(
            "cf_h2o/config/traffic_signal_tsc_v157b_feature_aligned_b100_runtime_freeze.json"
        ),
    }

    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        v157a_aggregate_remote=Path("/evidence/v157a.json"),
        v157a_prediction_artifact_remote=Path("/evidence/predictions.pkl"),
        fit_result_remote=Path("/fit/result.json"),
        source_cache_root=Path("/cache/source"),
        source_cache_audit_remote=Path("/audit/source.json"),
        target_cache_root=Path("/cache/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        selector_cache_root=Path("/cache/selector"),
        selector_cache_audit_remote=Path("/audit/selector.json"),
        remote_output_root=Path("/results/v157b"),
        node="node005",
        cache_workers=20,
        fit_workers=16,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == 20 and spec["ram_mb"] == 98304
    entrypoint = tokens[tokens.index("-c") + 1]
    assert (
        "from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze import main"
        in entrypoint
    )
    assert tokens[tokens.index("--fit-workers") + 1] == "16"
    assert "--legacy-v115-target-model" not in tokens
    assert "--legacy-v115-source-models" not in tokens
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")


def test_v157b_launcher_binds_derived_snapshot_and_reviewed_overlays() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "cf_h2o/config/traffic_signal_tsc_v157b_feature_aligned_b100_runtime_freeze.json"
        ).read_text()
    )
    digest = "a" * 64
    stage = {
        "protocol": derive_snapshot.DERIVATION_PROTOCOL,
        "snapshot_sha256": digest,
        "source_tree_sha256": "b" * 64,
        "snapshot_root": str(derive_snapshot.REMOTE_BASE / digest[:20]),
        "derived_from_snapshot_sha256": config["frozen_identities"][
            "v157a_snapshot_sha256"
        ],
        "derived_from_source_tree_sha256": config["frozen_identities"][
            "v157a_source_tree_sha256"
        ],
        "overlays": [
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
            }
            for relative in derive_snapshot.OVERLAYS
        ],
    }
    assert _validate_stage_manifest(stage, config=config) == Path(
        stage["snapshot_root"]
    )

    changed = json.loads(json.dumps(stage))
    changed["overlays"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="derived snapshot manifest"):
        _validate_stage_manifest(changed, config=config)
