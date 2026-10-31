from pathlib import Path
from types import SimpleNamespace

import pytest

from cf_h2o.eval import traffic_signal_tsc_mechanism_offline_ablation as offline


def _install_fake_cache(
    tmp_path: Path,
    monkeypatch,
    identities: list[tuple[str, int] | tuple[str, int, int, int]],
) -> tuple[Path, SimpleNamespace]:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    datasets = {}
    for index, identity in enumerate(identities):
        scenario, seed = identity[:2]
        shard, shard_count = identity[2:] if len(identity) == 4 else (0, 1)
        path = cache_root / f"cache_{index}.npz"
        path.write_bytes(b"test")
        datasets[path.name] = SimpleNamespace(
            size=1,
            metadata={
                "counterfactual_cache_identity": {
                    "scenario": scenario,
                    "seed": seed,
                    "collection_shard_index": shard,
                    "collection_shard_count": shard_count,
                },
                "action_group_ids": [f"{scenario}:seed{seed}:0:tls"],
            },
        )

    monkeypatch.setattr(
        offline,
        "load_mechanism_dataset",
        lambda path: datasets[Path(path).name],
    )
    monkeypatch.setattr(
        offline,
        "_validate_counterfactual_cache_dataset",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        offline,
        "_merge_collection_shards_v3",
        lambda scenario, seed, rows: rows[0],
    )

    def merge_seed_datasets(scenario, rows):
        return SimpleNamespace(
            size=sum(row.size for row in rows),
            metadata={
                "action_group_ids": [
                    group
                    for row in rows
                    for group in row.metadata["action_group_ids"]
                ]
            },
        )

    monkeypatch.setattr(
        offline,
        "_merge_scenario_seed_datasets_v3",
        merge_seed_datasets,
    )
    manifest = SimpleNamespace(
        sumocfgs={"source": tmp_path / "source.sumocfg", "target": tmp_path / "target.sumocfg"}
    )
    return cache_root, manifest


def test_mixed_seed_loader_uses_per_scenario_seed_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root, manifest = _install_fake_cache(
        tmp_path,
        monkeypatch,
        [("source", 2027), ("source", 3037), ("target", 5057), ("target", 6067)],
    )

    bank, audit = offline.load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(),
        scenario_seeds={"source": (2027, 3037), "target": (5057, 6067)},
        collection_shards=1,
        workers=2,
    )

    assert set(bank) == {"source", "target"}
    assert bank["source"].size == 2
    assert bank["target"].size == 2
    assert audit["protocol"] == (
        "embedded-identity-read-only-mixed-seed-shard-load-v2"
    )
    assert audit["used_file_count"] == 4
    assert audit["seeds_by_scenario"] == {
        "source": [2027, 3037],
        "target": [5057, 6067],
    }


def test_mixed_seed_loader_rejects_missing_target_seed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root, manifest = _install_fake_cache(
        tmp_path,
        monkeypatch,
        [("source", 2027), ("source", 3037), ("target", 5057)],
    )

    with pytest.raises(ValueError, match="coverage mismatch"):
        offline.load_frozen_counterfactual_bank(
            cache_root,
            manifest,
            seeds=(),
            scenario_seeds={"source": (2027, 3037), "target": (5057, 6067)},
            collection_shards=1,
            workers=2,
        )


def test_uniform_seed_loader_preserves_v1_audit_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root, manifest = _install_fake_cache(
        tmp_path,
        monkeypatch,
        [("source", 2027), ("target", 2027)],
    )

    _, audit = offline.load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(2027,),
        collection_shards=1,
        workers=2,
    )

    assert audit["protocol"] == "embedded-identity-read-only-complete-shard-load-v1"
    assert audit["seeds"] == [2027]
    assert "seeds_by_scenario" not in audit


def test_loader_accepts_scenario_specific_shard_counts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_root, manifest = _install_fake_cache(
        tmp_path,
        monkeypatch,
        [
            ("source", 2027, 0, 2),
            ("source", 2027, 1, 2),
            ("target", 2027, 0, 1),
        ],
    )

    bank, audit = offline.load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(2027,),
        collection_shards=1,
        scenario_collection_shards={"source": 2},
        workers=2,
    )

    assert set(bank) == {"source", "target"}
    assert audit["used_file_count"] == 3
    assert audit["collection_shards_by_scenario"] == {
        "source": 2,
        "target": 1,
    }
