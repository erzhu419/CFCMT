"""Strict, non-pickle persistence for traffic-signal mechanism datasets."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def save_mechanism_dataset(path: Path, dataset: MechanismDataset) -> None:
    """Atomically save a dataset without object arrays or executable pickle."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "features": np.asarray(dataset.features, dtype=float),
        "context": np.asarray(dataset.context, dtype=float),
        "domains": np.asarray(dataset.domains, dtype=str),
        "feature_names": np.asarray(dataset.feature_names, dtype=str),
        "context_names": np.asarray(dataset.context_names, dtype=str),
        "prior_names": np.asarray(sorted(dataset.priors), dtype=str),
        "target_names": np.asarray(sorted(dataset.targets), dtype=str),
        "metadata_json": np.asarray(json.dumps(dataset.metadata, sort_keys=True)),
    }
    for name, values in dataset.priors.items():
        arrays[f"prior__{name}"] = np.asarray(values, dtype=float)
    for name, values in dataset.targets.items():
        arrays[f"target__{name}"] = np.asarray(values, dtype=float)

    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def load_mechanism_dataset(path: Path) -> MechanismDataset:
    """Load and validate a dataset written by :func:`save_mechanism_dataset`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        prior_names = tuple(str(value) for value in archive["prior_names"].tolist())
        target_names = tuple(str(value) for value in archive["target_names"].tolist())
        metadata = json.loads(str(archive["metadata_json"].item()))
        dataset = MechanismDataset(
            feature_names=tuple(str(value) for value in archive["feature_names"].tolist()),
            features=np.asarray(archive["features"], dtype=float),
            context_names=tuple(str(value) for value in archive["context_names"].tolist()),
            context=np.asarray(archive["context"], dtype=float),
            priors={name: np.asarray(archive[f"prior__{name}"], dtype=float) for name in prior_names},
            targets={name: np.asarray(archive[f"target__{name}"], dtype=float) for name in target_names},
            domains=np.asarray(archive["domains"], dtype=str),
            metadata=metadata,
        )
    return dataset


def atomic_write_json(path: Path, values: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise
