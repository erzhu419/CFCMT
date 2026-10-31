import json
from pathlib import Path

from scripts.cluster.run_tsc_external_target_veto_closed_loop_development_shard import (
    shard_identities,
)


def test_target_veto_six_shards_cover_method_matrix_once() -> None:
    parent = json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v47_external_v9_long_horizon_target_veto.json"
        ).read_text(encoding="utf-8")
    )
    protocol = {
        "development": {
            **parent["closed_loop_development"],
            "method_only_matrix_size": 96,
        }
    }
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]

    assert {len(rows) for rows in shards} == {16}
    flattened = [identity for rows in shards for identity in rows]
    assert len(flattened) == 96
    assert len(set(flattened)) == 96
