import json
from pathlib import Path

from scripts.cluster.run_tsc_external_hierarchical_closed_loop_development_shard import (
    SHARD_PROTOCOL,
    _validated_completed_summary,
    shard_identities,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_closed_loop_development import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256


def test_hierarchical_shards_partition_matrix_exactly() -> None:
    protocol = json.loads(
        Path(
            "cf_h2o/config/traffic_signal_tsc_v46_external_v9_hierarchical_closed_loop_development.json"
        ).read_text(encoding="utf-8")
    )
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]

    flattened = [identity for shard in shards for identity in shard]
    assert len(flattened) == len(set(flattened)) == 416
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_completed_shard_resume_revalidates_result_hashes(tmp_path: Path) -> None:
    identity = ("jinan", "scenario", 7, "phase_pressure")
    protocol_sha = "a" * 64
    root = tmp_path / "jinan" / "scenario" / "seed_7" / "phase_pressure"
    root.mkdir(parents=True)
    tripinfo = root / "tripinfo.xml"
    tripinfo.write_text("<tripinfos />\n", encoding="utf-8")
    result = root / "result.json"
    result.write_text(
        json.dumps(
            {
                "protocol": RESULT_PROTOCOL,
                "city": "jinan",
                "scenario": "scenario",
                "seed": 7,
                "policy": "phase_pressure",
                "protocol_sha256": protocol_sha,
                "tripinfo_evidence": {"sha256": _sha256(tripinfo)},
                "metrics": {"ok": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "_shards" / "shard_0.json"
    summary_path.parent.mkdir()
    summary_path.write_text(
        json.dumps(
            {
                "protocol": SHARD_PROTOCOL,
                "passed": True,
                "shard_index": 0,
                "shard_count": 1,
                "protocol_sha256": protocol_sha,
                "rows": [
                    {
                        "city": "jinan",
                        "scenario": "scenario",
                        "seed": 7,
                        "policy": "phase_pressure",
                        "result_sha256": _sha256(result),
                        "tripinfo_sha256": _sha256(tripinfo),
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = _validated_completed_summary(
        summary_path=summary_path,
        identities=(identity,),
        results_root=tmp_path,
        protocol_sha256=protocol_sha,
        shard_index=0,
        shard_count=1,
    )

    assert payload["validated_summary_reuse"] is True
