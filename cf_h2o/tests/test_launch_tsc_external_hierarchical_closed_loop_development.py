import json
from pathlib import Path

from scripts.cluster.launch_tsc_external_hierarchical_closed_loop_development import (
    NODES,
    PROTOCOL_RELATIVE,
)


def test_hierarchical_development_launch_uses_six_nodes_for_full_matrix() -> None:
    protocol = json.loads(Path(PROTOCOL_RELATIVE).read_text(encoding="utf-8"))

    assert NODES == tuple(f"node{index:03d}" for index in range(1, 7))
    assert protocol["development"]["matrix_size"] == 416
    assert protocol["development"]["candidate_count"] == 12
    assert protocol["scientific_amendment"][
        "closed_loop_result_file_count_at_freeze"
    ] == 0
