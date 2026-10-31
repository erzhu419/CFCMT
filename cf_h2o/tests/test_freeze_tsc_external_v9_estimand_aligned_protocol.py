import hashlib
import json
from pathlib import Path

from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_protocol import (
    CITIES,
    build_protocol,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v9_protocol_freezer_separates_adaptation_diagnostics(tmp_path: Path) -> None:
    parent = tmp_path / "v41.json"
    parent.write_text(
        json.dumps(
            {
                "protocol": "tsc-v54r50-external-v9-full-budget-seed-blocked-refit-confirmation-v1",
                "target_protocol": {
                    "adaptation_seeds": [5057, 6067],
                    "closed_loop_development_seeds": [
                        5057,
                        6067,
                        52282,
                        41242,
                        63792,
                    ],
                    "closed_loop_validation_seeds": [73412, 61481],
                    "closed_loop_prospective_seeds": [10091, 11003, 12007],
                    "external_city_scenarios": {
                        "los_angeles": ["la_1x4"],
                        "jinan": ["jinan_3x4_real"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    freeze_root = tmp_path / "freeze"
    audit_cities = {}
    for city in CITIES:
        city_root = freeze_root / city
        city_root.mkdir(parents=True)
        certificate = city_root / "freeze.json"
        certificate.write_text(
            json.dumps(
                {
                    "protocol": "tsc-v55r51-external-v9-oof-validated-support-freeze-v1",
                    "city": city,
                    "selected_candidate": "constant_alpha_0p8",
                    "deployment": {
                        "policy": "phase_pressure",
                        "target_support": {"prototype_count": 0},
                    },
                    "freeze_gate": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
        model = city_root / "model.pkl"
        model.write_bytes(f"model:{city}".encode())
        audit_cities[city] = {
            "certificate": {"sha256": _sha(certificate)},
            "model": {"sha256": _sha(model)},
            "gate": {"passed": True},
        }
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "protocol": "tsc-v55r51-external-v9-oof-validated-support-joint-audit-v1",
                "status": "PASS",
                "decision": "authorize_v9_estimand_aligned_frozen_development_replay",
                "generation": "v9",
                "gate": {"passed": True},
                "cities": audit_cities,
            }
        ),
        encoding="utf-8",
    )

    payload = build_protocol(
        parent_protocol_path=parent,
        aligned_audit_path=audit,
        freeze_root=freeze_root,
        frozen_at_utc="2026-08-10T02:00:00Z",
    )

    development = payload["development"]
    assert development["adaptation_coupled_diagnostic_seeds"] == [5057, 6067]
    assert development["generalization_selection_seeds"] == [52282, 41242, 63792]
    assert set(payload["validation_reservation"]["closed_loop_seeds"]).isdisjoint(
        development["closed_loop_seeds"]
    )
