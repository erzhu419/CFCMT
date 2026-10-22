"""Read-only before/after continuation probes on the original held-out states.

Run on the server; print a small JSON report without exporting checkpoints.
The final critic uses an entropy backup, so its IQL residual is diagnostic only.
"""

import argparse
import json

import torch

from collect_values import compact_actor
from diagnose_values import (actor_summary, learner_from_payload, read_roles,
    tensors, value_summary)
from source_dense_adaptation import compact_values_with_ranges
from source_dense_continuation import campaign_protocol
from train_native import HERE, runtime_api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=("model-td", "real-td"), default="model-td")
    args = parser.parse_args()
    torch.set_num_threads(1)
    prefix, protocol = campaign_protocol(args.campaign)
    trainer_class, config_class, _, _ = runtime_api()
    raw_roles, contract, _ = read_roles(HERE / "target_pool.h5")
    roles = {name: tensors(raw) for name, raw in raw_roles.items()}
    results = {}
    for seed in protocol["training_seeds"]:
        stages = {}
        for stage, folder, filename in (
            ("adapted", "source_dense_adaptation", "adapted.pt"),
            ("continued", prefix, "final_learner.pt"),
        ):
            path = HERE / folder / f"seed{seed}" / filename
            payload = torch.load(path, map_location="cpu", weights_only=False)
            learner = learner_from_payload(trainer_class, config_class, contract, payload, seed)
            stage_report = dict(stored_updates=payload["total_updates"],
                temperature=float(learner.alpha.detach()), roles={})
            for index, (role, batch) in enumerate(roles.items()):
                values = value_summary(learner, batch, seed+80000+10000*index)
                compact = compact_values_with_ranges(values)
                for name in ("mixed_entropy_target", "mixed_entropy_residual_min",
                             "proposal_advantage_vs_zero_min"):
                    compact[name] = values[name]
                stage_report["roles"][role] = dict(values=compact,
                    actor=compact_actor(actor_summary(learner.actor, batch)))
            stages[stage] = stage_report
            del learner, payload
        results[str(seed)] = stages
    print(json.dumps(dict(complete=True, protocol=protocol["name"],
        updates=0, target_test_rows_read=0,
        role_rows={name: len(batch["actions"]) for name, batch in roles.items()},
        results=results,
        interpretation="Read-only numerical/actor probes on the original stop/calibrate states. Final Q uses the original mixed entropy backup; IQL residual and Q sign alone do not establish failure. Estimated proposal-versus-zero preferences are not counterfactual cost labels.")))


if __name__ == "__main__":
    main()
