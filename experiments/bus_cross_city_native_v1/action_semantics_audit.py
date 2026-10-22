"""Read-only replay of the frozen guard on target-fit action labels."""

import json

import h5py
import numpy as np

from native_runtime import HERE, bootstrap, project_executed_action


def main():
    api = bootstrap()
    protocol = json.loads((HERE / "protocol.json").read_text())
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == 0)
        raw = data["raw_observations"][ids]
        actions = data["actions"][ids]
        proposals = data["proposed_actions"][ids]
        proposed_seconds = data["proposed_hold_seconds"][ids].reshape(-1)
        executed_seconds = data["executed_hold_seconds"][ids].reshape(-1)
        config = api.contract.TransitMDPConfigV7(**json.loads(data.attrs["config"]))
    guard = protocol["smoke"]["completion_guard"]
    apply = lambda batch: np.asarray([api.completion_safety_action(s, a.copy(), guard, config)
        for s, a in zip(raw, batch)])
    replayed = apply(proposals)
    repeated = apply(actions)
    project = lambda batch: np.asarray([project_executed_action(api, s, a, guard, config)
        for s, a in zip(raw, batch)])
    projected = project(actions)
    projected_twice = project(projected)
    seconds = lambda batch: np.asarray(api.contract.policy_action_to_hold_seconds_v7(batch[:, 0], config))
    labelled = seconds(actions)
    replayed_seconds = seconds(replayed)
    repeated_seconds = seconds(repeated)
    positive = labelled > 0
    # Normalized float32 actions introduce a few microseconds of decode error.
    changed = np.abs(repeated_seconds-labelled) > 1e-4
    def describe(mask):
        count = int(np.sum(mask))
        return dict(rows=count, labelled_hold_seconds=float(np.mean(labelled[mask])) if count else None,
            after_repeated_guard_seconds=float(np.mean(repeated_seconds[mask])) if count else None)
    report = dict(complete=True, fit_rows=len(ids), model_updates=0, simulator_runs=0,
        source="target_pool.h5 fit rows only; no test states or checkpoint reads",
        recorded_actions_vs_executed_seconds_max_error=float(np.max(np.abs(labelled-executed_seconds))),
        recorded_proposals_vs_proposed_seconds_max_error=float(np.max(np.abs(seconds(proposals)-proposed_seconds))),
        replay_original_proposal_vs_recorded_action_max_error=float(np.max(np.abs(replayed-actions))),
        replay_original_proposal_vs_recorded_seconds_max_error=float(np.max(np.abs(replayed_seconds-labelled))),
        recorded_action_differs_from_proposal_rows=int(np.sum(np.max(np.abs(actions-proposals), axis=1)>1e-7)),
        labelled_hold_seconds_mean=float(np.mean(labelled)),
        proposed_hold_seconds_mean=float(np.mean(proposed_seconds)),
        repeated_guard_hold_seconds_mean=float(np.mean(repeated_seconds)),
        repeated_guard_changes=describe(changed),
        repeated_guard_decreases=describe(repeated_seconds < labelled-1e-4),
        repeated_guard_increases=describe(repeated_seconds > labelled+1e-4),
        unchanged_positive=describe(positive & ~changed),
        changed_positive=describe(positive & changed),
        positive_label_rows=int(np.sum(positive)),
        changed_positive_mean_scale=float(np.mean(repeated_seconds[positive & changed]/labelled[positive & changed])) if np.any(positive & changed) else None,
        execution_projection=dict(
            fit_labels_max_action_error=float(np.max(np.abs(projected-actions))),
            fit_labels_changed_rows=int(np.sum(np.max(np.abs(projected-actions), axis=1)>1e-7)),
            idempotence_max_action_error=float(np.max(np.abs(projected_twice-projected))),
            formerly_shrunk_rows_preserved=int(np.sum(changed & (np.max(np.abs(projected-actions), axis=1)==0)))),
        execution_layer=protocol["smoke"]["execution_layer"], completion_guard=guard,
        interpretation="Guard replay at the same recorded state, not an actor prediction or a closed-loop causal performance test.")
    (HERE / "action_semantics_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
