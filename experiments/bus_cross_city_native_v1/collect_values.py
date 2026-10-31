"""Return selected numerical diagnostics, keeping checkpoints and full data remote."""

import json
import shlex

def compact_values(row):
    fields = {
        "q_mean": row["q_recorded_mean"]["mean"],
        "q_min_mean": row["q_recorded_min"]["mean"],
        "positive_q_min_fraction": row["q_recorded_min"]["positive_fraction"],
        "value_mean": row["value"]["mean"],
        "iql_target_mean": row["iql_target"]["mean"],
        "iql_residual_min_mean": row["iql_residual_min"]["mean"],
        "iql_residual_min_mae": row["iql_residual_min"]["mae"],
    }
    for subset in ("terminal", "nonterminal"):
        if subset in row:
            fields[subset] = compact_values(row[subset])
    if "proposal_advantage_vs_zero_min" in row:
        fields["proposal_q_advantage_vs_zero_mean"] = row["proposal_advantage_vs_zero_min"]["mean"]
    return fields


def compact_actor(row):
    return dict(rows=row["rows"],
        gate_probability_mean=row["hold_probability"]["mean"],
        proposed_hold_mean_seconds=row["proposed_hold_seconds"]["mean"],
        proposed_positive_fraction=row["proposed_positive_fraction"],
        recorded_hold_mean_seconds=row["recorded_action_hold_seconds"]["mean"])


def compact_report(row):
    result = {key: row[key] for key in ("complete", "seed", "role_rows", "target_test_rows_read", "updates")}
    result["source_fit"] = {name: dict(rows=sample["rows"], values=compact_values(sample["values"]))
        for name, sample in row["source_initializer_on_source_fit"]["samples"].items()}
    result["stages"] = {}
    for stage, values in row["stages"].items():
        result["stages"][stage] = {}
        for role, data in values["roles"].items():
            result["stages"][stage][role] = dict(values=compact_values(data["values"]),
                actor=compact_actor(data["actor"]),
                model_minus_actual={name: {key: value[key] for key in ("mean", "mae")}
                    for name, value in data["same_recorded_action_model_pairs"].items()})
    result["model_predictions"] = row["model_predictions"]
    result["final_actors"] = {branch: dict(critic_available=data["critic_available"],
        roles={role: compact_actor(actor) for role, actor in data["roles"].items()})
        for branch, data in row["final_actors"].items()}
    return result


def main():
    from launch import HERE, R, run_remote

    code = "from pathlib import Path; import json; from collect_values import compact_report; "
    code += "root=Path(" + repr(R["remote_root"]) + "); "
    code += "rows={str(seed):compact_report(json.loads(p.read_text())) for seed in [75101,75102,75103] for p in [root/'diagnostics'/f'seed{seed}'/'summary.json'] if p.exists()}; "
    code += "print(json.dumps(rows))"
    command = [R["python"], "-c", "import sys; sys.path.insert(0," + repr(R["remote_root"]) + "); " + code]
    rows = json.loads(run_remote(shlex.join(command)))
    report = dict(protocol="cfcmt-bus-value-diagnostic-v1", complete=len(rows)==3,
        ready=len(rows), expected=3, results=rows,
        interpretation="Proposed holds and estimated value preferences only; recorded-action one-step model comparisons. Final critics are unavailable.")
    (HERE / "value_diagnostic_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    main()
