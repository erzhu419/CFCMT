"""Replace Salt Lake City cells with reruns on the corrected executor."""

import argparse
import json
from pathlib import Path


def replace_saltlake(old, new, keys):
    def key(row):
        return tuple(row[k] for k in keys)

    old_keys = [key(row) for row in old if row["family"] == "saltlake"]
    updates = {key(row): row for row in new if row["family"] == "saltlake"}
    if len(old_keys) != len(new) or len(new) != len(updates) or set(old_keys) != updates.keys():
        raise ValueError("The two Salt Lake City panels must contain the same distinct cells.")
    return [updates[key(row)] if row["family"] == "saltlake" else row for row in old]


def merge_summaries(archived: Path, runs: Path) -> dict:
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))

    merged = {}
    policies = (
        ("phase3_rl_baselines_repaired_result.json", "cv/phase3_rl_baselines_result.json"),
        ("phase3_rl_source_only_result.json", "source/phase3_rl_source_only_result.json"),
    )
    for name, rerun in policies:
        original, corrected = read(archived / name), read(runs / rerun)
        if corrected["training_audit"]["source_pretrain"] != {"reused_frozen_source_weights": True}:
            raise ValueError("Expected to reuse the archived source pretraining.")
        original["rows"] = replace_saltlake(
            original["rows"], corrected["rows"], ("scenario", "seed", "policy")
        )
        original["training_audit"]["families"]["saltlake"] = corrected["training_audit"]["families"]["saltlake"]
        original["family_summary"]["saltlake"] = corrected["family_summary"]["saltlake"]
        merged[name] = original

    name = "phase4_truth_reward_rl_result.json"
    original = read(archived / name)
    corrected = read(runs / "truth/truth_reward_rl_result.json")
    for field in ("repeat_eval_seeds", "target_training_seeds"):
        if original[field] != corrected[field]:
            raise ValueError(f"Truth-reward seeds changed: {field}")
    if original["acquisition"]["source_library"] != corrected["acquisition"]["source_library"]:
        raise ValueError("The source label library changed.")
    original["rows"] = replace_saltlake(
        original["rows"], corrected["rows"], ("scenario", "seed", "training_seed")
    )
    original["fits"] = replace_saltlake(original["fits"], corrected["fits"], ("seed",))
    original["acquisition"]["target"]["saltlake"] = corrected["acquisition"]["target"]["saltlake"]
    original["city_metrics"]["saltlake"] = corrected["city_metrics"]["saltlake"]
    merged[name] = original
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = merge_summaries(args.archived, args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, payload in summaries.items():
        (args.output / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
