"""Fixed continuation with recorded IQL actors and actor updates on controllable execution states."""

import argparse
import json
from pathlib import Path

from iql_actor_continuation import evaluate, train

HERE = Path(__file__).resolve().parent
PREFIX = "free_execution_actor"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--branch", required=True, choices=(*P["branches"], "strong_rule"))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    args = parser.parse_args()
    if args.mode == "train" and args.branch not in ("td0", "td025"):
        parser.error("Only the two free-execution actor arms train")
    if (args.branch == "strong_rule") != (args.seed is None):
        parser.error("Learned actors require their fixed seed; the rule has no training seed")
    if args.mode == "train":
        train(args.branch, args.seed, protocol=P, prefix=PREFIX, backup_entropy=False, free_execution_actor=True)
    else:
        evaluate(args.branch, args.seed, protocol=P, prefix=PREFIX)
