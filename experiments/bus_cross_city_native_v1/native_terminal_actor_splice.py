"""Evaluate the frozen75k gate with the same-seed55k native positive action."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

from native_runtime import HERE, bootstrap

PREFIX = "native_terminal_actor_splice"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class FrozenActorSplice(torch.nn.Module):
    """Compose native functions before the evaluator's one execution projection."""

    def __init__(self, gate_actor, amount_actor):
        super().__init__()
        self.gate_actor, self.amount_actor = gate_actor, amount_actor
        self.eval().requires_grad_(False)

    def forward(self, states, deterministic=False):
        if not deterministic:
            raise ValueError("The frozen intervention evaluates deterministic actions only")
        hold = self.gate_actor.distribution_parameters(states)[0].sigmoid() >= .5
        positive, _ = self.amount_actor.positive_action_and_log_prob(states, deterministic=True)
        action = torch.where(hold[:, None], positive, torch.full_like(positive, -1.))
        return action, None


def load_actors(seed):
    if seed not in P["training_seeds"] or P["variant"] != "gate75_amount55":
        raise ValueError("Actor composition or seed differs from the fixed protocol")
    bootstrap()
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    actors, paths, headers = {}, {}, {}
    components = (("gate75", "gate_actor", 75000, "cfcmt-bus-native-terminal-actor-unmasked-v1"),
        ("amount55", "positive_amount_actor", 55000, "cfcmt-bus-native-terminal-adaptation-v1"))
    for role, key, updates, protocol in components:
        path = HERE / P[key].format(seed=seed)
        actor, meta = load_actor_checkpoint_v7(path, "cpu", require_policy_artifact=True)
        if (meta.get("seed") != seed or meta.get("protocol") != protocol
                or meta.get("historical_policy_updates") != updates
                or type(actor).__name__ != "HurdleTransitActorV7"):
            raise RuntimeError(f"Wrong frozen native actor identity for {role}")
        actor.eval().requires_grad_(False)
        actors[role], paths[role] = actor, str(path.resolve())
        headers[role] = {key: meta[key] for key in ("seed", "protocol",
            "historical_policy_updates", "policy_artifact_schema")}
        headers[role]["actor_type"] = type(actor).__module__ + "." + type(actor).__name__
    return actors, dict(component_checkpoints=paths, component_metadata=headers)


@torch.inference_mode()
def admit_seed(seed):
    from canonical_execution_projection import project_canonical_executed_action
    from native_terminal_iql_drift import read_fit
    actors, identity = load_actors(seed)
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    diagnostic = json.loads((HERE / "native_terminal_iql_drift_protocol.json").read_text())
    batch, contract = read_fit(diagnostic)
    for key in ("max_hold_seconds", "scheduled_headway_scale_seconds"):
        if contract[key] != getattr(DEFAULT_CONFIG_V7, key):
            raise RuntimeError("The fixed fit records use a different action contract")
    splice = FrozenActorSplice(actors["gate75"], actors["amount55"])
    original = FrozenActorSplice(actors["gate75"], actors["gate75"])
    counts = dict(rows=0, gate75_positive_rows=0, gate75_zero_rows=0,
        gate75_positive_gate55_zero_rows=0, composition_differs_from55_full_forward_rows=0,
        requested_differs_from_projected_rows=0)
    for start in range(0, len(batch["actions"]), diagnostic["state_batch_size"]):
        states = batch["observations"][start:start + diagnostic["state_batch_size"]]
        gate75 = actors["gate75"].distribution_parameters(states)[0].sigmoid() >= .5
        gate55 = actors["amount55"].distribution_parameters(states)[0].sigmoid() >= .5
        amount55 = actors["amount55"].positive_action_and_log_prob(states, deterministic=True)[0]
        amount75 = actors["gate75"].positive_action_and_log_prob(states, deterministic=True)[0]
        direct75 = actors["gate75"](states, deterministic=True)[0]
        direct55 = actors["amount55"](states, deterministic=True)[0]
        requested = splice(states, deterministic=True)[0]
        expected = torch.where(gate75[:, None], amount55, torch.full_like(amount55, -1.))
        reconstruction = torch.where(gate75[:, None], amount75, torch.full_like(amount75, -1.))
        if not (torch.equal(original(states, deterministic=True)[0], direct75)
                and torch.equal(reconstruction, direct75) and torch.equal(requested, expected)
                and torch.equal(requested[~gate75], torch.full_like(requested[~gate75], -1.))
                and torch.equal(requested[gate75], amount55[gate75])
                and torch.isfinite(requested).all() and requested.shape == (len(states), 1)):
            raise RuntimeError("Native gate/positive-action composition differs from the fixed intervention")
        projected = project_canonical_executed_action(states, requested, DEFAULT_CONFIG_V7)
        if not torch.equal(projected, project_canonical_executed_action(states, expected, DEFAULT_CONFIG_V7)):
            raise RuntimeError("Composition and direct actions differ after one canonical projection")
        counts["rows"] += len(states)
        counts["gate75_positive_rows"] += int(gate75.sum())
        counts["gate75_zero_rows"] += int((~gate75).sum())
        counts["gate75_positive_gate55_zero_rows"] += int((gate75 & ~gate55).sum())
        counts["composition_differs_from55_full_forward_rows"] += int((requested != direct55).any(1).sum())
        counts["requested_differs_from_projected_rows"] += int((requested != projected).any(1).sum())
    return dict(complete=True, protocol=P["name"], seed=seed, **identity, **counts,
        new_training_updates=0, world_model_refit_updates=0, new_simulations=0,
        validations=dict(native75_deterministic_reconstruction_exact=True,
            native55_positive_action_exact=True, native75_gate_threshold_half=True,
            closed_gate_action_exact_minus_one=True, requested_composition_exact=True,
            single_canonical_projection_exact=True, separate_native_encoders=True),
        target_pool=str((HERE / diagnostic["target_pool"]).resolve()))


def admission():
    torch.set_num_threads(1)
    rows = {str(seed): admit_seed(seed) for seed in P["training_seeds"]}
    report = dict(complete=True, protocol=P["name"], variant=P["variant"],
        training_seeds=P["training_seeds"], results=rows,
        new_training_updates=0, world_model_refit_updates=0, new_simulations=0)
    write(HERE / (PREFIX + "_admission_summary.json"), report)
    print(json.dumps(report), flush=True)


def evaluate_case(seed, case_id):
    from evaluate_native import evaluate
    admitted = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
    if not (admitted["complete"] and admitted["protocol"] == P["name"]
            and admitted["variant"] == P["variant"]
            and admitted["training_seeds"] == P["training_seeds"]
            and set(admitted["results"]) == {str(value) for value in P["training_seeds"]}
            and all(row["complete"] for row in admitted["results"].values())):
        raise RuntimeError("All three frozen actor compositions must pass fit-state admission")
    actors, identity = load_actors(seed)
    if any(identity[key] != admitted["results"][str(seed)][key] for key in identity):
        raise RuntimeError("Evaluation component identities differ from the admitted actors")
    folder = HERE / PREFIX / P["variant"] / f"seed{seed}" / f"case{case_id:02d}"
    args = SimpleNamespace(method="cross_city", seed=seed, case=case_id, checkpoint=None,
        assets=Path(R["assets_root"]) / P["asset_folder"] / f"case{case_id:02d}" / "asset_manifest.json",
        output=folder, evaluation_protocol=HERE / (PREFIX + "_protocol.json"), actor_action_space="executed")
    extra = dict(method=P["variant"], variant=P["variant"], **identity,
        new_training_updates=0, world_model_refit_updates=0,
        admission_protocol=admitted["protocol"], admission_fit_rows=admitted["results"][str(seed)]["rows"],
        admission_all_training_seeds_complete=True,
        intervention=P["intervention"], interpretation=P["interpretation"],
        actor_composition="native75_gate_native55_positive_action_each_with_own_encoder",
        requested_action_semantics="75k native hold probability>=0.5 selects55k native deterministic positive action; otherwise exact-1. Existing evaluation execution projection once; no hold division.")
    evaluate(args, actor=FrozenActorSplice(actors["gate75"], actors["amount55"]), result_extra=extra)


def run(seed):
    for case in P["evaluation"]["cases"]:
        subprocess.run([sys.executable, "-u", str(HERE / (PREFIX + ".py")), "case",
            "--seed", str(seed), "--case", str(case["case"])], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("admission", "run", "case"))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    parser.add_argument("--case", type=int, choices=[row["case"] for row in P["evaluation"]["cases"]])
    args = parser.parse_args()
    if args.mode == "admission":
        admission()
    elif args.seed is None or (args.mode == "case" and args.case is None):
        parser.error("Evaluation requires its fixed seed, and case mode requires its fixed case")
    elif args.mode == "case":
        evaluate_case(args.seed, args.case)
    else:
        run(args.seed)
