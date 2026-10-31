"""Read-only fit-state action ranking and frozen SAC objective decomposition."""

import argparse
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from canonical_execution_projection import canonical_execution_bounds, project_canonical_executed_action
from diagnose_values import learner_from_payload
from train_native import runtime_api, write

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "fit_action_ranking_protocol.json").read_text())
STAGES = tuple(P["stages"])


def average(values, mask):
    return float(np.asarray(values)[mask].mean())


def describe(values, mask):
    values = np.asarray(values)[mask]
    return dict(mean=float(values.mean()), median=float(np.median(values)),
        p10=float(np.quantile(values, .1)), p90=float(np.quantile(values, .9)),
        positive_fraction=float((values > 0).mean()))


@torch.no_grad()
def diagnose(seed, admission=False):
    started = time.monotonic()
    torch.set_num_threads(1)
    base, config_class, _, _ = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == P["fit_role"])
        assert len(ids) == P["target_fit_rows"]
        if admission:
            ids = ids[np.linspace(0, len(ids)-1, 64, dtype=int)]
        states = torch.as_tensor(data["observations"][ids], dtype=torch.float32)
        labels = torch.as_tensor(data["actions"][ids], dtype=torch.float32).reshape(-1, 1)
        contract = json.loads(data.attrs["config"])
    lower, upper = canonical_execution_bounds(states, DEFAULT_CONFIG_V7)
    projector = lambda s,a: project_canonical_executed_action(s, a, DEFAULT_CONFIG_V7)
    assert float((projector(states, labels)-labels).abs().max()) < P["action_equality_tolerance"]
    learners, artifacts = {}, {}
    for stage, spec in P["stages"].items():
        path = HERE / spec["folder"] / f"seed{seed}" / spec["filename"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["total_updates"] == spec["stored_updates"]
        learners[stage] = learner_from_payload(base, config_class, contract, payload, seed)
        assert learners[stage].config.actor_q_reduction == "min-all"
        assert learners[stage].config.ensemble_size == 2
        artifacts[stage] = str(path)
        del payload
    fields = {}
    def add(key, value):
        fields.setdefault(key, []).append(value.detach().cpu().numpy().reshape(-1))
    k = P["positive_samples_per_state"]
    size = P["state_batch_size"]
    for start in range(0, len(states), size):
        batch, recorded = states[start:start+size], labels[start:start+size]
        actions = dict(recorded=recorded)
        components = {}
        for stage, learner in learners.items():
            actor = learner.actor
            actions[stage] = projector(batch, actor(batch, deterministic=True)[0])
            with torch.random.fork_rng():
                torch.manual_seed(seed + P["common_draw_seed_offset"] + start)
                part = actor.repeated_soft_action_components(batch, k)
            repeated_states = batch.repeat_interleave(k, dim=0)
            part["zero_action"] = projector(repeated_states, part["zero_action"])
            part["positive_action"] = projector(repeated_states, part["positive_action"])
            components[stage] = part
        for candidate, action in actions.items():
            add("action/"+candidate, action)
        for critic_stage, learner in learners.items():
            critic = learner.critic
            features = critic.encoder(batch)
            q = lambda feature, action: critic.forward_features(feature, action).min(0).values
            if admission:
                for action in actions.values():
                    assert torch.equal(q(features,action), critic(batch,action).min(0).values)
            for candidate, action in actions.items():
                add(f"q/{critic_stage}/{candidate}", q(features, action))
            repeated_features = features.repeat_interleave(k, dim=0)
            for actor_stage, part in components.items():
                p = part["hold_probability"].reshape(-1,k)[:,0]
                log_zero = part["zero_log_prob"].reshape(-1,k)[:,0]
                log_positive = part["positive_log_prob"].reshape(-1,k).mean(1)
                conditional_log = part["positive_conditional_log_prob"].reshape(-1,k).mean(1)
                q0 = q(features, part["zero_action"].reshape(-1,k,1)[:,0])
                qp = q(repeated_features, part["positive_action"]).reshape(-1,k).mean(1)
                eq = (1-p)*q0 + p*qp
                elog = (1-p)*log_zero + p*log_positive
                alpha = learner.alpha.detach()
                gate_q = -p*(1-p)*(qp-q0)
                log_odds = log_positive-conditional_log-log_zero
                gate_entropy = alpha*p*(1-p)*(log_odds+conditional_log)
                prefix = f"objective/{critic_stage}/{actor_stage}/"
                for name, value in dict(expected_q=eq, expected_log_pi=elog,
                    q_loss=-eq, entropy_loss=alpha*elog, loss=-eq+alpha*elog,
                    hold_probability=p, gate_q=gate_q, gate_entropy=gate_entropy,
                    gate_total=gate_q+gate_entropy).items():
                    add(prefix+name, value)
                if admission:
                    direct = (1-p)*(alpha*log_zero-q0)+p*(alpha*log_positive-qp)
                    assert torch.allclose(-eq+alpha*elog, direct, atol=1e-5, rtol=1e-5)
                    assert torch.allclose(gate_q+gate_entropy,
                        p*(1-p)*(alpha*(log_positive-log_zero)-(qp-q0)), atol=1e-6, rtol=1e-5)
        print(json.dumps(dict(seed=seed, diagnostic_rows=min(start+size,len(states)),
            total_rows=len(states))), flush=True)
    fields = {key:np.concatenate(value) for key,value in fields.items()}
    assert all(np.isfinite(value).all() for value in fields.values())
    lo, hi = lower[:,0].numpy(), upper[:,0].numpy()
    groups = dict(all=np.ones(len(states),dtype=bool), free=hi>lo,
        forced60=(lo==1)&(hi==1), other_collapsed=(hi==lo)&(hi!=1))
    groups = {name:mask for name,mask in groups.items() if mask.any()}
    report = dict(complete=True, protocol=P["name"], seed=seed, admission=admission,
        rows=len(states), updates=0, evaluation_rows_read=0, artifacts=artifacts,
        alpha_source="critic/checkpoint stage, shared across candidate actors within each row",
        alphas={stage:float(learner.alpha.detach()) for stage,learner in learners.items()}, groups={})
    tol = P["action_equality_tolerance"]
    for name, mask in groups.items():
        result = dict(rows=int(mask.sum()), actions={}, q_scores={}, rankings={}, objectives={})
        for candidate in P["candidates"]:
            action = fields["action/"+candidate]
            result["actions"][candidate] = dict(mean_hold_seconds=average(30*(action+1), mask),
                positive_fraction=average(action>-1+tol,mask),
                lower_bound_fraction=average(abs(action-lo)<tol,mask),
                upper_bound_fraction=average(abs(action-hi)<tol,mask))
        for critic_stage in STAGES:
            result["q_scores"][critic_stage] = {candidate:average(fields[f"q/{critic_stage}/{candidate}"],mask)
                for candidate in P["candidates"]}
        for final in ("td0","td025"):
            selected = mask & (abs(fields['action/'+final]-fields['action/adapted'])>tol)
            row = dict(different_execution_rows=int(selected.sum()),
                hold_drift_seconds=average(30*(fields['action/'+final]-fields['action/adapted']),mask))
            if selected.any():
                gaps = {c:fields[f'q/{c}/{final}']-fields[f'q/{c}/adapted'] for c in STAGES}
                row['q_gaps'] = {c:describe(value,selected) for c,value in gaps.items()}
                row['initial_not_preferred_to_final_preferred_fraction'] = average((gaps['adapted']<=0)&(gaps[final]>0),selected)
                row['actor_minus_recorded'] = {c:describe(fields[f'q/{c}/{final}']-fields[f'q/{c}/recorded'],selected) for c in STAGES}
            result['rankings'][final] = row
        for critic_stage in STAGES:
            cells = {}
            for actor_stage in STAGES:
                prefix = f"objective/{critic_stage}/{actor_stage}/"
                cell = {key:average(fields[prefix+key],mask) for key in
                    ('expected_q','expected_log_pi','q_loss','entropy_loss','loss','hold_probability')}
                for key in ('gate_q','gate_entropy','gate_total'):
                    values = fields[prefix+key]
                    cell[key] = dict(mean=average(values,mask),mean_abs=average(abs(values),mask),
                        negative_fraction=average(values<0,mask))
                qg,hg = fields[prefix+'gate_q'],fields[prefix+'gate_entropy']
                cell['entropy_gate_larger_abs_fraction'] = average(abs(hg)>abs(qg),mask)
                cells[actor_stage] = cell
            result['objectives'][critic_stage] = cells
        report['groups'][name] = result
    report['elapsed_seconds'] = time.monotonic()-started
    destination = HERE / 'fit_action_ranking' / ('admission' if admission else f'seed{seed}') / 'summary.json'
    write(destination, report)
    print(json.dumps(dict(complete=True, seed=seed, admission=admission, rows=len(states),
        elapsed_seconds=report['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int, choices=P['training_seeds'])
    parser.add_argument('--admission', action='store_true')
    args = parser.parse_args()
    diagnose(args.seed, args.admission)
