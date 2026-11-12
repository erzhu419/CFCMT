"""Offline physical-arrival residuals; no fitting, simulation or per-vehicle export."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean, median
import sys

ORIGINAL = Path(__file__).resolve().parents[1] / "tsc_corridor_arrival_transfer_20261107"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ORIGINAL))
from arrival_policy import ArrivalPolicy, fit_hours  # noqa: E402
from model_gate import hour, load_training, release_id, rows, verify_protocol  # noqa: E402
from history_baseline import causal_predictions  # noqa: E402

SCOPES = ("all_observable_risk", "original_common_scored")


def conditions(row, flight, previous):
    """Features use this decision and its immediately preceding 10s snapshot only."""
    eta = flight["analytic_eta_s"]
    delay = (flight["age_s"] + eta + flight["tail_s"] - flight["freeflow_s"]) / flight["freeflow_s"]
    prior = previous.get(release_id(flight))
    ratio = None if prior is None else (prior["remaining_m"] - flight["remaining_m"]) / 10 / flight["speed_limit_mps"]
    protected = int(row["protected"][flight["edge"]][row["pp"]]).bit_count()
    return {
        "current_lane": "straight" if flight["on_straight_lane"] else "other",
        "analytic_eta_s": "<=5" if eta <= 5 else "(5,10]" if eta <= 10 else "(10,20]" if eta <= 20 else ">20",
        "entry_queue": str(bool(flight["queued_at_entry"])).lower(),
        "normalized_elapsed_delay": "<=0" if delay <= 0 else "(0,0.25]" if delay <= .25 else ">0.25",
        "past_10s_speed_ratio": "missing" if ratio is None else "<=0.5" if ratio <= .5 else "(0.5,0.9]" if ratio <= .9 else "(0.9,1.1]" if ratio <= 1.1 else ">1.1",
        "nominal_PP_protected_seconds": "0" if protected == 0 else "10" if protected == 10 else "1-9",
    }


def observation(row, flight, outcome, previous, identity_prefix):
    t = row["t"]
    if outcome["end_s"] <= t or flight["age_s"] != t - outcome["entry_s"]:
        raise ValueError("Outcome does not belong to the pre-decision risk set")
    if outcome["status"] == "censored" and outcome["end_s"] < t + 10:
        raise ValueError("Ten-second observation window is incomplete")
    pred = bool(flight["on_straight_lane"] and flight["analytic_eta_s"] <= 10)
    actual = bool(outcome["status"] == "straight" and outcome["end_s"] <= t + 10)
    category = "true_positive" if pred and actual else "true_negative" if not pred and not actual else "false_positive" if pred else "false_negative"
    reason = None
    if category == "false_positive":
        reason = "timing_fp" if outcome["status"] == "straight" else "censored_fp" if outcome["status"] == "censored" else "competing_fp"
    elif category == "false_negative":
        reason = "timing_fn" if flight["on_straight_lane"] else "lanechange_fn"
    return dict(identity=(*identity_prefix, *release_id(flight)), t=t, eta=flight["analytic_eta_s"],
                current_straight=bool(flight["on_straight_lane"]), actual=actual, pred=pred,
                status=outcome["status"], category=category, reason=reason,
                residual=outcome["end_s"] - t - flight["analytic_eta_s"]
                if flight["on_straight_lane"] and outcome["status"] == "straight" else None,
                conditions=conditions(row, flight, previous))


class Summary:
    def __init__(self):
        self.counts = Counter()
        self.vehicles = set()
        self.residuals = []

    def add(self, item):
        self.counts["snapshots"] += 1
        self.vehicles.add(item["identity"])
        self.counts[item["category"]] += 1
        self.counts["actual_arrivals_10s"] += item["actual"]
        self.counts["predicted_arrivals_10s"] += item["pred"]
        self.counts["final_" + item["status"]] += 1
        if item["reason"]:
            self.counts[item["reason"]] += 1
        if item["residual"] is not None:
            value = item["residual"]
            self.residuals.append(value)
            self.counts["early_beyond_1s"] += value < -1
            self.counts["late_beyond_1s"] += value > 1
            self.counts["within_1s"] += -1 <= value <= 1

    def merge(self, other):
        self.counts.update(other.counts)
        self.vehicles.update(other.vehicles)
        self.residuals.extend(other.residuals)

    def result(self):
        keys = ("snapshots", "true_positive", "true_negative", "false_positive", "false_negative",
                "actual_arrivals_10s", "predicted_arrivals_10s", "timing_fp", "timing_fn", "lanechange_fn",
                "competing_fp", "censored_fp", "final_straight", "final_other", "final_terminal", "final_censored",
                "early_beyond_1s", "late_beyond_1s", "within_1s")
        result = {key: self.counts[key] for key in keys}
        result.update(unique_vehicles=len(self.vehicles), repeated_snapshots=self.counts["snapshots"] - len(self.vehicles),
                      residual_n=len(self.residuals), residual_mean=mean(self.residuals) if self.residuals else None,
                      residual_median=median(self.residuals) if self.residuals else None)
        return result


class Panel:
    def __init__(self):
        self.cells = defaultdict(Summary)
        self.nearest = {}
        self.edge_error = 0.0
        self.history_edge_error = 0.0
        self.edge_epochs = 0

    def add(self, item):
        self.cells["overall"].add(item)
        for name, value in item["conditions"].items():
            self.cells[name + ":" + value].add(item)
        if item["current_straight"]:
            old = self.nearest.get(item["identity"])
            if old is None or (abs(item["eta"] - 10), item["t"]) < (abs(old["eta"] - 10), old["t"]):
                self.nearest[item["identity"]] = item

    def result(self):
        nearest = defaultdict(Summary)
        for item in self.nearest.values():
            nearest["overall"].add(item)
            for name, value in item["conditions"].items():
                nearest[name + ":" + value].add(item)
        return dict(snapshot_weighted={key: cell.result() for key, cell in sorted(self.cells.items())},
                    nearest_eta10_current_straight={key: cell.result() for key, cell in sorted(nearest.items())},
                    edge_epochs=self.edge_epochs, analytic_count_mae=self.edge_error / self.edge_epochs if self.edge_epochs else None,
                    history_count_mae=self.history_edge_error / self.edge_epochs if self.edge_epochs else None)

    def merge(self, other):
        for key, cell in other.cells.items():
            self.cells[key].merge(cell)
        self.nearest.update(other.nearest)
        self.edge_error += other.edge_error
        self.history_edge_error += other.history_edge_error
        self.edge_epochs += other.edge_epochs


def analyze_hour(directory, *, city, seed, part="dev", block=None, policy=None):
    labels = list(rows(directory / "labels.jsonl.gz"))
    outcomes = {release_id(record): record for record in labels}
    if len(outcomes) != len(labels) or any(r["status"] == "unknown" for r in labels):
        raise ValueError("Duplicated or unresolved vehicle labels")
    receipt = json.loads((directory / "result.json").read_text())
    horizon = receipt["horizon_sec"]
    panels = {name: Panel() for name in SCOPES if name != "original_common_scored" or policy is not None}
    counts = Counter()
    previous_rows = {}
    baseline_history = {}
    identity_prefix = (city, part, seed)
    for row in rows(directory / "decisions.jsonl.gz"):
        counts["all_decisions"] += 1
        counts["locked_decisions"] += len(row["states"]) == 1
        counts["spillback_decisions"] += row["spillback"]
        counts["unmatched_decisions"] += row["unmatched"]
        counts["no_flight_decisions"] += not row["flights"]
        counts["all_moving_flight_snapshots"] += len(row["flights"])
        previous_time, previous_flights = previous_rows.get(row["tls"], (None, {}))
        previous = previous_flights if previous_time == row["t"] - 10 else {}
        baseline = causal_predictions(row, baseline_history)
        observable = bool(not row["spillback"] and not row["unmatched"] and row["flights"] and row["t"] + 10 <= horizon)
        forecasts = policy.forecast(row) if policy is not None else None
        common = bool(forecasts) and observable
        counts["observable_decisions"] += observable
        counts["original_common_decisions"] += common
        counts["observable_noncommon_decisions"] += observable and not common if policy is not None else 0
        if observable:
            observations = [observation(row, flight, outcomes[release_id(flight)], previous, identity_prefix)
                            for flight in row["flights"]]
            edges = defaultdict(list)
            for flight, item in zip(row["flights"], observations):
                edges[flight["edge"]].append(item)
            for scope in ("all_observable_risk", "original_common_scored") if common else ("all_observable_risk",):
                panel = panels[scope]
                for item in observations:
                    panel.add(item)
                for edge, items in edges.items():
                    observed = sum(item["actual"] for item in items)
                    predicted = float(forecasts[edge][3, 10]) if scope == "original_common_scored" else sum(item["pred"] for item in items)
                    panel.edge_error += abs(predicted - observed)
                    panel.history_edge_error += abs(baseline["history"][edge][-1] - observed)
                    panel.edge_epochs += 1
        previous_rows[row["tls"]] = row["t"], {release_id(flight): flight for flight in row["flights"]}
    record = dict(city=city, seed=seed, part=part, block=block, counts=dict(counts),
                  release_labels_by_status=dict(Counter(r["status"] for r in labels)),
                  scopes={name: panel.result() for name, panel in panels.items()})
    return record, panels


def diagnose(root, protocol, *, parts="dev"):
    cities = protocol["cities"]
    verify_protocol(cities)
    training = {}
    for city in cities:
        source_hours = load_training(root, "donor", city, cities[city]["donor"])
        training[city] = dict(source=fit_hours(source_hours), placebo=fit_hours(source_hours, permute=True),
                              target=[fit_hours(load_training(root, "adapt", city, block)) for block in cities[city]["adapt"]])
    results = []
    combined = defaultdict(Panel)
    for city, donor_city in (("hangzhou", "jinan"), ("jinan", "hangzhou")):
        spec = cities[city]
        for part in (("donor", "adapt", "dev") if parts == "all" else ("dev",)):
            seeds = sum(spec[part], []) if part == "adapt" else spec[part]
            for index, seed in enumerate(seeds):
                block = index // 2 if part in ("dev", "adapt") else None
                policy = (ArrivalPolicy(training[city]["target"][block], training[donor_city]["source"], training[donor_city]["placebo"])
                          if part == "dev" else None)
                record, panels = analyze_hour(hour(root, part, city, seed), city=city, seed=seed, part=part, block=block, policy=policy)
                results.append(record)
                for scope, panel in panels.items():
                    combined[city, part, scope].merge(panel)
                print(f"Analyzed {part}/{city}/{seed}", flush=True)
    city_results = defaultdict(lambda: defaultdict(dict))
    for (city, part, scope), panel in combined.items():
        result = panel.result()
        per_seed = [r["scopes"][scope]["analytic_count_mae"] for r in results if r["city"] == city and r["part"] == part]
        result["equal_seed_analytic_count_mae"] = mean(per_seed)
        result["equal_seed_history_count_mae"] = mean(
            r["scopes"][scope]["history_count_mae"] for r in results if r["city"] == city and r["part"] == part)
        result["seed_count"] = len(per_seed)
        city_results[city][part][scope] = result
    return dict(status="OFFLINE_DESCRIPTIVE_DIAGNOSTIC", parts=parts, new_simulations=0,
                original_gate_unchanged="HOLD_DEVELOPMENT", seed_results=results, city_summaries=city_results,
                definitions={
                    "all_observable_risk": "All current moving paired flights at unblocked/matched decisions with a complete next 10s; no learned-support filter.",
                    "original_common_scored": "Dev only: same ArrivalPolicy.forecast support and analytic prediction as the original model_gate.py.",
                    "time_residual": "First straight detector crossing 30m before stop line minus decision time minus remaining_m/posted_speed; current-straight and observed-final-straight only. Current-other to final-straight stays in lane-change counts, not timing residuals. Descriptive, not censor-adjusted.",
                    "early_late": "Early <-1s, late >1s; inclusive +/-1s tolerance for 1s observation.",
                    "nearest": "Among current-straight observations of each release, minimize (abs(analytic_eta_s-10),decision_time), without conditioning selection on future outcome.",
                    "speed": "Past 10s mean position advance / posted speed from immediately preceding snapshot; absent observation remains missing, not zero.",
                    "weights": "Nearest-ETA10 current-straight observation per release is the main timing description; repeated-snapshot summaries are sensitivity analyses. City predictive MAE also equally averages complete seeds. No formal inference or learned correction.",
                })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parts", choices=("dev", "all"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.root, json.loads(args.protocol.read_text()), parts=args.parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
