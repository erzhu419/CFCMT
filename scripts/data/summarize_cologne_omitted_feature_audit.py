"""Describe omitted-feature differences for the fixed48 queries and144 neighbors."""

from collections import Counter
import math
from statistics import median

FEATURE_NAMES = ["current_green_elapsed_norm"] + [
    prefix + name for name in ("green_q_max", "red_q_max", "green_q_cv", "red_q_cv")
    for prefix in ("", "reference_", "delta_")]
SEED_COUNTS = {2027: 17, 3037: 17, 4047: 14}
OUTCOME_COUNTS = {"beneficial": 31, "harmful": 15, "tie": 2}
RELATIONS = ("agree", "disagree", "tie_involved")
TOLERANCE = 1e-12


def gain_sign(value):
    return 1 if value > TOLERANCE else -1 if value < -TOLERANCE else 0


def gain_sign_relation(query_gain, neighbor_gain):
    query_sign, neighbor_sign = gain_sign(query_gain), gain_sign(neighbor_gain)
    return "tie_involved" if not query_sign or not neighbor_sign else "agree" if query_sign == neighbor_sign else "disagree"


def distribution(values):
    return {"count": len(values), "minimum": min(values) if values else None,
            "median": median(values) if values else None, "maximum": max(values) if values else None}


def feature_distributions(pairs, names):
    return {name: {
        "absolute_raw_difference": distribution([pair["raw"][index] for pair in pairs]),
        "absolute_standardized_difference": distribution([pair["standardized"][index] for pair in pairs
                                                           if pair["standardized"][index] is not None]),
        "constant_pair_count": sum(pair["standardized"][index] is None for pair in pairs),
        "constant_mismatch_count": sum(pair["standardized"][index] is None and pair["raw"][index] > 0 for pair in pairs)}
        for index, name in enumerate(names)}


def summarize_pairs(pairs, names):
    return {"query_count": len({pair["group_id"] for pair in pairs}), "pair_count": len(pairs),
            "gain_sign_relation_counts": {relation: sum(pair["relation"] == relation for pair in pairs) for relation in RELATIONS},
            "features": feature_distributions(pairs, names),
            "by_gain_sign_relation": {relation: {
                "pair_count": sum(pair["relation"] == relation for pair in pairs),
                "features": feature_distributions([pair for pair in pairs if pair["relation"] == relation], names)}
                for relation in RELATIONS}}


def summarize_audit(audit):
    """Keep every frozen query-neighbor pair; constant columns have no scaled gap."""
    names, queries = audit["feature_names"], audit["queries"]
    if (names != FEATURE_NAMES or audit["query_count"] != 48 or len(queries) != 48
            or len({query["group_id"] for query in queries}) != 48
            or Counter(query["seed"] for query in queries) != SEED_COUNTS
            or Counter(query["outcome"] for query in queries) != OUTCOME_COUNTS
            or audit["pair_count"] != 144 or any(len(query["neighbors"]) != 3 for query in queries)):
        raise ValueError("Require all48 original queries,144 pairs and the fixed13 omitted features")
    folds = {fold["heldout_seed"]: fold for fold in audit["folds"]}
    if len(audit["folds"]) != 3 or set(folds) != set(SEED_COUNTS):
        raise ValueError("All three corresponding training folds are required")
    for fold in folds.values():
        arrays = [fold[key] for key in ("feature_std", "feature_min", "feature_max")]
        if (any(len(values) != len(names) or not all(math.isfinite(value) for value in values) for values in arrays)
                or fold["constant_features"] != [name for name, lo, hi in zip(names, fold["feature_min"], fold["feature_max"]) if lo == hi]
                or any(lo > hi or (lo != hi and std <= 0)
                       for std, lo, hi in zip(*arrays))):
            raise ValueError("Feature scaling and constant columns must come from the query's training fold")
    pairs = []
    for query in queries:
        fold = folds[query["seed"]]
        if (not math.isfinite(query["actual_gain"])
                or query["outcome"] != {1: "beneficial", -1: "harmful", 0: "tie"}[gain_sign(query["actual_gain"])]
                or len({neighbor["group_id"] for neighbor in query["neighbors"]}) != 3):
            raise ValueError("Query outcomes and three distinct neighbor groups must remain intact")
        values = query["feature_values"]
        for neighbor in query["neighbors"]:
            neighbor_values = neighbor["feature_values"]
            if (any(len(vector) != len(names) or not all(math.isfinite(value) for value in vector)
                    for vector in (values, neighbor_values)) or not math.isfinite(neighbor["raw_gain"])
                    or neighbor["seed"] == query["seed"]):
                raise ValueError("Each pair needs13 finite feature values and a training-fold neighbor")
            raw = [abs(left - right) for left, right in zip(values, neighbor_values)]
            scaled = [None if lo == hi else difference / std for difference, std, lo, hi
                      in zip(raw, fold["feature_std"], fold["feature_min"], fold["feature_max"])]
            pairs.append({"group_id": query["group_id"], "seed": query["seed"], "outcome": query["outcome"],
                          "relation": gain_sign_relation(query["actual_gain"], neighbor["raw_gain"]),
                          "raw": raw, "standardized": scaled})
    return {"feature_names": names, "query_count": len(queries), "pair_count": len(pairs),
            "query_outcome_counts": dict(Counter(query["outcome"] for query in queries)),
            "query_seed_counts": {str(seed): sum(query["seed"] == seed for query in queries) for seed in SEED_COUNTS},
            "overall": summarize_pairs(pairs, names),
            "per_seed": {str(seed): summarize_pairs([pair for pair in pairs if pair["seed"] == seed], names) for seed in SEED_COUNTS},
            "per_query_outcome": {outcome: summarize_pairs([pair for pair in pairs if pair["outcome"] == outcome], names)
                                  for outcome in OUTCOME_COUNTS},
            "checks": {"all48_queries_retained": len(queries) == 48, "all144_pairs_retained": len(pairs) == 144,
                       "original_query_outcomes_exact": Counter(query["outcome"] for query in queries) == OUTCOME_COUNTS,
                       "original_query_seed_counts_exact": Counter(query["seed"] for query in queries) == SEED_COUNTS},
            "standardization": "Absolute query-neighbor difference divided by the query's training-fold feature standard deviation. "
                               "Columns with equal training minimum and maximum are excluded from standardized distributions; "
                               "their raw differences and nonzero mismatch counts are retained.",
            "gain_sign_relation_definition": "Agree/disagree compare two nonzero raw label gain signs; either gain within1e-12 of zero is tie_involved."}
