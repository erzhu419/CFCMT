"""Post hoc comparison of paired movement responses on the seven shared count points."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path
from statistics import mean


def group(cases):
    rows = defaultdict(list)
    for row in cases:
        rows[tuple(row["near_counts"])].append((row["delta_right"], row["delta_straight"]))
    return rows


def vector_mean(vectors):
    return tuple(Fraction(sum(values), len(values)) for values in zip(*vectors))


def errors(prediction, outcomes):
    return vector_mean([tuple(abs(x - y) for x, y in zip(prediction, observed))
                        for observed in outcomes])


def labels(vector):
    return dict(right=float(vector[0]), straight=float(vector[1]), joint=float(mean(vector)))


def readout(source_cases, target_cases, detail=False):
    source, target = group(source_cases), group(target_cases)
    points = sorted(source)
    observed = [i for i, point in enumerate(points) if point in target]
    predictions = [vector_mean(source[p]) for p in points]
    constant = vector_mean(predictions)
    # Compare every target row before averaging rows within its count point.
    matrix = {i: [errors(p, target[points[i]]) for p in predictions] for i in observed}
    matched_errors = [matrix[i][i] for i in observed]
    constant_errors = [errors(constant, target[points[i]]) for i in observed]
    matched_mae, constant_mae = vector_mean(matched_errors), vector_mean(constant_errors)
    costs = {i: [mean(v) for v in matrix[i]] for i in observed}
    distribution = Counter(sum(costs[i][perm[i]] for i in observed) / len(observed)
                           for perm in permutations(range(len(points))))
    total = sum(distribution.values())
    matched = mean(matched_mae)
    permuted = dict(count=total, mean=float(sum(v*n for v,n in distribution.items()) / total),
                    at_most_matched=sum(n for v,n in distribution.items() if v <= matched),
                    equal_matched=distribution[matched])
    output = dict(target_rows=len(target_cases), represented_points=len(observed),
                  matched_mae=labels(matched_mae), constant_mae=labels(constant_mae),
                  matched_minus_constant_joint_mae=float(matched - mean(constant_mae)),
                  permutation_joint_mae=permuted)
    if detail:
        permuted["distribution"] = [dict(joint_mae=float(v), exact=str(v), assignments=n)
                                    for v,n in sorted(distribution.items())]
        output.update(constant_prediction=dict(right=float(constant[0]), straight=float(constant[1])),
                      per_point=[dict(near_counts=list(points[i]), target_rows=len(target[points[i]]),
                                      predicted_delta=list(map(float, predictions[i])),
                                      observed_deltas=target[points[i]],
                                      matched_mae=labels(matrix[i][i]),
                                      constant_mae=labels(errors(constant, target[points[i]])))
                                 for i in observed])
    return output


def run(summary):
    source, target = summary["hangzhou"]["cases"], summary["jinan_common"]["cases"]
    return dict(status="POST_HOC_DESCRIPTIVE", unit="absolute error in vehicle counts per 10 seconds",
                scoring="Use all scheduled observed responses, including valid 0/0 clearance extensions. Average absolute errors across observed Jinan rows within each point, then across points and movements equally. Each permutation moves paired movement predictions together across the seven source labels. The available target points alone receive weight after a target seed is omitted; all seven source predictions and their constant remain fixed.",
                scope="Development observations only. Permutation counts describe assignment ranks, without confirmatory significance.",
                main=readout(source, target, detail=True),
                single_source_seed={seed:readout([r for r in source if r["seed"]==seed], target)
                                    for seed in sorted({r["seed"] for r in source})},
                omit_target_seed={seed:readout(source, [r for r in target if r["seed"]!=seed])
                                  for seed in sorted({r["seed"] for r in target})})


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    result = run(json.loads((directory / "summary.json").read_text()))
    print(json.dumps(result, indent=2))
