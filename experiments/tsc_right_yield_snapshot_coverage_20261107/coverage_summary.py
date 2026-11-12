"""Count preaction coverage on a movement, junction-decision and snapshot basis."""

GATES = ("stopped_right", "both_straight_stopped", "allowed_point", "eight",
         "contrast", "pp_on_pair", "sensitive")
IDENTITY = ("city", "seed", "time_sec", "tls_id", "right_id", "straight_id")
POINTS = ((1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 2))


def verdict(row, gates):
    values = [row.get(g) for g in gates]
    if False in values:
        return False
    return None if None in values else True


def count(rows):
    epochs = {(r["seed"], r["time_sec"]) for r in rows}
    junctions = {(r["seed"], r["time_sec"], r["tls_id"]) for r in rows}
    focal = sum(r["focal"] for r in rows)
    return dict(movements=len(rows), junction_decisions=len(junctions), snapshot_epochs=len(epochs),
                seeds=len({r["seed"] for r in rows}), focal_movements=focal, nonfocal_movements=len(rows)-focal)


def stages(rows):
    result = {"registered": count(rows)}
    for i, g in enumerate(GATES, 1):
        selected, unknown = [], []
        for row in rows:
            v = verdict(row, GATES[:i])
            if v is True:
                selected.append(row)
            elif v is None:
                unknown.append(row)
        result[g] = dict(**count(selected), unknown_movements=len(unknown))
    return result


def detail(row):
    cols = (*IDENTITY, "point", "focal", "n", "hi", "lo", "pp", "b_hi", "b_lo",
            "pp_on_pair", "sensitive")
    return {k: row.get(k) for k in cols}


def summarize(rows):
    if len({tuple(r[k] for k in IDENTITY) for r in rows}) != len(rows):
        raise ValueError("Duplicate registered movement at one junction snapshot")
    by_city = {}
    support = {}
    strata = {}
    for city, gate in (("hangzhou", "contrast"), ("jinan", "sensitive")):
        city_rows = [r for r in rows if r["city"] == city]
        if not city_rows:
            raise ValueError(f"Missing {city} snapshot rows")
        comparable = [r for r in city_rows if verdict(r, GATES[:5]) is True]
        gates = GATES[:GATES.index(gate) + 1]
        selected = [r for r in city_rows if verdict(r, gates) is True]
        unknown = [r for r in city_rows if verdict(r, gates) is None]
        support[city] = True if selected else (None if unknown else False)
        strata[city] = {tuple(r["point"]) for r in selected}
        by_city[city] = dict(
            stages=stages(city_rows),
            per_seed=[dict(seed=seed, stages=stages([r for r in city_rows if r["seed"] == seed]))
                      for seed in sorted({r["seed"] for r in city_rows})],
            terminal_gate=gate, terminal_support=support[city],
            terminal_by_point=[dict(point=p, **count([r for r in selected if tuple(r["point"]) == p]))
                               for p in POINTS],
            comparable_rows=[detail(r) for r in comparable],
            terminal_unknown_rows=[dict(**detail(r), missing=[g for g in gates if r.get(g) is None])
                                   for r in unknown],
            originally_nominated=[{k: r[k] for k in (*IDENTITY, "point")} for r in city_rows if r["focal"]])
    if False in support.values():
        status = "CLOSE_NO_COVERAGE"
    elif None in support.values():
        status = "UNKNOWN_COVERAGE"
    else:
        status = "COVERAGE_PRESENT"
    return dict(status=status, support=support, shared_observed_points=sorted(strata["hangzhou"] & strata["jinan"]),
                by_city=by_city)
