# V154B Feature-Aligned Rigid Smoke Correction Protocol

V154B measures the effect of binding the existing rigid ranker's inputs by
stored feature names across the original V150L four-city smoke roster. Cologne
has already completed this correction as task `t90756`; its result is reused.
This protocol freezes the other three runs before their corrected outcomes
are observed.

| City | Scenario | Seed | Corrected run |
|---|---|---:|---|
| Atlanta | `atlanta_1x5` | 41242 | New V154B run |
| Cologne | `cologne1` | 41242 | Reuse V154 task `t90756` |
| New York | `manhattan_28x7` | 41242 | New V154B run |
| RESCO synthetic | `grid4x4` | 41242 | New V154B run |

All runs use snapshot `589fd20266b7265b6f2f`, the original V150K runtime model
for that city, original manifest and conversion data, and the original V150L
rigid target-only arguments: SUMO 1.22.0/libsumo, 3600 seconds, 60-second
warmup, ten-second control interval, 450-second prediction horizon, direct
control and zero cooldown. Each new task requests one CPU, 8192 MB RAM and
no GPU. Model identities and original launch arguments are retained in
`cf_h2o/config/traffic_signal_tsc_v154b_feature_aligned_rigid_smoke.json`.

The only controller change is resolving the anchor's saved feature names in
the current dataset instead of reading saved column positions. Original
fitted models, feature values, occupancy formulas, demand, vehicle parameters,
clearance and collision handling remain fixed. The runs use the
`rigid_target_only` arm; the unrecomputed V150K source-utility payload supplies
no source correction.

For each city, report the corrected result beside the retained original rigid
and original PhasePressure result from the same scenario and seed. Report
waiting together with due demand, departures, arrivals, pending insertion,
active vehicles and system load, so a low waiting value is not interpreted
without its served-demand population. Retain collision incidents and starting
teleports, including adverse results. Present paired numerical differences
without introducing a new performance admission threshold.

This is an implementation-correction diagnostic on an existing development
roster. It does not establish a source benefit, safe-controller admission,
independent confirmation or general superiority to PhasePressure. The known
Cologne correction improved service while retaining collision incidents;
that outcome does not predetermine the three remaining cities.

New signatures are
`CFCMT/v154b/feature-aligned-rigid-smoke-v1/<city>/<scenario>/41242`.
Only the three missing rigid corrections are submitted. Outputs use the new
directory
`cf_h2o/results/cluster/tsc_v154b_feature_aligned_rigid_smoke_20260909/rigid_v1/`.
The four-city comparison artifact will be
`cf_h2o/results/paper_artifacts/tsc_v154b_feature_aligned_rigid_smoke_v1.json`.
All original results and the existing Cologne correction remain retained.
Retrieval is limited to small result JSON files and short logs; runtime models
and temporary tripinfo data stay on the server.
