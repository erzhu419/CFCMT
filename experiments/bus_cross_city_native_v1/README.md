# Halifax native interface

This stage adapts a fixed 12-service Halifax GTFS/APC subset to the frozen
transit-v7 SUMO interface. It records actual passenger state and ledger rewards,
raw 245-dimensional joint transitions and the corresponding 229-dimensional
policy input. The rolling arrival-rate feature uses observed arrivals only.

`protocol.json` fixes the 07:00–11:00 subset rule, seeds and two 1800-second
collection diagnostics. The selected services use the earliest nonholiday
weekday in the supplied GTFS calendar, 2026-05-19: 119 actual trips. The
32 KB `selection/weekday_timetables.json` retains trip/service IDs and stop
signatures; mutually exclusive Sunday/weekday trips are not merged.
Run `launch.py stage`, `stage-inputs`, `stage-timetables`, `prepare`,
`smoke`, then `collect`. Transition HDF5 files remain on the server; local
collection retrieves small summaries only. `runtime_supplement` owns copies
of the seven asset-side simulator modules, alongside the earlier frozen H2O
evaluation package. Existing Thesis source is not changed.

For the formal target pool, `launch.py prepare-data` builds eight separate
demand realizations, `submit-data` collects their complete 24000-second days,
and `collect-pool` assembles 2048 original transitions per day. Six days are
fit, one is stop, and one is calibrate. All final terminal rows are retained;
sampled rows keep their original adjacent next state. Canonical 229-dimensional
observations and raw 245-dimensional states are both stored. The actor reads
`split_id=0`; the world model reads the explicit target roles rather than
splitting the actor's fit-only replay again. The resulting HDF5 remains remote.

The planned three-branch comparison uses the same 16384-row target union and
equal update counts. Cross-city initialization replaces target initialization.
The world-model cap is 8192 eligible fit rows, not permission to obtain extra
target observations. Training and test-day protocols are finalized separately.

`training_protocol.json` freezes the next bounded offline pilot: three seeds,
50000 initialization + 5000 target adaptation + 20000 continuation updates per
route. `target_only` uses real target replay; `target_model` adds normalized
Bellman-confidence model TD; `cross_city` replaces initialization with source
initialization and otherwise follows `target_model`. The latter two physically
share their target world model. This learned-model diagnostic differs from the
Thesis live low-fidelity interaction experiment.

Source initialization uses a fixed 16384-row subset of 240 old source shards.
Unavailable causal source arrival rates are omitted, and city-specific line
columns plus the arrival-rate column are reinitialized on transfer. The target
model supports its actual 58 station categories. Only the independent CFCMT
runtime snapshots are imported. `training_launch.py stage`, `prepare-source`,
`prepare-tests`, `smoke`, and `collect-smoke` admit the real interface before
`train`; complete-day policy evaluation uses the three predeclared new demand
seeds. Checkpoints and pools stay on the server.

The network comprises generated route corridors. Route-level APC totals are
observed; directional allocation and OD destinations are generated. Diagnostics
here do not establish cross-city policy gains or full-city traffic performance.
