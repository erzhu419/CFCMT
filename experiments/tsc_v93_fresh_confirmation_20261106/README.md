# V93 fresh-seed confirmation

The frozen V93 controller, MaxPressure, PhasePressure and exact generalized
pressure reference are evaluated on 56 new seeds and the original four scenarios
(896 one-hour runs). The previous V93 panel remains a separate result.

`protocol.json` fixes seeds, controllers, metrics and the analysis before new
outcomes. Only simulator seeds change. Technical smoke uses previously used
seeds. `seed_novelty.json` records zero candidate-token matches in 18,130 existing
local research text files, including archived summaries and reserved protocols.
It excludes downloaded raw datasets, environments, binary archives and Git
history. `seed_novelty_server.json` records zero matches in 17,760 server-side
small text files and 59,786 names, with the exclusions recorded in that receipt.
`smoke_equivalence.json` records exact agreement for all 80 historical replay
metric values. Formal task IDs are t213637 through t213641.

The primary effect first averages Jinan's three scenarios within each seed,
forms a relative waiting change per city, then averages cities and seeds equally.
Report 20,000 paired-seed percentile bootstrap intervals and Holm9-adjusted
two-sided Wilcoxon tests (three comparators by three scopes). Fixed sample size;
no early stopping, replacement seeds or pooling with historical results.
Retain observed mean improvements when superiority is unconfirmed; this study
does not isolate source-data contribution.

After all runs are valid and complete, export compact scalar records with the
CSV schema in `protocol.json`, then run from the repository root:

```bash
python3 experiments/tsc_v93_fresh_confirmation_20261106/finalize.py
python3 experiments/tsc_v93_fresh_confirmation_20261106/analyze.py \
  --records experiments/tsc_v93_fresh_confirmation_20261106/evaluation/rollouts.csv \
  --out-dir experiments/tsc_v93_fresh_confirmation_20261106/analysis
```

The analyzer rejects incomplete, duplicate or substituted identities and writes
`comparisons.csv`, `summary.csv`, `paired_seed.csv` and `analysis.json`. A favorable
mean, an interval below zero and Holm9 p < 0.05 are all required for each confirmed
waiting advantage. Raw logs, tripinfo XML, models and networks stay on the server.

Completed: all 896 identities are present. `final_collection.json` retains 833
original completions plus 63 cells from t213664 (four interrupted retries and
59 first executions); all 73 unplanned scheduler-rerun completions are excluded.
Fresh macro waiting is lowest for V93 (145.5874 s), but no macro comparison
passes Holm9: MP -1.2373% (p=.0690), PP -0.5687% (p=.6066), exact GP -0.4195%
(p=1). Intervals and all nine comparisons are in `analysis/comparisons.csv`.
Jinan repeats 167/168 exact ten-metric ties with GP, with one executed override.
Collision incidents total 16/23/11/12 for V93/MP/PP/GP; all teleport counts are
zero. The waiting-mean ranking does not establish dominance on every outcome.

The separate `guard_diagnosis_adaptation.json` contains 911 fitted Jinan groups:
all 695 model/GP disagreements are confidence-rejected. Correcting the cycle
scale or removing its term changes no action; the 0.432793 base penalty alone
exceeds every proposed gain. The subsequent seed-separated final-action
calibration completed with HOLD_NO_EFFECT; see the sibling selected-action
calibration experiment and [current research direction](../RESEARCH_DIRECTION_20261106.md).
These fitted, 60-second PP-continuation labels establish neither new control
benefit nor source attribution. The first diagnostic task t213625 failed on a
policy-name interface; the corrected diagnostic completed as t213629.
