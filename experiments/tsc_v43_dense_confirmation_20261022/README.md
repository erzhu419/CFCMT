# V43 CFCMT / dense: 10-seed confirmation

Original two city geometries and four demand scenarios; ten preregistered new
SUMO seeds; two frozen policies; 80 completed 3600-second rollouts in ten paired
seed blocks.

`protocol.json` fixes the cohort, estimand and paired-seed bootstrap before
outcomes. `evaluation_protocol.json` copies the original V43 specification,
changing only the evaluation-seed list and resolving its parent path inside
the same original server snapshot. The original evaluator, converted inputs,
full-refit models, blend selection, executor and timing remain unchanged.
No training, adaptation, new label collection or parameter selection occurs.

Original model files and tripinfo XML stay on the server. Only compact JSON
records are collected. Results are reported separately from the original
56-rollout matrix and the later V91/V98 studies; no failed seed is replaced.
