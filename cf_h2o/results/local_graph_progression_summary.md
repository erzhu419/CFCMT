# Local Graph Progression Validation

Run: `cf_h2o/results/local_graph_progression_validation.json`

Scope:
- Full-route city bundles, no fixed single line.
- 6 cross-city splits.
- Training cap: 10k rows per source city; eval cap: 30k target rows; graph cap: 20k rows.
- Source selector cap: 3k train/eval/graph rows per held-out source fold.
- 8 data-loading workers; PyTorch training remains single-process CPU.

## Tested Steps

1. Deterministic static local features were added: route position, previous/next stop distance, previous/current/next stop demand, same-line nearby demand, segment speed quantile, headway imbalance, co-line imbalance, local headway pressure, demand/action per target headway.
2. Local features are used only in the latent mechanism factor input `theta`; residual parent sets remain controlled by AutoDAG.
3. Source-city leave-one selector was added. Static local is enabled only when held-out source folds show both sufficient win rate and mean improvement.
4. Austin APC was checked as a real-data proxy. It has stop-event demand and vehicle positions, but no synchronized all-vehicle snapshot and no counterfactual holding action, so it supports external/few-shot APC validation but not a strong true local-graph claim.

## Main Results

| Method | Mean total MSE | Mean vs H2O+ | Wins vs H2O+ |
|---|---:|---:|---:|
| H2O+ dense ridge | 884.8525 | 1.0000 | 0/6 |
| CFCMT ridge template | 851.8929 | 0.7651 | 4/6 |
| Neural AutoDAG + latent + CFCMT fallback trust | 865.4106 | 0.9677 | 5/6 |
| + deterministic static local | 867.4736 | 0.9588 | 4/6 |
| + source selector | 864.8044 | 0.9675 | 5/6 |
| Old proxy encoder local ablation | 857.1496 | 0.9005 | 4/6 |

The deterministic local graph is not universally useful on static-derived data: it improves target MSE in 3/6 splits and hurts in 3/6. The source selector avoids the worst static-local failures and gives a tiny mean-total-MSE gain over no-local neural CFCMT, but the effect is small.

The old proxy encoder ablation scores better in this capped run, but it is not a defensible local-graph result because it learns an embedding directly from the row-level observation/action vector rather than from a real synchronized neighborhood graph. Keep it as an ablation only.

## Source Selector

| Split | Selected | Source static/no-local mean | Source win rate | Target vs H2O+ |
|---|---|---:|---:|---:|
| source_open_na_to_singapore | no local | 1.1236 | 0/3 | 0.9995 |
| source_singapore_austin_halifax_to_mbta | no local | 0.9827 | 1/3 | 0.7092 |
| leave_one_city_out_all::singapore_lta_all | static local | 0.9759 | 3/3 | 0.9941 |
| leave_one_city_out_all::austin_capmetro_all | no local | 1.0206 | 2/3 | 1.4614 |
| leave_one_city_out_all::halifax_transit_all | no local | 1.0136 | 2/3 | 0.9871 |
| leave_one_city_out_all::mbta_all | no local | 1.0772 | 0/3 | 0.6534 |

Selector rule after the fix: static local requires both win-rate threshold and mean source-fold improvement. This prevents enabling local when two small wins are outweighed by one large source-domain loss.

## APC Snapshot Check

Austin APC summary has 2 datasets, 11,567,532 transitions, 140 route entries, 4,891 stop entries, and 641,266 trips. It contains observed stop-event vehicle positions and demand, but it does not contain synchronized all-vehicle snapshots or counterfactual holding actions. Conclusion: use it for real passive/few-shot APC validation, not as final evidence for full local graph transfer.
