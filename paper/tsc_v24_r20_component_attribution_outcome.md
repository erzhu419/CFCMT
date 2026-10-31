# TSC v24/r20 Physical-Component Attribution Outcome

Generated: 2026-08-08T17:02:48.081292+00:00

## Integrity decision

**PASS**: six target shards and all 42 independently fitted component results passed SHA-256, zero-target-budget, complete-group, finite-metric, and frozen-cache checks.

- source snapshot SHA-256: `427a778306958f08bc03e57d842e6b92e6c70a2806fb8878ebc02fad315bf9fd`
- source tree SHA-256: `0c6710bf1e8d660d38b57a2544225ffd927a83cf698e61ea80d30e8c20b4319c`
- cache aggregate SHA-256: `a6ba8169f78f629a9aef6f3ac4c0321860e51ea77aefb9d2d474e8039a6937e2`
- cache coverage: 768 files, 32,394 rows, 16 scenarios, 3 seeds
- evaluated action groups: 2,626 across six held-out targets

## Family attribution

| Family | Macro regret | Improvement vs rigid | Cities improved | Max regression | Stage A |
|---|---:|---:|---:|---:|---:|
| Rigid | 0.291137 | +0.00% | 0/6 | 0.000000 | FAIL |
| Served only | 0.294655 | -1.21% | 2/6 | 0.069338 | FAIL |
| Mobility only | 0.274374 | +5.76% | 5/6 | 0.011636 | FAIL |
| Full five mechanisms | 0.315446 | -8.35% | 4/6 | 0.297545 | FAIL |
| Queue only | 0.302070 | -3.76% | 4/6 | 0.240289 | FAIL |
| Red accumulation only | 0.312655 | -7.39% | 3/6 | 0.297198 | FAIL |
| Spillback only | 0.327771 | -12.58% | 0/6 | 0.219802 | FAIL |
| Mobility + queue | 0.306122 | -5.15% | 3/6 | 0.218485 | FAIL |
| Mobility + red | 0.313155 | -7.56% | 3/6 | 0.296862 | FAIL |
| Mobility + spillback | 0.303569 | -4.27% | 4/6 | 0.214667 | FAIL |
| Mobility + served | 0.311766 | -7.09% | 5/6 | 0.300684 | FAIL |

## Per-target regret

| Family | RESCO grid4x4 | Cologne1 | Ingolstadt1 | Atlanta 1x5 | Hangzhou 4x4 | Manhattan 28x7 |
|---|---:|---:|---:|---:|---:|---:|
| Rigid | 0.345286 | 0.285399 | 0.199679 | 0.215572 | 0.367733 | 0.333154 |
| Served only | 0.345286 | 0.285399 | 0.269017 | 0.215572 | 0.348816 | 0.303841 |
| Mobility only | 0.305752 | 0.256173 | 0.186253 | 0.186161 | 0.379369 | 0.332536 |
| Full five mechanisms | 0.244438 | 0.298180 | 0.156925 | 0.513117 | 0.357718 | 0.322300 |
| Queue only | 0.228430 | 0.261839 | 0.170245 | 0.455861 | 0.357643 | 0.338400 |
| Red accumulation only | 0.226887 | 0.286825 | 0.156925 | 0.512770 | 0.358612 | 0.333910 |
| Spillback only | 0.345286 | 0.285399 | 0.199679 | 0.435374 | 0.367733 | 0.333154 |
| Mobility + queue | 0.227836 | 0.281992 | 0.182224 | 0.434057 | 0.374882 | 0.335741 |
| Mobility + red | 0.223801 | 0.286825 | 0.156925 | 0.512434 | 0.368570 | 0.330376 |
| Mobility + spillback | 0.227836 | 0.254720 | 0.197470 | 0.430239 | 0.378614 | 0.332536 |
| Mobility + served | 0.227322 | 0.265892 | 0.176625 | 0.516256 | 0.361087 | 0.323413 |

## Scientific decision

No family passes the preregistered Stage-A gate. Mobility only is the sole robust diagnostic: it improves five cities, reduces city-macro regret by 5.76%, and has only 0.0116 maximum regression. It still misses the required 10% macro improvement.

Queue propagation, red accumulation, and spillback each produce a large Atlanta regression when used alone. Adding any one of them to mobility also recreates the failure. The full model's failure is therefore not merely a high-order interaction; the terminal local-state mechanisms themselves are non-invariant under this estimand.

The admissible next step is to replace the 60 s terminal-local labels and 60 s analytic priors with a policy-consistent one-control-interval (10 s) mechanism estimand, while retaining the 60 s rollout cost as the action-ranking outcome. No current physical family is promoted to full-budget or closed-loop evaluation.
