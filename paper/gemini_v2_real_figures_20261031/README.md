# V93 figures from recorded experiments

Six figures use the implemented V93 pipeline and existing experiment records. The original draft illustrations remain in a separate directory. No simulations, training, parameter tuning, synthetic trajectories or generated statistical samples are used in this revision.

| Figure | Content and evidence |
|---|---|
| 1 | Offline 60-second action-label branch and online 10-second decisions; current method, frozen configuration and executor. |
| 2 | Actual 18-network/seven-source-group fitting, target adaptation, development selection and frozen deployment. |
| 3 | Observed trip waiting by departure minute, fixed Los Angeles seed 1020485; four original tripinfo files, aggregated on the server. |
| 4 | Source identity screening in both cities and the actual Jinan source-weight curve; all eight development seeds. |
| 5 | All 56 recorded run-mean waiting values for every controller in each of the four confirmation scenarios (1,344 runs). |
| 6 | Actual guard configuration comparison, all eight development seeds; this replaces the unsupported switch-cost ablation. |

`source_data/` includes the plotted records and their provenance. Development screens and held-out confirmation stay separate. Source weight is the executed source prediction mass, distinct from the reference/pairwise blend coefficient. Plot 4 reports means and standard deviations; plot 6 reports means, all seed points and their full range. Plot 5 estimates density from recorded run means and displays the original observations, without generating extra samples.

In figures 1–2, `g` indexes a matched action group, `r_g` is its pressure-reference phase, `Y_g` is the branch cost, and `sigma_g` is the numerical-floor-adjusted range of that group's action costs. The normalized label shown applies to source components; the target anchor retains its separate domain scale. Figure 4 keeps the Los Angeles source screen, but does not portray its zero-weight fallback as a varying source-weight experiment. Figure 6 keeps the frozen raw guard in Los Angeles: risk 0.5 reduces its mean waiting by 1.11%, but its worst seed worsens by 4.16%, exceeding the development limit.

## Data limits

The archived runs contain tripinfo but no FCD or full per-step queue history; figure 3 therefore shows departure-cohort waiting, not spatial trajectories. Waiting includes unfinished trips observed up to 3,600 seconds; later departure cohorts have shorter observation windows. The case is the smallest prespecified evaluation seed, chosen before its outcomes were inspected. The weight and guard screens are sequential development stages, not a full factorial sweep. No 1.25× peak-demand test or switch-cost removal experiment exists in these records.

Run `python3 build_real_figures.py` here to regenerate PNG, editable SVG, vector PDF and the combined PDFs. `export_development_data.py` reproduces the compact screen CSVs from the original project records.
