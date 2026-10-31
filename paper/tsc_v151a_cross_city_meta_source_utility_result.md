# V151A Cross-City Meta Source Utility Result

## Decision

**REJECT.** The frozen leave-target-city-out meta-utility gate admitted no
target city. All seven selected arms therefore recovered rigid target-only
CFCMT exactly. No closed-loop V151A matrix is authorized.

## Frozen Development Result

| Evaluated target | Mean meta gain vs rigid | vs placebo | vs source-blind | Nondegrading vs rigid | Interventions | Admitted |
|---|---:|---:|---:|---:|---:|---:|
| Atlanta | 0.000036 | -0.000061 | 0.000031 | 6/6 | 254 | No |
| Cologne | -0.000259 | 0.000138 | 0.000634 | 5/6 | 314 | No |
| Hangzhou | -0.004330 | 0.000130 | 0.000597 | 5/6 | 245 | No |
| Ingolstadt | -0.000008 | -0.000203 | -0.000074 | 3/6 | 308 | No |
| New York | -0.004978 | -0.000183 | -0.000078 | 3/6 | 417 | No |
| RESCO synthetic | -0.004729 | 0.000210 | 0.000186 | 4/6 | 462 | No |
| Salt Lake City | -0.003361 | 0.000074 | 0.000308 | 4/6 | 288 | No |

Positive values denote lower cost for source than the named comparator. The
frozen gate required every mean gain to be at least `0.0005`, at least five of
six nondegrading pseudo-target cities for every comparison, and at least six
intervention groups. No evaluated target met all conditions.

The aggregate source admission count was `0/7`. Consequently, the selected
effect versus rigid, matched placebo and source-blind was exactly zero in every
city. The no-regression and exact-fallback checks passed by construction; the
source-contribution checks failed.

## Interpretation

V151A rules out the tested strategy: a low-complexity utility relation trained
on other cities cannot reliably decide whether source evidence should trigger a
PhasePressure-versus-rigid deviation in a held-out city. This is not an
unstandardized-cost artifact. The utility labels use the existing
action-group-range normalized pressure target before cross-city fitting.

The result does not negate the earlier structural result that rigid CFCMT beats
the dense H2O+-style residual, nor the descriptive existence of source
headroom. It shows that V151A cannot identify and deploy that headroom from its
available features. Thresholds will not be relaxed after observing this result.

## Next Method Boundary

The next candidate will not add another state-utility gate. It will estimate a
robust equal-city hierarchical prior over mechanism parameters, select only a
mechanism block and prior strength using leave-one-source-city-out evaluation,
and apply that frozen choice to B100 target adaptation. Source-null must recover
the same rigid target model exactly; shuffled-source and zero-centred priors are
matched controls. A rejected meta prior returns rigid CFCMT exactly.

## Evidence

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v151a_cross_city_meta_source_utility.json`
- Atlanta equivalence-verified target result: `cf_h2o/results/cluster/tsc_v151_cross_city_meta_utility_20260908/preflight_v2_optimized/atlanta/result.json`
- Six-city matrix: `cf_h2o/results/cluster/tsc_v151_cross_city_meta_utility_20260908/development_v1_optimized/`
