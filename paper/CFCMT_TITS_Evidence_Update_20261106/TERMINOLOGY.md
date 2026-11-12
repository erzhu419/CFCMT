# Manuscript terminology

CFCMT = Causal-Factored Cross-City Model Transfer: supervised phase-cost prediction with predeployment target adaptation.
Alpha_g is the component blend; omega_n,k is the source-mixture weight; beta_n is the heuristic guard multiplier.
The target anchor T_g differs from the execution reference r_g.
PP = PhasePressure; MP = MaxPressure; GP = exact generalized-pressure reference.
The dense linear residual is not the published H2O+ algorithm.

Historical panel: seven controllers, four scenarios, 56 reused seeds, 1,568 runs; six comparators by three scopes use Holm18.
Fresh panel: four frozen controllers, the same four fixed network and demand scenarios, 56 new SUMO seeds, 896 runs; three comparators by three scopes use Holm9.
Panels remain separate. Both show best CFCMT equal-city performance means; no fresh macro comparison passes the joint confirmation criterion.
The historical MP result remains favorable. Jinan matches GP in 167 of 168 runs in each panel.
Source-data and learned-intervention incremental benefits remain unestablished.

An effective override is an accepted non-reference switch request or effective current-green hold, not a completed green-phase transition.
Collision events count raw records; incidents deduplicate vehicle pair, type and lane within a run. Teleport starts and ends remain separate.
Main figures: framework and source selection. Supplementary S1: departure cohort; Table S1: historical execution; S2: historical waiting distributions; S3: development guard.

Supplementary Note S1: separate 450-second native-label transfer pilot on reused development groups; T=target, S=true source+target, P=permuted source+target. Its failed conjunctive gate is not part of V93 Holm9 or Holm18.
