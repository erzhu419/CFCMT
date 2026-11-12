# Manuscript terminology

CFCMT = Causal-Factored Cross-City Model Transfer: supervised phase-cost prediction
with predeployment target adaptation. Alpha_g is the within-component blend;
omega_n,k is the source-mixture weight; beta_n is the heuristic guard multiplier.
The target anchor T_g differs from the execution reference r_g.
PP = PhasePressure; MP = MaxPressure; GP = exact generalized-pressure reference
(green queue proxy minus 0.02 times downstream occupancy, with the shared executor).
The dense linear residual is not the published H2O+ algorithm.
The current panel has seven controllers, four scenarios, 56 reused seeds and 1,568 formal runs.
Current eight-seed development and 56-seed evaluation panels are disjoint but
reuse an earlier 64-seed benchmark. Six comparators and three scopes use Holm18.
MP remains favorable; PP and GP paired intervals cross zero.
An effective override is an accepted non-reference switch request or effective
current-green hold, not a completed green-phase transition.
Collision events sum raw getCollisions() records; incidents deduplicate vehicle
pair, type and lane within a run. Teleport starts and ends remain separate.
Four main figures show the framework, source selection, waiting distributions and guard.
Supplementary Fig. S1 and Table S1 retain cohort waiting and learned-policy counts.
