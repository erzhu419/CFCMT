# One registered shared-beta support initialization

This stage permits one parameter-estimation operation and one HiGHS LP call,
with a 60-second limit. It permits no NLL fitting, native simulation or retry.
The current discrete physics, feature scales, priors, geometry and labels stay fixed.

Register the six goal pairs already recorded in support_conditions.json.
Within each corresponding held-goal feasible example, take the registered
executable tuple whose focal action is the observed lateral class, not stay.
An actor assigned -b in that example receives the linear cell below; an actor
assigned zero receives the positive plateau. Other local landing-gap actors
retain that example's zero intervention and are also assigned the plateau.

For controls 491/728/1477, inspect intent_marginal_summary.json in saved pair order.
Take its first pair with positive zero-residual observed-class mass. Within that
pair, take the first executable zero-residual observed-action tuple in the
original static enumeration order. Register every prefix landing constraint,
including a front actor's departure before the focal action where applicable.
Do not replace these choices after seeing LP status or the resulting beta.

For each registered goal configuration, recompute canonical current-only phi_j
and the zero-residual physical proposal P_j. All quantities below use full
precision; printed gap deficits are not new numerical margins.
One beta in R^38 belongs to every configuration and actor; write s_j=beta^T phi_j.
The registered local affine endpoint is X_j=x_j+P_j+e_j, with cells

    linear:    -b_j <= s_j <= 0,  e_j=s_j,  P_j+s_j >= 0;
    plateau:    s_j >= 0,         e_j=0.

The positive plateau means zero effective displacement residual, not necessarily
zero raw residual: positive clipped residuals are reapplied below the P_j ceiling.
The linear cell also enforces the nonnegative-speed floor. The six registered
braking actors have P_j>b_j=4.5, so their floor is inactive throughout this cell.
If another actor has P_j<b_j, the branch d=P_j+s_j is valid only for s_j>=-P_j;
the saturated zero-speed branch is different and is not silently substituted.

The affine construction requires the registered same-road endpoint coordinates,
lane/body remapping and assumed nonbinding additional caps to remain applicable.
A different body, receiving or reservation branch is not represented by this LP.
It is assessed by the subsequent shared physical readout.

At every prefix of the registered lateral sequence, keep its registered lane
assignments and longitudinal actor order. For each lane, impose the adjacent
front/rear pair inequality, with front u and rear v,

    X_u - length_u - X_v >= min_gap_v.

Use all local landing-gap actors, including actors whose action is stay, and
check all lane adjacent pairs once more after the complete sequence. These
constraints preserve the registered order as well as the landing gaps. Static
goal legality and defined body remapping are prerequisites, not extra soft scores.
Retain the six direct-denier inequalities as registered checks, without claiming
that those six inequalities alone suffice for the sequence or shared dynamics.

Let A beta <= c denote these cell and prefix constraints. Add 38 absolute-value
auxiliaries t and solve exactly the single registered objective

    minimize sum_k t_k
    subject to A beta <= c, -t_k <= beta_k <= t_k, t_k >= 0.

The 38 beta coordinates are free variables; nonnegative bounds apply only to t.
No positive gap margin, epsilon floor, post-solve perturbation, replacement branch
or second solve is introduced. A minimum-L1 solution can lie at gap/cell boundaries;
its existence does not establish a differentiable or stable NLL initialization.
Infeasibility concerns this selected set of goal/action/cap cells, not the full
clipped model family or all latent branch alternatives.

If a beta is returned, apply that identical vector to every packet in each
current frame, including background, receiving and conflict dependencies.
Rebuild the canonical all-packet features under each of the same 144 goal pairs,
then use a fresh shared resolver and original ordered lateral probabilities.
Report actual clips, caps, gaps, total probability and goal-prior class masses;
the nine observed classes, including all three controls, need positive mass for
this local support initialization to pass. Keep every failed case and prior control.
An LP-feasible but physically rejected point is reported as such without repair.
This validation cannot establish finite full-reservoir/full-190 loss, convergence
or control improvement. The original training HOLD remains independently recorded.
