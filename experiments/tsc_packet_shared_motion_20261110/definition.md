# Shared-motion readout for the fixed nine cases

The six unresolved cases and restored controls 491/728/1477 remain fixed.
No coefficient is estimated and no per-vehicle residual is selected in this stage.
The held-neighbour-intent control and complete positive-prior control are retained.

For each focal/denier goal pair z=(z_f,z_n), install those current intents and
reconstruct all connections and current-only caches. Other current intents stay
at the frozen feature-seed configuration; they are conditioned nuisance variables.
For every packet j in the entire current frame, including background and dependency
packets, compute the original scaled 38-vector from the canonical context:

    phi_j(z) = model.features(kernel._context(packet_j, all_current_packets,
                                               connection_j, compatible_j)).
    r_j(beta,z) = clip(beta^T phi_j(z), -b_j, a_j).

The same supplied beta belongs to every packet. A fresh shared discrete resolver
receives this complete decision map and potential start/yield permissions.
All queried receiving, conflict, leader and landing-neighbour dependencies use
their own r_j from that same beta. Background motion is not held at zero.
Accepted states are never reused across beta values or goal pairs. This retains
the declared local resolver/precedence convention, not a claim of equality to
every globally ordered full-network forward step.

After shared longitudinal resolution, enumerate the focal and denier's lateral
actions in original packet order, updating the branch after each action.
All other lateral actions are do(stay), with probability one; their bodies remain
in the gap checks and their longitudinal responses still use beta.
At each node use the production goal/gap mask A_t and normalized masked softmax.
For fixed z,beta,gamma the path probability q_seq(a|S,z,beta,gamma) is the product of node
probabilities. At zero lane coefficients each factor is 1/|A_t|.
An unresolved positive-mass branch makes the readout unresolved, not probability zero.
Here gamma=(gamma_left,gamma_right) contains the separate 38-dimensional lane vectors;
gamma=0 in the current readout and varies only for fixed-motion gradient checks.

    p_{beta,gamma}(y|S) = sum_{z_f,z_n} pi_f(z_f|S) pi_n(z_n|S)
                    sum_{a:a_f=y} q_seq(a|S,z,beta,gamma).

All positive-prior goals, including END and lane-incompatible intentions, enter.
Each goal-pair action tree sums to one, so this is a normalized local class law.
Endpoint-compatible mass is separately sum pi_f*pi_n*q_seq*F; it does not
renormalize p_{beta,gamma}. The observed next goal/speed/lane enters only as a label.

An optional local working joint density is

    f_{beta,gamma}(v,y|S) = sum_z pi_f*pi_n sum_{a:a_f=y}
                      q_seq(a|S,z,beta,gamma) Normal(v; mu_f(beta,z), 1).

This integrates/sums to one over v,y when all branches are defined. Normal noise
is an observation density, not an additional physical residual draw. The density
is conditional on frozen outside intents and lateral interventions; it is not
the full-network, full-four-head 190-parameter likelihood or an already fitted model.

Six representative fixed-goal/action, stable-cap necessary conditions follow
(displayed rounded; a numerical readout keeps full precision). Let
e_j=clip(beta^T phi_j,-b_j,0): positive scores cannot increase the physical P_j.
For these recorded baseline proposals P_j>b_j=4.5, a branch without further
body/spatial caps has displacement d_j=P_j+e_j. At the focal lateral turn:

| Focal | Necessary relative effective response (m) |
| --- | --- |
| 1198 | e_f-e_n >= 1.460250112507 |
| 278 | e_f-e_n >= 1.195330588411 |
| 306 | e_f-e_n >= 4.043573840135 |
| 1178 | e_f-e_n >= 2.087335260922 |
| 2255 | e_n-e_f >= 2.726803593477 |
| 377 | e_n-e_f >= 1.669362114803 |

These follow from x_f+d_f-L_f-x_n-d_n>=g_n for a rear denier, or
x_n+d_n-L_n-x_f-d_f>=g_f for a front denier. The recorded order puts the focal
before its direct denier in these six cases. Their goal/cap/order conditions and
all other landing gaps still need checking; the table is not sufficient support
and does not establish that one beta can satisfy all rows across every goal pair.

For a fixed active branch, an optional support diagnostic is the sum over cases
of the minimum, over positive-prior lawful observed-action branches, of squared
positive gap deficits. This is a feasibility diagnostic, not a likelihood.
Inside a fixed unsaturated negative-score cell, grad e_j=phi_j; flat cells give
zero. For deficit h=max(delta-s(beta),0), grad h^2=-2h*grad s when h>0.
At beta=0, e_j has directional derivative min(d^T phi_j,0); a conventional
smooth gradient is not defined there. Additional active caps need their own
branch derivative, and reservation/order switches can invalidate that branch.

With fixed motion/masks, lateral-logit gradients are the branch-posterior mean
of the sequential chosen-logit-minus-masked-expectation score. A differentiable
motion mean gives a posterior-weighted Normal score (v-mu)*Dmu. Six zero class probabilities
still yield infinite NLL and no smooth NLL gradient at beta=0; no epsilon floor
or reuse of the previous optimizer/gradient follows from these formulas.

The minimal executable check supplies beta rather than solving for it. Predeclare
the six directions +/-unit(bias), +/-unit(speed_mps), +/-unit(acceleration_mps2),
using existing feature scales; no labels determine these vectors.
Read beta=0 and beta=h*d at h=1e-4,5e-5 as numerical one-sided derivative probes,
not a support search. Rebuild all-frame phi/decisions per goal configuration,
then report shared means, goal-weighted class masses, exact active caps/order,
gap deficits and one-sided differences against applicable branch derivatives.
Retain all nine cases and both prior controls even when a direction worsens them.
This verifies a shared readout and derivative scope; it neither finds a usable
coefficient vector nor establishes finite full-reservoir loss or control benefit.
