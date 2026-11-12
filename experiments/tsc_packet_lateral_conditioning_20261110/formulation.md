# Local lateral conditioning: definition before implementation

The fixed discrete-braking readout still excludes all nine original lateral positives.
Their recorded mechanisms are five rear overadvances, two front vehicles leaving
the target lane, and two focal overadvances. This classification uses poststep
records for diagnosis; their observed responses cannot become predictor inputs.

Let S contain only the current packets, topology, signals and frozen history laws.
Choose a finite local actor set J containing the focal i and declared neighbours;
condition on all outside actors keeping their existing intents, zero residuals
and stay actions. This is a local conditional model, not a whole-network marginal.
Neighbour inclusion and all alternatives must be selected from current geometry
and the outer progress bounds [0,P_j], not observed next lanes or speeds.
Mu_j(0) and mu_j(-b_j) are not proven extrema of the residual response.

For every actor j, latent road intent z_j has its frozen prior pi_j(z_j|S).
Committed internal paths retain their current committed configuration.
If a branch reaches a new road, its next intent is another latent draw with that
road's prior: an OR over its legal intents supplies no probability weight.
The finite z tuple includes these draws; its probability is their prior product.

For fixed z and parameters theta, r_j(theta)=clip(beta_motion^T phi_j,-b_j,a_j).
All actors share beta_motion. Obtain longitudinal endpoints M_theta(S,z) with
the shared discrete-braking resolver from their true current lanes and bodies.
No actor is moved to a proposed target lane before this longitudinal calculation.
Potential start/yield permissions retain the current conditional-readout meaning.

For each z, enumerate the joint lateral tuple a in {left,stay,right}^|J|.
Let L_j(S,z;M_theta) be static topology/road-intent legal classes on the branch's
landing road, including stay;
it does not contain the forecast target-gap test. Using the existing lane logits,

    q_theta(a|S,z) = product_j softmax_{L_j(S,z;M_theta)}(eta_theta,j)[a_j].

Thus q is normalized before imposing gaps. It gives a front neighbour's lawful
departure a probability; selecting its observed departure would be leakage.
Apply the proposed lateral tuple only to the calculated endpoints. Define F as
one exactly when all selected landings have consistent body remapping, legal
lanes and the frozen front/rear minimum gaps against the joint landing state,
including conditioned outside actors. Any positive-q branch with undefined
remapping leaves the distribution unresolved; it cannot silently receive F=0.
All-stay can witness Z>0 only after its endpoint conditions have actually passed.

    w_theta(z,a|S) = pi(z|S) q_theta(a|S,z) F_theta(S,z,a)
    Z_theta(S) = sum_{z,a} w_theta(z,a|S)
    p_theta(a_i=y|S) = sum_{z,a:a_i=y} w_theta(z,a|S) / Z_theta(S).

When Z=0 this candidate distribution is undefined and fitting remains on HOLD.
No epsilon floor or fallback label is introduced. A nonempty feasible union
proves only that an admissible branch exists; its mass also needs pi and q.
The branch count is finite, at most |Z|*3^|J|, but full-data cost is not yet known.

This endpoint-conditioned joint teacher induces correlation by conditioning on
F. It does not say the focal lane change caused the rear vehicle to brake.
The present witness trials r_j in {0,-b_j} are external interventions only:
the 190-coefficient model assigns no probabilities to those two residual atoms,
and independently chosen endpoints need not be realizable by one shared beta.
Consequently those witnesses cannot be averaged into this likelihood or establish
that its theta=0 loss is finite. Five rear and two focal cases can remain excluded
at theta=0; lateral marginalization alone addresses only the missing action branch.

For training, use the normalized p above as the candidate lateral likelihood.
For recursive use, sample the same joint branch w/Z and apply its same endpoint
map; independently sampling old pointwise masked heads would define another law.
A motion observation can have Normal(M_theta,i,1) conditional on a branch, then
be marginalized with w/Z. This remains an observation density, not a distribution
over physical residuals. It preserves Gaussian support without inventing brakes.
This joint construction changes the former working likelihood and recursive law;
the existing four-head optimizer and forward equivalence do not follow from it.

With r and F fixed locally, logit gradients are finite conditional-expectation
differences: E[grad log q | a_i=y,F] - E[grad log q | F].
Motion means are piecewise responses to clipped beta; cap and reservation branch
changes are nonsmooth. F can change discontinuously, producing zero likelihood
and no usable old smooth gradient. No general-theta derivative is implemented.

Endpoint gaps do not establish a one-second lane-change trajectory or event order.
Initially parallel bodies must not be treated as already merged at time zero;
front departure and focal arrival in the same sampled second remain unordered.
The finite audit supplies an endpoint witness or reports none in its sampled set.
Only an infeasible full [0,P] outer envelope can certify impossibility, conditional
on the specified actions and fixed outside configuration. A prospective-response operator,
its seconds-within-step ordering, and a tractable shared fitting objective remain
unresolved. Overlapping local teachers need a consistent global joint law or an
explicit update order; separate local normalizations do not supply either.
No production change or new fitting is justified by this definition.

## Conservative alternative: preserve the production lateral order
Fix z, an explicit residual vector r, and the resulting longitudinal endpoints.
Intervene that outside actors stay; do(stay) contributes probability one and is
not conditioning on an observed outside action. Process local actors in the
original _packet_order, retaining all outside bodies in each branch state S_t.
At actor j_t, recompute A_t with the production goal/gap/body-remapping rules
after all earlier selected lateral actions, then use its masked softmax:

    q_seq(a|S,z,r,do(outside=stay)) = product_t softmax_{A_t(S_t)}(eta_j_t)[a_j_t].

An unavailable class has probability zero; successful actions update S_t before
the next actor. Undefined remapping leaves the calculation unresolved.
Enumerating the entire legal branch tree gives sum_a q_seq(a)=1 because every
node is normalized and stay is available. At theta=0, each factor is 1/|A_t|.
The focal class probability sums q_seq over all neighbour-action branches.
Report sum_a q_seq(a)*F(a) separately as endpoint-compatible mass; multiplying
by F and renormalizing would instead return to the earlier conditioned teacher.
This sequence can admit a front vehicle leaving before the focal enters while
rejecting a simultaneous endpoint tuple that cannot execute in the original order.
For fixed motion, grad log q_seq sums chosen-logit gradients minus masked-softmax
expectations of logit gradients; positive focal likelihood uses their posterior mean.
Residual endpoints remain interventions without mixture weights, and fixed-motion
probabilities neither solve shared-beta expressibility nor define a new full
190-parameter likelihood, prospective braking response or control benefit.

Integrating the direct denier's complete positive-prior goal set changes the
estimand from a fixed-goal conditional to a latent-goal marginal, using frozen
history weights rather than substituting its observed goal. For each fixed r,
the class mass is sum_{z_i,z_n} pi_i(z_i|S) pi_n(z_n|S) p_seq(y|S,z_i,z_n,r),
with a fresh joint-intent context and resolver per configuration; no residual
intervention is averaged into this marginal. Retain the fixed-goal control.
