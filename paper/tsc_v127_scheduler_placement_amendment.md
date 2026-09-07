# V127 Scheduler Placement Amendment

## Trigger

Corrected V127 task `t88306` was initially submitted with a hard requirement
for `node003`. The scientific inputs passed their remote identity preflight, but
the task remained queued because that node had no 20-core allocation available.

## Placement-only change

Before execution began, the hard node requirement was removed and replaced by
the explicit allowlist `node001,node002,node003,node005,node006`. `node004`
remained excluded. The immutable source snapshot, command, 20-worker execution,
RAM budget, input hashes, output directory, model, folds, calibration and gates
were unchanged.

Each allowed node passed a lightweight shared-filesystem visibility check for
the snapshot wrapper, frozen prediction artifact, selector cache and selector
audit. Task `t88306` then launched on `node005`. This amendment changes only
where the same frozen command runs and has no scientific effect.

The local launcher was subsequently advanced to placement protocol v3. It now
uses this five-node allowlist directly, preflights visibility on every allowed
node, requests the observed 24,576 MB RAM envelope and leaves final placement
to the scheduler. This launcher improvement postdates the immutable execution
snapshot and does not alter task `t88306`.
