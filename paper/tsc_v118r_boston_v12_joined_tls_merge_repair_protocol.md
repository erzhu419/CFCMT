# TSC V118r Boston V12 Joined-TLS Merge Repair Protocol

## Purpose

Create Boston complete-published package v12 by repairing the single
collision-observed `joinedS_514` movement in rejected package v11. This is an
input-admission repair only; it is not a controller experiment and supplies no
CFCMT efficacy evidence.

## Frozen Evidence

Package v11 failed complete-day admission at simulation time `13,146 s` when
vehicles from two straight movements entered receiving lane `8647416#3_0`:

- uncontrolled state-`M` movement `8647416#1_0 -> 8647416#3_0`;
- `joinedS_514` link 0 movement `128013681#0_0 -> 8647416#3_0`, served as
  protected green `G` in phase 6.

A read-only full-network audit was completed before declaring this repair. It
confirmed the focused shared-lane topology and its two-entry foe relation. The
broad audit also found many topology candidates, so it does not authorize a
network-wide automatic rewrite.

## Declared Change

Link 0 is active under phases 0, 2 and 6. Because the uncontrolled mainline is
not phase-specific, v12 changes exactly these three characters from protected
green `G` to yielding green `g`:

| TLS | Phase | Link | Before | After |
|---|---:|---:|:---:|:---:|
| `joinedS_514` | 0 | 0 | `G` | `g` |
| `joinedS_514` | 2 | 0 | `G` | `g` |
| `joinedS_514` | 6 | 0 | `G` | `g` |

Yellow clearances in phases 1, 3 and 7 remain unchanged. Every other network
character and every other package file must remain exact.

## Admission Sequence

Package v12 must pass, in order:

1. manifest identity and exact three-character static compatibility checks;
2. the existing 21 static package gates;
3. complete-demand route admission for all `3,806,510` published trips;
4. the existing trigger-window microscopic admission, crossing `13,146 s`;
5. complete-day microscopic admission with zero collisions and zero teleports.

Failure at any stage rejects v12. A pass establishes package integrity only.
V149 remains permanently closed and cannot be revived by this repair.
