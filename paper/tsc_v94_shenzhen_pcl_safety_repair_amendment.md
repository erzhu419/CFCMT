# Shenzhen PCL pre-efficacy safety-repair amendment

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This amendment governs network construction and simulator-safety admission only.
No CFCMT, H2O+, source-selection, policy, reward, queue, delay, or throughput
outcome from Shenzhen PCL has been computed or inspected.

## Evidence preceding the amendment

- RoadnetSZ is fixed at commit `80a2785961b9439b6b92637e5cecd42ab5733052`.
- The repository's compiled SUMO 1.0 network retained 36 TLS programs and passed
  the all-seed zero-collision/zero-teleport gate in 10 of 28 complete one-hour
  windows.
- Rebuilding the native node, edge, connection, type, and TLS components with
  SUMO 1.22 restored all 140 nodes declared as traffic lights while preserving
  the IDs and phase counts of the 36 explicit source programs. It passed 16 of
  28 windows.
- A 5.5 m/s junction turn-speed repair also passed 16 of 28 windows under the
  exact RNG protocol and is rejected.
- Safety-only diagnostics localized most remaining incidents to permissive
  minor-green movements in automatically generated TLS programs. The standard
  netconvert `--tls.layout incoming` construction removes minor greens from the
  104 generated programs while leaving the 36 explicit source programs intact.

## Frozen development gate

- Construction protocol:
  `cfcmt-roadnetsz-native-sumo122-window-package-v4`.
- All 28 contiguous one-hour windows remain in the matrix.
- Development seeds: `5057, 6067, 7103, 8171, 9277, 12001, 13007, 14009`.
- A window is a candidate only when all eight seeds complete the full 3,600 s,
  load the explicit demand consistently, expose 140 controllable TLS, and report
  zero collision incidents and zero teleports.
- At least 20 candidate windows are required. The threshold is unchanged from
  the earlier PCL admission protocols.

## Frozen disjoint confirmation gate

If and only if the development gate passes, evaluate its candidate windows with
the disjoint seeds `15101, 16001, 17011, 18013, 19001, 20011, 21013, 22003`.
A confirmed window must pass every check for all eight confirmation seeds. At
least 20 windows must remain confirmed. Failure blocks all Shenzhen controller
efficacy experiments; the seed list, collision threshold, and minimum-window
threshold will not be revised again for this dataset.

Only the confirmed windows may enter later target-offline construction and
closed-loop efficacy evaluation. Rejected windows and all failed repair audits
remain part of the provenance record.
