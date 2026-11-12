"""38-feature shared-beta fixed-pair algebra; squared hinge is not an NLL.

Actor keys: phi[38], b, a, P, x, length, min_gap. P is the current physical
zero-residual proposal, not the accepted shared-resolver mean. Geometry and
other constraints must be re-resolved before interpreting a branch as accepted.
Directions are right derivatives at beta+t*d, t down to zero from above.
"""
import numpy as np


DIMENSION = 38


def _clip(score, low, high, derivative):
    value = max(low, min(score, high))
    if low == high:
        return value, 0., None if derivative is None else 0.
    slope = 1. if low < score < high else 0. if score < low or score > high else None
    if derivative is None:
        directed = None
    elif score < low or score > high:
        directed = 0.
    elif score == low:
        directed = max(derivative, 0.)
    elif score == high:
        directed = min(derivative, 0.)
    else:
        directed = derivative
    return value, slope, directed


def motion_response(beta, actor, *, direction=None):
    beta, phi = np.asarray(beta, dtype=float), np.asarray(actor['phi'], dtype=float)
    if beta.shape != (DIMENSION,) or phi.shape != (DIMENSION,):
        raise ValueError('The frozen shared coefficient and feature vectors each have38 entries')
    if direction is not None:
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (DIMENSION,):
            raise ValueError('A shared-beta direction has38 entries')
    score = float(beta@phi)
    directed_score = None if direction is None else float(direction@phi)
    raw, raw_slope, raw_directed = _clip(score, -actor['b'], actor['a'], directed_score)
    effective, slope, directed = _clip(score, -actor['b'], 0., directed_score)
    before_floor = actor['P']+effective
    proposal = max(before_floor, 0.)
    jacobian = None if slope is None else slope*phi
    if actor['P'] == 0 or before_floor < 0:
        proposal_jacobian = np.zeros(DIMENSION)
    elif before_floor > 0:
        proposal_jacobian = jacobian
    else:
        proposal_jacobian = np.zeros(DIMENSION) if slope == 0. else None
    proposal_directed = (None if directed is None else directed if before_floor > 0 else
                         0. if before_floor < 0 else max(directed, 0.))
    return dict(score=score, raw_r=raw, effective_e=effective, physical_proposal_mps=proposal,
        score_directional=directed_score, raw_r_directional=raw_directed,
        effective_e_directional=directed, physical_proposal_directional=proposal_directed,
        raw_r_jacobian=None if raw_slope is None else (raw_slope*phi).tolist(),
        effective_e_jacobian=None if jacobian is None else jacobian.tolist(),
        physical_proposal_jacobian=None if proposal_jacobian is None else proposal_jacobian.tolist())


def shared_gap(beta, focal, neighbour, relation, *, direction=None):
    """Use relation='rear' for a rear neighbour and 'front' for an ahead one.

    Rear margin = xF+vF-LF-xN-vN-gN; front margin has the opposite velocity
    contrast, xN+vN-LN-xF-vF-gF. Positive margins satisfy this pair's end gap.
    Kink components emit no ordinary Jacobian; their one-sided derivative is
    evaluated exactly, including min(d.phi,0) at beta0 and the zero-speed floor.
    """
    f = motion_response(beta, focal, direction=direction)
    n = motion_response(beta, neighbour, direction=direction)
    if relation == 'rear':
        sign, required = 1., neighbour['min_gap']
        constant = focal['x']-focal['length']-neighbour['x']
    elif relation == 'front':
        sign, required = -1., focal['min_gap']
        constant = neighbour['x']-neighbour['length']-focal['x']
    else:
        raise ValueError('The fixed neighbour relation is rear or front')
    gap = constant+sign*(f['physical_proposal_mps']-n['physical_proposal_mps'])
    margin = gap-required
    violation = max(-margin, 0.)
    jf, jn = f['physical_proposal_jacobian'], n['physical_proposal_jacobian']
    jacobian = None if jf is None or jn is None else sign*(np.asarray(jf)-np.asarray(jn))
    directed = (None if direction is None else
                sign*(f['physical_proposal_directional']-n['physical_proposal_directional']))
    hinge_jacobian = (np.zeros(DIMENSION) if margin >= 0 else
                      None if jacobian is None else -2*violation*jacobian)
    return dict(relation=relation, focal=f, neighbour=n, relative_gap_m=gap,
        required_gap_m=required, gap_margin_m=margin, squared_hinge=violation**2,
        gap_jacobian=None if jacobian is None else jacobian.tolist(),
        gap_directional=directed,
        squared_hinge_jacobian=None if hinge_jacobian is None else hinge_jacobian.tolist(),
        squared_hinge_directional=None if directed is None else -2*violation*directed,
        jacobian_available=jacobian is not None, loss_kind='Squared-hinge diagnostic, not NLL',
        scope='Fixed pair and physical-proposal endpoint algebra under one shared beta. Accepted means, active constraints and whole-branch feasibility require the shared resolver; no coefficients or probabilities are estimated.')
