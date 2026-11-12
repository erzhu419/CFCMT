"""Generate the one registered min-L1 shared-beta LP; never solve it here.

build_min_l1(actors, gaps): actors is {id: {phi[38],P,x,length,min_gap,b,cell}}
with cell linear/plateau/zero_floor. Gaps are [{label,front:id,rear:id}].
Variables are beta38 followed by38 nonnegative absolute-value auxiliaries.
Endpoints are fixed-cell physical proposals; shared-resolver validation remains
separate from these affine constraints. No gap epsilon is introduced.
"""
import numpy as np


DIMENSION = 38


def build_min_l1(actors, gaps):
    rows, limits, labels, endpoints = [], [], [], {}

    def append(coefficients, limit, label):
        row = np.zeros(2*DIMENSION)
        row[:DIMENSION] = coefficients
        rows.append(row)
        limits.append(float(limit))
        labels.append(label)

    for actor_id, actor in actors.items():
        phi = np.asarray(actor['phi'], dtype=float)
        if phi.shape != (DIMENSION,):
            raise ValueError('Each registered actor has the original38 features')
        P, x, b, cell = float(actor['P']), float(actor['x']), float(actor['b']), actor['cell']
        coefficient, constant = np.zeros(DIMENSION), x
        if P == 0.:
            effective_cell = 'constant_zero_proposal'
        elif cell == 'linear':
            coefficient, constant = phi.copy(), x+P
            effective_cell = cell
            append(-phi, b, f'{actor_id}/linear/clip_lower')
            append(phi, 0., f'{actor_id}/linear/clip_upper')
            append(-phi, P, f'{actor_id}/linear/proposal_nonnegative')
        elif cell == 'plateau':
            constant, effective_cell = x+P, cell
            append(-phi, 0., f'{actor_id}/plateau/nonnegative_score')
        elif cell == 'zero_floor':
            if P > b:
                raise ValueError(f'{actor_id}: the physical zero-floor cell requires P<=b')
            effective_cell = cell
            append(phi, -P, f'{actor_id}/zero_floor/nonpositive_proposal')
        else:
            raise ValueError(f'{actor_id}: unknown registered cell {cell}')
        endpoints[actor_id] = dict(constant_m=constant, beta_coefficients=coefficient,
                                  registered_cell=cell, effective_cell=effective_cell)

    for gap in gaps:
        front, rear = actors[gap['front']], actors[gap['rear']]
        f, r = endpoints[gap['front']], endpoints[gap['rear']]
        # front_end-L_front-rear_end >= min_gap_rear.
        coefficient = r['beta_coefficients']-f['beta_coefficients']
        limit = f['constant_m']-front['length']-r['constant_m']-rear['min_gap']
        append(coefficient, limit, 'gap/'+gap['label'])

    for index in range(DIMENSION):
        row = np.zeros(2*DIMENSION)
        row[index], row[DIMENSION+index] = 1., -1.
        rows.append(row); limits.append(0.); labels.append(f'absolute/beta{index}/positive')
        row = np.zeros(2*DIMENSION)
        row[index], row[DIMENSION+index] = -1., -1.
        rows.append(row); limits.append(0.); labels.append(f'absolute/beta{index}/negative')
    objective = np.r_[np.zeros(DIMENSION), np.ones(DIMENSION)]
    return dict(A_ub=np.asarray(rows), b_ub=np.asarray(limits), c=objective,
        bounds=[(None, None)]*DIMENSION+[(0., None)]*DIMENSION,
        constraint_labels=labels, actor_endpoints=endpoints,
        variable_names=[f'beta{index}' for index in range(DIMENSION)]+
                       [f'abs_beta{index}' for index in range(DIMENSION)],
        objective='min sum absolute shared beta coefficients', gap_epsilon_m=0.)
