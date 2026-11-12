"""Regression for NumPy density comparisons escaping into the finite-case count."""
import json

import numpy as np
import pytest

from experiments.tsc_packet_shared_motion_20261110.diagnose import aggregate, probes


@pytest.mark.parametrize('finite', [True, False])
def test_aggregate_numpy_points_and_finite_case_summary_serialize(finite):
    phi = np.zeros(38); phi[0] = 1.
    points = []
    for name, beta, direction, h in probes():
        mass = np.asarray([.25, .75, 0.] if finite else [0., 1., 0.])
        mean = np.float64(5.+min(float(beta@phi), 0.))
        points.append(dict(class_mass=mass, endpoint_class_mass=mass.copy(),
            grad_mass_left=np.zeros((3, 38)), grad_mass_right=np.zeros((3, 38)),
            lateral_signature=(('stay',),), motion=dict(focal=dict(mean_mps=mean,
                mean_matches_proposal=True, body_slack_m=10., spatial_slack_m=10.)),
            decision_count=3, negative_residual_background_count=int(beta[0] < 0), resolved_background=0))
    assert len(points) == 13
    components = [dict(weight=1., focal=dict(phi=phi.tolist()), points=points)]
    readouts = aggregate(components, dict(packet_id='focal', y=0), native_speed=5.2)
    assert len(readouts) == 13
    cases = [dict(point_readouts=readouts)]
    # This deliberately retains the original uncast sum: with a NumPy density
    # its comparisons produce np.bool_, and sum becomes non-JSON np.int64.
    finite_count = sum(case['point_readouts'][0]['joint_density'] > 0 for case in cases)
    summary = dict(cases=cases, finite_zero_joint_cases=finite_count)
    restored = json.loads(json.dumps(summary, allow_nan=False))
    assert type(finite_count) is int
    assert restored['finite_zero_joint_cases'] == int(finite)
    assert restored['cases'][0]['point_readouts'][0]['joint_status'] == (
        'FINITE' if finite else 'ZERO_DENSITY_UNDEFINED_LOG_GRADIENT')
