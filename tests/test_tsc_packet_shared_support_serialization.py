"""Serialize a hand-built solver return without invoking any optimizer."""
import json

import numpy as np
from scipy.optimize import OptimizeResult

from experiments.tsc_packet_shared_support_lp_20261110.solve import native


def test_native_preserves_infinite_free_bounds_and_numpy_solver_fields_as_json():
    result = OptimizeResult(x=np.zeros(76), success=np.bool_(True), status=np.int64(0),
        nit=np.int32(3), fun=np.float64(1.5),
        lower=OptimizeResult(residual=np.full(76, np.inf), marginals=np.zeros(76)),
        upper=OptimizeResult(residual=np.full(76, np.inf), marginals=np.zeros(76, dtype=np.int64)),
        ineqlin=OptimizeResult(residual=np.asarray([1., 0.]), marginals=np.asarray([0., -1.])))
    converted = native(result)
    restored = json.loads(json.dumps(converted, allow_nan=False))
    assert restored['lower']['residual'] == ['Infinity']*76
    assert restored['upper']['residual'] == ['Infinity']*76
    assert restored['x'] == [0.]*76
    assert restored['success'] is True and type(restored['status']) is int
    assert restored['nit'] == 3 and restored['fun'] == 1.5
    assert restored['upper']['marginals'] == [0]*76
    assert restored['ineqlin']['marginals'] == [0., -1.]
