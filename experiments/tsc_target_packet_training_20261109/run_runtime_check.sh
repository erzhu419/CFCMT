set -o pipefail
env PYTHONPATH=/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/target_packet_training:/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/repaired_action_cost_pilot/producer:/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/repaired_action_cost_pilot/branch_helpers:/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime LD_LIBRARY_PATH=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/python/lib CFCMT_SOURCE_ROOT=/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109/repaired_action_cost_pilot/producer CFCMT_RUNTIME_ROOT=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime CFCMT_PYTHON=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/python/bin/python3.10 SUMO_HOME=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime/sumo PATH=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime/bin:/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CFCMT_EXPECTED_SUMO_VERSION=1.22.0 CFCMT_SUMO_SCRATCH_DIR=/dev/shm TMPDIR=/dev/shm TMP=/dev/shm TEMP=/dev/shm LC_ALL=C LANG=C OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/python/bin/python3.10 -c 'import json
import numpy as np
from experiments.tsc_target_packet_training_20261109 import collect,prepare_and_fit,training_data,evaluate,fit_model
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import corridor,executor_seed,initial_observation,packet,coefficients
from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
s,l=corridor(incoming_length=20,internal_lengths=(5,))
e={"J":executor_seed(("G",),clearance=("I0",))}
k=PacketKernel(s,initial_observation(s,[packet("synthetic","in",17,5)],executors=e),e,l,coefficients())
a=evaluate.StructuralAudit(s,k.state,e)
from copy import deepcopy
for _ in range(20):
 b=deepcopy(k.state); row=k.step();a.check(b,k.state,row)
heads={}
for name in fit_model.HEAD_NAMES:
 X=np.zeros((1,2,38));X[:,:,0]=1
 heads[name]=dict(X=X,prior=np.array([[.5,.5]]),risk=np.ones((1,2),bool),y=np.array([0.]))
heads["motion"].update(upper=np.ones((1,2))*2,a=np.ones((1,2))*2,b=np.ones((1,2))*4)
heads["lane_change"].update(mask=np.ones((1,2,3),bool),y=np.array([1.]))
loss,grad=fit_model.objective(np.zeros(190),heads)
assert np.isfinite(loss) and np.isfinite(grad).all()
print(json.dumps(dict(status="PASS_RUNTIME_PIPELINE",synthetic_audited_steps=a.steps,likelihood_finite=True,native_starts=0,real_fits=0)))
'
