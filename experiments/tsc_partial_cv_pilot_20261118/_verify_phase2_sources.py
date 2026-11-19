"""Quick 300s smoke & collision verification for Phase 2 expanded source scenarios."""

from __future__ import annotations

from pathlib import Path
import shutil

from cf_h2o.sumo_runtime import load_libsumo
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _start_sumo, _tls_phase_infos
from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors
from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import PartialCvObserver, select_observed_pressure_phase

STAGED = Path("/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase0/staged_scenarios")
SRC = Path("/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/source_scenarios")

# Stage hangzhou_4x4_hetero_repaired bound to repaired zero-conflict hangzhou_aligned.net.xml
hz_rep_net = STAGED / "hangzhou_4x4_repaired/hangzhou_aligned.net.xml"
hz_het_src = SRC / "hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.rou.xml"
hz_het_dst = SRC / "hangzhou_4x4_hetero_repaired"
hz_het_dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(hz_rep_net, hz_het_dst / "hangzhou_aligned.net.xml")
shutil.copy2(hz_het_src, hz_het_dst / "hangzhou_hetero.rou.xml")
(hz_het_dst / "hangzhou_hetero.sumocfg").write_text(
    '<configuration><input><net-file value="hangzhou_aligned.net.xml"/>'
    '<route-files value="hangzhou_hetero.rou.xml"/></input>'
    '<time><begin value="0"/><end value="3600"/></time></configuration>\n',
    encoding="utf-8",
)

CANDS = {
    "hz_het_rep": hz_het_dst / "hangzhou_hetero.sumocfg",
    "hz_1x1_bc": SRC / "hangzhou_1x1_bc-tyc_18041610_1h/hangzhou_1x1_bc-tyc_18041610_1h.sumocfg",
    "hz_1x1_kn": SRC / "hangzhou_1x1_kn-hz_18041608_1h/hangzhou_1x1_kn-hz_18041608_1h.sumocfg",
    "hz_1x1_qc": SRC / "hangzhou_1x1_qc-yn_18041608_1h/hangzhou_1x1_qc-yn_18041608_1h.sumocfg",
    "hz_1x1_sb": SRC / "hangzhou_1x1_sb-sx_18041607_1h/hangzhou_1x1_sb-sx_18041607_1h.sumocfg",
    "slc_400s": SRC / "saltlake2_400sX200w/saltlake2_400sX200w.sumocfg",
    "slc_state": SRC / "saltlake2_stateXuniversity/saltlake2_stateXuniversity.sumocfg",
}

sumo_api = load_libsumo()
for name, cfg in CANDS.items():
    _start_sumo(sumo_api, cfg, 2027, tripinfo_output=None)
    try:
        infos = _tls_phase_infos(sumo_api)
        execs = build_safe_phase_executors(sumo_api, infos)
        obs = PartialCvObserver(sumo_api=sumo_api, infos=infos, seed=2027, p_cv=0.10)
        obs.observe_step(sumo_api, 1.0)
        cols = 0
        for step in range(600):
            if step % 10 == 0:
                for tid in sorted(infos):
                    p = select_observed_pressure_phase(info=infos[tid], executor=execs[tid], observer=obs, mode="conservation_gp")
                    execs[tid].request(p)
            sumo_api.simulation.step()
            for ex in execs.values():
                ex.advance(1.0)
            obs.observe_step(sumo_api, 1.0)
            cols += len(sumo_api.simulation.getCollisions())
        ks = sorted({len(ex.feasible_states_now()) for ex in execs.values()})
        sample_st = next(iter(execs.values())).feasible_states_now()[0][:24]
        print(f"OK {name:12s} | n_tls={len(infos):2d} | K={ks} | cols_600s={cols:3d} | sample={sample_st}")
    finally:
        sumo_api.close()
