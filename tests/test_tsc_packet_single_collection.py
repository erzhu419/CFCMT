"""Cohort accounting tests using fake native boundaries, never SUMO."""
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from experiments.tsc_target_packet_training_20261109 import collect


def test_first_failed_trajectory_stops_and_consumes_one_start(monkeypatch,tmp_path):
    from cf_h2o import sumo_runtime
    fake_api=SimpleNamespace(close=lambda:None)
    monkeypatch.setattr(sumo_runtime,'load_libsumo',lambda:fake_api)
    monkeypatch.setattr(sumo_runtime,'libsumo_version',lambda api:'1.22.0')
    monkeypatch.setitem(sys.modules,'static_context',SimpleNamespace(read_static_context=lambda cfg:{'conflicts_by_tls':{}}))
    monkeypatch.setitem(sys.modules,'cf_h2o.eval.traffic_signal_resco_cfcmt_v3',SimpleNamespace())
    monkeypatch.setattr(collect,'read_static',lambda cfg:{})
    monkeypatch.setattr(collect,'due_schedule',lambda cfg:({},[]))
    calls=[]
    def fail(api,v3,case,output,static,due):
        calls.append(case['seed'])
        return dict(status='HOLD_COLLECTION',completed_steps=3,error='Synthetic essential observation failure')
    monkeypatch.setattr(collect,'collect_one',fail)
    protocol=dict(cohorts=dict(training_seeds=[1,2,3,4],calibration_seeds=[11,12]),
                  proposed_budget=dict(seconds_per_single_trajectory=1800))
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    result=collect.collect_group(tmp_path,Path('/synthetic-input'),'training')
    assert result['status']=='HOLD_COLLECTION' and result['new_native_starts']==1
    assert calls==[1] and not (tmp_path/'training/seed2').exists()
    saved=json.loads((tmp_path/'training/seed1/result.json').read_text())
    assert saved['error']=='Synthetic essential observation failure'
    with pytest.raises(ValueError,match='already started'):
        collect.collect_group(tmp_path,Path('/synthetic-input'),'training')
    assert calls==[1]


def test_calibration_refuses_an_unfrozen_model(monkeypatch,tmp_path):
    monkeypatch.setitem(sys.modules,'cf_h2o.eval.traffic_signal_resco_cfcmt_v3',SimpleNamespace())
    monkeypatch.setitem(sys.modules,'static_context',SimpleNamespace(read_static_context=lambda cfg:{}))
    (tmp_path/'protocol.json').write_text('{}')
    (tmp_path/'training').mkdir()
    (tmp_path/'training/result.json').write_text('{"status":"COMPLETE"}')
    (tmp_path/'fit_result.json').write_text('{"status":"HOLD_FIT"}')
    with pytest.raises(ValueError,match='Freeze'):
        collect.collect_group(tmp_path,Path('/synthetic-input'),'calibration')
    assert not (tmp_path/'calibration').exists()
