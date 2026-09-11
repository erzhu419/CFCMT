"""The native refit cannot proceed with unsafe or incomplete original B100 data."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def collection(tmp_path, monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / 'scripts/data'))
    module = importlib.import_module('fit_cologne_native_b100')
    protocol = json.loads((root / 'cf_h2o/results/cluster/'
        'tsc_v154p_cologne_native_b100_20260909/protocol_v1.json').read_text())
    cells = {}
    for seed in module.SEEDS:
        groups = [g for g in protocol['training_group_ids'] if g.startswith(f'cologne1:seed{seed}:')]
        cell = {'protocol': module.PROTOCOL, 'status': 'PASS', 'seed': seed,
            'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'],
            'requested_groups': groups.copy(), 'retained_groups': groups.copy(),
            'rows': 8 * len(groups), 'unsafe_groups': [], 'invalid_groups': [],
            'unsafe_native_branches': 0, 'groups': []}
        path = tmp_path / f'collection/seed_{seed}/result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(cell))
        cells[seed] = (path, cell)
    return module, protocol, cells


def test_complete_original_b100_is_admitted(collection, tmp_path):
    module, protocol, _ = collection
    cells, failures = module.collection_gate(protocol, tmp_path)
    assert failures == []
    assert [g for c in cells for g in c['retained_groups']] == protocol['training_group_ids']
    assert [len(c['retained_groups']) for c in cells] == [34, 33, 33]
    assert sum(c['rows'] for c in cells) == 800


@pytest.mark.parametrize('failure', ['unsafe', 'missing_seed', 'reduced_roster'])
def test_failed_collection_stops_before_bank_assembly_and_fit(collection, tmp_path, monkeypatch, failure):
    module, protocol, cells = collection
    path, cell = cells[3037]
    missing = cell['retained_groups'][0]
    if failure == 'missing_seed':
        path.unlink()
    else:
        cell['retained_groups'].remove(missing)
        cell['rows'] -= 8
        if failure == 'unsafe':
            cell.update(status='FAIL_UNSAFE', unsafe_groups=[missing], unsafe_native_branches=1,
                        groups=[{'group_id': missing, 'status': 'FAIL_UNSAFE'}])
        else:
            # Internally consistent 99-group collection is still below the fixed budget.
            cell['requested_groups'].remove(missing)
        path.write_text(json.dumps(cell))
    _, failures = module.collection_gate(protocol, tmp_path)
    assert failures
    assert missing in {g for item in failures for g in item.get('missing_groups', [])}
    if failure == 'unsafe':
        assert any(item.get('unsafe_groups') == [missing] for item in failures)
    monkeypatch.setenv('CFCMT_SOURCE_ROOT', protocol['source_root'])
    monkeypatch.setattr(module, 'assemble_native_bank',
                        lambda *a, **k: pytest.fail('A failed collection reached bank assembly'))
    output = tmp_path / 'fit'
    result = module.run(protocol, tmp_path, output)
    assert result['status'] == 'FAIL_B100'
    assert result['new_model_fits'] == 0
    assert json.loads((output / 'result.json').read_text()) == result
    assert sorted(p.name for p in output.iterdir()) == ['result.json']
