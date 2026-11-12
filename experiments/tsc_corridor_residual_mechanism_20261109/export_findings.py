"""Extract small explanatory aggregates; leave full records and tables on the server."""
import argparse
import json
from pathlib import Path


def nonzero_loss_cells(table, key='weighted_attributed_count_loss_delta'):
    return {label: cell for label, cell in table.items()
            if abs(cell['counts'].get(key, 0)) > 1e-15 or cell['counts'].get('decoded10_changes', 0)}


def export(path, output):
    full = json.loads(path.read_text())
    out = {key: full[key] for key in ('status', 'previous_status', 'new_simulations', 'new_candidates',
           'new_labels_collected', 'independent_confirmation', 'reproduced_seed_hours', 'model_changed')}
    out['city_decisions'] = full['city_decisions']
    out['loss_context'] = dict(by_city={}, definitions=full['loss_context']['definitions'])
    for city, scopes in full['loss_context']['by_city'].items():
        out['loss_context']['by_city'][city] = {}
        for scope, panel in scopes.items():
            tables = panel['tables']
            compact = dict(overall=panel['overall'], attribution_checks=panel['attribution_checks'],
                           by_seed=tables['seed'], by_block=tables['block'],
                           context_attribution=nonzero_loss_cells(panel['attribution']['context']),
                           changed_contexts=nonzero_loss_cells(tables['context']),
                           changed_block_contexts=nonzero_loss_cells(tables['block_context']),
                           feature_attribution=panel['attribution'])
            edges = sorted(tables['edge'].items(), key=lambda kv: -abs(kv[1]['counts'].get('weighted_count_loss_delta', 0)))
            compact['largest_edge_contributions'] = dict(edges[:5])
            # Marginal feature partitions are separately additive, never summed across features.
            compact['feature_contexts'] = {name: nonzero_loss_cells(tables[name])
                                           for name in ('current_lane', 'history_eta_bin', 'speed_ratio', 'entry_queue')}
            out['loss_context']['by_city'][city][scope] = compact
    out['seed_reproduction'] = []
    for record in full['seed_results']:
        out['seed_reproduction'].append({key: record[key] for key in ('city', 'seed', 'block', 'reproduction')})
    out['storage'] = dict(full_server_result=str(path), full_result_bytes=path.stat().st_size,
                          local_raw_records_bytes=0, local_model_bytes=0)
    output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=out['status'], full_bytes=path.stat().st_size,
                         compact_bytes=output.stat().st_size, file=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.input, args.output)
