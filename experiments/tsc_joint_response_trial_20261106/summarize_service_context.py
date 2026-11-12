"""Actual preaction joint supply points under nominal G/g; no response labels."""
import json
from pathlib import Path

from tsc_joint_response_features import FEATURE_NAMES, build_joint_features


def run(spec, root):
    columns = {name: i for i, name in enumerate(FEATURE_NAMES)}
    cells = {}
    for case in spec['cases']:
        graph = json.loads((root / f"seed_{case['seed']}" / case['case_id'] / 'joint_interface_server_only.json').read_text())
        features = build_joint_features(graph)
        for mode in ['G', 'g']:
            cell = cells.setdefault((case['city'], case['seed'], mode), set())
            for action in range(8):
                for movement in range(features.shape[1]):
                    row = features[action, movement]
                    for step in range(10):
                        prefix = f't{step:02d}'
                        if row[columns[f'own_{prefix}_{mode}_link_fraction']] <= 0:
                            continue
                        own = row[columns[f'own_{prefix}_can_serve_near_vehicle_count']]
                        if own <= 0:
                            continue
                        foe = row[columns[f'foe_{prefix}_can_serve_near_vehicle_count']]
                        yielding = row[columns[f'yields_to_{prefix}_priority_near_vehicle_count']]
                        cell.add((int(own), int(foe), int(yielding)))
    rows = [dict(city=city, seed=seed, mode=mode, points=sorted(points))
            for (city, seed, mode), points in cells.items()]
    unique = {city: {mode: sorted({tuple(p) for row in rows if row['city'] == city and row['mode'] == mode
                                  for p in row['points']}) for mode in ['G', 'g']}
              for city in ['hangzhou', 'jinan']}
    return dict(protocol=spec['protocol'], new_fits=0, new_simulations=0,
        point_fields=['own_near_supply', 'foe_can_serve_near_supply', 'yields_to_priority_near_supply'],
        rows=rows, unique_points=unique,
        scope='Observed joint preaction supply points, grouped by complete seed and nominal own G/g. Repeated times/actions deduplicated within seed. No labels, costs, target-response filtering, score or automatic gate.')
