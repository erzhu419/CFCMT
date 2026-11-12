"""Inventory existing numerical pools using their saved scalar receipts."""

import json
from pathlib import Path


def _read(path):
    return json.loads(path.read_text())


def _key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def _receipt(kind, source, row, path, particles, participants, frame_actors):
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(kind=kind, source=source, particle_path=str(path), particle_count=particles,
                source_case_index=row['case_index'], reference=row['reference'],
                stochastic_participants=participants, frame_actor_count=frame_actors)


def cache_inventory(base, wanted_keys):
    """Return per-reference receipts and distinct-reference/pool counts.

    Only metadata and path existence are read. The old pilot's merged exact
    atoms cannot recover parameter-dependent branches. The complete exact
    caches in the 904 holdout belong to a separate seed.
    """
    base = Path(base)
    wanted = set(wanted_keys)
    by_key = {key: dict(exact=[], q0=[], mixture=[]) for key in wanted}
    fit_root = base / 'target_packet_particle_first_fit' / 'particles'
    fit = _read(fit_root / 'summary.json')
    pilot_root = base / 'target_packet_real_particle_pilot' / 'evaluation'
    pilot = _read(pilot_root / 'summary.json')
    mixed_root = base / 'target_packet_mixture_confirmation' / 'evaluation'
    mixed = _read(mixed_root / 'summary.json')
    registered = _read(mixed_root / 'registered.json')['cases']
    if (fit['status'] != 'PASS_FROZEN_PARTICLE_TRAINING_POOLS'
            or pilot['status'] != 'PASS_BOUNDED_REAL_PARTICLE_PILOT'
            or mixed['status'] != 'COMPLETE_FIXED_MIXTURE_CONFIRMATION'):
        raise ValueError('The archived numerical pools must have completed receipts')
    pilot_by_key = {_key(row['reference']): row for row in pilot['cases']}
    registered_by_key = {_key(row['reference']): row for row in registered}

    for row in fit['cases']:
        key = _key(row['reference'])
        if key not in wanted:
            continue
        if row['status'] != 'PASS_FIRST_FIT_PARTICLE_CASE' or row['particle_count'] != 256:
            raise ValueError(f'Incomplete original first-fit pool at {key}')
        receipt = _receipt('q0', 'first_fit', row, fit_root / row['particle_path'], 256,
                           row['stochastic_participants'], row['frame_actor_count'])
        receipt['reused_pilot_pool'] = row['reused_pilot_pool']
        receipt['reused_pilot_case_index'] = row.get('reused_pilot_case_index')
        by_key[key]['q0'].append(receipt)

    for key, row in pilot_by_key.items():
        if key not in wanted:
            continue
        if (row['status'] != 'PASS_REAL_PARTICLE_CASE'
                or row['physical_calls'] != 1024
                or not {'256', '1024'} <= row['readouts'].keys()):
            raise ValueError(f'Incomplete original pilot pool at {key}')
        path = pilot_root / f"case{row['case_index']:02d}_seed{key[0]}_t{key[1]}" / 'particles_server_only.json.gz'
        by_key[key]['q0'].append(_receipt('q0', 'pilot', row, path, 1024,
                                         row['stochastic_participants'], row['frame_actor_count']))

    anchor = base / 'target_packet_particle_first_fit' / 'fit' / 'coefficients_server_only.json'
    if not anchor.is_file():
        raise FileNotFoundError(anchor)
    for row in mixed['cases']:
        key = _key(row['reference'])
        if key not in wanted:
            continue
        prior, registration = pilot_by_key[key], registered_by_key[key]
        sizes = row['mixture']
        if (row['status'] != 'COMPLETE_MIXTURE_CASE' or row['physical_calls'] != 256
                or sizes['256']['particle_count'] != 256
                or sizes['128']['component_counts'] != {'0': 64, 'f': 64}
                or sizes['256']['component_counts'] != {'0': 128, 'f': 128}
                or registration['physical_limit'] != 256
                or registration['required_participants'] != prior['stochastic_participants']):
            raise ValueError(f'Incomplete equal-mixture pool at {key}')
        path = mixed_root / f"case{row['case_index']:02d}_seed{key[0]}_t{key[1]}" / 'particles_server_only.json.gz'
        receipt = _receipt('mixture', 'mixture_confirmation', row, path, 256,
                           prior['stochastic_participants'], prior['frame_actor_count'])
        receipt.update(anchor_file=str(anchor),
                       denominator='0.5*q0 + 0.5*qf at the saved frozen anchor',
                       seed_zero=registration['seed_zero'], seed_fitted=registration['seed_fitted'])
        by_key[key]['mixture'].append(receipt)

    summary = dict(
        exact_pool_count=sum(len(row['exact']) for row in by_key.values()),
        q0_pool_count=sum(len(row['q0']) for row in by_key.values()),
        mixture_pool_count=sum(len(row['mixture']) for row in by_key.values()),
        references_with_exact=sum(bool(row['exact']) for row in by_key.values()),
        references_with_q0=sum(bool(row['q0']) for row in by_key.values()),
        references_with_mixture=sum(bool(row['mixture']) for row in by_key.values()),
        references_with_any_pool=sum(any(row.values()) for row in by_key.values()),
        references_without_pool=sum(not any(row.values()) for row in by_key.values()),
        q0_pilot_prefix_copies=sum(receipt.get('reused_pilot_pool', False)
            for row in by_key.values() for receipt in row['q0']),
        interpretation='Counts inventory existing numerical pools. Each exact reference still requires an exact cache. Particle pools retain their proposal-specific denominator.',
        new_full_forward_calls=0, new_canonical_feature_evaluations=0,
    )
    return by_key, summary
