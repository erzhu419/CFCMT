"""Original PP requests precede labels and preserve the frozen training barrier."""
from collections import Counter
import gzip
import json
from pathlib import Path

from experiments.tsc_corridor_pp_plan_20261109 import run_plan as r


def test_original_pp_plan_precedes_labels_and_preserves_training_barrier(tmp_path, monkeypatch):
    context = [True, 1, '(0.5,0.9]', False]
    cohort, receipts, cache, reads, instances, features_seen = {}, [], {}, [], [], []
    for city, offset in [('hangzhou', 100), ('jinan', 200)]:
        spec = dict(donor=list(range(offset, offset+8)),
                    adapt=[[offset+10+2*b, offset+11+2*b] for b in range(3)],
                    dev=list(range(offset+20, offset+26)))
        cohort[city] = spec
        order = [('donor', seed, None) for seed in spec['donor']]
        for block in range(3):
            order += [('adapt', seed, block) for seed in spec['adapt'][block]]
            order += [('dev', seed, block) for seed in spec['dev'][2*block:2*block+2]]
        for part, seed, block in order:  # Deliberately interleave training and dev receipts.
            n = (16 if city == 'hangzhou' else 12)*360
            moving = 'moving_straight_at10' if seed % 2 else 'moving_other_lane_at10'
            receipts.append(dict(city=city, part=part, seed=seed, block=block,
                panel={'donor': 'source', 'adapt': 'local_adapt', 'dev': 'dev'}[part],
                counts=dict(decisions=n, unlocked_decisions=n, locked_decisions=0,
                    inflight_decisions=4, spillback_decisions=3, unmatched_decisions=0,
                    observable_flights=2, selected_context_flights=2),
                state_recovery=dict(identified=2, unlocked_flights=2),
                late_motion_counts=dict(stopped_at10=1, **{moving: 1}),
                slowing_counts=dict(slowing=2), slowing_reasons=dict(available=2)))
    plans_seen = []
    program, output = tmp_path/'program', tmp_path/'output'
    program.mkdir()
    (program/'protocol.json').write_text(json.dumps(dict(frozen_contexts=[context])))
    cohort_file, previous_file = tmp_path/'cohort.json', tmp_path/'previous.json'
    cohort_file.write_text(json.dumps(dict(cities=cohort)))
    previous_file.write_text(json.dumps(dict(protocol=dict(frozen_contexts=[context]), hours=receipts,
        targets={city: dict(old_magnitude=city, training=dict(
            cuts_by_block={str(b): [.2, .2, .2] for b in range(3)}, training_max=.2)) for city in cohort},
        previous_development_status='DEVELOPMENT_SIGNAL_ONLY')))

    class Stats:
        def __init__(self, contexts):
            self.city = list(cohort)[len(instances)]
            self.contexts = tuple(tuple(item) for item in contexts)
            self.hours, self.values, self.frozen = [], [], False
            instances.append(self)

        def register_hour(self, panel, seed, block):
            assert self.frozen == (panel == 'dev')
            owner = ('jinan' if self.city == 'hangzhou' else 'hangzhou') if panel == 'source' else self.city
            allowed = cohort[owner]['donor' if panel == 'source' else 'dev'] if panel != 'local_adapt' else cohort[owner]['adapt'][block]
            assert seed in allowed
            self.hours.append((panel, seed, block))

        def update(self, panel, seed, block, ctx, stratum, category, motion, unlocked,
                   progress_m=None, slowing=None, magnitude=None, plan=None):
            assert magnitude['value'] == (.8 if panel == 'dev' else .2)
            assert slowing['group'] == 'slowing' and tuple(ctx) in self.contexts
            assert plan == dict(protected_seconds=6, mask=1008, planned_state='rG', pp_index=1)
            self.values.append((panel, magnitude['value']))

        def freeze_training(self, expected_prior_training):
            assert len(self.hours) == 14 and all(panel != 'dev' for panel, _, _ in self.hours)
            self.frozen = True
            self.artifact = dict(cuts_by_block={str(block): [.2, .2, .2] for block in range(3)},
                                 training_max=max(value for _, value in self.values))
            assert self.artifact == expected_prior_training
            return dict(previous_training_reproduced=True, training=self.artifact)

        def result(self, previous):
            assert previous == dict(old_magnitude=self.city, training=self.artifact) and len(self.hours) == 20
            assert self.artifact['training_max'] == .2  # Dev amplitudes are .8.
            return dict(binary_reproduced=True, **self.artifact)

    def raw_rows(path):
        part, city, seed = path.parents[2].name, path.parents[1].name, int(path.parent.name)
        reads.append(str(path))
        if part == 'dev':
            assert all(stats.frozen for stats in instances)
            frozen = json.loads((output/'training_result.json').read_text())
            assert frozen['development_hours_read'] == 0 and set(frozen['targets']) == set(cohort)
        key = city, part == 'dev', seed % 2
        if key not in cache:
            old = 14 if part == 'dev' else 8
            positions, decisions = [1+60+10*old, 61, 1], []
            for t in range(0, 3600, 10):
                for tls in range(16 if city == 'hangzhou' else 12):
                    flights = []
                    if tls == 0 and t in (0, 10, 20):
                        flights = [dict(edge='road', vid=vid, entry_s=0, age_s=t,
                            remaining_m=positions[t//10], on_straight_lane=True,
                            queued_at_entry=False, speed_limit_mps=10,
                            analytic_eta_s=positions[t//10]/10) for vid in ('a', 'b')]
                    elif tls == 0 and t == 30:
                        flights = [dict(edge='road', vid='b', entry_s=0, age_s=t,
                            remaining_m=.2, on_straight_lane=bool(seed % 2))]
                    decisions.append(dict(t=t, tls=str(tls), states=['Gr', 'rG', 'GG'], pp=1 if t == 20 else 0, gp=2,
                        current_request_index=0, current_state='Gr',
                        flights=flights, spillback=tls == 0 and t in (0, 10, 30),
                        unmatched=False, protected={'road': [1023, 1008, 0]}))
            labels = [dict(edge='road', vid=vid, entry_s=0, end_s=35, status='straight') for vid in ('a', 'b')]
            cache[key] = labels, decisions
        return iter(cache[key][0 if path.name.startswith('labels') else 1])

    def recovery(row, previous):
        assert row['t'] == 20 and previous == 'Gr'
        return dict(identified=True, by_edge={'road': dict(stratum=['protected', 'zero'])})

    actual_magnitude, actual_label, actual_motion = r.normalized_magnitude, r.classify_outcome, r.classify_late_motion
    def capture_magnitude(flight, slowing):
        result = actual_magnitude(flight, slowing)
        features_seen.append((flight['vid'], result['value']))
        return result
    def label(flight, *args):
        assert features_seen[-1][0] == plans_seen[-1][0] == flight['vid']
        return actual_label(flight, *args)
    def motion(flight, *args):
        assert features_seen[-1][0] == plans_seen[-1][0] == flight['vid']
        return actual_motion(flight, *args)
    actual_plan = r.pp_plan_feature
    def plan(row, flight):
        assert features_seen[-1][0] == flight['vid']
        assert row['pp'] != row['gp'] and row['pp'] != row['current_request_index']
        assert row['protected'][flight['edge']] == [1023, 1008, 0]
        result = actual_plan(row, flight)
        plans_seen.append((flight['vid'], result))
        return result
    monkeypatch.setattr(r, 'HERE', program)
    monkeypatch.setattr(r, 'PlanStats', Stats)
    monkeypatch.setattr(r, 'verify_protocol', lambda actual: actual == cohort)
    monkeypatch.setattr(r, 'hour', lambda root, part, city, seed: Path('/records')/part/city/str(seed))
    monkeypatch.setattr(r, 'rows', raw_rows)
    monkeypatch.setattr(r, 'recover_state_features', recovery)
    monkeypatch.setattr(r, 'normalized_magnitude', capture_magnitude)
    monkeypatch.setattr(r, 'pp_plan_feature', plan)
    monkeypatch.setattr(r, 'classify_outcome', label)
    monkeypatch.setattr(r, 'classify_late_motion', motion)
    result = r.run(Path('/records'), cohort_file, previous_file, output)
    assert len(features_seen) == len(plans_seen) == 80
    assert len(reads) == 80 and set(Counter(reads).values()) == {1}
    assert result['training_frozen_before_development']
    assert all(receipt['pp_plan_cohort_reproduced'] for receipt in result['hours'])
    assert [{key: value for key, value in receipt.items() if key != 'pp_plan_cohort_reproduced'}
            for receipt in result['hours']] == receipts
    with gzip.open(output/'aggregate_result.json.gz', 'rt') as stream:
        assert json.load(stream) == json.loads((output/'result.json').read_text())
