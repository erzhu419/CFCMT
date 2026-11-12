"""The domain precheck decodes current inputs, never next-observation pairs."""
import gzip
import json

from experiments.tsc_packet_conditional_domain_20261110 import register


def test_selected_frames_use_same_interval_signals_and_do_not_decode_the_next_frame(tmp_path, monkeypatch):
    with gzip.open(tmp_path/'frames_server_only.jsonl.gz', 'wt') as stream:
        for t in range(6):
            stream.write(json.dumps(dict(time_s=t, packets={}))+'\n')
    with gzip.open(tmp_path/'execution_server_only.jsonl.gz', 'wt') as stream:
        for t in range(5):
            stream.write(json.dumps(dict(time_sec=t, applied_states={'J': 'g' if t % 2 else 'r'}))+'\n')
    parsed = []
    original = json.loads
    def record(line):
        value = original(line)
        parsed.append((value.get('time_s'), value.get('time_sec')))
        return value
    monkeypatch.setattr(register.json, 'loads', record)
    rows = list(register.selected_current_records(tmp_path, [1, 3]))
    assert [(frame['time_s'], action['time_sec'], action['applied_states']) for frame, action, unused in rows] == [
        (1, 1, {'J': 'g'}), (3, 3, {'J': 'g'})]
    assert parsed == [(1, None), (None, 1), (3, None), (None, 3)]
