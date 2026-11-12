"""One fixture for joined TLS request spaces, bit order and partial context."""
from types import SimpleNamespace
from static_context import _conflicts


class Connection:
    def __init__(self, junction, request_index, tls, link_index):
        self.junction, self.request_index, self.tls, self.link_index = junction, request_index, tls, link_index
    def getJunction(self): return SimpleNamespace(getID=lambda: self.junction)
    def getJunctionIndex(self): return self.request_index
    def getTLSID(self): return self.tls
    def getTLLinkIndex(self): return self.link_index


def test_joined_tls_bit_order_and_partial_context():
    connections = [Connection('A', 0, 'joined', 5), Connection('A', 1, 'joined', 9),
                   Connection('A', 2, '', -1), Connection('B', 0, 'joined', 2)]
    requests = {('A', 0): {'foes': '110', 'response': '010'},
                ('A', 1): {'foes': '001', 'response': '000'},
                ('A', 2): {'foes': '001', 'response': '001'},
                ('B', 0): {'foes': '0', 'response': '0'}}
    result = _conflicts(connections, requests, 'fixture')['joined']
    assert result['foes'] == [[5, 9], [9, 5]]
    assert result['response'] == [[5, 9]]
    assert result['mapping_complete'] and not result['full_context_complete']
    assert result['external_foe_connection_count'] == 1
    del requests[('B', 0)]
    result = _conflicts(connections, requests, 'fixture')['joined']
    assert not result['mapping_complete'] and result['missing_controlled_request_count'] == 1
