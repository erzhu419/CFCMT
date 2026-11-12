"""Forward variant using the same local body dependency resolver as admission."""
from experiments.tsc_packet_spatial_conflict_20261110.kernel import PacketKernel as SpatialKernel
from .motion import physical_proposal, resolve_forward


class PacketKernel(SpatialKernel):
    def __init__(self, *args, conflict_geometry=None, **kwargs):
        if conflict_geometry is None or not conflict_geometry.get('entity_width_m'):
            raise ValueError('Pass the fixed explicit entity-width geometry table')
        super().__init__(*args, conflict_geometry=conflict_geometry, **kwargs)

    def _speed(self, packet, decision, packets, connection, compatible, travel=None):
        proposal = physical_proposal(self, packet, decision, packets, connection, compatible)
        return proposal if travel is None else min(proposal, max(travel, 0.))

    def _move(self, old, decisions, speeds):
        # _step has already drawn every head uniform; the shared resolver
        # applies the same start/yield decisions without advancing RNG.
        return resolve_forward(self, old, decisions)
