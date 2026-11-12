"""Forward kernel using the same discrete physical proposal as the readout."""
from experiments.tsc_packet_mean_alignment_20261110.kernel import PacketKernel as FrozenKernel
from .physics import physical_proposal
from .motion import resolve_forward


class PacketKernel(FrozenKernel):
    def _speed(self, packet, decision, packets, connection, compatible, travel=None):
        proposal = physical_proposal(self, packet, decision, packets, connection, compatible)
        return proposal if travel is None else min(proposal, max(travel, 0.))

    def _move(self, old, decisions, speeds):
        return resolve_forward(self, old, decisions)
