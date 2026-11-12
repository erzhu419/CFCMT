"""The frozen packet kernel with the local processed-movement correction."""
from experiments.tsc_packet_same_step_receiving_20261110.kernel import PacketKernel as SharedKernel
from .motion import resolve_forward


class PacketKernel(SharedKernel):
    def _move(self, old, decisions, speeds):
        return resolve_forward(self, old, decisions)
