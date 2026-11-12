"""Construct the fixed physical kernel without any proposal/response query."""
from experiments.tsc_packet_discrete_braking_20261110.kernel import PacketKernel
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation


def prepare_zero_kernel(static, frame, executor_seeds, laws, action, random_seed, geometry):
    kernel = PacketKernel(static, initialize_observation(frame, frame['time_s']), executor_seeds,
        laws, random_seed=random_seed, conflict_geometry=geometry)
    for tls, request in sorted(action['requests'].items()):
        kernel.executors[tls].request(request)
    if kernel.signals != action['applied_states']:
        raise ValueError('The fixed current interval executor signals did not reproduce')
    kernel._connections = {key: kernel._connection(packet) for key, packet in kernel.state['packets'].items()}
    # No _conditional_* indices, phi, lane-mask cache or zero proposal is built.
    return kernel
