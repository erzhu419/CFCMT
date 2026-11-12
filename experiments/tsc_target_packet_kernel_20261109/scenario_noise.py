"""Scenario-coupled streams independent of action-dependent packet activity."""
from math import isfinite
from random import Random


class ScenarioNoise:
    """Draw boundary noise separately and advance packet decisions by time.

    Initial identities are sorted before stream assignment. The kernel must
    register newly due identities in the same arrival order for every action,
    even when those identities remain pending or leave the network early.
    """
    def __init__(self, seed, initial_ids, start_time):
        self.seed, self.start_time = seed, start_time
        seed_rng = Random(seed)
        self.arrival_rng = Random(seed_rng.getrandbits(128))
        self.entry_rng = Random(seed_rng.getrandbits(128))
        self.profile_rng = Random(seed_rng.getrandbits(128))
        self.packet_seed_rng = Random(seed_rng.getrandbits(128))
        self._packets = {}
        for packet_id in sorted(initial_ids):
            self.register(packet_id, start_time)

    def register(self, packet_id, born_time):
        """Assign streams once at birth, independently of actual insertion."""
        if packet_id in self._packets:
            raise ValueError('A scenario packet identity must be registered once')
        self._packets[packet_id] = dict(born_time=born_time,
            intent_rng=Random(self.packet_seed_rng.getrandbits(128)),
            decision_rng=Random(self.packet_seed_rng.getrandbits(128)),
            next_offset=0, cache_time=None, cache=None)

    def intent_uniform(self, packet_id):
        """The kth road-entry intent has the same draw across candidate actions."""
        return self._packets[packet_id]['intent_rng'].random()

    def step_uniforms(self, packet_id, time_s):
        """Return the fixed start/yield/lane bundle for this absolute second.

        Skipped pending/inactive seconds still consume all three draws. Calls
        may interleave packets arbitrarily; each packet's clock is monotone and
        its current-second bundle is cached for repeated reads.
        """
        packet = self._packets[packet_id]
        offset = time_s-packet['born_time']
        if offset < 0 or int(offset) != offset:
            raise ValueError('Decision time must be a whole second at/after packet birth')
        if packet['cache_time'] is not None and time_s < packet['cache_time']:
            raise ValueError('A packet decision clock cannot move backwards')
        while packet['next_offset'] <= offset:
            rng = packet['decision_rng']
            packet['cache'] = {'start': rng.random(), 'yield': rng.random(), 'lane': rng.random()}
            packet['cache_time'] = packet['born_time']+packet['next_offset']
            packet['next_offset'] += 1
        return dict(packet['cache'])

    def poisson_due(self, rate):
        """Poisson count from unit-rate exponential arrivals over interval rate."""
        if not isfinite(rate) or rate < 0:
            raise ValueError('Poisson rate must be finite and nonnegative')
        if rate == 0:
            return 0
        elapsed, count = 0., 0
        while True:
            elapsed += self.arrival_rng.expovariate(1.)
            if elapsed > rate:
                return count
            count += 1

    def entry_uniform(self):
        return self.entry_rng.random()

    def profile_index(self, count):
        return self.profile_rng.randrange(count)
