"""Causal-prefix and same-time accounting checks, without running SUMO."""

import unittest
from types import MethodType, SimpleNamespace

from native_runtime import CausalArrivalRates, _full_horizon_termination
from collect_native import proposed_hold
from collect_native import collection_spec


class CausalArrivalTests(unittest.TestCase):
    def bridge(self):
        present = SimpleNamespace(start_stop_id_s="s1", passable_line_l=["100X"], arriver_time_n=10)
        future = SimpleNamespace(start_stop_id_s="s1", passable_line_l=["100X"], arriver_time_n=1000)
        return SimpleNamespace(departed_passenger_times={"present": 10},
            passenger_obj_dic={"present": present, "future": future},
            passenger_arriver_rate_d={"future-plan": [100000]})

    def test_future_plan_and_undeclared_departure_do_not_change_prefix(self):
        bridge = self.bridge()
        a = CausalArrivalRates()
        a.observe(bridge, 60, {"s1": ["present"]})
        bridge.passenger_obj_dic["future"].arriver_time_n = 99999
        bridge.passenger_arriver_rate_d = {"future-plan": [99999999]}
        bridge.departed_passenger_times["future"] = 99999
        b = CausalArrivalRates()
        b.observe(bridge, 60, {"s1": ["present"]})
        self.assertEqual(a.rate("s1", "100X", 60), 1/60)
        self.assertEqual(a.rate("s1", "100X", 60), b.rate("s1", "100X", 60))

    def test_same_time_batch_does_not_count_twice(self):
        bridge = self.bridge()
        rates = CausalArrivalRates()
        for _ in range(3):
            rates.observe(bridge, 60, {"s1": ["present"]})
        self.assertEqual(len(rates.seen), 1)
        self.assertEqual(rates.rate("s1", "100X", 60), 1/60)

    def test_elapsed_window_and_transfer_observation(self):
        rates = CausalArrivalRates()
        self.assertEqual(rates.rate("s1", "100X", 0), 0)
        rates.record("p", "s1", ["100X"], 10, 10)
        rates.record("p", "s2", ["101S"], 1900, 1900)
        self.assertEqual(rates.rate("s1", "100X", 1900), 0)
        self.assertEqual(rates.rate("s2", "101S", 1900), 1/1800)

    def test_zero_does_not_consume_action_rng(self):
        class NoRng:
            def random(self):
                raise AssertionError("zero method should not sample")
        self.assertEqual(proposed_hold("zero", NoRng(), {}), 0)

    def test_formal_case_uses_its_own_seeds_and_horizon(self):
        case = dict(case=3, trajectory_id=12204, split="fit", method="mixture", episode_seconds=24000,
            demand_seed=80103, sumo_seed=81103, action_seed=82103)
        protocol = dict(smoke=dict(demand_seed=72101, sumo_seed=73101, action_seed=74101),
            collection_cases=[case])
        spec = collection_spec(SimpleNamespace(case=3, method=None, episode_seconds=None), protocol)
        self.assertEqual(spec["demand_seed"], 80103)
        self.assertEqual(spec["sumo_seed"], 81103)
        self.assertEqual(spec["action_seed"], 82103)
        self.assertEqual(spec["episode_seconds"], 24000)
        self.assertEqual(spec["global_trajectory_id"], 12204)
        with self.assertRaises(ValueError):
            collection_spec(SimpleNamespace(case=3, method=None, episode_seconds=1800), protocol)

    def test_existing_smoke_defaults_are_unchanged(self):
        protocol = dict(smoke=dict(demand_seed=72101, sumo_seed=73101, action_seed=74101))
        spec = collection_spec(SimpleNamespace(case=None, method="zero", episode_seconds=None), protocol)
        self.assertEqual(spec["method"], "zero")
        self.assertEqual(spec["episode_seconds"], 1800)
        self.assertEqual(spec["action_seed"], 74101)

    def test_empty_sumo_tail_processes_last_arrival_before_horizon(self):
        # SUMO reports its last arrival after the t=2 simulationStep. The
        # bridge processes that report on its next tick, before advancing time.
        class Bridge:
            steps = 0
            max_steps = 5
            done = False
            waiting = 1
            accrued_wait = 0
            completed = 0
            cleanups = 0

            def _cleanup_departed_buses(self):
                self.cleanups += 1

            def advance(self):
                if self.done:
                    return
                if self.steps == 2:
                    self.completed = 1
                    self.waiting = 0
                self.accrued_wait += self.waiting
                self.steps += 1
                self._check_termination()

        bridge = Bridge()
        bridge._check_termination = MethodType(_full_horizon_termination, bridge)
        bridge.advance()
        bridge.advance()
        self.assertFalse(bridge.done)
        self.assertEqual(bridge.waiting, 1)  # final SUMO arrival still pending
        while not bridge.done:
            bridge.advance()
        self.assertEqual(bridge.steps, 5)
        self.assertEqual(bridge.completed, 1)
        self.assertEqual(bridge.waiting, 0)
        self.assertEqual(bridge.accrued_wait, 2)
        self.assertEqual(bridge.cleanups, 1)


if __name__ == "__main__":
    unittest.main()
