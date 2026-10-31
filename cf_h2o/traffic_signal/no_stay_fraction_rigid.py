"""Keep PP switches when the frozen model proposes holding the current green."""

from dataclasses import replace

if __package__:
    from cf_h2o.traffic_signal.thresholded_fraction_rigid import ThresholdedFractionRigidOriginator
else:
    from thresholded_fraction_rigid import ThresholdedFractionRigidOriginator


ORIGINATOR_PROTOCOL = "tsc-v154s-no-stay-fraction-rigid-originator-v1"


class NoStayFractionRigidOriginator(ThresholdedFractionRigidOriginator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.originator_protocol = ORIGINATOR_PROTOCOL
        self._execution_states = {}
        self._model_stay_veto_count = 0

    def prepare_interval(self, *, states, executors, routing_graph, control_interval_sec):
        super().prepare_interval(
            states=states, executors=executors, routing_graph=routing_graph,
            control_interval_sec=control_interval_sec,
        )
        self._execution_states = {
            str(tls_id): (str(executor.mode), str(executor.current_green_state))
            for tls_id, executor in executors.items()
        }

    def select(self, contrast, *, reference_index: int):
        decision = super().select(contrast, reference_index=reference_index)
        if not (decision.eligible and decision.learned_differs):
            return decision
        tls_id = str(contrast.metadata["row_tls"][decision.reference_index])
        mode, current_green_state = self._execution_states[tls_id]
        candidate_states = contrast.metadata["candidate_states"]
        if (mode == "green"
                and str(candidate_states[decision.selected_index]) == current_green_state
                and str(candidate_states[decision.reference_index]) != current_green_state):
            self._model_stay_veto_count += 1
            self._override_count -= 1
            return replace(
                decision, selected_index=decision.reference_index,
                learned_differs=False, eligible=False, priority=0.0,
                predicted_score=decision.reference_score, rejection="model_stay_veto",
            )
        return decision

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "model_stay_veto_count": self._model_stay_veto_count,
        }
