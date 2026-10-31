"""Recorded-action IQL actor updates inside the frozen SAC continuation loop.

The existing value update supplies detached IQL weights for the existing replay
batch. Only the actor objective changes. SAC temperature sampling and updates,
critic/model losses, value updates, and online scheduling remain in place.
"""

import torch


def iql_actor_update_trainer(base_class):
    """Return the actor-only variant for this pilot's fit-only, latest batch."""

    class IQLActorUpdateTrainer(base_class):
        def _update_actor_for_batch(self, *, online, offline_batch, online_batch,
                                    iql_actor_weights, collect_metrics):
            assert online_batch is None
            assert offline_batch is not None and iql_actor_weights is not None
            self._recorded_actor_update = (offline_batch, iql_actor_weights)
            try:
                return super()._update_actor_for_batch(
                    online=online, offline_batch=offline_batch,
                    online_batch=online_batch, iql_actor_weights=iql_actor_weights,
                    collect_metrics=collect_metrics,
                )
            finally:
                del self._recorded_actor_update

        def _update_sac_actor(self, actor_states, *, temperature_states=None,
                              temperature_action_samples=1, collect_metrics=True):
            batch, weights = self._recorded_actor_update
            assert actor_states.data_ptr() == batch["observations"].data_ptr()
            if temperature_states is None:
                temperature_states = actor_states
            self._set_critic_grad(False)
            try:
                # Preserve the frozen hybrid actor's one positive draw at this
                # exact point in the SAC loop, including its latent log density.
                # Its discarded min-all Q query consumes no random numbers.
                with torch.no_grad():
                    components = self.actor.soft_action_components(actor_states)
                    actor_log_prob = (
                        components["zero_probability"] * components["zero_log_prob"]
                        + components["hold_probability"] * components["positive_log_prob"]
                    )
                    hold_log_prob = (components["positive_log_prob"]
                                     - components["positive_conditional_log_prob"])
                    gate_entropy = -(
                        components["zero_probability"] * components["zero_log_prob"]
                        + components["hold_probability"] * hold_log_prob
                    )
                    weighted_positive_entropy = -(
                        components["hold_probability"]
                        * components["positive_conditional_log_prob"]
                    )
                    diagnostics = {
                        "train/hybrid_entropy": (-actor_log_prob).mean().detach(),
                        "train/hybrid_gate_entropy": gate_entropy.mean().detach(),
                        "train/hybrid_weighted_positive_entropy": weighted_positive_entropy.mean().detach(),
                        "train/hold_probability": components["hold_probability"].mean().detach(),
                        "train/positive_action_mean": components["positive_action"].mean().detach(),
                    }
                    if (self.config.temperature_state_source == "actor"
                            and temperature_action_samples == 1):
                        temperature_log_prob = actor_log_prob.detach()
                    else:
                        temperature_log_prob = self._temperature_log_probability(
                            temperature_states, temperature_action_samples,
                        )
                # This is the actual frozen weighted-likelihood method, using
                # recorded execution labels and the already computed weights.
                actor_loss, actor_grad_norm = self._update_iql_actor(
                    batch, weights, collect_metrics=collect_metrics,
                )
                self._last_actor_diagnostics = diagnostics
            finally:
                self._set_critic_grad(True)

            alpha_loss = -(
                self.log_alpha
                * (temperature_log_prob + self.config.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            return actor_loss, alpha_loss, actor_grad_norm

    return IQLActorUpdateTrainer
