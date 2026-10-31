"""Remove actor gradients where the canonical executed action is constant."""

from canonical_execution_projection import canonical_execution_bounds


def free_execution_actor_trainer(base):
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7

    class FreeExecutionActorTrainer(base):
        def _update_iql_actor(self, batch, weights, *, collect_metrics=True):
            lower, upper = canonical_execution_bounds(batch["observations"], DEFAULT_CONFIG_V7)
            free = (upper[:, 0] > lower[:, 0]).to(weights.dtype)
            return super()._update_iql_actor(
                batch, weights * free, collect_metrics=collect_metrics)

    return FreeExecutionActorTrainer
