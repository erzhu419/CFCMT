"""Project frozen hurdle-actor samples into the executed-action domain."""

from types import MethodType


def install_projected_sampling(actor, projector):
    """Project sampled actions while preserving latent probabilities and entropy.

    The frozen actor's forward and repeated-action methods delegate to these
    component methods. Recorded-action likelihoods and actor parameters remain
    unchanged; the projector retains its ordinary differentiable clip behavior.
    """
    original_soft = actor.soft_action_components
    original_repeated = actor.repeated_soft_action_components

    def projected_soft(self, states, *, deterministic_positive=False):
        components = original_soft(
            states, deterministic_positive=deterministic_positive
        )
        components["zero_action"] = projector(states, components["zero_action"])
        components["positive_action"] = projector(
            states, components["positive_action"]
        )
        return components

    def projected_repeated(
        self, states, repeats, *, deterministic_positive=False
    ):
        components = original_repeated(
            states, repeats, deterministic_positive=deterministic_positive
        )
        # The frozen sampler expands parameters in state-major, repeat-minor
        # order: (batch, repeats) -> (batch * repeats).
        repeated_states = states.repeat_interleave(repeats, dim=0)
        components["zero_action"] = projector(
            repeated_states, components["zero_action"]
        )
        components["positive_action"] = projector(
            repeated_states, components["positive_action"]
        )
        return components

    actor.soft_action_components = MethodType(projected_soft, actor)
    actor.repeated_soft_action_components = MethodType(projected_repeated, actor)
