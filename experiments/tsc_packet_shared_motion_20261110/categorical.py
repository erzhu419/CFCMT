"""Exact lateral tuple probability/gradient conditional on frozen motion masks."""
from math import exp, fsum, log


def ordered_tuple_probability(ordered, phi_by_packet, gamma_left, gamma_right):
    """Shared38 logits; stay logit zero; masks/actor order are fixed inputs.

    Returns both mass derivatives and log-mass scores. A structurally illegal
    tuple has zero mass/derivative and undefined log mass/score (None).
    """
    if len(gamma_left) != 38 or len(gamma_right) != 38:
        raise ValueError('Both lateral coefficient vectors must have38 entries')
    legal = bool(ordered['all_requested_actions_accepted']) and all(
        step['action'] in step['allowed_choices'] for step in ordered['steps'])
    if not legal:
        return dict(legal=False, q=0., logq=None, dq_left=[0.]*38, dq_right=[0.]*38,
                    dlogq_left=None, dlogq_right=None)
    scores_left, scores_right, log_terms = [0.]*38, [0.]*38, []
    for step in ordered['steps']:
        phi = phi_by_packet[step['packet_id']]
        if len(phi) != 38:
            raise ValueError('Canonical lateral features must have38 entries')
        logits = dict(left=fsum(a*b for a, b in zip(gamma_left, phi)), stay=0.,
                      right=fsum(a*b for a, b in zip(gamma_right, phi)))
        allowed = step['allowed_choices']
        largest = max(logits[choice] for choice in allowed)
        weights = {choice: exp(logits[choice]-largest) for choice in allowed}
        total = fsum(weights.values())
        log_terms.append(logits[step['action']]-largest-log(total))
        left = (step['action'] == 'left')-weights.get('left', 0.)/total
        right = (step['action'] == 'right')-weights.get('right', 0.)/total
        for index, value in enumerate(phi):
            scores_left[index] += left*value
            scores_right[index] += right*value
    logq = fsum(log_terms)
    q = exp(logq)
    return dict(legal=True, q=q, logq=logq,
                dq_left=[q*value for value in scores_left], dq_right=[q*value for value in scores_right],
                dlogq_left=scores_left, dlogq_right=scores_right)
