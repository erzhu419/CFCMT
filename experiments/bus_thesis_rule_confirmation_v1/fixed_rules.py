"""Fixed holding formulas; missing neighbors use the current scheduled gap."""


def hold_seconds(method, forward, backward, target_forward, target_backward,
                 forward_present, backward_present):
    forward = forward if forward_present else target_forward
    backward = backward if backward_present else target_backward
    if method == "daganzo_duet":
        hold = 0.5 * (backward - forward)
    elif method == "forward_proportional":
        hold = 0.6 * (target_forward - forward)
    else:
        raise ValueError(method)
    return min(60.0, max(0.0, hold))
