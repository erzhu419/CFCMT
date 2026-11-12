"""Frozen190-coefficient local inference; no optimization or simulator access."""
from math import exp, fsum, isfinite


FEATURES = (
    ('bias', 1), ('speed_mps', 15), ('acceleration_mps2', 3),
    ('allowed_speed_mps', 15), ('accel_mps2', 3), ('decel_mps2', 3),
    ('tau_s', 2), ('min_gap_m', 5), ('length_m', 5), ('imperfection', 1),
    ('speed_factor', 1), ('stopline_distance_m', 100), ('net_leader_gap_m', 50),
    ('leader_speed_mps', 15), ('leader_present', 1), ('internal_lane', 1),
    ('signal_G', 1), ('signal_g', 1), ('signal_Y', 1), ('signal_y', 1),
    ('signal_r', 1), ('signal_s', 1), ('signal_O', 1), ('signal_uncontrolled', 1),
    ('foe_min_TTA_s', 10), ('foe_present', 1), ('foe_internal_count', 10),
    ('receiving_free_space_m', 50), ('left_front_gap_m', 50), ('left_rear_gap_m', 50),
    ('right_front_gap_m', 50), ('right_rear_gap_m', 50), ('left_rear_speed_mps', 15),
    ('right_rear_speed_mps', 15), ('left_legal', 1), ('right_legal', 1),
    ('lateral_position_m', 4), ('lateral_speed_mps', 3),
)
FEATURE_NAMES = tuple(name for name, divisor in FEATURES)
FIXED_DIVISORS = tuple(divisor for name, divisor in FEATURES)
COEFFICIENT_VECTORS = ('motion', 'start', 'yield_entry', 'lane_left', 'lane_right')
LANE_CHOICES = ('left', 'stay', 'right')


def _sigmoid(value):
    if value >= 0:
        return 1./(1.+exp(-value))
    scaled = exp(value)
    return scaled/(1.+scaled)


def masked_lane_probabilities(logits, allowed):
    """Softmax raw logits over allowed classes; both inputs are name mappings.

    Mask before exponentiation so a forbidden high logit cannot underflow the
    probabilities of the remaining legal lane choices.
    """
    legal = [choice for choice in LANE_CHOICES if allowed[choice]]
    if not legal:
        raise ValueError('At least one lane choice must be allowed')
    largest = max(logits[choice] for choice in legal)
    weights = {choice: exp(logits[choice]-largest) for choice in legal}
    total = fsum(weights.values())
    return {choice: weights[choice]/total if choice in weights else 0. for choice in LANE_CHOICES}


class FrozenLocalModel:
    """Predict protocol heads from raw physical context.

    Coefficients are five named38-vectors in protocol feature order. Default
    zeros are an untrained synthetic placeholder, not a fitted traffic model.
    Hard movement/start/yield/lateral admissibility belongs to the traffic
    kernel; this module returns unconstrained head probabilities.
    """
    feature_names = FEATURE_NAMES
    fixed_divisors = FIXED_DIVISORS
    fit_performed = False

    def __init__(self, coefficients=None):
        self.parameter_origin = 'untrained_zero_placeholder' if coefficients is None else 'supplied_coefficients'
        if coefficients is None:
            coefficients = {name: (0.,)*len(FEATURES) for name in COEFFICIENT_VECTORS}
        if set(coefficients) != set(COEFFICIENT_VECTORS):
            raise ValueError('Expected the five frozen protocol coefficient vectors')
        self.coefficients = {name: tuple(float(value) for value in coefficients[name])
                             for name in COEFFICIENT_VECTORS}
        if any(len(vector) != len(FEATURES) or not all(map(isfinite, vector))
               for vector in self.coefficients.values()):
            raise ValueError('Each coefficient vector requires38 finite values')

    def features(self, context):
        """Read only the declared current-state fields; force bias to one."""
        values = tuple(1. if name == 'bias' else float(context[name])/divisor
                       for name, divisor in FEATURES)
        if not all(map(isfinite, values)):
            raise ValueError('Protocol physical features must be finite')
        return values

    def predict(self, context):
        phi = self.features(context)
        scores = {name: fsum(coefficient*value for coefficient, value in zip(vector, phi))
                  for name, vector in self.coefficients.items()}
        residual = max(-float(context['decel_mps2']),
                       min(scores['motion'], float(context['accel_mps2'])))
        logits = dict(left=scores['lane_left'], stay=0., right=scores['lane_right'])
        return dict(motion_residual_mps=residual,
                    start_probability=_sigmoid(scores['start']),
                    yield_probability=_sigmoid(scores['yield_entry']),
                    lane_logits=logits,
                    lane_probabilities=masked_lane_probabilities(logits, dict.fromkeys(LANE_CHOICES, True)))
