"""Load the B195 raw anchor while keeping the original threshold and S veto."""

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL

if __package__:
    from .no_stay_fraction_rigid import NoStayFractionRigidOriginator
else:
    from no_stay_fraction_rigid import NoStayFractionRigidOriginator

MODEL_PROTOCOL = 'tsc-v155l-cologne-expanded-raw-target-model-v1'
FIT_PROTOCOL = 'tsc-v155l-cologne-expanded-data-full-fit-v1'
ORIGINATOR_PROTOCOL = 'tsc-v155m-expanded-data-no-stay-originator-v1'
SCORE_UNITS = 'raw_native_halted_per_lane_cost_difference'


class ExpandedDataNoStayOriginator(NoStayFractionRigidOriginator):
    """Only payload admission changes; inherited methods retain top1 then veto."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self.threshold != 0. or self.threshold_mode != 'minimum_advantage'
                or self.frozen_threshold.get('score_units') != SCORE_UNITS
                or self.frozen_threshold.get('calibration_protocol') != 'tsc-v155k-cologne-expanded-data-oof-v1'):
            raise ValueError('Use the K-fixed raw-score threshold zero')
        self.expected_runtime_protocol = MODEL_PROTOCOL
        self.originator_protocol = ORIGINATOR_PROTOCOL

    def _validate_payload(self):
        groups = self.payload.get('selected_groups', [])
        model = self.payload.get('rigid_model')
        if (self.payload.get('protocol') != MODEL_PROTOCOL or self.payload.get('fit_protocol') != FIT_PROTOCOL
                or self.payload.get('target_budget') != 195 or len(groups) != 195 or len(set(groups)) != 195
                or self.payload.get('city') != 'cologne' or self.payload.get('score_units') != SCORE_UNITS
                or self.payload.get('occupancy_equation_protocol') != OCCUPANCY_EQUATION_PROTOCOL
                or not isinstance(model, FrozenAnchoredBlendModel)
                or model.candidate != 'constant_alpha_0' or model.anchor_model is not model.correction_model):
            raise ValueError('Require the L raw-target payload with195 groups and one shared alpha0 anchor')

    def diagnostics(self):
        return {**super().diagnostics(), 'target_budget': 195, 'model_protocol': MODEL_PROTOCOL,
                'training_group_count': len(self.payload['selected_groups'])}
