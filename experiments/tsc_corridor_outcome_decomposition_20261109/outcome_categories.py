"""Split detector category zero without changing its ten-second arrival label."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_corridor_residual_transfer_20261109'))
from residual_model import label_category

OUTCOME_CLASSES = ('straight_by10', 'straight_after10', 'other_by10', 'other_after10',
                   'terminal_by10', 'terminal_after10', 'censored')


def classify_outcome(flight, outcome, time_s):
    """Return (original category 0..10, observed outcome class).

    ``other`` records detection-line crossing in a lane other than the monitored
    straight lane. ``terminal`` records departure from the tracked road when
    SUMO reports the vehicle arrived. Neither establishes turn intent. In this
    study's currently-straight contexts, a later other-lane detection implies an
    intervening lane change, without identifying its time. A complete censored
    window remains an observed category-zero label.
    """
    category = label_category(flight, outcome, time_s)
    if flight['age_s'] != time_s-outcome['entry_s']:
        raise ValueError('Label age differs from the current in-flight release')
    status = outcome['status']
    if status == 'censored':
        return category, 'censored'
    period = 'by10' if outcome['end_s']-time_s <= 10 else 'after10'
    return category, status+'_'+period
