from dataclasses import replace
from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardAudit, ContrastProposal, _coordinate_contrast_proposals,
)


def switch_proposal():
    return ContrastProposal(
        prior_candidate=SimpleNamespace(state='Grr'),
        selected_candidate=SimpleNamespace(state='rrG'),
        learned_differs=True, eligible=True, priority=.8,
    )


def coordinate(proposal, cooldowns, audit):
    return _coordinate_contrast_proposals(
        {'A': proposal}, cooldowns=cooldowns, cooldown_intervals=44,
        max_simultaneous_overrides=None, audit=audit,
        current_green_states={'A': 'rGr'},
    )['A']


def test_model_switch_spacing_preserves_440_seconds_of_pp_continuation():
    proposal = switch_proposal()
    cooldowns, audit = {}, ContrastGuardAudit()
    actions = {time_sec: coordinate(proposal, cooldowns, audit).state
               for time_sec in range(0, 451, 10)}
    assert actions[0] == actions[450] == proposal.selected_candidate.state
    assert all(actions[time_sec] == proposal.prior_candidate.state
               for time_sec in range(10, 441, 10))
    assert audit.accepted_overrides == audit.accepted_switch_overrides == 2
    assert audit.accepted_stay_overrides == 0
    assert audit.rejected_cooldown == 44
    assert audit.proposed_overrides == 46


def test_pp_agreements_and_ineligible_proposals_do_not_extend_spacing():
    proposal = switch_proposal()
    agrees = replace(proposal, selected_candidate=proposal.prior_candidate,
                     learned_differs=False, eligible=False, priority=0.)
    ineligible = replace(proposal, selected_candidate=proposal.prior_candidate,
                         eligible=False, priority=0., rejection='minimum_priority')
    cooldowns, audit = {}, ContrastGuardAudit()
    assert coordinate(proposal, cooldowns, audit) == proposal.selected_candidate
    for index, _time_sec in enumerate(range(10, 431, 10), 1):
        intermediate = agrees if index % 2 else ineligible
        assert coordinate(intermediate, cooldowns, audit) == proposal.prior_candidate
    assert coordinate(proposal, cooldowns, audit) == proposal.prior_candidate  # t=440
    assert coordinate(proposal, cooldowns, audit) == proposal.selected_candidate  # t=450
    assert audit.accepted_overrides == audit.accepted_switch_overrides == 2
    assert audit.rejected_cooldown == 1
    assert audit.prior_agreements == 22
    assert audit.rejected_minimum_priority == 21
    assert audit.proposed_overrides == 3
