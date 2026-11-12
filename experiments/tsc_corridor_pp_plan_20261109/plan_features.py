"""Read the recorded PP request's original ten-second nominal protection."""


def pp_plan_feature(row, flight):
    """Describe this PP plan, without reconstructing actual future signal states.

    Protection means all monitored straight link indices are G in a nominal
    second. The mask belongs to the original ten-second row, before any service
    window extension. Its count is planned exposure, not observed future green.
    """
    pp = row['pp']
    mask = int(row['protected'][flight['edge']][pp])
    if not 0 <= mask < (1 << 10):
        raise ValueError('PP plan requires the original ten-bit protection mask')
    return dict(protected_seconds=mask.bit_count(), mask=mask,
                planned_state=row['states'][pp], pp_index=pp)
