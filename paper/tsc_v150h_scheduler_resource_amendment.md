# TSC V150H Scheduler Resource Amendment

V150H was submitted with the inherited V150D request of 65,536 MB per target.
The immediately preceding, code-equivalent V150G seven-city jobs peaked between
851 MB and 1,985 MB. Early V150H telemetry was also below 2 GB. The declaration
was therefore reduced to 8,192 MB for queued jobs and fixed in the launcher for
future execution.

This amendment changes scheduler placement only. It does not change source
code, inputs, target budgets, random seeds, fit parameters, selectors or output
paths. Tasks already launching or running retained their original reservation;
four still-queued tasks were edited in place by the scheduler.
