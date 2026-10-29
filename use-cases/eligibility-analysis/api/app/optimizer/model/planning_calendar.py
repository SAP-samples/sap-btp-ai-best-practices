"""Explicit Monday calendar for new workspace runs; legacy results retain their calendar."""
from datetime import timedelta


def effective_release_week(release_date):
    """Return the first Monday on or after a repayment's effective date."""
    return release_date+timedelta(days=(-release_date.weekday())%7)


def planning_weeks(start,horizon_weeks):
    """Build an explicit positive-length Monday horizon without silently moving dates."""
    if start.weekday()!=0 or type(horizon_weeks) is not int or not 1<=horizon_weeks<=104:
        raise ValueError('Planning start must be Monday and horizon must be 1–104 weeks')
    return [start+timedelta(weeks=index) for index in range(horizon_weeks)]
