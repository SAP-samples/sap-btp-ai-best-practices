"""Replay fixed funding decisions with alternate lifetimes; never refit on observed truth."""
import math
from datetime import timedelta


def exposure_series(schedule,lifetimes_days,week_starts):
    """Compute integer-cent outstanding balances under the solver's whole-week convention."""
    output=[]
    for week in week_starts:
        total=0
        for row in schedule:
            days=lifetimes_days[row['row_id']]
            if not math.isfinite(days) or days<=0:raise ValueError('Observed lifetimes must be finite and positive')
            release=row['funding_week']+timedelta(weeks=math.ceil(days/7))
            if row['funding_week']<=week<release:total+=row['amount_cents']
        output.append(total)
    return output


def capacity_breaches(used_cents,limits_cents):
    """Return exact positive weekly excess for aligned usage and limit series."""
    if len(used_cents)!=len(limits_cents):raise ValueError('Usage and limit series must align')
    excess=[max(0,used-limit) for used,limit in zip(used_cents,limits_cents)]
    return dict(breach_weeks=sum(value>0 for value in excess),max_excess_cents=max(excess,default=0),total_excess_cents=sum(excess))


def replay_capacity(schedule,lifetimes,week_starts,limits,opening,customer_to_group):
    """Replay every constrained level, adding the supplied weekly opening/repayment schedule.

    Missing observed outcomes are reported and omitted from observed replay, so callers
    cannot interpret incomplete coverage as proof of realized compliance.
    """
    missing=[row['row_id'] for row in schedule if row['row_id'] not in lifetimes]
    complete=[row for row in schedule if row['row_id'] in lifetimes];results={}
    for kind,entities in limits.items():
        results[kind]={}
        for entity,limit in entities.items():
            selected=[row for row in complete if (customer_to_group.get(row['customer_id']) if kind=='group' else row[kind+'_id'])==entity]
            new=exposure_series(selected,lifetimes,week_starts)
            total=[new[index]+opening.get(week,{}).get(kind,{}).get(entity,0) for index,week in enumerate(week_starts)]
            results[kind][entity]={**capacity_breaches(total,[limit]*len(week_starts)),'used_cents':total}
    return {'levels':results,'missing_outcome_row_ids':missing,'observed_selected_count':len(complete),'selected_count':len(schedule)}
