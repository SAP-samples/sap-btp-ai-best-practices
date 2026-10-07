"""Validated cumulative projections of optional repayments against opening exposure."""
from copy import deepcopy
from .planning_calendar import effective_release_week


def build_opening_schedule(opening_cents,repayments,weeks,customer_to_group,customer_to_facility=None):
    """Project integer-cent opening balances across facility/customer/group constraints.

    Repayments use customer_id, facility_id, release_date and amount_cents. Missing
    schedules preserve constant exposure. An optional validated association map
    prevents releasing another facility's historical balance.
    """
    if not weeks: raise ValueError('A planning horizon is required')
    opening={kind:dict(opening_cents.get(kind,{})) for kind in ('facility','customer','group')}
    for balances in opening.values():
        if any(type(value) is not int or value<0 for value in balances.values()):
            raise ValueError('Opening exposure must use nonnegative integer cents')
    cumulative=deepcopy(opening)
    seen=set()
    effects=[]
    for number,row in enumerate(repayments,1):
        customer,facility,release,amount=(row[key] for key in ('customer_id','facility_id','release_date','amount_cents'))
        identity=(customer,facility,release,amount)
        if identity in seen: raise ValueError(f'Repayment row {number}: duplicate repayment')
        seen.add(identity)
        if type(amount) is not int or amount<=0: raise ValueError(f'Repayment row {number}: amount must be positive cents')
        if release<weeks[0]: raise ValueError(f'Repayment row {number}: release precedes the opening snapshot')
        if customer_to_facility is not None and customer_to_facility.get(customer)!=facility:
            raise ValueError(f'Repayment row {number}: customer/facility association is missing or inconsistent')
        affected=[('customer',customer),('facility',facility)]
        if group:=customer_to_group.get(customer): affected.append(('group',group))
        for kind,entity in affected:
            if entity not in cumulative[kind]: raise ValueError(f'Repayment row {number}: unknown opening {kind} {entity}')
            cumulative[kind][entity]-=amount
            if cumulative[kind][entity]<0: raise ValueError(f'Repayment row {number}: cumulative release exceeds {kind} opening exposure')
        effects.append((effective_release_week(release),amount,affected))
    output={}
    for week in weeks:
        balances=deepcopy(opening)
        for release,amount,affected in effects:
            if release<=week:
                for kind,entity in affected: balances[kind][entity]-=amount
        output[week]=balances
    return output
