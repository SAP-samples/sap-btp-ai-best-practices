"""Normalize manual credit settings and preview opening exposure before mutations."""
from copy import deepcopy
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from ...optimizer.model.planning_calendar import planning_weeks
from ...optimizer.model.repayments import build_opening_schedule
from ...models.workspace import RevisionConflict, WorkspaceValidationError


def money(value):
    """Normalize finite nonnegative money into an exact two-decimal string."""
    try:
        amount=Decimal(str(value))
        if not amount.is_finite() or amount<0: raise ValueError('Money must be finite and nonnegative')
        return str(amount.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))
    except (InvalidOperation,ValueError,TypeError):
        raise ValueError('Provide a finite nonnegative amount') from None


def cents(value):
    """Convert normalized decimal money to integer cents at a solver boundary."""
    return int(Decimal(money(value))*100)


def validate_settings(settings,rows):
    """Return reviewed settings, readiness issues and an optional weekly opening preview.

    Input rows are the exact saved candidate population. Missing limits, associations,
    exposure and FX remain explicit readiness issues; no automatic limits are generated.
    """
    if not isinstance(settings,dict): raise WorkspaceValidationError('Settings must be an object','settings')
    map_keys=('facility_limits_by_company_code','customer_limits','group_limits','customer_to_group',
              'seller_to_facility','customer_to_facility','currency_rates','base_exposure')
    for key in map_keys:
        if key in settings and not isinstance(settings[key],dict):
            raise WorkspaceValidationError('Provide a mapping of IDs to values',key)
    for key in ('seller_to_facility','customer_to_facility','customer_to_group'):
        if any(not isinstance(value,str) or not value.strip() for value in settings.get(key,{}).values()):
            raise WorkspaceValidationError('Associations must contain nonempty text IDs',key)
    if any(not isinstance(value,dict) for value in settings.get('base_exposure',{}).values()):
        raise WorkspaceValidationError('Opening exposure must contain facility/customer/group mappings','base_exposure')
    if not isinstance(settings.get('expected_repayments',[]),list) or any(not isinstance(row,dict) for row in settings.get('expected_repayments',[])):
        raise WorkspaceValidationError('Expected repayments must be a list of records','expected_repayments')
    normalized=deepcopy(settings)
    issues,warnings=[],[]
    preview=[]
    normalized.update(calendar_version='monday-v1',planning_mode='multi_week',synthetic_generation={'enabled':False})
    normalized.setdefault('expected_repayments',[])
    for key in ('facility_limits_by_company_code','customer_limits','group_limits','customer_to_group',
                'seller_to_facility','customer_to_facility','currency_rates'):
        normalized.setdefault(key,{})
    normalized.setdefault('base_exposure',{})
    for kind in ('facility','customer','group'): normalized['base_exposure'].setdefault(kind,{})
    weeks=[]
    try:
        normalized['planning_start']=str(normalized.get('planning_start',''))
        weeks=planning_weeks(date.fromisoformat(normalized['planning_start']),normalized.get('horizon_weeks',12))
        normalized['horizon_weeks']=len(weeks)
    except (ValueError,TypeError): issues.append({'path':'planning_start','message':'Choose a Monday start and 1–104 planning weeks','row_ids':[]})
    for key in ('facility_limits_by_company_code','customer_limits','group_limits'):
        for entity,value in list(normalized[key].items()):
            try: normalized[key][entity]=money(value)
            except ValueError as error: issues.append({'path':f'{key}.{entity}','message':str(error),'row_ids':[]})
    for kind,values in normalized['base_exposure'].items():
        if kind not in ('facility','customer','group'):
            issues.append({'path':'base_exposure','message':f'Unknown exposure level {kind}','row_ids':[]}); continue
        for entity,value in list(values.items()):
            try: values[entity]=money(value)
            except ValueError as error: issues.append({'path':f'base_exposure.{kind}.{entity}','message':str(error),'row_ids':[]})
    # Repayment/import-only currencies need the same as-of policy as invoice currencies.
    for currency,rate in normalized['currency_rates'].items():
        try:
            number=Decimal(str(rate['eur_per_unit']))
            effective=date.fromisoformat(str(rate['as_of']))
            if not number.is_finite() or number<=0 or (weeks and effective>weeks[0]):raise ValueError()
            rate.update(eur_per_unit=str(number),as_of=effective.isoformat())
        except (KeyError,ValueError,InvalidOperation,TypeError,AttributeError):
            issues.append({'path':f'currency_rates.{currency}','message':'Provide a positive EUR rate and as-of date no later than planning start','row_ids':[]})
    customers,facilities,groups=set(),set(),set()
    for row in rows:
        invoice=row['invoice']; customer=invoice.get('debtor_id'); seller=invoice.get('seller_id')
        if not customer:
            issues.append({'path':'debtor_id','message':'Customer ID is required for optimization','row_ids':[row['row_id']]})
            continue
        facility=normalized['seller_to_facility'].get(seller)
        customers.add(customer)
        if not facility: issues.append({'path':f'seller_to_facility.{seller}','message':'Map this seller to its credit facility','row_ids':[row['row_id']]})
        else: facilities.add(facility)
        if group:=normalized['customer_to_group'].get(customer): groups.add(group)
        for key,entity in (('customer_limits',customer),('facility_limits_by_company_code',facility)):
            if entity and entity not in normalized[key]: issues.append({'path':f'{key}.{entity}','message':'Provide an explicit credit limit','row_ids':[row['row_id']]})
        currency=invoice['original_currency']
        if currency!='EUR':
            try:
                rate=normalized['currency_rates'][currency]
                number=Decimal(str(rate['eur_per_unit']))
                if not number.is_finite() or number<=0: raise ValueError()
                effective=date.fromisoformat(str(rate['as_of']))
                if weeks and effective>weeks[0]: raise ValueError()
                rate['eur_per_unit']=str(number)
            except (KeyError,ValueError,InvalidOperation,TypeError):
                issues.append({'path':f'currency_rates.{currency}','message':'Provide a positive EUR exchange rate and as-of date no later than planning start','row_ids':[row['row_id']]})
    for group in groups:
        if group not in normalized['group_limits']: issues.append({'path':f'group_limits.{group}','message':'Provide a limit for the mapped group','row_ids':[]})
    # Explicit zero is a user acknowledgement; omission alone never becomes zero.
    for kind,entities in (('customer',customers),('facility',facilities),('group',groups)):
        for entity in entities:
            if entity not in normalized['base_exposure'][kind]:
                if normalized.get('opening_confirmed_zero') is True: normalized['base_exposure'][kind][entity]='0.00'
                else: issues.append({'path':f'base_exposure.{kind}.{entity}','message':'Provide opening exposure or explicitly confirm omitted balances are zero','row_ids':[]})
    repayments=[]
    for index,row in enumerate(normalized['expected_repayments']):
        try:
            rate=Decimal('1') if row['currency']=='EUR' else Decimal(str(normalized['currency_rates'][row['currency']]['eur_per_unit']))
            if not rate.is_finite() or rate<=0: raise ValueError('Provide a positive exchange rate')
            amount=cents(Decimal(money(row['amount']))*rate)
            release=date.fromisoformat(str(row['release_date']))
            repayments.append(dict(customer_id=row['customer_id'],facility_id=row['facility_id'],release_date=release,amount_cents=amount))
            row.update(amount=money(row['amount']),release_date=release.isoformat())
            if weeks and release>weeks[-1]: warnings.append(f'Repayment row {index+1} is beyond the planning horizon')
        except (ValueError,KeyError,InvalidOperation,TypeError) as error:
            issues.append({'path':f'expected_repayments.{index}','message':f'Provide customer, facility, date, positive amount and currency: {error}','row_ids':[]})
    if not issues:
        try:
            opening={kind:{key:cents(value) for key,value in values.items()} for kind,values in normalized['base_exposure'].items()}
            schedule=build_opening_schedule(opening,repayments,weeks,normalized['customer_to_group'],normalized['customer_to_facility'])
            limits={'facility':normalized['facility_limits_by_company_code'],'customer':normalized['customer_limits'],'group':normalized['group_limits']}
            for kind,values in opening.items():
                for key,value in values.items():
                    if key not in limits[kind]: raise ValueError(f'Opening {kind} {key} requires an explicit limit')
                    if value>cents(limits[kind][key]): raise ValueError(f'Opening exposure exceeds the {kind} limit for {key}')
            preview=[{'week_start':week.isoformat(),'opening':{kind:{key:money(Decimal(value)/100) for key,value in values.items()} for kind,values in balances.items()}}
                     for week,balances in schedule.items()]
        except ValueError as error: issues.append({'path':'base_exposure','message':str(error),'row_ids':[]})
    return dict(normalized_settings=normalized,readiness_issues=issues,warnings=warnings,weekly_opening_preview=preview)


def preview_run_settings(service,run_id,expected_revision,settings):
    """Validate a panel draft without changing its saved run revision."""
    run=service.runs.get(run_id)
    if run['revision']!=expected_revision or run['status']!='draft': raise RevisionConflict('Refresh this draft before changing settings')
    rows=[row for row in service.analyses.all_rows(run['analysis_id']) if row['row_id'] in set(run['row_ids'])]
    return validate_settings(settings,rows)


def save_run_settings(service,run_id,expected_revision,settings):
    """Persist a reviewed settings draft and its authoritative readiness issues."""
    preview=preview_run_settings(service,run_id,expected_revision,settings)
    return service.runs.compare_and_swap(run_id,expected_revision,{'settings':preview['normalized_settings'],
                                                                  'readiness_issues':preview['readiness_issues']})
