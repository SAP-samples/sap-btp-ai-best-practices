"""Read uploaded YAML/JSON/Excel credit settings with explicit currency conversions."""
from copy import deepcopy
from datetime import date
from decimal import Decimal
from io import BytesIO
from pathlib import Path
import pandas as pd
import yaml
from ...optimizer.model.limits_import import EXCEL_REQUIRED_COLUMNS, _normalize_id
from .settings import money


def convert_amount(value,currency,rates):
    """Convert a supplied amount to EUR using only a reviewed positive explicit rate."""
    currency=str(currency).strip().upper()
    amount=Decimal(money(value))
    if currency!='EUR':
        if currency not in rates: raise ValueError(f'Provide an explicit Currency Rates entry for {currency}')
        rate=Decimal(str(rates[currency]['eur_per_unit']))
        if not rate.is_finite() or rate<=0: raise ValueError(f'Invalid exchange rate for {currency}')
        amount*=rate
    return money(amount)


def set_consistent(mapping,key,value):
    """Accept repeated identical settings but reject conflicting amounts or associations."""
    if not key: raise ValueError('Settings contain an empty entity ID')
    if key in mapping and mapping[key]!=value: raise ValueError(f'Conflicting settings for {key}')
    mapping[key]=value


def import_settings(content,filename,current):
    """Return a preview draft from uploaded bytes; no database mutation occurs here."""
    extension=Path(filename).suffix.lower()
    if extension in ('.yaml','.yml','.json'):
        payload=yaml.safe_load(content)
        if not isinstance(payload,dict): raise ValueError('Settings file must contain a mapping')
        return {**deepcopy(current),**payload}
    if extension!='.xlsx': raise ValueError('Upload YAML, JSON or XLSX settings')
    result=deepcopy(current)
    with pd.ExcelFile(BytesIO(content),engine='openpyxl') as workbook:
        rates=deepcopy(current.get('currency_rates',{}))
        if 'Currency Rates' in workbook.sheet_names:
            for row in pd.read_excel(workbook,sheet_name='Currency Rates').to_dict('records'):
                rates[str(row['Currency']).upper()]={'eur_per_unit':str(row['EUR Per Unit']),
                                                   'as_of':pd.Timestamp(row['As Of']).date().isoformat()}
        result['currency_rates']=rates
        frame=pd.read_excel(workbook,sheet_name=0)
        frame.columns=[str(column).strip() for column in frame.columns]
        if missing:=set(EXCEL_REQUIRED_COLUMNS)-set(frame.columns): raise ValueError('Missing required columns: '+', '.join(sorted(missing)))
        for key in ('facility_limits_by_company_code','customer_limits','group_limits','customer_to_group','seller_to_facility','customer_to_facility'):
            result[key]={}
        for index,row in enumerate(frame.to_dict('records'),2):
            try:
                seller,customer,group=(_normalize_id(row.get(key),allow_empty=True) for key in ('ID Seller','ID Debtor','Group Debtor'))
                if not seller or not customer: raise ValueError('Seller and debtor IDs are required')
                facility=_normalize_id(row.get('Facility ID'),allow_empty=True) or seller
                set_consistent(result['seller_to_facility'],seller,facility)
                set_consistent(result['customer_to_facility'],customer,facility)
                facility_amount=row['Facility Limit'] if pd.notna(row['Facility Limit']) else row['Seller Limit']
                facility_currency=row['Currency Facility'] if pd.notna(row['Facility Limit']) else row['Currency Seller LM']
                set_consistent(result['facility_limits_by_company_code'],facility,convert_amount(facility_amount,facility_currency,rates))
                set_consistent(result['customer_limits'],customer,convert_amount(row['Debtor Limit'],row['Currency Debtor LM'],rates))
                if group:
                    set_consistent(result['customer_to_group'],customer,group)
                    set_consistent(result['group_limits'],group,convert_amount(row['Group Limit'],row['Currency Group LM'],rates))
            except (KeyError,ValueError) as error: raise ValueError(f'Limits row {index}: {error}') from None
        if 'Opening Exposure' in workbook.sheet_names:
            result['base_exposure']={'facility':{},'customer':{},'group':{}}
            for row in pd.read_excel(workbook,sheet_name='Opening Exposure').to_dict('records'):
                kind=str(row['Entity Type']).lower()
                if kind not in result['base_exposure']: raise ValueError('Opening Exposure Entity Type must be facility, customer or group')
                set_consistent(result['base_exposure'][kind],_normalize_id(row['Entity ID']),convert_amount(row['Amount'],row['Currency'],rates))
        result['expected_repayments']=[]
        if 'Expected Repayments' in workbook.sheet_names:
            for row in pd.read_excel(workbook,sheet_name='Expected Repayments').to_dict('records'):
                result['expected_repayments'].append({'customer_id':_normalize_id(row['Customer ID']),
                    'facility_id':_normalize_id(row['Facility ID']),'release_date':pd.Timestamp(row['Release Date']).date().isoformat(),
                    'amount':money(row['Amount']),'currency':str(row['Currency']).upper()})
    return result
