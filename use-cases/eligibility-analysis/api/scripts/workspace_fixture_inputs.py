"""Project explicit optimizer fixtures into user-uploadable workspace test files.

Example: PYTHONPATH=api .venv/bin/python api/scripts/validate_workspace_packages.py --output /tmp/workspace-packages

This is development-test input preparation only. It excludes marked historical rows,
preserves candidate dates and amounts, and never feeds local files to application logic.
"""
from io import BytesIO
from pathlib import Path
import json
import pandas as pd
import yaml

SCENARIOS=('stress_balanced_limits','stress_tight_capacity','synthetic_2025-01-28_w8_n40_seed42')


def project_package(path, output):
    """Return a candidate-only offer projection and explicit original credit settings."""
    frame=pd.read_excel(path/'synthetic_extraction.xlsx')
    candidates=frame[frame['Synthetic Row Type'].eq('candidate')].copy()
    history=frame[frame['Synthetic Row Type'].eq('history')]
    if len(candidates)+len(history)!=len(frame):raise ValueError('Unexpected synthetic row labels')
    mapping={'PROGRAMA':'PROGRAMA','Company Code':'ID SELLER','Customer':'ID DEBTOR',
             'Invoice Reference':'REFERENCE NUMBER','Document Number':'DOC NUMBER','Fiscal Year':'FISCAL YEAR',
             'Currency':'ORIGINAL CURRENCY','Issuance date':'ISSUANCE DATE','Due Date':'DUE DATE',
             'Amount':'AMOUNT ORIGINAL','Purchase Price':'TOTAL NET VALUE (ORIGINAL CCY)'}
    offer=candidates[list(mapping)].rename(columns=mapping)
    offer['SELLER']=offer['ID SELLER'];offer['DEBTOR']=offer['ID DEBTOR']
    output.mkdir(parents=True,exist_ok=True);offer_path=output/'candidate-offer.xlsx';offer.to_excel(offer_path,index=False)
    config=yaml.safe_load((path/'limits.yaml').read_text())
    pairs=candidates[['Customer','Company Code']].drop_duplicates()
    if pairs.Customer.duplicated().any():raise ValueError('Ambiguous customer/facility mapping')
    context={'planning_start':'2025-02-03','horizon_weeks':json.loads((path/'scenario_manifest.json').read_text())['generator_config']['weeks'],
             'seller_to_facility':{value:value for value in candidates['Company Code'].unique()},
             'customer_to_facility':dict(zip(pairs.Customer,pairs['Company Code']))}
    combined={**config,**context,'expected_repayments':[]}
    (output/'credit-settings.yaml').write_text(yaml.safe_dump(combined))
    excel=limits_excel(combined);(output/'credit-settings.xlsx').write_bytes(excel)
    return dict(offer_path=offer_path,source_rows=len(frame),candidate_rows=len(candidates),history_rows=len(history),
                context=context,settings=combined,excel=excel)


def limits_excel(settings):
    """Encode reviewed limits and opening exposure in the supported Excel upload schema."""
    records=[]
    for customer,facility in settings['customer_to_facility'].items():
        group=settings.get('customer_to_group',{}).get(customer)
        records.append({'ID Seller':facility,'ID Debtor':customer,'Group Debtor':group,
            'Seller Limit':settings['facility_limits_by_company_code'][facility],'Currency Seller LM':'EUR',
            'Facility Limit':settings['facility_limits_by_company_code'][facility],'Currency Facility':'EUR',
            'Debtor Limit':settings['customer_limits'][customer],'Currency Debtor LM':'EUR',
            'Group Limit':settings.get('group_limits',{}).get(group),'Currency Group LM':'EUR'})
    balances=[{'Entity Type':kind,'Entity ID':entity,'Amount':amount,'Currency':'EUR'}
              for kind,values in settings['base_exposure'].items() for entity,amount in values.items()]
    buffer=BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        pd.DataFrame(records).to_excel(writer,sheet_name='Limits',index=False)
        pd.DataFrame(balances).to_excel(writer,sheet_name='Opening Exposure',index=False)
    return buffer.getvalue()
