"""Exercise large invoice packages through the authenticated local workspace API.

Examples:
  PYTHONPATH=api .venv/bin/python api/scripts/validate_workspace_packages.py --direct --require-rpt1 --output /tmp/fixed-reference-selection
  PYTHONPATH=api .venv/bin/python api/scripts/validate_workspace_packages.py --output /tmp/workspace-packages

Requires the local API, HANA and api/.env. Creates clearly named test analyses/runs;
no deployment or live operational offer is modified. Use --require-rpt1 to require actual fixed-reference predictions for every candidate.
Without that option, the harness explicitly accepts any saved 28-day fallback.
"""
import argparse
import json
import time
import uuid
from decimal import Decimal
from pathlib import Path
import requests
from dotenv import dotenv_values
from tqdm import tqdm
from workspace_fixture_inputs import SCENARIOS,project_package


class WorkspaceCheck:
    """Perform authenticated HTTP operations without exposing credentials in evidence."""
    def __init__(self,base,require_rpt1=False):
        """Read the existing local API key and scope calls to the workspace endpoints."""
        self.require_rpt1=require_rpt1
        self.base=base.rstrip('/')+'/api/workspace';self.session=requests.Session()
        self.session.headers['X-API-Key']=dotenv_values('api/.env')['API_KEY']

    def call(self,method,path,**kwargs):
        """Require successful responses and return parsed API data."""
        started=time.monotonic()
        response=self.session.request(method,self.base+path,timeout=90,**kwargs)
        if method!='GET':tqdm.write(f'{method} {path}: {response.status_code} ({time.monotonic()-started:.2f}s)')
        if not response.ok:raise RuntimeError(f'{method} {path}: {response.status_code} {response.text[:1200]}')
        return response.json()

    def wait(self,run):
        """Poll bounded background work while showing terminal progress."""
        with tqdm(total=1800,desc=run['run_id'][:8],unit='s',leave=False) as progress:
            started=time.monotonic()
            while run['status'] in ('estimating_lifetimes','optimizing'):
                if time.monotonic()-started>1800:raise TimeoutError(run['run_id'])
                time.sleep(2);progress.update(2);run=self.call('GET','/runs/'+run['run_id'])
        return run

    def solve(self,run):
        """Verify an explicit acknowledgement gate, then solve the exact saved scope."""
        path='/runs/'+run['run_id'];run=self.wait(self.call('POST',path+'/prepare',json={'expected_revision':run['revision']}))
        if self.require_rpt1:
            assert run['preparation']['fallback_count']==0,run['preparation']['history']
            assert run['preparation']['history']['context_policy']=='fixed_reference'
            assert run['preparation']['history']['api_calls']>0
        elif run['status']=='awaiting_lifetime_acknowledgement':
            request={'expected_revision':run['revision'],'preparation_id':run['preparation_id'],'accepted':True}
            run=self.wait(self.call('POST',path+'/acknowledgement',json=request))
        assert run['status']=='completed',run.get('result')
        result=run['result'];selected={row['row_id'] for row in result['selected']}
        assert selected<=set(run['row_ids'])
        for weeks in result['exposure'].values():
            for entities in weeks.values():
                for value in entities.values():assert value['used_total']<=value['limit']+.005
        # Files may finish shortly after the immutable result becomes visible.
        for attempt in range(40):
            artifacts=self.call('GET',path+'/artifacts')['items']
            if all(item['status']=='ready' for item in artifacts):break
            time.sleep(1)
        assert len(artifacts)==9 and all(item['status']=='ready' for item in artifacts),artifacts
        response=self.session.get(self.base+path+'/artifacts/all-files',timeout=90)
        assert response.status_code==200 and response.content.startswith(b'PK')
        return run


def summarize(run):
    """Report immutable scope, proof status, deferrals and artifact evidence."""
    result=run['result'];first=result['week_starts'][0]
    dates=[row['planned_week_start_iso'] for row in result['weekly_plan']]
    assert {row['row_id'] for row in result['weekly_plan']}=={row['row_id'] for row in result['selected']}
    for row in result['weekly_plan']:
        assert row['planned_week_start_iso'] in result['week_starts']
        assert row['planned_week_start_iso']<=str(row['Due Date'])[:10]
        assert row['planned_week_start_iso']>=str(row['Offer File Date (UTC)'])[:10]
    return dict(history_dataset=run['preparation']['history'].get('dataset_id'),context_policy=run['preparation']['history'].get('context_policy'),
        rpt1_predictions=sum(row['source']=='rpt1' for row in run['preparation']['predictions']),fallback_count=run['preparation']['fallback_count'],
        run_id=run['run_id'],candidates=len(run['row_ids']),selected=len(result['selected']),
        solver_status=result['solver_status'],funded_eur=result['objective_amount'],
        forecast_capacity_violations=0,artifact_count=9,fallback_acknowledged=bool(run['preparation'].get('acknowledgement')),
        deferred_selected_invoices=sum(day>first for day in dates),funding_dates=sorted(set(dates)))


def main():
    """Validate original YAML, equivalent Excel, full eligible scope and a repayment subset."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='/tmp/workspace-packages');parser.add_argument('--base',default='http://127.0.0.1:8000');parser.add_argument('--direct',action='store_true',help='Upload original extraction directly for invoice selection');parser.add_argument('--require-rpt1',action='store_true',help='Require fixed-reference RPT-1 predictions with no fallback');args=parser.parse_args()
    output=Path(args.output);client=WorkspaceCheck(args.base,args.require_rpt1);evidence=[]
    for name in tqdm(SCENARIOS,desc='Workspace package checks',unit='package'):
        source=Path('data/synthetic')/name;package=project_package(source,output/name)
        analysis=client.call('POST','/selections' if args.direct else '/analyses',files={'file':(name+'-extraction.xlsx' if args.direct else name+'-candidate-offer.xlsx',(source/'synthetic_extraction.xlsx').read_bytes() if args.direct else package['offer_path'].read_bytes())},
            data={'analysis_date':'2025-01-28','settings':json.dumps({'nddt':6,'teih':365,'isspur':0,'eligible_currencies':['EUR']}),'request_key':'package-check-'+str(uuid.uuid4())})
        assert analysis['total_invoices']==package['candidate_rows']
        if args.direct:
            assert analysis['eligible_count']==package['candidate_rows']
            assert analysis['settings']['historical_rows_excluded']==package['history_rows']
        run=client.call('POST','/runs',json={'analysis_id':analysis['analysis_id'],'candidate_scope':{'mode':'all_eligible','row_ids':[]}})
        assert len(run['row_ids'])==analysis['eligible_count']
        path='/runs/'+run['run_id']
        run=client.call('PUT',path+'/settings',json={'expected_revision':run['revision'],'settings':{**run['settings'],'horizon_weeks':package['context']['horizon_weeks']} if args.direct else package['context']})
        previews=[]
        for filename,content in [('limits.yaml',(source/'limits.yaml').read_bytes()),('credit-settings.xlsx',package['excel'])]:
            preview=client.call('POST',path+'/settings/import',files={'file':(filename,content)},data={'expected_revision':run['revision']})
            assert not preview['readiness_issues'],preview['readiness_issues'];previews.append(preview)
        for key in ('facility_limits_by_company_code','customer_limits','group_limits','customer_to_group','base_exposure'):
            assert previews[0]['normalized_settings'][key]==previews[1]['normalized_settings'][key],key
        constant=previews[0]['weekly_opening_preview'];assert all(row['opening']==constant[0]['opening'] for row in constant)
        run=client.call('PUT',path+'/settings',json={'expected_revision':run['revision'],'settings':previews[0]['normalized_settings']})
        run=client.solve(run)
        record={'source_mode':'direct_selection' if args.direct else 'eligibility_projection','package':name,'analysis_id':analysis['analysis_id'],'uploaded_candidate_rows':package['candidate_rows'],
                'excluded_historical_rows':package['history_rows'],'eligible_rows':analysis['eligible_count'],
                'yaml_excel_equivalent':True,'unscheduled_opening_constant':True,'all_eligible':summarize(run)}
        if name=='stress_tight_capacity':
            subset=client.call('POST','/runs',json={'analysis_id':analysis['analysis_id'],'candidate_scope':{'mode':'selected','row_ids':run['row_ids'][:24]}})
            settings=previews[0]['normalized_settings'];customer=next(iter(settings['customer_limits']));facility=settings['customer_to_facility'][customer]
            settings['expected_repayments']=[{'customer_id':customer,'facility_id':facility,'release_date':'2025-02-12','amount':'10000.00','currency':'EUR'}]
            subpath='/runs/'+subset['run_id'];preview=client.call('POST',subpath+'/settings/preview',json={'expected_revision':subset['revision'],'settings':settings})
            assert not preview['readiness_issues'],preview['readiness_issues']
            series=preview['weekly_opening_preview'];group=settings['customer_to_group'][customer]
            for kind,entity in [('customer',customer),('facility',facility),('group',group)]:
                initial=Decimal(series[0]['opening'][kind][entity]);assert Decimal(series[1]['opening'][kind][entity])==initial
                assert all(Decimal(row['opening'][kind][entity])==initial-10000 for row in series[2:])
            subset=client.call('PUT',subpath+'/settings',json={'expected_revision':subset['revision'],'settings':settings})
            record['selected_subset_with_partial_repayment']=summarize(client.solve(subset));record['repayment_releases_next_monday_at_all_levels']=True
        evidence.append(record);(output/'evidence.json').write_text(json.dumps(evidence,indent=2))
        print(json.dumps(record),flush=True)
    print('Evidence: '+str(output/'evidence.json'))


if __name__=='__main__':main()
