"""Run structural capacity fixtures and persist benchmark evidence in HANA.

Example: PYTHONPATH=api .venv/bin/python api/scripts/validate_capacity_scenarios.py --report docs/reports/2026-09-07-capacity-validation.md

Local Excel/YAML inputs are explicit development fixtures, not application data storage.
Synthetic lifetime outcomes must not be interpreted as RPT-1 prediction accuracy.
"""
import argparse
import json
from datetime import date
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from app.services.database.backend import get_backend
from app.services.lifecycle.evaluation_store import EvaluationStore
from app.optimizer.model.limits import load_limits_config,resolve_limits
from app.optimizer.model.planning_calendar import effective_release_week,planning_weeks
from app.optimizer.opt.optimizer_multi_week import optimize_multi_week,MultiWeekOptimizerSettings


SCENARIOS=('stress_balanced_limits','stress_tight_capacity','synthetic_2025-01-28_w8_n40_seed42')


def run_scenario(path):
    """Solve current fixture rows with fixed lifetimes and original explicit credit limits."""
    frame=pd.read_excel(path/'synthetic_extraction.xlsx')
    current=frame[frame['Synthetic Row Type'].astype(str).str.lower().eq('candidate')].copy()
    if current.empty:
        # Inspect actual fixture labels instead of injecting historical rows as candidates.
        raise ValueError('No current rows; observed labels: '+str(frame['Synthetic Row Type'].unique().tolist()))
    current['row_id']=['fixture-'+str(index) for index in range(len(current))]
    current['expected_lifetime_weeks']=4
    config=load_limits_config(path/'limits.yaml');limits=resolve_limits(current,config)
    start=effective_release_week(pd.to_datetime(current['Offer File Date (UTC)']).min().date())
    horizon=json.loads((path/'scenario_manifest.json').read_text())['generator_config']['weeks']
    weeks=[pd.Timestamp(week) for week in planning_weeks(start,horizon)]
    opening={week:{kind:{str(key):float(value) for key,value in config['base_exposure'].get(kind,{}).items()} for kind in ('facility','customer','group')} for week in weeks}
    result=optimize_multi_week(current,limits,weeks,opening,MultiWeekOptimizerSettings(horizon_weeks=horizon,attempt_cap=horizon,calendar_version='monday-v1',max_time_seconds=5))
    violations=[dict(level=kind,week=week,entity=entity,excess=values['used_total']-values['limit'])
        for kind,usage in [('facility',result.facility_weekly_usage),('customer',result.customer_weekly_usage),('group',result.group_weekly_usage)]
        for week,entities in usage.items() for entity,values in entities.items() if values['used_total']>values['limit']+.005]
    if violations:raise ValueError('Forecast plan violates supplied constraints')
    return dict(row_id=path.name,source_current_rows=len(current),source_history_rows=len(frame)-len(current),selected=len(result.selected_df),
        amount=result.objective_amount,solver_status=result.status,forecast_capacity_violations=len(violations),
        horizon_weeks=horizon,planning_start=start.isoformat(),lifetime_policy='Deterministic four-week structural test; no RPT-1 call')


def main():
    """Run named fixtures with bounded solver time and publish honest proof/status evidence."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',default='docs/reports/2026-09-07-capacity-validation.md');args=parser.parse_args()
    load_dotenv('api/.env');backend=get_backend()
    if not backend.is_hana:parser.error('HANA required for experiment evidence')
    store=EvaluationStore(backend);identity=store.save_experiment({'kind':'structural-capacity-v1','scenarios':SCENARIOS,'lifetime_weeks':4,'max_solver_seconds':5})
    results=[]
    for name in tqdm(SCENARIOS,desc='Capacity fixtures',unit='scenario'):
        results.append(run_scenario(Path('data/synthetic')/name));store.save_predictions(identity,[results[-1]])
    store.save_metrics(identity,{'scenarios':results})
    report=Path(args.report);report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text('# Capacity validation\n\nHANA experiment: `'+identity+'`\n\nThese are structural fixture checks with supplied credit settings and deterministic four-week lifetimes. Historical rows are excluded from candidates. Original dates and all facility/customer/group constraints are retained. Time-limited FEASIBLE is not an optimality claim.\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\nThe observed-release replay unit test holds decisions fixed and demonstrates a 40-unit capacity breach when a seven-day forecast actually takes fourteen days. The reusable replay includes opening exposure and scheduled repayments at every constrained level.\n\nHistorical validation does not contain corresponding operational credit limits, opening exposures or an actual funded portfolio. A measured historical portfolio-breach rate cannot be inferred from it. The eight-week synthetic fixture also lacks realized release outcomes. Consequently this report does not claim realized credit-policy acceptance.\n')
    print(json.dumps({'experiment_id':identity,'report':str(report),'results':results},indent=2))


if __name__=='__main__':main()
