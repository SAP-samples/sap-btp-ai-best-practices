"""Run resumable HANA-backed chronological RPT-1 validation.

Examples (repository root):
  PYTHONPATH=api .venv/bin/python api/scripts/evaluate_lifetime_chronological.py --dataset-id reference-runtime-20260907 --dry-run
  PYTHONPATH=api .venv/bin/python api/scripts/evaluate_lifetime_chronological.py --dataset-id invoice-lifetime-reference-v1 --folds 3 --test-rows 200 --observable-days 110 --report docs/reports/2026-09-25-rpt1-lifetime-validation.md
  PYTHONPATH=api .venv/bin/python api/scripts/evaluate_lifetime_chronological.py --dataset-id reference-runtime-20260907 --resume EXPERIMENT_ID
"""
import argparse
import json
import math
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from app.services.database.backend import get_backend
from app.services.lifecycle.store import LifecycleStore
from app.services.lifecycle.evaluation_store import EvaluationStore
from app.services.lifecycle.evaluation import chronological_folds,lifetime_metrics,median_baseline,error_metrics
from app.optimizer.model.lifetime_estimation import LifetimeEstimationConfig,estimate_candidate_lifetime_with_rpt1


def aggregate(rows):
    """Compute all errors from persisted prediction rows, including transparent baselines."""
    actual=[row['actual_days'] for row in rows]
    result=lifetime_metrics(actual,[row['predicted_days'] for row in rows],[row['source'] for row in rows])
    result['four_week_baseline']=error_metrics(actual,[28]*len(rows))
    result['median_baseline']=error_metrics(actual,[row['median_days'] for row in rows])
    result['segments']={}
    for key in ('fold','customer','program'):
        result['segments'][key]={str(value):lifetime_metrics(
            [row['actual_days'] for row in rows if row[key]==value],
            [row['predicted_days'] for row in rows if row[key]==value],
            [row['source'] for row in rows if row[key]==value]) for value in sorted({row[key] for row in rows})}
    return result


def write_report(path,identity,metadata,metrics):
    """Export evidence and limitations, keeping detailed tabular predictions in HANA."""
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True)
    lines=['# Chronological RPT-1 lifetime validation','',f'Generated {datetime.now(timezone.utc).isoformat()}', '',
        f'HANA experiment: `{identity}`', '',f"Dataset: `{metadata['dataset_id']}`; source content hash: `{metadata['source_hash']}`.",'',
        '## Method and source boundaries','',
        'Three disjoint forward windows are requested. Context is frozen at the earliest credit-start timestamp of each window, conservatively preceding every query. '
        f"Queries are drawn only from fundings at least {metadata['observable_days']} days before the last observed outcome, so recent right-censored cohorts (only short lifetimes observed) do not bias the test set. "
        'Only positive, completed, uniquely identified lifecycles are evaluated. Reconciliation-file time proxies outcome availability; actual ingestion times and historical feature revisions are unavailable. This is not a random holdout or business acceptance.','',
        'Original outcome/target fields are removed from query input. Customer/program/global median baselines use the same admissible context; the four-week baseline is unchanged. API failures remain fallback observations and are excluded from model-only accuracy.','',
        '## Results','', '| Method | Count | MAE days | RMSE days |','|---|---:|---:|---:|']
    for label,key in [('RPT-1 only','model_only'),('Operational including fallback','operational'),('Four weeks','four_week_baseline'),('Hierarchical median','median_baseline')]:
        row=metrics[key];lines.append(f"| {label} | {row['count']} | {row['mae_days']} | {row['rmse_days']} |")
    lines.extend(['',f"Model coverage: {metrics['model_coverage']:.1%}; fallback rows: {metrics['fallback_count']}; invalid rows: {metrics['invalid_count']}.",'',
        '## Full aggregate evidence','', '```json',json.dumps(metrics,indent=2),'```','',
        '## Interpretation','',
        'Compare model-only error and coverage together. A correct optimizer using forecasts does not establish realized exposure compliance. No fallback duration or automatic lifetime policy is changed by this evaluation. Capacity replay is a separate experiment; synthetic stress fixture lifetimes do not establish predictive accuracy.'])
    target.write_text('\n'.join(lines)+'\n')


def main():
    """Inspect folds first, then measure/resume bounded batches with HANA persistence."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-id',required=True);parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--test-rows',type=int,default=100);parser.add_argument('--min-context',type=int,default=500)
    parser.add_argument('--max-context',type=int,default=800);parser.add_argument('--query-batch-size',type=int,default=50)
    parser.add_argument('--observable-days',type=float,default=110,help='Skip fundings closer than this to the last outcome (right-censoring guard)')
    parser.add_argument('--context-min-rows',type=int,default=100,help='Top-up size for customers with little history')
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--dry-run',action='store_true');parser.add_argument('--resume')
    parser.add_argument('--report',default='docs/reports/2026-09-07-rpt1-lifetime-validation.md');args=parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1]/'.env');backend=get_backend()
    if not backend.is_hana:parser.error('HANA is required for validation predictions')
    history=LifecycleStore(backend);source=history.get_dataset(args.dataset_id);frame=pd.DataFrame(history.records(args.dataset_id))
    folds=chronological_folds(frame,args.folds,args.test_rows,args.min_context,observable_days=args.observable_days)
    if not folds:parser.error('No chronological windows have enough admissible outcomes')
    config=LifetimeEstimationConfig(context_min_rows=args.context_min_rows,context_max_rows=args.max_context,
        query_batch_size=args.query_batch_size,env_path=str(Path(__file__).resolve().parents[1]/'.env'),max_parallel_calls=1)
    metadata=dict(dataset_id=args.dataset_id,source_hash=source['content_hash'],folds=folds,config=asdict(config),seed=args.seed,
        observable_days=args.observable_days,method='frozen-forward-window-v2')
    print(json.dumps({'folds':[{'as_of':fold['as_of'],'queries':len(fold['query_keys']),'context':len(fold['context_keys'])} for fold in folds],
        'estimated_calls':sum(math.ceil(len(fold['query_keys'])/args.query_batch_size) for fold in folds)},indent=2),flush=True)
    if args.dry_run:return
    store=EvaluationStore(backend);identity=store.save_experiment(metadata)
    if args.resume and args.resume!=identity:
        store.load(args.resume);parser.error('Resume configuration does not match this experiment fingerprint')
    saved={row['row_id'] for row in store.predictions(identity)};print('Experiment '+identity,flush=True)
    with tqdm(total=sum(len(fold['query_keys']) for fold in folds),initial=len(saved),desc='Chronological predictions',unit='invoice') as progress:
        for fold_index,fold in enumerate(folds):
            context=frame[frame.invoice_key.isin(fold['context_keys'])].copy()
            query=frame.set_index('invoice_key').loc[fold['query_keys']].reset_index()
            query=query[~query.invoice_key.isin(saved)]
            for start in range(0,len(query),args.query_batch_size):
                batch=query.iloc[start:start+args.query_batch_size].copy()
                features=batch.drop(columns=['credit_duration_days','credit_release','outcome_known_at','Reconciliation File Date (UTC)'],errors='ignore')
                output,report=estimate_candidate_lifetime_with_rpt1(features,context,config=config)
                measurements=[]
                for actual,prediction in zip(batch.to_dict('records'),output.to_dict('records')):
                    days=prediction.get('expected_lifetime_days',28);source_name=prediction.get('expected_lifetime_source')
                    valid=not pd.isna(days) and math.isfinite(float(days)) and float(days)>0
                    model=source_name=='RPT-1' and valid
                    measurements.append(dict(row_id=actual['invoice_key'],fold=fold_index+1,prediction_at=actual['credit_start'],
                        context_as_of=fold['as_of'],context_rows=len(context),customer=str(actual.get('Customer','')),program=str(actual.get('PROGRAMA','')),
                        actual_days=actual['credit_duration_days'],predicted_days=float(days) if model else 28,
                        source='rpt1' if model else 'fallback_default_weeks',median_days=median_baseline(context,actual),report=report))
                store.save_predictions(identity,measurements);progress.update(len(measurements))
    rows=store.predictions(identity);metrics=aggregate(rows);store.save_metrics(identity,metrics);write_report(args.report,identity,metadata,metrics)
    print(json.dumps({'experiment_id':identity,'model_coverage':metrics['model_coverage'],'model_only':metrics['model_only'],'report':args.report},indent=2))


if __name__=='__main__':main()
