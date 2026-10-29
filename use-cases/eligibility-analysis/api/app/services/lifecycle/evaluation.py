"""Forward-only lifecycle windows and transparent source-aware prediction metrics."""
import math
import numpy as np
import pandas as pd
from .context import admissible_history


def chronological_folds(rows, folds=3, test_rows=100, min_context=500, observable_days=None):
    """Select the last disjoint windows with context frozen before each window starts.

    A conservative common cutoff permits batched queries without exposing outcomes
    learned during the window. Every query retains its own prediction-time features.

    observable_days guards against right-censoring: a history extract only contains
    invoices that were already released, so the most recent fundings are biased toward
    short lifetimes. With observable_days set, queries are drawn only from fundings at
    least that many days before the last observed outcome (use a high lifetime
    percentile, e.g. 110 days). None keeps every funding eligible.
    """
    if min(folds,test_rows,min_context)<1: raise ValueError('Fold sizes must be positive')
    ordered=rows.copy();ordered['_instant']=pd.to_datetime(ordered.credit_start,utc=True,errors='coerce')
    ordered=ordered[ordered['_instant'].notna()].sort_values(['_instant','invoice_key'])
    if observable_days is not None:
        last_outcome=pd.to_datetime(rows.outcome_known_at,utc=True,errors='coerce').max()
        ordered=ordered[ordered['_instant']<=last_outcome-pd.Timedelta(days=float(observable_days))]
    selected=[];end=len(ordered)
    while end>0 and len(selected)<folds:
        query=ordered.iloc[max(0,end-test_rows):end];end-=len(query)
        at=query['_instant'].min();keys=query.invoice_key.tolist()
        context=admissible_history(rows,at,set(keys))
        if len(context)>=min_context:
            selected.append(dict(as_of=at.isoformat(),query_keys=keys,context_keys=context.invoice_key.tolist()))
    return list(reversed(selected))


def error_metrics(actual,predicted):
    """Summarize finite duration errors with direction, percentiles and week rounding."""
    if not len(actual): return {'count':0,'mae_days':None,'rmse_days':None}
    actual=np.asarray(actual,float);predicted=np.asarray(predicted,float);error=predicted-actual
    return dict(count=len(actual),mae_days=float(np.mean(abs(error))),rmse_days=float(np.sqrt(np.mean(error**2))),
        p90_absolute_error_days=float(np.quantile(abs(error),.9)),underestimate_rate=float(np.mean(error<0)),
        mean_underestimate_days=float(np.mean(np.maximum(-error,0))),
        week_rounding_mae=float(np.mean(abs(np.ceil(predicted/7)-np.ceil(actual/7)))))


def lifetime_metrics(actual,predicted,sources):
    """Separate model-only performance from operational predictions that include fallbacks."""
    if not len(actual)==len(predicted)==len(sources): raise ValueError('Metric inputs must align')
    valid=[i for i,(a,p) in enumerate(zip(actual,predicted)) if a is not None and p is not None and math.isfinite(float(a)) and math.isfinite(float(p)) and float(a)>0 and float(p)>0]
    model=[i for i in valid if sources[i]=='rpt1']
    return dict(sample_count=len(actual),invalid_count=len(actual)-len(valid),
        model_coverage=len(model)/len(actual) if actual else 0,
        fallback_count=sum(source!='rpt1' for source in sources),
        model_only=error_metrics([actual[i] for i in model],[predicted[i] for i in model]),
        operational=error_metrics([actual[i] for i in valid],[predicted[i] for i in valid]))


def median_baseline(context,query,min_group=5):
    """Use customer, then program, then global medians from admissible context only."""
    values=pd.to_numeric(context.credit_duration_days);global_median=float(values.median())
    for key in ('Customer','PROGRAMA'):
        if key not in context or not query.get(key):continue
        matches=values[context[key].astype(str)==str(query[key])]
        if len(matches)>=min_group:return float(matches.median())
    return global_median
