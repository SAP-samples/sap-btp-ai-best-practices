"""Bridge saved temporal HANA context to the existing RPT-1 lifetime estimator."""
import pandas as pd
from .context import invoice_key
from ...optimizer.model.lifetime_estimation import LifetimeEstimationConfig, estimate_candidate_lifetime_with_rpt1


def estimate_from_history(store,candidates,prediction_at,config=None,dataset_id=None,context_policy="chronological"):
    """Estimate current rows using an explicit fixed-reference or chronological policy.

    Returns the estimator frame and provenance report. Missing history is returned
    explicitly for the preparation service's acknowledgement gate; no history rows
    are ever appended to candidates. Existing estimator grouping/ranking is retained.
    """
    excluded={key for row in candidates.to_dict('records') if (key:=invoice_key(row))}
    try:
        history,provenance=store.context(prediction_at,excluded,dataset_id,context_policy=context_policy)
    except LookupError as error:
        return candidates.copy(),{'status':'missing_history','errors':[str(error)],'dataset_id':None}
    # Date columns persisted as ISO strings are converted by the existing estimator.
    # Every query here shares this as-of instant; evaluation groups distinct instants.
    output,report=estimate_candidate_lifetime_with_rpt1(candidates,history,
        config=config or LifetimeEstimationConfig())
    return output,{**report,**provenance}
