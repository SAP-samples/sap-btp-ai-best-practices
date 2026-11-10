/** Pure calculations for the saved recommendation dashboard and its drill-downs. */

export const RESULT_PAGE_SIZE=50;

/**
 * Convert a possibly absent numeric value without allowing NaN into the UI.
 * @param {unknown} value Candidate numeric value.
 * @returns {number} A finite number, or zero when unavailable.
 */
function finiteNumber(value) {
  const number=Number(value);
  return Number.isFinite(number) ? number : 0;
}

/**
 * Format a saved amount in its result currency.
 * @param {unknown} amount Numeric amount from the saved result.
 * @param {string} currency ISO currency code from the saved result.
 * @returns {string} Localized amount with exactly two decimals.
 */
export function formatMoney(amount,currency='EUR') {
  return new Intl.NumberFormat('en-US',{
    style:'currency',currency:currency||'EUR',minimumFractionDigits:2,maximumFractionDigits:2,
  }).format(finiteNumber(amount));
}

/**
 * Flatten all saved exposure levels into rows suitable for ranking and pagination.
 * @param {Record<string, Record<string, Record<string, object>>>} exposure Saved exposure cube.
 * @returns {Array<object>} Weekly entity rows across facility, customer and group levels.
 */
function capacityRows(exposure={}) {
  const rows=[];
  for(const [level,byWeek] of Object.entries(exposure||{})) {
    for(const [week,entities] of Object.entries(byWeek||{})) {
      for(const [entity,values] of Object.entries(entities||{})) {
        rows.push({
          week,level,entity,
          opening:finiteNumber(values.used_base),
          recommended:finiteNumber(values.used_new),
          total:finiteNumber(values.used_total),
          limit:finiteNumber(values.limit),
          utilizationPct:finiteNumber(values.utilization_pct),
        });
      }
    }
  }
  return rows;
}

/**
 * Retain the most constrained week for each entity and rank closest limits first.
 * @param {Array<object>} rows Flattened capacity rows.
 * @returns {Array<object>} One peak row per entity and level, descending by utilization.
 */
function rankedConstraints(rows) {
  const peaks=new Map();
  for(const row of rows) {
    const key=`${row.level}\u0000${row.entity}`;
    if(!peaks.has(key)||peaks.get(key).utilizationPct<row.utilizationPct)peaks.set(key,row);
  }
  return [...peaks.values()].sort((left,right)=>right.utilizationPct-left.utilizationPct);
}

/**
 * Aggregate the facility level once so customer and group views do not double-count exposure.
 * @param {object} result Saved optimizer result.
 * @returns {Array<{week:string,opening:number,recommended:number,total:number}>} Weekly chart rows.
 */
function weeklyRows(result) {
  const facility=result.exposure?.facility||{};
  const weeks=result.week_starts?.length ? result.week_starts : Object.keys(facility).sort();
  return weeks.map(week=>{
    const values=Object.values(facility[week]||{});
    return {
      week,
      opening:values.reduce((sum,item)=>sum+finiteNumber(item.used_base),0),
      recommended:values.reduce((sum,item)=>sum+finiteNumber(item.used_new),0),
      total:values.reduce((sum,item)=>sum+finiteNumber(item.used_total),0),
    };
  });
}

/**
 * Derive the explicit lifetime acknowledgement meaning from the saved preparation snapshot.
 * @param {object} run Saved workspace run.
 * @returns {string} One of not_required, pending, accepted, or inconsistent.
 */
function acknowledgementStatus(run) {
  const preparation=run.preparation||{};
  const predictions=preparation.predictions||[];
  const fallbackCount=finiteNumber(preparation.fallback_count)||predictions.filter(row=>row.source!=='rpt1').length;
  if(!fallbackCount)return 'not_required';
  if(preparation.acknowledgement)return 'accepted';
  return run.status==='awaiting_lifetime_acknowledgement' ? 'pending' : 'inconsistent';
}

/**
 * Build one executive and drill-down model from a completed immutable run.
 * @param {object} run Saved workspace run returned by the existing REST endpoint.
 * @returns {object} Dashboard metrics, chart rows, ranked constraints and detail populations.
 */
export function buildResultsSummary(run) {
  const result=run?.result||{};
  const candidates=run?.row_ids||[];
  const selected=result.selected||[];
  const capacity=capacityRows(result.exposure);
  const constraints=rankedConstraints(capacity);
  const predictions=run?.preparation?.predictions||[];
  const fallbackCount=finiteNumber(run?.preparation?.fallback_count)||predictions.filter(row=>row.source!=='rpt1').length;
  return {
    runId:run?.run_id||'',
    solverStatus:result.solver_status||'UNAVAILABLE',
    currency:result.currency||'EUR',
    guidance:result.solver_status==='OPTIMAL'
      ? 'Optimal recommendation: the primary objective is proven optimal for this saved scope and its assumptions.'
      : 'Feasible recommendation: the solver found a valid plan, but this is not proof of global optimality.',
    metrics:{
      selectedCount:selected.length,
      recommendedAmount:finiteNumber(result.objective_amount),
      selectionRatePct:candidates.length ? selected.length/candidates.length*100 : 0,
      peakUtilizationPct:constraints[0]?.utilizationPct||0,
    },
    weekly:weeklyRows(result),
    constraints,
    capacityRows:capacity,
    selectedRows:result.weekly_plan||selected,
    notSelectedRows:[
      ...(result.not_selected||[]).map(row=>({...row,outcome:'Not recommended'})),
      ...(result.pre_excluded||[]).map(row=>({...row,outcome:'Screened before optimization'})),
    ],
    assumptions:{
      planningStart:run?.settings?.planning_start||null,
      horizonWeeks:run?.settings?.horizon_weeks||result.week_starts?.length||0,
      rpt1Count:predictions.filter(row=>row.source==='rpt1').length,
      fallbackCount,
      acknowledgementStatus:acknowledgementStatus(run||{}),
      acknowledgedAt:run?.preparation?.acknowledgement?.accepted_at||null,
      history:run?.preparation?.history||{},
    },
  };
}

/**
 * Return a clamped 50-row page for one results drill-down.
 * @param {Array<object>} rows Complete detail population.
 * @param {number} requestedPage One-based requested page.
 * @param {number} pageSize Rows per page; defaults to the required 50.
 * @returns {{items:Array<object>,page:number,totalPages:number,total:number,pageSize:number}} Page metadata and rows.
 */
export function paginateRows(rows=[],requestedPage=1,pageSize=RESULT_PAGE_SIZE) {
  const total=rows.length;
  const totalPages=Math.max(1,Math.ceil(total/pageSize));
  const page=Math.min(totalPages,Math.max(1,Number(requestedPage)||1));
  const start=(page-1)*pageSize;
  return {items:rows.slice(start,start+pageSize),page,totalPages,total,pageSize};
}
