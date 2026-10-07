/** Verify that the executive results view is derived only from the saved run snapshot. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildResultsSummary,
  formatMoney,
  paginateRows,
} from '../src/pages/workspace/components/run-results-model.js';

/** Build a compact completed run with overlapping facility, customer and group exposure. */
function completedRun() {
  return {
    run_id:'run-12345678', status:'completed', row_ids:['a','b','c','d'],
    settings:{planning_start:'2025-02-03',horizon_weeks:2},
    preparation:{
      predictions:[{source:'rpt1'},{source:'rpt1'},{source:'rpt1'},{source:'fallback_default_weeks'}],
      fallback_count:1,
      acknowledgement:{accepted_at:'2026-09-08T10:00:00Z'},
      history:{dataset_id:'reference-v1'},
    },
    result:{
      solver_status:'FEASIBLE', objective_amount:3000, currency:'EUR',
      week_starts:['2025-02-03','2025-02-10'],
      selected:[{row_id:'a'},{row_id:'b'},{row_id:'c'}],
      weekly_plan:[
        {row_id:'a','Invoice Reference':'INV-A','Customer Name':'Alpha','Original Amount':1000,'Original Currency':'EUR','Normalized Amount':1000,planned_week_start_iso:'2025-02-03'},
        {row_id:'b','Invoice Reference':'INV-B','Customer Name':'Beta','Original Amount':1000,'Original Currency':'EUR','Normalized Amount':1000,planned_week_start_iso:'2025-02-10'},
      ],
      not_selected:[{row_id:'d','Invoice Reference':'INV-D',excluded_reason:'FACILITY_CAP_BINDING'}],
      pre_excluded:[{row_id:'x','Invoice Reference':'INV-X',exclusion_reason:'Nonpositive purchase price'}],
      exposure:{
        facility:{
          '2025-02-03':{F1:{used_base:5000,used_new:1000,used_total:6000,limit:10000,utilization_pct:60}},
          '2025-02-10':{F1:{used_base:5000,used_new:4000,used_total:9000,limit:10000,utilization_pct:90}},
        },
        customer:{
          '2025-02-03':{C1:{used_base:1000,used_new:850,used_total:1850,limit:2000,utilization_pct:92.5}},
          '2025-02-10':{C1:{used_base:1000,used_new:500,used_total:1500,limit:2000,utilization_pct:75}},
        },
        group:{
          '2025-02-03':{G1:{used_base:500,used_new:450,used_total:950,limit:1000,utilization_pct:95}},
          '2025-02-10':{G1:{used_base:500,used_new:300,used_total:800,limit:1000,utilization_pct:80}},
        },
      },
    },
  };
}

test('results summary calculates executive metrics and facility-only weekly totals', () => {
  const summary=buildResultsSummary(completedRun());

  assert.deepEqual(summary.metrics,{
    selectedCount:3,
    recommendedAmount:3000,
    selectionRatePct:75,
    peakUtilizationPct:95,
  });
  assert.deepEqual(summary.weekly,[
    {week:'2025-02-03',opening:5000,recommended:1000,total:6000},
    {week:'2025-02-10',opening:5000,recommended:4000,total:9000},
  ]);
  assert.equal(summary.constraints[0].entity,'G1');
  assert.equal(summary.constraints[1].entity,'C1');
  assert.equal(summary.constraints[2].entity,'F1');
  assert.match(summary.guidance,/feasible/i);
  assert.equal(summary.assumptions.acknowledgementStatus,'accepted');
});

test('results summary handles empty saved results without NaN or invented utilization', () => {
  const run=completedRun();
  run.row_ids=[];
  run.result={...run.result,selected:[],weekly_plan:[],not_selected:[],pre_excluded:[],objective_amount:0,week_starts:[],exposure:{}};

  const summary=buildResultsSummary(run);

  assert.equal(summary.metrics.selectionRatePct,0);
  assert.equal(summary.metrics.peakUtilizationPct,0);
  assert.deepEqual(summary.weekly,[]);
  assert.deepEqual(summary.constraints,[]);
  assert.equal(formatMoney(summary.metrics.recommendedAmount,'EUR'),'€0.00');
});

test('drill-down pagination uses 50 rows and clamps invalid pages', () => {
  const rows=Array.from({length:121},(_,index)=>({row_id:String(index+1)}));

  assert.equal(paginateRows(rows,1).items.length,50);
  assert.equal(paginateRows(rows,2).items[0].row_id,'51');
  assert.equal(paginateRows(rows,3).items.length,21);
  assert.equal(paginateRows(rows,99).page,3);
  assert.equal(paginateRows([],1).totalPages,1);
});

test('currency formatting uses the saved run currency', () => {
  assert.equal(formatMoney(1234567.8,'EUR'),'€1,234,567.80');
  assert.match(formatMoney(25,'USD'),/\$25\.00/);
});
