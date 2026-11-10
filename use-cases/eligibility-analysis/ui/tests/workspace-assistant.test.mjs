/** Check the reference-only boundary between a visible workspace and chat requests. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {workspaceReferences} from '../src/services/assistant-context.js';

test('assistant metadata omits workbook content, credentials and unknown filter fields',()=>{
  const result=workspaceReferences({analysis_id:'offer',run_id:'run',revision:0,row_ids:['source'],
    workbook:'private bytes',api_key:'credential',filters:{seller_id:'seller',api_key:'credential'}});
  assert.deepEqual(result,{analysis_id:'offer',run_id:'run',revision:0,row_ids:['source'],filters:{seller_id:'seller'}});
});

test('captured request scope does not change when the visible selection changes',()=>{
  const source={analysis_id:'offer',row_ids:['first'],filters:{status:'eligible'}};
  const captured=workspaceReferences(source);source.row_ids.push('second');source.filters.status='not_eligible';
  assert.deepEqual(captured.row_ids,['first']);assert.equal(captured.filters.status,'eligible');
});

test('leaving the offer drops orphaned run references',()=>{
  assert.deepEqual(workspaceReferences({run_id:'old-run',revision:3,row_ids:['old']}),{});
  assert.deepEqual(workspaceReferences({analysis_id:'new-offer'}),{analysis_id:'new-offer',run_id:null,revision:null,row_ids:[],filters:{}});
});
