import test from 'node:test';
import assert from 'node:assert/strict';
import {watchRun} from '../src/pages/workspace/components/run-polling.js';

test('polling stops on completion without acknowledging pending assumptions',async()=>{
  const states=['awaiting_lifetime_acknowledgement','optimizing','completed'];let calls=0;const seen=[];
  await watchRun('r',{api:{getRun:async()=>({status:states[calls++]})},onState:run=>seen.push(run.status),intervalMs:1});
  assert.equal(calls,3);assert.deepEqual(seen,states);
});
test('navigation aborts polling',async()=>{
  const controller=new AbortController();let calls=0;
  await watchRun('r',{api:{getRun:async()=>{calls++;return {status:'estimating_lifetimes'};}},
    onState:()=>controller.abort(),signal:controller.signal,intervalMs:1});
  assert.equal(calls,1);
});
