/** Browser regression probe. Run with:
 * npx @playwright/cli -s=snapshot-lifecycle run-code "$(cat ui/tests/lifecycle.browser.js)"
 * Requires local Vite at http://127.0.0.1:5187. All API requests use disposable fixtures.
 */
async (page) => {
  const snapshots = [{dataset_id:'reference', name:'Reference snapshot', status:'published', metadata:{counts:{fini_master:1}}}];
  const runs = [{run_id:'finished', dataset_id:'imported', status:'completed', stage:'completed'}, {run_id:'active', dataset_id:'imported', status:'running', stage:'solving'}];
  const deleted = [];
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.unroute('**/api/**');
  await page.route('**/api/**', async route => {
    const req=route.request(), path=req.url().replace(/^https?:\/\/[^/]+/, '').split('?')[0], method=req.method();
    let data={};
    if (path==='/api/datasets' && method==='POST') {
      data={dataset_id:'imported',name:'New import',status:'review',metadata:{counts:{fini_master:1}}}; snapshots.push(data);
    } else if(path==='/api/datasets') data={items:snapshots};
    else if(path.endsWith('/publish')) {data=snapshots.find(d=>path.includes(d.dataset_id));data.status='published';}
    else if(path.startsWith('/api/datasets/')) {
      const id=path.split('/')[3];
      if(method==='DELETE') {deleted.push(id);snapshots.splice(snapshots.findIndex(d=>d.dataset_id===id),1);data={deleted:true};}
      else data=snapshots.find(d=>d.dataset_id===id);
    } else if(path==='/api/query') data={rows:[{material:'001'}],fields:['material'],total:1};
    else if(path==='/api/run-drafts') data={draft_id:'draft',dataset_id:JSON.parse(req.postData()).dataset_id,revision:1,request:{},budget:{}};
    else if(path==='/api/runs') data={items:runs};
    else if(path.endsWith('/results')) data={points:[]};
    else if(path.startsWith('/api/runs/')) {
      const id=path.split('/')[3];
      if(method==='DELETE') {deleted.push(id);runs.splice(runs.findIndex(r=>r.run_id===id),1);data={deleted:true};}
      else data=runs.find(r=>r.run_id===id);
    }
    await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(data)});
  });
  await page.goto('http://127.0.0.1:5187');
  await page.getByRole('button',{name:/Reference snapshot published/}).click();
  await page.waitForFunction(()=>document.querySelector('#review-title').textContent==='Reference snapshot');
  await page.getByRole('button',{name:/Reference snapshot published/}).click();
  await page.waitForFunction(()=>document.querySelector('#review-title').textContent==='Select a snapshot');
  await page.getByRole('button',{name:/Reference snapshot published/}).click();
  await page.locator('[name=name]').fill('New import');
  await page.locator('[name=primary]').evaluate(el=>{const transfer=new DataTransfer();transfer.items.add(new File(['fixture'],'wheel.xlsx'));el.files=transfer.files;});
  await page.locator('#upload').click();
  await page.waitForFunction(()=>document.querySelector('#review-title').textContent==='New import' && !document.querySelector('#publish').disabled);
  if(!await page.locator('#use-dataset').evaluate(el=>el.disabled)) throw Error('Unpublished import can enter workspace');
  await page.locator('#publish').click();
  await page.waitForFunction(()=>document.querySelector('#publish').textContent==='Published');
  await page.locator('#use-dataset').click();
  await page.waitForFunction(()=>document.querySelector('#workspace-dataset')?.value==='imported');
  await page.getByRole('button',{name:'completed · completed finished'}).click();
  await page.waitForFunction(()=>!document.querySelector('#deselect-run').disabled);
  await page.getByRole('button',{name:'completed · completed finished'}).click();
  await page.waitForFunction(()=>document.querySelector('#deselect-run').disabled);
  await page.getByRole('button',{name:'running · solving active'}).click();
  await page.waitForFunction(()=>document.querySelector('#run-status').textContent.includes('solving'));
  await page.locator('#deselect-run').click();
  await page.waitForTimeout(4500);
  if((await page.locator('#run-status').textContent()).includes('solving')) throw Error('Polling revived deselected run');
  const finishedRow=page.locator('.history-entry').filter({hasText:'finished'});
  page.once('dialog',dialog=>dialog.dismiss());
  await finishedRow.getByRole('button',{name:'Delete permanently'}).click();
  if(deleted.length) throw Error('Cancel confirmation still deleted data');
  page.once('dialog',dialog=>dialog.accept());
  await finishedRow.getByRole('button',{name:'Delete permanently'}).click();
  await page.waitForFunction(()=>!document.querySelector('#run-list').textContent.includes('finished'));
  if(!deleted.includes('finished')) throw Error('Run was not deleted');
  await page.locator('#workspace-dataset').selectOption('');
  await page.waitForFunction(()=>document.querySelector('#draft-revision').textContent==='No draft selected');
  await page.locator('[data-nav=datasets]').click();
  await page.getByRole('button',{name:/New import published/}).click();
  page.once('dialog',dialog=>{if(!dialog.message().includes('ALL associated runs')) throw Error('Missing cascade warning');return dialog.accept();});
  await page.locator('.history-entry').filter({hasText:'New import'}).getByRole('button',{name:'Delete permanently'}).click();
  await page.waitForFunction(()=>document.querySelector('#review-title').textContent==='Select a snapshot');
  if(!deleted.includes('imported')) throw Error('Snapshot was not deleted');
  if(errors.length) throw Error(errors.join('\n'));
  await page.evaluate(()=>{document.body.dataset.lifecycleCheck='passed';});
  console.log('PASS: import selection, explicit publish, workspace availability, snapshot/run deselection, polling stop, deletion cancel/confirm, cascade warning, dataset dropdown clear; no browser errors.');
}
