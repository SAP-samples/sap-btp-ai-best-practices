/** Observe authoritative run state; polling never sends approvals or solve mutations. */
export async function watchRun(runId,{api,onState,signal,intervalMs=1500}) {
  while(!signal?.aborted) {
    try {
      const run=await api.getRun(runId,signal);if(signal?.aborted)return;
      await onState(run);if(['completed','failed','cancelled'].includes(run.status)||signal?.aborted)return;
      await new Promise(resolve=>{
        const done=()=>{clearTimeout(timer);signal?.removeEventListener('abort',done);resolve();};
        const timer=setTimeout(done,intervalMs);signal?.addEventListener('abort',done,{once:true});
      });
    } catch(error) {if(error.name==='AbortError')return;throw error;}
  }
}
