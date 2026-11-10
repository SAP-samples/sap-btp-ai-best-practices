/** Serialize source references without copying uploaded data or credentials into chat metadata. */
export function workspaceReferences(context={}) {
  if(!context?.analysis_id)return {};
  const keys=['status','search','seller_id','debtor_id','programa','insurer_id','original_currency'];
  return {
    analysis_id:context.analysis_id,
    run_id:context.run_id??null,
    revision:context.run_id?context.revision:null,
    row_ids:[...(context.row_ids||[])],
    filters:Object.fromEntries(Object.entries(context.filters||{}).filter(([key,value])=>keys.includes(key)&&typeof value==='string')),
  };
}
