/** Safe editable settings tables shared by the credit and repayment panels. */

/** Create a UI5 input with an accessible name, initial value and direct draft binding. */
export function inputField(value,label,onChange,type='Text') {
  const input=document.createElement('ui5-input'); input.value=String(value??''); input.type=type;
  input.accessibleName=label; input.addEventListener('input',()=>onChange(input.value)); return input;
}

/** Render scalar object mappings as editable rows without interpreting uploaded IDs as HTML. */
export function mappingTable(host,heading,mapping,extra={}) {
  host.replaceChildren(); const title=document.createElement('h3');title.textContent=heading;host.append(title);
  const table=document.createElement('table');table.className='ws-edit-table';
  const head=table.createTHead().insertRow();
  for(const label of ['ID','Limit (EUR)',...(extra.opening?['Opening exposure (EUR)']:[])]) {const th=document.createElement('th');th.textContent=label;head.append(th);}
  for(const [id,value] of Object.entries(mapping)) {
    const row=table.insertRow(); row.insertCell().textContent=id;
    row.insertCell().append(inputField(value,`${heading} ${id} limit`,v=>mapping[id]=v,'Number'));
    if(extra.opening) row.insertCell().append(inputField(extra.opening[id],`${heading} ${id} opening exposure`,v=>extra.opening[id]=v,'Number'));
  }
  host.append(table);
}

/** Render known associations, preserving blank values as validation issues until mapped. */
export function associationTable(host,title,mapping) {
  const heading=document.createElement('h3');heading.textContent=title;host.append(heading);
  for(const id of Object.keys(mapping)) {
    const label=document.createElement('label');label.textContent=id;
    label.append(inputField(mapping[id],`${title} ${id}`,value=>mapping[id]=value));host.append(label);
  }
}
