/** Pure grouping and status helpers for saved recommendation artifacts. */

const DESCRIPTIONS={
  eligibility:'Eligibility outcome for the saved source population.',
  selected:'Invoices included in the recommendation.',
  excluded:'Invoices not recommended or screened before optimization.',
  'weekly-plan':'Recommended invoices organized by planned week.',
  exposure:'Weekly opening, recommended, total, and limit exposure.',
  'report-pdf':'Readable recommendation summary for review.',
  'report-markdown':'Markdown source used to create the report.',
  snapshot:'Run settings, provenance, preparation, and saved result fields.',
  'all-files':'All currently available files in one package.',
};

const DISPLAY_LABELS={
  selected:'Recommended invoices',
  excluded:'Not recommended and pre-excluded invoices',
  'weekly-plan':'Weekly recommendation plan',
  exposure:'Weekly exposure',
  'report-pdf':'Recommendation report',
  'report-markdown':'Report source',
  snapshot:'Run snapshot and assumptions',
};

const GROUPS=[
  ['Recommendation data',['eligibility','selected','excluded','weekly-plan','exposure']],
  ['Report',['report-pdf','report-markdown']],
  ['Audit and assumptions',['snapshot']],
];

/**
 * Convert an artifact filename into a compact file-type label.
 * @param {string} filename Saved artifact filename.
 * @returns {string} Uppercase extension or FILE when unavailable.
 */
function fileType(filename='') {
  const extension=String(filename).split('.').pop();
  return extension&&extension!==filename?extension.toUpperCase():'FILE';
}

/**
 * Project one manifest entry into the download dialog model.
 * @param {object} item Existing artifact manifest entry.
 * @returns {object} Display and action flags for the artifact.
 */
function artifactRow(item) {
  return {
    artifactId:item.artifact_id,
    label:DISPLAY_LABELS[item.artifact_id]||item.label||item.filename||item.artifact_id,
    filename:item.filename||'',
    fileType:fileType(item.filename),
    description:DESCRIPTIONS[item.artifact_id]||'Saved recommendation artifact.',
    status:item.status||'pending',
    error:item.error||null,
    canDownload:item.status==='ready',
    canPreview:item.artifact_id==='report-pdf'&&item.status==='ready',
  };
}

/**
 * Group the unchanged artifact manifest with ZIP first and retry derived from failures.
 * @param {Array<object>} items Manifest entries from the existing artifacts endpoint.
 * @returns {{primary:object|null,groups:Array<object>,retryVisible:boolean}} Dialog view model.
 */
export function buildDownloadGroups(items=[]) {
  const rows=items.map(artifactRow);
  return {
    primary:rows.find(row=>row.artifactId==='all-files')||null,
    groups:GROUPS.map(([label,ids])=>({label,items:ids.map(id=>rows.find(row=>row.artifactId===id)).filter(Boolean)})).filter(group=>group.items.length),
    retryVisible:rows.some(row=>row.status==='failed'),
  };
}
