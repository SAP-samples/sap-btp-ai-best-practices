/** Serialize an offer upload without changing the user's date or retry identity. */
export function analysisForm(file, {analysisDate, settings, requestKey}) {
  const form = new FormData();
  form.append('file', file);
  form.append('analysis_date', analysisDate);
  form.append('settings', JSON.stringify(settings));
  form.append('request_key', requestKey);
  return form;
}
