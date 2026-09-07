const LANGUAGE_STORAGE_KEY = "document-assessment-assistant-language";
const DEFAULT_LANGUAGE = "en";
const SUPPORTED_LANGUAGES = new Set(["en", "it"]);

const TRANSLATIONS = {
  en: {
    "app.title": "Evaluation Assessment Assistant",
    "nav.home": "Document Manager",
    "nav.adminDocuments": "Super Admin Document Manager",
    "nav.assessment": "Assessment",
    "nav.score": "Score",
    "shell.search": "Search Apps, Products",
    "shell.help": "Help",
    "shell.language": "Language",
    "assessment.title": "Assessment",
    "assessment.versionWarning": "Version to submit",
    "assessment.score": "Score",
    "assessment.reset": "Reset changes",
    "assessment.batchUpload": "Analyze all questions",
    "assessment.saveDraft": "Analyze dimension",
    "assessment.dimensionLabel": "Dimension",
    "assessment.customerClassLabel": "Company size",
    "assessment.nace1Label": "NACE-1 sector",
    "assessment.noNaceOptions": "No NACE-1 cohort available",
    "assessment.profileSaving": "Saving benchmark context to HANA...",
    "assessment.profileSaved": "Benchmark class and NACE-1 context saved.",
    "assessment.profileSaveFailed": "Benchmark context could not be saved: {message}",
    "assessment.profileOptionsFailed": "Benchmark cohort options could not be loaded: {message}",
    "assessment.profileUnavailable": "Benchmark comparison is unavailable until an active dataset supplies a class and NACE-1 cohort.",
    "assessment.customerClassQuestionUnavailable": "This question is not available for the selected company size.",
    "assessment.customerClassDimensionUnavailable": "No questions in this dimension are available for the selected company size.",
    "assessment.answered": "{answered} of {total} answered",
    "assessment.handled": "Handled: YES",
    "assessment.currentAnswers": "Current questionnaire answers",
    "assessment.aiVerified": "AI verified answers",
    "assessment.applyAiMarks": "Apply AI Recommendations",
    "assessment.applyAiMarksDone": "AI marks applied to the user form.",
    "assessment.applyAiMarksUnavailable": "Analyze this question before applying AI marks.",
    "assessment.attachFile": "Analyze question",
    "assessment.level": "LEVEL {level}",
    "assessment.noFramework": "Assessment questions are not available. Confirm HANA is configured and the framework import has been run.",
    "assessment.noAi": "Analyze this question to generate AI verified answers from Document Manager evidence.",
    "assessment.aiFailed": "AI review failed",
    "assessment.aiFailedFallback": "The worker could not complete this question review.",
    "assessment.legacyResultTitle": "Run this AI review again",
    "assessment.legacyResultMessage": "This result was created with an older answer-decision contract. It remains visible for reference, but its AI marks cannot be applied until a new review completes.",
    "assessment.noDecisions": "No answer-item decisions returned for this level.",
    "assessment.noLevelResults": "The worker completed this question but the model returned no level results.",
    "assessment.highestSupportedLevel": "Highest supported level {level}",
    "assessment.noSupportedLevel": "No supported level",
    "assessment.reasoning": "Reasoning",
    "assessment.reasoningEvidence": "Reasoning and evidence",
    "assessment.evidence": "Evidence: {refs}",
    "assessment.noEvidence": "Evidence: none linked",
    "assessment.loadingError": "Assessment framework could not be loaded from HANA. {message}",
    "assessment.responsesSaveFailed": "Answer selection was updated locally, but could not be saved to HANA: {message}",
    "assessment.attachRequired": "Select at least one question in the current dimension.",
    "assessment.batchAttachRequired": "Assessment questions are not available for all-question analysis.",
    "assessment.submitting": "Starting AI analysis for this dimension...",
    "assessment.submittingQuestion": "Starting AI analysis for this question...",
    "assessment.submittingBatch": "Starting AI analysis for all questions...",
    "assessment.createdJob": "AI review job {jobId} created with {taskCount} task(s).",
    "assessment.createdBatchJob": "AI review job {jobId} created with {taskCount} question task(s).",
    "assessment.pollFailed": "AI review polling failed: {message}",
    "assessment.staleJobCleared": "The previous AI review job is no longer available. Start a new analysis to refresh these answers.",
    "assessment.jobStatus": "AI review {status}: {completed}/{total} question task(s) completed{failedText}.",
    "assessment.failedText": ", {failed} failed",
    "assessment.resetting": "Resetting all draft changes and AI review jobs...",
    "assessment.resetDone": "All draft changes and AI review jobs were reset. Removed {jobs} review job(s) and {tasks} question task(s).",
    "assessment.startFailed": "AI review failed to start: {message}",
    "assessment.resetFailed": "Reset failed: {message}",
    "file.remove": "Remove {fileName}",
    "label.completed": "Completed",
    "label.failed": "Failed",
    "label.in_progress": "In progress",
    "label.insufficient_evidence": "Insufficient evidence",
    "label.keep_selected": "Keep selected",
    "label.low_confidence": "Low Confidence",
    "label.partial_failed": "Partially failed",
    "label.partially_supported": "Partially supported",
    "label.pending": "Pending",
    "label.select": "Select",
    "label.unclear": "Unclear",
    "label.unsupported": "Not evidenced",
    "assessment.activeProgress": "{message} {completed}/{total} completed{failedText}.",
    "assessment.questionInProgress": "AI review in progress",
    "assessment.questionProgressFallback": "This question is being reviewed.",
    "admin.benchmark.title": "Benchmark data",
    "admin.benchmark.description": "Validate and activate versioned assessment benchmark workbooks stored in HANA. This uses the existing shared API-key protection.",
    "admin.benchmark.refresh": "Refresh status",
    "admin.benchmark.select": "Select XLSX",
    "admin.benchmark.validate": "Validate workbook",
    "admin.benchmark.activate": "Activate dataset",
    "admin.benchmark.none": "None",
    "admin.benchmark.loadingHistory": "Loading active benchmark status...",
    "admin.benchmark.noActiveTitle": "Benchmark unavailable",
    "admin.benchmark.noActiveDetail": "No active benchmark dataset is stored in HANA. Select and validate an XLSX workbook before activation.",
    "admin.benchmark.activeTitle": "Active benchmark dataset",
    "admin.benchmark.file": "Source file",
    "admin.benchmark.status": "Status",
    "admin.benchmark.activated": "Activated",
    "admin.benchmark.importId": "Import ID",
    "admin.benchmark.rows": "Rows",
    "admin.benchmark.companies": "Companies",
    "admin.benchmark.questionnaires": "Questionnaires",
    "admin.benchmark.questions": "Questions",
    "admin.benchmark.accepted": "Accepted",
    "admin.benchmark.rejected": "Rejected",
    "admin.benchmark.classes": "Classes",
    "admin.benchmark.naceCohorts": "NACE-1 cohorts",
    "admin.benchmark.warnings": "Aggregated warnings ({count})",
    "admin.benchmark.rowsList": "rows {rows}",
    "admin.benchmark.noWarnings": "No warnings.",
    "admin.benchmark.sampledErrors": "Sampled row errors ({count})",
    "admin.benchmark.row": "Row {row}",
    "admin.benchmark.noErrors": "No sampled row errors.",
    "admin.benchmark.noSelection": "No workbook selected.",
    "admin.benchmark.historyFailed": "Active benchmark status could not be loaded. Check the API connection and try again.",
    "admin.benchmark.selectRequired": "Select one XLSX workbook before validation.",
    "admin.benchmark.validating": "Validating the workbook without writing to HANA...",
    "admin.benchmark.validationPassed": "Validation passed",
    "admin.benchmark.validationFailed": "Validation failed",
    "admin.benchmark.validationReady": "Validation passed. Review the results before activating this dataset.",
    "admin.benchmark.validationRequestFailed": "The workbook validation request could not be completed. Check the API connection and try again.",
    "admin.benchmark.validationRequired": "Validate the currently selected workbook successfully before activation.",
    "admin.benchmark.confirmActivation": "Activate this validated benchmark dataset? It will replace the current active version for all assessment comparisons.",
    "admin.benchmark.activating": "Creating and populating the HANA tables, then activating the validated dataset...",
    "admin.benchmark.alreadyActive": "This identical dataset is already active.",
    "admin.benchmark.activationDone": "The benchmark dataset was activated successfully.",
    "admin.benchmark.activationFailed": "Benchmark activation failed; the previous active version was retained. Review the validation result or server logs and try again.",
    "admin.benchmark.xlsxOnly": "Benchmark data must be provided as one .xlsx workbook.",
    "admin.benchmark.importStatus.active": "Active",
    "admin.benchmark.importStatus.inactive": "Inactive",
    "admin.benchmark.importStatus.validated": "Validated",
    "admin.benchmark.importStatus.rejected": "Rejected",
    "admin.benchmark.importStatus.failed": "Failed",
    "admin.benchmark.importStatus.unknown": "Unknown",
    "admin.benchmark.issue.warningFallback": "A non-blocking source-data issue was found; review the code and affected rows.",
    "admin.benchmark.issue.errorFallback": "The row or workbook does not satisfy the benchmark import contract.",
    "admin.benchmark.issue.answer_text_mismatch": "Source answer text differs from the canonical framework text; canonical mapping was retained.",
    "admin.benchmark.issue.dimension_mismatch": "The source dimension differs from the canonical framework dimension; the framework value was retained.",
    "admin.benchmark.issue.duplicate_row": "A duplicate source row was ignored.",
    "admin.benchmark.issue.profile_placeholder_normalized": "A source profile placeholder was normalized to an empty value.",
    "admin.benchmark.issue.score_reconciliation": "A supplied score differs from the score recalculated with the application formula.",
    "admin.benchmark.issue.conflicting_answer_catalog": "The same source answer position contains conflicting values.",
    "admin.benchmark.issue.conflicting_row": "A duplicate logical response contains conflicting data.",
    "admin.benchmark.issue.empty_workbook": "The required worksheet contains no benchmark response rows.",
    "admin.benchmark.issue.invalid_boolean": "A boolean field contains an unsupported value.",
    "admin.benchmark.issue.invalid_class": "The company class is not supported by the assessment framework.",
    "admin.benchmark.issue.invalid_date": "A date field contains an invalid value.",
    "admin.benchmark.issue.invalid_extension": "The selected file is not an XLSX workbook.",
    "admin.benchmark.issue.invalid_headers": "The worksheet headers do not match the exact import contract.",
    "admin.benchmark.issue.invalid_level": "An answer maturity level is outside the supported range.",
    "admin.benchmark.issue.invalid_profile_value": "A company profile field contains an invalid value.",
    "admin.benchmark.issue.invalid_score": "A supplied score is not a valid finite number.",
    "admin.benchmark.issue.malformed_xlsx": "The file could not be opened as a valid XLSX workbook.",
    "admin.benchmark.issue.missing_identifier": "A required company, questionnaire, question, or answer identifier is missing.",
    "admin.benchmark.issue.missing_sheet": "The workbook does not contain the required Estrazione worksheet.",
    "admin.benchmark.issue.mixed_company_metadata": "Rows for the same company contain inconsistent profile metadata.",
    "admin.benchmark.issue.mixed_dimension_score": "Rows for the same questionnaire dimension contain inconsistent supplied scores.",
    "admin.benchmark.issue.mixed_questionnaire_metadata": "Rows for the same questionnaire contain inconsistent submission metadata.",
    "admin.benchmark.issue.source_too_large": "The workbook exceeds the permitted upload size.",
    "admin.benchmark.issue.unknown_question": "The source question does not exist in the canonical assessment framework.",
    "admin.benchmark.issue.unmappable_answer": "The source answer cannot be mapped to a canonical answer position.",
    "admin.benchmark.issue.unsafe_zip": "The XLSX archive is unsafe or exceeds extraction limits.",
    "score.title": "Score",
    "score.subtitle": "Assessment positioning against the active peer cohort.",
    "score.context": "Company class: {customerClass} | NACE-1: {nace1}",
    "score.contextUnavailable": "No persisted benchmark class and NACE-1 profile is available.",
    "score.profileMissing": "Not configured",
    "score.refresh": "Refresh",
    "score.score": "Score",
    "score.finalScore": "Final score",
    "score.benchmark": "Benchmark",
    "score.delta": "Delta",
    "score.sameSector": "Same sector peers",
    "score.sameSize": "Same size peers",
    "score.sectorDelta": "Sector delta",
    "score.sizeDelta": "Size delta",
    "score.dimensionScores": "Dimension scores",
    "score.questionPositioning": "Question positioning",
    "score.question": "Question",
    "score.dimension": "Dimension",
    "score.selected": "Selected",
    "score.applicable": "Applicable",
    "score.loadingError": "Score could not be loaded: {message}",
    "score.overviewTitle": "Overall performance overview",
    "score.bestPeer": "Best peer",
    "score.peerAverage": "Peer average",
    "score.company": "Company",
    "score.legendLabel": "Benchmark chart legend",
    "score.chartDescription": "Each group uses three nested bars on the same hidden maturity scale. Bar thickness and the textual positioning below identify best peer, peer average, and company; exact values are withheld here.",
    "score.dimensionAnalysis": "Dimension analysis",
    "score.dimensionAnalysisDescription": "Expand a dimension to review its applicable questionnaire topics and deterministic peer commentary.",
    "score.noApplicableTopics": "No applicable topics are available for this company class.",
    "score.position.below_peers": "Below your peers",
    "score.position.in_line_with_peers": "In line with your peers",
    "score.position.above_peers": "Above your peers",
    "score.position.unavailable": "Benchmark unavailable",
    "score.benchmarkUnavailable": "Benchmark unavailable",
    "score.unavailableNoDataset": "No active benchmark dataset is available. Company results remain visible without synthetic peer data.",
    "score.unavailableNoProfile": "Save a company class and NACE-1 profile on the Assessment page to select an exact peer cohort.",
    "score.unavailableSmallCohort": "This cohort does not contain enough eligible peer data for comparison.",
    "score.unavailableGeneric": "Peer comparison is unavailable for the selected context.",
    "score.benchmarkAvailableDetail": "Comparison uses eligible peer data from the dataset activated on {datasetDate}.",
    "score.datasetDate": "Active dataset date: {datasetDate}.",
    "score.calificationReport": "Calification Report",
    "score.reportSubmitting": "Preparing the assessment snapshot...",
    "score.reportQueued": "Calification report queued.",
    "score.reportGenerating": "Preparing the deterministic calification report...",
    "score.reportRendering": "Rendering the calification report PDF...",
    "score.reportResuming": "Resuming calification report generation...",
    "score.reportReady": "Calification report ready. The PDF download has started.",
    "score.reportFailed": "The calification report could not be generated. Please try again.",
    "score.reportStartFailed": "Calification report generation could not start: {message}",
    "score.reportPollFailed": "The report status could not be refreshed; polling will retry: {message}",
    "score.reportDownloadFailed": "The report is ready, but the download failed: {message}",
    "score.reportExpired": "The previous report is no longer available. Generate a new report.",
    "score.reportUnknownStatus": "The report returned an unsupported status. Generate a new report.",
    "label.yes": "Yes",
    "label.no": "No",
    "label.pending_indexing": "Preparing documents",
    "label.extracting_documents": "Extracting documents",
    "label.embedding_documents": "Embedding documents",
    "label.reviewing_questions": "Reviewing questions",
    "label.retrieving_evidence": "Retrieving evidence",
    "label.finalizing_answer": "Finalizing answer",
    "label.retrying": "Retrying"
  },
  it: {
    "app.title": "Evaluation Assessment Assistant",
    "nav.home": "Documenti",
    "nav.adminDocuments": "Super Admin Document Manager",
    "nav.assessment": "Valutazione",
    "nav.score": "Punteggio",
    "shell.search": "Cerca app, prodotti",
    "shell.help": "Aiuto",
    "shell.language": "Lingua",
    "assessment.title": "Valutazione",
    "assessment.versionWarning": "Versione da inviare",
    "assessment.score": "Punteggio",
    "assessment.reset": "Reset modifiche",
    "assessment.batchUpload": "Analizza tutte le domande",
    "assessment.saveDraft": "Analizza dimensione",
    "assessment.dimensionLabel": "Dimensione",
    "assessment.customerClassLabel": "Dimensione aziendale",
    "assessment.nace1Label": "Settore NACE-1",
    "assessment.noNaceOptions": "Nessuna coorte NACE-1 disponibile",
    "assessment.profileSaving": "Salvataggio del contesto di benchmark in HANA...",
    "assessment.profileSaved": "Classe e contesto NACE-1 del benchmark salvati.",
    "assessment.profileSaveFailed": "Impossibile salvare il contesto di benchmark: {message}",
    "assessment.profileOptionsFailed": "Impossibile caricare le opzioni della coorte di benchmark: {message}",
    "assessment.profileUnavailable": "Il confronto di benchmark non è disponibile finché un dataset attivo non fornisce una coorte per classe e NACE-1.",
    "assessment.customerClassQuestionUnavailable": "Questa domanda non è disponibile per la classe aziendale selezionata.",
    "assessment.customerClassDimensionUnavailable": "Nessuna domanda in questa dimensione è disponibile per la classe aziendale selezionata.",
    "assessment.answered": "{answered} di {total} risposte",
    "assessment.handled": "Gestito: SI",
    "assessment.currentAnswers": "Risposte correnti del questionario",
    "assessment.aiVerified": "Risposte verificate dall'AI",
    "assessment.applyAiMarks": "Applica raccomandazioni AI",
    "assessment.applyAiMarksDone": "Risposte AI applicate al modulo utente.",
    "assessment.applyAiMarksUnavailable": "Analizza questa domanda prima di applicare le risposte AI.",
    "assessment.attachFile": "Analizza domanda",
    "assessment.level": "LIVELLO {level}",
    "assessment.noFramework": "Le domande di valutazione non sono disponibili. Verifica che HANA sia configurato e che l'import del framework sia stato eseguito.",
    "assessment.noAi": "Analizza questa domanda per generare risposte verificate dall'AI usando le evidenze del Document Manager.",
    "assessment.aiFailed": "Revisione AI non riuscita",
    "assessment.aiFailedFallback": "Il worker non ha completato la revisione di questa domanda.",
    "assessment.legacyResultTitle": "Esegui nuovamente questa revisione AI",
    "assessment.legacyResultMessage": "Questo risultato e stato creato con un contratto precedente per le decisioni sulle risposte. Rimane visibile come riferimento, ma i suggerimenti AI non possono essere applicati finche non viene completata una nuova revisione.",
    "assessment.noDecisions": "Nessuna decisione sugli elementi di risposta restituita per questo livello.",
    "assessment.noLevelResults": "Il worker ha completato questa domanda, ma il modello non ha restituito risultati per livello.",
    "assessment.highestSupportedLevel": "Livello massimo supportato {level}",
    "assessment.noSupportedLevel": "Nessun livello supportato",
    "assessment.reasoning": "Motivazione",
    "assessment.reasoningEvidence": "Motivazione ed evidenze",
    "assessment.evidence": "Evidenze: {refs}",
    "assessment.noEvidence": "Evidenze: nessun riferimento collegato",
    "assessment.loadingError": "Impossibile caricare il framework di valutazione da HANA. {message}",
    "assessment.responsesSaveFailed": "La risposta e stata aggiornata localmente, ma non e stato possibile salvarla in HANA: {message}",
    "assessment.attachRequired": "Seleziona almeno una domanda nella dimensione corrente.",
    "assessment.batchAttachRequired": "Le domande di valutazione non sono disponibili per l'analisi completa.",
    "assessment.submitting": "Avvio dell'analisi AI per questa dimensione...",
    "assessment.submittingQuestion": "Avvio dell'analisi AI per questa domanda...",
    "assessment.submittingBatch": "Avvio dell'analisi AI per tutte le domande...",
    "assessment.createdJob": "Job di revisione AI {jobId} creato con {taskCount} task.",
    "assessment.createdBatchJob": "Job di revisione AI {jobId} creato con {taskCount} task domanda.",
    "assessment.pollFailed": "Polling della revisione AI non riuscito: {message}",
    "assessment.staleJobCleared": "Il job di revisione AI precedente non e piu disponibile. Avvia una nuova analisi per aggiornare queste risposte.",
    "assessment.jobStatus": "Revisione AI {status}: {completed}/{total} task domanda completati{failedText}.",
    "assessment.failedText": ", {failed} non riusciti",
    "assessment.resetting": "Reset di tutte le modifiche bozza e dei job di revisione AI...",
    "assessment.resetDone": "Tutte le modifiche bozza e i job di revisione AI sono stati azzerati. Rimossi {jobs} job di revisione e {tasks} task domanda.",
    "assessment.startFailed": "Avvio della revisione AI non riuscito: {message}",
    "assessment.resetFailed": "Reset non riuscito: {message}",
    "file.remove": "Rimuovi {fileName}",
    "label.completed": "Completato",
    "label.failed": "Non riuscito",
    "label.in_progress": "In corso",
    "label.insufficient_evidence": "Evidenze insufficienti",
    "label.keep_selected": "Mantieni selezionata",
    "label.low_confidence": "Bassa confidenza",
    "label.partial_failed": "Parzialmente non riuscito",
    "label.partially_supported": "Parzialmente supportato",
    "label.pending": "In attesa",
    "label.select": "Seleziona",
    "label.unclear": "Non chiaro",
    "label.unsupported": "Non evidenziato",
    "assessment.activeProgress": "{message} {completed}/{total} completati{failedText}.",
    "assessment.questionInProgress": "Revisione AI in corso",
    "assessment.questionProgressFallback": "Questa domanda e in revisione.",
    "admin.benchmark.title": "Dati di benchmark",
    "admin.benchmark.description": "Convalida e attiva i file di benchmark della valutazione, versionati e archiviati in HANA. Viene usata la protezione esistente con chiave API condivisa.",
    "admin.benchmark.refresh": "Aggiorna stato",
    "admin.benchmark.select": "Seleziona XLSX",
    "admin.benchmark.validate": "Convalida file",
    "admin.benchmark.activate": "Attiva dataset",
    "admin.benchmark.none": "Nessuno",
    "admin.benchmark.loadingHistory": "Caricamento dello stato del benchmark attivo...",
    "admin.benchmark.noActiveTitle": "Benchmark non disponibile",
    "admin.benchmark.noActiveDetail": "Nessun dataset di benchmark attivo è archiviato in HANA. Seleziona e convalida un file XLSX prima dell'attivazione.",
    "admin.benchmark.activeTitle": "Dataset di benchmark attivo",
    "admin.benchmark.file": "File sorgente",
    "admin.benchmark.status": "Stato",
    "admin.benchmark.activated": "Attivato",
    "admin.benchmark.importId": "ID importazione",
    "admin.benchmark.rows": "Righe",
    "admin.benchmark.companies": "Aziende",
    "admin.benchmark.questionnaires": "Questionari",
    "admin.benchmark.questions": "Domande",
    "admin.benchmark.accepted": "Accettate",
    "admin.benchmark.rejected": "Rifiutate",
    "admin.benchmark.classes": "Classi",
    "admin.benchmark.naceCohorts": "Coorti NACE-1",
    "admin.benchmark.warnings": "Avvisi aggregati ({count})",
    "admin.benchmark.rowsList": "righe {rows}",
    "admin.benchmark.noWarnings": "Nessun avviso.",
    "admin.benchmark.sampledErrors": "Errori di riga campionati ({count})",
    "admin.benchmark.row": "Riga {row}",
    "admin.benchmark.noErrors": "Nessun errore di riga campionato.",
    "admin.benchmark.noSelection": "Nessun file selezionato.",
    "admin.benchmark.historyFailed": "Impossibile caricare lo stato del benchmark attivo. Verifica la connessione API e riprova.",
    "admin.benchmark.selectRequired": "Seleziona un file XLSX prima della convalida.",
    "admin.benchmark.validating": "Convalida del file senza scrittura in HANA...",
    "admin.benchmark.validationPassed": "Convalida superata",
    "admin.benchmark.validationFailed": "Convalida non riuscita",
    "admin.benchmark.validationReady": "Convalida superata. Esamina i risultati prima di attivare il dataset.",
    "admin.benchmark.validationRequestFailed": "Impossibile completare la richiesta di convalida del file. Verifica la connessione API e riprova.",
    "admin.benchmark.validationRequired": "Convalida correttamente il file attualmente selezionato prima dell'attivazione.",
    "admin.benchmark.confirmActivation": "Attivare questo dataset di benchmark convalidato? Sostituirà la versione attiva corrente per tutti i confronti della valutazione.",
    "admin.benchmark.activating": "Creazione e popolamento delle tabelle HANA, quindi attivazione del dataset convalidato...",
    "admin.benchmark.alreadyActive": "Questo dataset identico è già attivo.",
    "admin.benchmark.activationDone": "Il dataset di benchmark è stato attivato correttamente.",
    "admin.benchmark.activationFailed": "Attivazione del benchmark non riuscita; la versione attiva precedente è stata mantenuta. Esamina il risultato della convalida o i log del server e riprova.",
    "admin.benchmark.xlsxOnly": "I dati di benchmark devono essere forniti in un unico file .xlsx.",
    "admin.benchmark.importStatus.active": "Attivo",
    "admin.benchmark.importStatus.inactive": "Inattivo",
    "admin.benchmark.importStatus.validated": "Convalidato",
    "admin.benchmark.importStatus.rejected": "Rifiutato",
    "admin.benchmark.importStatus.failed": "Non riuscito",
    "admin.benchmark.importStatus.unknown": "Sconosciuto",
    "admin.benchmark.issue.warningFallback": "È stato rilevato un problema non bloccante nei dati sorgente; esamina il codice e le righe interessate.",
    "admin.benchmark.issue.errorFallback": "La riga o il file non rispetta il contratto di importazione del benchmark.",
    "admin.benchmark.issue.answer_text_mismatch": "Il testo della risposta sorgente differisce dal framework canonico; è stata mantenuta la mappatura canonica.",
    "admin.benchmark.issue.dimension_mismatch": "La dimensione sorgente differisce da quella del framework canonico; è stato mantenuto il valore del framework.",
    "admin.benchmark.issue.duplicate_row": "Una riga sorgente duplicata è stata ignorata.",
    "admin.benchmark.issue.profile_placeholder_normalized": "Un segnaposto del profilo sorgente è stato normalizzato come valore vuoto.",
    "admin.benchmark.issue.score_reconciliation": "Un punteggio fornito differisce dal punteggio ricalcolato con la formula dell'applicazione.",
    "admin.benchmark.issue.conflicting_answer_catalog": "La stessa posizione di risposta sorgente contiene valori in conflitto.",
    "admin.benchmark.issue.conflicting_row": "Una risposta logica duplicata contiene dati in conflitto.",
    "admin.benchmark.issue.empty_workbook": "Il foglio richiesto non contiene righe di risposta del benchmark.",
    "admin.benchmark.issue.invalid_boolean": "Un campo booleano contiene un valore non supportato.",
    "admin.benchmark.issue.invalid_class": "La classe aziendale non è supportata dal framework di valutazione.",
    "admin.benchmark.issue.invalid_date": "Un campo data contiene un valore non valido.",
    "admin.benchmark.issue.invalid_extension": "Il file selezionato non è un file XLSX.",
    "admin.benchmark.issue.invalid_headers": "Le intestazioni del foglio non corrispondono al contratto di importazione esatto.",
    "admin.benchmark.issue.invalid_level": "Un livello di maturità della risposta non rientra nell'intervallo supportato.",
    "admin.benchmark.issue.invalid_profile_value": "Un campo del profilo aziendale contiene un valore non valido.",
    "admin.benchmark.issue.invalid_score": "Un punteggio fornito non è un numero finito valido.",
    "admin.benchmark.issue.malformed_xlsx": "Impossibile aprire il file come file XLSX valido.",
    "admin.benchmark.issue.missing_identifier": "Manca un identificativo obbligatorio di azienda, questionario, domanda o risposta.",
    "admin.benchmark.issue.missing_sheet": "Il file non contiene il foglio Estrazione richiesto.",
    "admin.benchmark.issue.mixed_company_metadata": "Le righe della stessa azienda contengono metadati di profilo incoerenti.",
    "admin.benchmark.issue.mixed_dimension_score": "Le righe della stessa dimensione del questionario contengono punteggi forniti incoerenti.",
    "admin.benchmark.issue.mixed_questionnaire_metadata": "Le righe dello stesso questionario contengono metadati di invio incoerenti.",
    "admin.benchmark.issue.source_too_large": "Il file supera la dimensione massima consentita per il caricamento.",
    "admin.benchmark.issue.unknown_question": "La domanda sorgente non esiste nel framework di valutazione canonico.",
    "admin.benchmark.issue.unmappable_answer": "La risposta sorgente non può essere associata a una posizione di risposta canonica.",
    "admin.benchmark.issue.unsafe_zip": "L'archivio XLSX non è sicuro o supera i limiti di estrazione.",
    "score.title": "Punteggio",
    "score.subtitle": "Posizionamento della valutazione rispetto alla coorte di peer attiva.",
    "score.context": "Classe aziendale: {customerClass} | NACE-1: {nace1}",
    "score.contextUnavailable": "Non è disponibile un profilo di benchmark salvato con classe e NACE-1.",
    "score.profileMissing": "Non configurato",
    "score.refresh": "Aggiorna",
    "score.score": "Punteggio",
    "score.finalScore": "Punteggio finale",
    "score.benchmark": "Benchmark",
    "score.delta": "Scostamento",
    "score.sameSector": "Stesso settore",
    "score.sameSize": "Stessa classe",
    "score.sectorDelta": "Scost. settore",
    "score.sizeDelta": "Scost. classe",
    "score.dimensionScores": "Punteggi per dimensione",
    "score.questionPositioning": "Posizionamento per domanda",
    "score.question": "Domanda",
    "score.dimension": "Dimensione",
    "score.selected": "Selezionate",
    "score.applicable": "Applicabile",
    "score.loadingError": "Impossibile caricare il punteggio: {message}",
    "score.overviewTitle": "Panoramica delle prestazioni complessive",
    "score.bestPeer": "Miglior peer",
    "score.peerAverage": "Media dei peer",
    "score.company": "Azienda",
    "score.legendLabel": "Legenda del grafico di benchmark",
    "score.chartDescription": "Ogni gruppo utilizza tre barre sovrapposte sulla stessa scala di maturità nascosta. Lo spessore delle barre e il posizionamento testuale identificano miglior peer, media dei peer e azienda; i valori esatti non sono mostrati qui.",
    "score.dimensionAnalysis": "Analisi per dimensione",
    "score.dimensionAnalysisDescription": "Espandi una dimensione per esaminare i temi applicabili del questionario e il commento deterministico sui peer.",
    "score.noApplicableTopics": "Nessun tema applicabile è disponibile per questa classe aziendale.",
    "score.position.below_peers": "Sotto i tuoi peer",
    "score.position.in_line_with_peers": "In linea con i tuoi peer",
    "score.position.above_peers": "Sopra i tuoi peer",
    "score.position.unavailable": "Benchmark non disponibile",
    "score.benchmarkUnavailable": "Benchmark non disponibile",
    "score.unavailableNoDataset": "Nessun dataset di benchmark attivo è disponibile. I risultati aziendali restano visibili senza dati peer sintetici.",
    "score.unavailableNoProfile": "Salva una classe aziendale e un profilo NACE-1 nella pagina Valutazione per selezionare una coorte di peer esatta.",
    "score.unavailableSmallCohort": "Questa coorte non contiene dati peer idonei sufficienti per il confronto.",
    "score.unavailableGeneric": "Il confronto con i peer non è disponibile per il contesto selezionato.",
    "score.benchmarkAvailableDetail": "Il confronto utilizza dati peer idonei dal dataset attivato il {datasetDate}.",
    "score.datasetDate": "Data del dataset attivo: {datasetDate}.",
    "score.calificationReport": "Report di valutazione",
    "score.reportSubmitting": "Preparazione dell'istantanea della valutazione...",
    "score.reportQueued": "Report di valutazione in coda.",
    "score.reportGenerating": "Preparazione del report di valutazione deterministico...",
    "score.reportRendering": "Creazione del PDF del report di valutazione...",
    "score.reportResuming": "Ripresa della generazione del report di valutazione...",
    "score.reportReady": "Report di valutazione pronto. Il download del PDF è iniziato.",
    "score.reportFailed": "Impossibile generare il report di valutazione. Riprova.",
    "score.reportStartFailed": "Impossibile avviare la generazione del report di valutazione: {message}",
    "score.reportPollFailed": "Impossibile aggiornare lo stato del report; il polling verrà riprovato: {message}",
    "score.reportDownloadFailed": "Il report è pronto, ma il download non è riuscito: {message}",
    "score.reportExpired": "Il report precedente non è più disponibile. Genera un nuovo report.",
    "score.reportUnknownStatus": "Il report ha restituito uno stato non supportato. Genera un nuovo report.",
    "label.yes": "Si",
    "label.no": "No",
    "label.pending_indexing": "Preparazione documenti",
    "label.extracting_documents": "Estrazione documenti",
    "label.embedding_documents": "Creazione embedding",
    "label.reviewing_questions": "Revisione domande",
    "label.retrieving_evidence": "Recupero evidenze",
    "label.finalizing_answer": "Finalizzazione risposta",
    "label.retrying": "Nuovo tentativo"
  }
};

/**
 * Read the active UI language from browser session storage.
 *
 * @returns {"en" | "it"} Supported language code for the current browser tab.
 */
export function getLanguage() {
  const storedLanguage = window.sessionStorage.getItem(LANGUAGE_STORAGE_KEY);
  return SUPPORTED_LANGUAGES.has(storedLanguage) ? storedLanguage : DEFAULT_LANGUAGE;
}

/**
 * Store a language selection for the current browser session.
 *
 * @param {string} language - Requested language code.
 * @returns {void}
 */
export function setLanguage(language) {
  const normalizedLanguage = SUPPORTED_LANGUAGES.has(language) ? language : DEFAULT_LANGUAGE;
  window.sessionStorage.setItem(LANGUAGE_STORAGE_KEY, normalizedLanguage);
  document.dispatchEvent(
    new CustomEvent("language-change", {
      detail: { language: normalizedLanguage }
    })
  );
}

/**
 * Translate one UI text key and interpolate named placeholders.
 *
 * @param {string} key - Translation key.
 * @param {Record<string, unknown>} params - Placeholder values.
 * @returns {string} Localized UI string.
 */
export function t(key, params = {}) {
  const language = getLanguage();
  const template = TRANSLATIONS[language][key] || TRANSLATIONS.en[key] || key;
  return Object.entries(params).reduce(
    (value, [paramKey, paramValue]) => value.replaceAll(`{${paramKey}}`, String(paramValue)),
    template
  );
}

/**
 * Apply shell and navigation labels for the active language.
 *
 * @returns {void}
 */
export function applyGlobalTranslations() {
  const language = getLanguage();
  document.documentElement.lang = language;

  const branding = document.querySelector("ui5-shellbar-branding");
  if (branding?.childNodes?.[0]) {
    branding.childNodes[0].nodeValue = `\n            ${t("app.title")}\n            `;
  }

  const search = document.querySelector("ui5-shellbar-search");
  if (search) {
    search.setAttribute("placeholder", t("shell.search"));
  }

  const help = document.getElementById("help-shell-item");
  if (help) {
    help.setAttribute("text", t("shell.help"));
  }

  const languageButton = document.getElementById("language-menu-button");
  if (languageButton) {
    languageButton.setAttribute("text", language.toUpperCase());
    languageButton.setAttribute("title", t("shell.language"));
  }

  const homeItem = document.querySelector("ui5-side-navigation-item[href='/home']");
  if (homeItem) {
    homeItem.setAttribute("text", t("nav.home"));
  }

  const adminDocumentsItem = document.querySelector(
    "ui5-side-navigation-item[href='/admin-documents']"
  );
  if (adminDocumentsItem) {
    adminDocumentsItem.setAttribute("text", t("nav.adminDocuments"));
  }

  const assessmentItem = document.querySelector("ui5-side-navigation-item[href='/assessment']");
  if (assessmentItem) {
    assessmentItem.setAttribute("text", t("nav.assessment"));
  }

  const scoreItem = document.querySelector("ui5-side-navigation-item[href='/score']");
  if (scoreItem) {
    scoreItem.setAttribute("text", t("nav.score"));
  }
}

/**
 * Wire the shell language button and popover to session language state.
 *
 * @returns {void}
 */
export function initLanguageSelector() {
  const button = document.getElementById("language-menu-button");
  const popover = document.getElementById("language-popover");
  if (!button || !popover) {
    return;
  }

  button.addEventListener("click", () => {
    popover.open = !popover.open;
  });

  document.querySelectorAll("[data-language-option]").forEach((option) => {
    option.addEventListener("click", () => {
      setLanguage(option.dataset.languageOption);
      popover.open = false;
      applyGlobalTranslations();
    });
  });

  document.addEventListener("language-change", applyGlobalTranslations);
  applyGlobalTranslations();
}
