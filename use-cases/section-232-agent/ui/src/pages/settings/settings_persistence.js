import { readJsonStorage, removeStorageItem, writeJsonStorage } from "../../utils/storage.js";

const DRAFT_BATCH_STORAGE_KEY = "section232.settings.currentDraftBatch";

export function buildDraftBatchSnapshot(batch) {
  if (!batch?.batch_id) {
    return null;
  }
  const sourceFilenames = Array.isArray(batch.source_filenames)
    ? batch.source_filenames.map((value) => String(value ?? "").trim()).filter(Boolean)
    : [];
  const acceptedCount = Number(batch.accepted_count || 0);
  const rejectedCount = Number(batch.rejected_count || 0);
  const pendingCount = Number(batch.pending_count || 0);
  const warningCount = Number(batch.warning_count || 0);
  const totalCount = Number(batch.rule_candidate_count || acceptedCount + rejectedCount + pendingCount || 0);
  return {
    batch_id: batch.batch_id,
    status: batch.status || "pending_review",
    source_count: Number(batch.source_count || sourceFilenames.length || 0),
    source_filenames: sourceFilenames,
    rule_candidate_count: totalCount,
    pending_count: pendingCount,
    accepted_count: acceptedCount,
    rejected_count: rejectedCount,
    warning_count: warningCount,
    created_at: batch.created_at || null
  };
}

export function persistDraftBatch(batch) {
  if (!batch?.batch_id) {
    clearPersistedDraftBatch();
    return;
  }
  writeJsonStorage(DRAFT_BATCH_STORAGE_KEY, batch);
}

export function readPersistedDraftBatch() {
  const parsed = readJsonStorage(DRAFT_BATCH_STORAGE_KEY);
  if (parsed && typeof parsed.batch_id === "string" && parsed.batch_id.trim()) {
    return parsed;
  }
  return null;
}

export function clearPersistedDraftBatch() {
  removeStorageItem(DRAFT_BATCH_STORAGE_KEY);
}
