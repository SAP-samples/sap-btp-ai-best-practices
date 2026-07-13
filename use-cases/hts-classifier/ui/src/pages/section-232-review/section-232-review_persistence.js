import { getBrowserStorage, readJsonStorage, removeStorageItem, writeJsonStorage } from "../../utils/storage.js";

import { trimText } from "./section-232-review_shared.js";

const PUBLISHED_BY_STORAGE_KEY = "section232.review.publishedBy";
const SETTINGS_DRAFT_BATCH_STORAGE_KEY = "section232.settings.currentDraftBatch";

export function readPersistedPublishedBy(batchId) {
  const normalizedBatchId = trimText(batchId);
  if (!normalizedBatchId) return "";
  const parsed = readJsonStorage(PUBLISHED_BY_STORAGE_KEY);
  if (parsed && trimText(parsed.batch_id) === normalizedBatchId && typeof parsed.published_by === "string") {
    return parsed.published_by;
  }
  return "";
}

export function persistPublishedBy(batchId, publishedBy) {
  const normalizedBatchId = trimText(batchId);
  const normalizedPublishedBy = trimText(publishedBy);
  if (!normalizedBatchId || !normalizedPublishedBy) {
    removeStorageItem(PUBLISHED_BY_STORAGE_KEY);
    return;
  }
  writeJsonStorage(PUBLISHED_BY_STORAGE_KEY, {
    batch_id: normalizedBatchId,
    published_by: normalizedPublishedBy
  });
}

export function clearPersistedPublishedBy(batchId) {
  const normalizedBatchId = trimText(batchId);
  const storage = getBrowserStorage();
  if (!storage) return;
  if (!normalizedBatchId) {
    storage.removeItem(PUBLISHED_BY_STORAGE_KEY);
    return;
  }
  const rawValue = storage.getItem(PUBLISHED_BY_STORAGE_KEY);
  if (!rawValue) return;
  try {
    const parsed = JSON.parse(rawValue);
    if (trimText(parsed?.batch_id) === normalizedBatchId) {
      removeStorageItem(PUBLISHED_BY_STORAGE_KEY);
    }
  } catch (_error) {
    removeStorageItem(PUBLISHED_BY_STORAGE_KEY);
  }
}

export function clearPersistedSettingsDraftBatch(batchId) {
  const normalizedBatchId = trimText(batchId);
  const storage = getBrowserStorage();
  if (!storage) return;
  const parsed = readJsonStorage(SETTINGS_DRAFT_BATCH_STORAGE_KEY);
  if (!parsed) {
    return;
  }
  if (!normalizedBatchId || trimText(parsed.batch_id) === normalizedBatchId) {
    removeStorageItem(SETTINGS_DRAFT_BATCH_STORAGE_KEY);
  }
}
