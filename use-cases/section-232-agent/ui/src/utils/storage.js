export function getBrowserStorage() {
  try {
    return window.localStorage;
  } catch (_error) {
    return null;
  }
}

export function readJsonStorage(key) {
  const storage = getBrowserStorage();
  const rawValue = storage?.getItem(key);
  if (!rawValue) {
    return null;
  }
  try {
    return JSON.parse(rawValue);
  } catch (_error) {
    storage?.removeItem(key);
    return null;
  }
}

export function writeJsonStorage(key, value) {
  const storage = getBrowserStorage();
  if (!storage) {
    return false;
  }
  storage.setItem(key, JSON.stringify(value));
  return true;
}

export function removeStorageItem(key) {
  getBrowserStorage()?.removeItem(key);
}
