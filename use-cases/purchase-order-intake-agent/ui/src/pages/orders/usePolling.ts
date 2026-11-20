/**
 * Polls a loader while a condition holds (and the browser tab is visible).
 *
 * Polling is data-driven: every completed load (automatic or manual `reload`) schedules the
 * next one if `shouldPoll` is true for the new data, so a manual refresh after an upload
 * restarts polling automatically.
 */
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Loads data once, then reloads every `intervalMs` while `shouldPoll(data)` is true.
 *
 * @param load - Async loader.
 * @param shouldPoll - Decides from the latest data whether to keep polling.
 * @param intervalMs - Poll interval.
 * @returns Latest data, loading flag, last error and a manual `reload`.
 */
export function usePolling<T>(load: () => Promise<T>, shouldPoll: (data: T) => boolean, intervalMs = 3000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);  // increments after every load attempt
  const loadRef = useRef(load);
  const pollRef = useRef(shouldPoll);
  loadRef.current = load;
  pollRef.current = shouldPoll;

  const reload = useCallback(async () => {
    try {
      const result = await loadRef.current();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
      setVersion((v) => v + 1);
    }
  }, []);

  // Initial load.
  useEffect(() => {
    void reload();
  }, [reload]);

  // After each load: schedule the next one while the condition holds (retry on errors).
  useEffect(() => {
    if (version === 0) return;
    if (error === null && (data === null || !pollRef.current(data))) return;
    const timer = setTimeout(() => {
      if (document.visibilityState === "visible") void reload();
      else setVersion((v) => v + 1);
    }, intervalMs);
    return () => clearTimeout(timer);
  }, [version, data, error, reload, intervalMs]);

  return { data, error, loading, reload };
}
