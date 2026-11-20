/** Poll one read operation serially until terminal state, abort, or explicit stop. */
export function startAnalysisPolling({
  load,
  onUpdate,
  onError,
  isFinished,
  intervalMs = 3000,
  signal,
}) {
  let stopped = false;
  let timer = null;

  const stop = () => {
    stopped = true;
    if (timer !== null) clearTimeout(timer);
  };
  signal?.addEventListener("abort", stop, { once: true });

  /** Complete one request before scheduling the next poll. */
  const tick = async () => {
    if (stopped || signal?.aborted) return;
    try {
      const value = await load();
      if (stopped || signal?.aborted) return;
      onUpdate(value);
      if (isFinished(value)) return stop();
    } catch (error) {
      if (stopped || signal?.aborted || error?.name === "AbortError") return;
      onError(error);
    }
    timer = setTimeout(tick, intervalMs);
  };
  void tick();
  return stop;
}
