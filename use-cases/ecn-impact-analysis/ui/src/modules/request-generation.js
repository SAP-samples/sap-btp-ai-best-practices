/** Create one independent last-write-wins request lane bound to a page signal. */
export function createLatestRequestGate(parentSignal) {
  let generation = 0;
  let activeController = null;
  let disposed = false;

  const abortActive = () => activeController?.abort();
  parentSignal?.addEventListener("abort", abortActive, { once: true });

  return {
    /** Abort the prior request in this lane and return a guarded child request. */
    next() {
      activeController?.abort();
      const controller = new AbortController();
      activeController = controller;
      const currentGeneration = ++generation;
      if (disposed || parentSignal?.aborted) controller.abort();
      return {
        signal: controller.signal,
        isCurrent: () => !disposed
          && !controller.signal.aborted
          && currentGeneration === generation,
      };
    },
    /** Abort the current operation but keep this lane reusable. */
    cancel() {
      generation += 1;
      activeController?.abort();
      activeController = null;
    },
    /** Abort this lane permanently without affecting sibling lanes. */
    dispose() {
      disposed = true;
      generation += 1;
      activeController?.abort();
      activeController = null;
      parentSignal?.removeEventListener("abort", abortActive);
    },
  };
}
