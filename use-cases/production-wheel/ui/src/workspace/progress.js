/** Describe queue ownership and phase progress from a public run record. */
export function runProgress(run, now = Date.now()) {
  if (run.status === "queued") {
    const age = Math.max(0, Math.floor((now - Date.parse(run.created_at)) / 1000)) || 0;
    return {
      fraction: null,
      text: `Queued for ${age}s. Solver execution has not started.${age >= 60 ? " Check that an optimizer worker is running; it may also be busy with another run. Do not submit a duplicate." : " Waiting for a worker."}`,
    };
  }
  const progress = run.progress;
  if (progress && typeof progress === "object" && progress.total > 0 && Number.isFinite(progress.current)) {
    return {fraction: Math.max(0, Math.min(1, progress.current / progress.total)), text: `${run.stage || "Processing"}: ${progress.current} / ${progress.total} in this phase.`};
  }
  if (typeof progress === "number") return {fraction: Math.max(0, Math.min(1, progress > 1 ? progress / 100 : progress)), text: run.stage || "Processing"};
  return {fraction: null, text: run.stage || "Processing"};
}
