import { afterEach, expect, it, vi } from "vitest";
import { startAnalysisPolling } from "../src/modules/analysis-polling.js";

afterEach(() => vi.useRealTimers());

it("loads immediately, never overlaps, retries errors, and stops at terminal state", async () => {
  vi.useFakeTimers();
  let release;
  const load = vi.fn()
    .mockRejectedValueOnce(new Error("temporary"))
    .mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
  const updates = vi.fn();
  const errors = vi.fn();
  startAnalysisPolling({
    load,
    onUpdate: updates,
    onError: errors,
    isFinished: (value) => value.unfinished_count === 0,
    intervalMs: 3000,
  });

  await vi.waitFor(() => expect(errors).toHaveBeenCalledOnce());
  await vi.advanceTimersByTimeAsync(3000);
  await vi.advanceTimersByTimeAsync(9000);
  expect(load).toHaveBeenCalledTimes(2);
  release({ unfinished_count: 0 });
  await Promise.resolve();
  await vi.advanceTimersByTimeAsync(3000);
  expect(updates).toHaveBeenCalledWith({ unfinished_count: 0 });
  expect(load).toHaveBeenCalledTimes(2);
});
