import { beforeEach, describe, expect, it, vi } from "vitest";

import initHomePage from "../src/pages/home/home.js";

describe("home controller", () => {
  beforeEach(() => {
    document.body.innerHTML = `
      <input id="reset-extra-account-input" />
      <button id="reset-extra-account"></button>
      <button id="reset-extra-all"></button>
      <div id="reset-extra-status"></div>
    `;
  });

  it("resets saved extra information for one billing account", async () => {
    const api = {
      resetSavedAnswersForAccount: vi.fn().mockResolvedValue({ deleted_count: 2 }),
      resetAllSavedAnswers: vi.fn()
    };

    initHomePage(api, { confirm: vi.fn() });
    document.getElementById("reset-extra-account-input").value = "104";
    document.getElementById("reset-extra-account").click();
    await Promise.resolve();
    await Promise.resolve();

    expect(api.resetSavedAnswersForAccount).toHaveBeenCalledWith("104");
    expect(api.resetAllSavedAnswers).not.toHaveBeenCalled();
    expect(document.getElementById("reset-extra-status").textContent).toContain(
      "Reset 2 saved answers for account 104."
    );
  });

  it("confirms before resetting all saved extra information", async () => {
    const api = {
      resetSavedAnswersForAccount: vi.fn(),
      resetAllSavedAnswers: vi.fn().mockResolvedValue({ deleted_count: 5 })
    };
    const browser = { confirm: vi.fn().mockReturnValue(true) };

    initHomePage(api, browser);
    document.getElementById("reset-extra-all").click();
    await Promise.resolve();
    await Promise.resolve();

    expect(browser.confirm).toHaveBeenCalled();
    expect(api.resetAllSavedAnswers).toHaveBeenCalledTimes(1);
    expect(api.resetSavedAnswersForAccount).not.toHaveBeenCalled();
    expect(document.getElementById("reset-extra-status").textContent).toContain(
      "Reset 5 saved answers across all accounts."
    );
  });
});
