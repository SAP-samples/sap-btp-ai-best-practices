import {
  resetAllSavedAnswers,
  resetSavedAnswersForAccount
} from "../../services/offerApi.js";

/**
 * Initialize the home page reset controls for saved customer answers.
 *
 * @param {object} api - API methods used by the reset controls.
 * @param {Window|object} browser - Browser object providing confirm().
 * @returns {void}
 */
export default function initHomePage(
  api = {
    resetAllSavedAnswers,
    resetSavedAnswersForAccount
  },
  browser = window
) {
  const accountInput = document.getElementById("reset-extra-account-input");
  const resetAccountButton = document.getElementById("reset-extra-account");
  const resetAllButton = document.getElementById("reset-extra-all");
  const status = document.getElementById("reset-extra-status");
  let requestInFlight = false;

  /**
   * Render a short status message for the reset tool.
   *
   * @param {string} message - Text to show in the status area.
   * @returns {void}
   */
  function setStatus(message) {
    if (status) {
      status.textContent = message;
    }
  }

  /**
   * Synchronize disabled state while a reset request is running.
   *
   * @returns {void}
   */
  function syncBusyState() {
    if (accountInput) {
      accountInput.disabled = requestInFlight;
    }
    if (resetAccountButton) {
      resetAccountButton.disabled = requestInFlight;
    }
    if (resetAllButton) {
      resetAllButton.disabled = requestInFlight;
    }
  }

  /**
   * Run one reset request while preventing overlapping operations.
   *
   * @param {Function} operation - Async operation to run exclusively.
   * @returns {Promise<object|null>} Operation result or null when skipped.
   */
  async function runExclusive(operation) {
    if (requestInFlight) {
      return null;
    }

    requestInFlight = true;
    syncBusyState();
    try {
      return await operation();
    } finally {
      requestInFlight = false;
      syncBusyState();
    }
  }

  resetAccountButton?.addEventListener("click", async () => {
    const billingAccount = accountInput?.value.trim() || "";
    if (!billingAccount) {
      setStatus("Enter a billing account to reset saved answers for one account.");
      return;
    }

    const response = await runExclusive(() =>
      api.resetSavedAnswersForAccount(billingAccount)
    );
    if (!response) {
      return;
    }
    setStatus(`Reset ${response.deleted_count} saved answers for account ${billingAccount}.`);
  });

  resetAllButton?.addEventListener("click", async () => {
    const confirmed = browser.confirm(
      "Reset saved extra information for all billing accounts?"
    );
    if (!confirmed) {
      return;
    }

    const response = await runExclusive(() => api.resetAllSavedAnswers());
    if (!response) {
      return;
    }
    setStatus(`Reset ${response.deleted_count} saved answers across all accounts.`);
  });
}
