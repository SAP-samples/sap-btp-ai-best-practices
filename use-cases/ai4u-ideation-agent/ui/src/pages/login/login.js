import { loginXsuaa, loginWithCdcToken, getReturnUrl } from "../../services/auth.js";
import { sendOtpCode, loginWithOtp, getCdcJwt, isCdcConfigured, trackLoginServer } from "../../services/cdc.js";

/**
 * Initialize the login page.
 * Renders into the element with id="login-root".
 */
export default function initLoginPage() {
  const root = document.getElementById("login-root");
  if (!root) return;

  root.innerHTML = _renderLoginHTML();
  _attachListeners(root);
}

function _renderLoginHTML() {
  const showCdc = isCdcConfigured();

  return `
    <div class="login-page">
      <div class="login-page__background"></div>
      <div class="login-page__container">

        <ui5-card class="login-card">
          <div class="login-card__header">
            <img src="/images/SAP_logo.svg" alt="SAP" class="login-card__logo" />
            <ui5-title level="H2" class="login-card__title">AI4U Ideation Agent</ui5-title>
            <ui5-text class="login-card__subtitle">Your AI-powered use case ideation partner</ui5-text>
          </div>

          <!-- Method selection (initial view) -->
          <div class="login-card__body" id="login-view-methods">
            <div class="login-methods">
              <button class="login-method" id="login-sap-employee">
                <div class="login-method__icon-wrap login-method__icon-wrap--sap">
                  <ui5-icon name="employee"></ui5-icon>
                </div>
                <div class="login-method__content">
                  <ui5-title level="H5">SAP Login</ui5-title>
                  <ui5-text>Sign in with your organization’s account.</ui5-text>
                </div>
                <ui5-icon name="navigation-right-arrow" class="login-method__arrow"></ui5-icon>
              </button>

              ${showCdc ? `
              <button class="login-method" id="login-external-user">
                <div class="login-method__icon-wrap login-method__icon-wrap--external">
                  <ui5-icon name="email"></ui5-icon>
                </div>
                <div class="login-method__content">
                  <ui5-title level="H5">External User</ui5-title>
                  <ui5-text>Sign in with a one-time email code</ui5-text>
                </div>
                <ui5-icon name="navigation-right-arrow" class="login-method__arrow"></ui5-icon>
              </button>
              ` : ""}
            </div>

            <p class="login-consent-notice">
              By continuing, you agree to the
              <a href="/privacy" target="_blank" rel="noopener">Privacy Statement</a>
              and
              <a href="https://www.sap.com/about/legal/terms-of-use.html" target="_blank" rel="noopener noreferrer">Terms of Use</a>.
            </p>
          </div>

          <!-- OTP flow (shown when External User selected) -->
          ${showCdc ? `
          <div class="login-card__body login-transition" id="login-view-otp" style="display: none;">
            <div class="otp-header">
              <ui5-button design="Transparent" icon="nav-back" id="otp-back-btn" tooltip="Back to login options"></ui5-button>
              <ui5-title level="H5">Email Verification</ui5-title>
            </div>

            <!-- Step 1: Email -->
            <div id="otp-step-email" class="otp-step">
              <div class="otp-step__illustration">
                <ui5-icon name="email" class="otp-step__icon otp-step__icon--email"></ui5-icon>
              </div>
              <ui5-text class="otp-step__description">
                Enter your email address and we'll send you a verification code.
              </ui5-text>
              <div class="otp-step__form">
                <ui5-input id="otp-email-input"
                  placeholder="your@email.com"
                  type="Email"
                  class="otp-input"
                  accessible-name="Email address">
                </ui5-input>
                <ui5-button id="otp-send-btn" design="Emphasized" class="otp-submit-btn">
                  Send Verification Code
                </ui5-button>
              </div>
            </div>

            <!-- Step 2: Code verification -->
            <div id="otp-step-verify" class="otp-step" style="display: none;">
              <div class="otp-step__illustration">
                <ui5-icon name="message-success" class="otp-step__icon otp-step__icon--verify"></ui5-icon>
              </div>
              <ui5-text class="otp-step__description">
                We sent a 6-digit code to
              </ui5-text>
              <ui5-text class="otp-step__email-highlight" id="otp-sent-email"></ui5-text>
              <div class="otp-step__form">
                <ui5-input id="otp-code-input"
                  placeholder="000000"
                  maxlength="6"
                  class="otp-input otp-input--code"
                  accessible-name="Verification code">
                </ui5-input>
                <ui5-button id="otp-verify-btn" design="Emphasized" class="otp-submit-btn">
                  Verify & Sign In
                </ui5-button>
                <ui5-button id="otp-resend-btn" design="Transparent" class="otp-resend-btn">
                  Didn't receive it? Resend code
                </ui5-button>
              </div>
            </div>

            <!-- Error display -->
            <ui5-message-strip id="otp-error" design="Negative" hide-close-button class="otp-error" style="display: none;">
            </ui5-message-strip>

            <!-- Loading overlay -->
            <div id="otp-loading" class="otp-loading" style="display: none;">
              <ui5-busy-indicator active size="Medium"></ui5-busy-indicator>
            </div>
          </div>
          ` : ""}
        </ui5-card>

        <div class="login-page__footer">
          <ui5-text>Powered by SAP AI</ui5-text>
        </div>
      </div>
    </div>
  `;
}

function _attachListeners(root) {
  // SAP Login
  const sapBtn = root.querySelector("#login-sap-employee");
  if (sapBtn) {
    sapBtn.addEventListener("click", () => {
      const returnUrl = getReturnUrl() || "/";
      loginXsuaa(returnUrl);
    });
  }

  // External User login
  const extBtn = root.querySelector("#login-external-user");
  if (extBtn) {
    extBtn.addEventListener("click", () => _showOtpView(root));
  }

  // Back button
  const backBtn = root.querySelector("#otp-back-btn");
  if (backBtn) {
    backBtn.addEventListener("click", () => _showMethodsView(root));
  }

  // Send OTP
  const sendBtn = root.querySelector("#otp-send-btn");
  if (sendBtn) {
    sendBtn.addEventListener("click", () => _handleSendOtp(root));
  }

  // Verify OTP
  const verifyBtn = root.querySelector("#otp-verify-btn");
  if (verifyBtn) {
    verifyBtn.addEventListener("click", () => _handleVerifyOtp(root));
  }

  // Resend
  const resendBtn = root.querySelector("#otp-resend-btn");
  if (resendBtn) {
    resendBtn.addEventListener("click", () => _handleSendOtp(root));
  }

  // Enter key on inputs
  const emailInput = root.querySelector("#otp-email-input");
  if (emailInput) {
    emailInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") _handleSendOtp(root);
    });
  }

  const codeInput = root.querySelector("#otp-code-input");
  if (codeInput) {
    codeInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") _handleVerifyOtp(root);
    });
  }
}

// --- View transitions ---

function _showOtpView(root) {
  const methods = root.querySelector("#login-view-methods");
  const otp = root.querySelector("#login-view-otp");
  if (methods) methods.style.display = "none";
  if (otp) {
    otp.style.display = "";
    otp.classList.remove("login-transition");
    void otp.offsetWidth; // reflow
    otp.classList.add("login-transition");
  }
  _clearError(root);
  setTimeout(() => {
    const input = root.querySelector("#otp-email-input");
    if (input) input.focus();
  }, 150);
}

function _showMethodsView(root) {
  const methods = root.querySelector("#login-view-methods");
  const otp = root.querySelector("#login-view-otp");
  if (otp) otp.style.display = "none";
  if (methods) methods.style.display = "";
  // Reset OTP state
  root.querySelector("#otp-step-email").style.display = "";
  root.querySelector("#otp-step-verify").style.display = "none";
  _clearError(root);
  _currentVToken = null;
  _currentEmail = "";
}

// --- State ---
let _currentVToken = null;
let _currentEmail = "";

function _showError(root, message) {
  const el = root.querySelector("#otp-error");
  if (el) {
    el.textContent = message;
    el.style.display = "";
  }
}

function _clearError(root) {
  const el = root.querySelector("#otp-error");
  if (el) {
    el.textContent = "";
    el.style.display = "none";
  }
}

function _setLoading(root, loading) {
  const loadingEl = root.querySelector("#otp-loading");
  const sendBtn = root.querySelector("#otp-send-btn");
  const verifyBtn = root.querySelector("#otp-verify-btn");

  if (loadingEl) loadingEl.style.display = loading ? "flex" : "none";
  if (sendBtn) sendBtn.disabled = loading;
  if (verifyBtn) verifyBtn.disabled = loading;
}

async function _handleSendOtp(root) {
  const emailInput = root.querySelector("#otp-email-input");
  const email = (emailInput?.value || "").trim();

  if (!email || !email.includes("@")) {
    _showError(root, "Please enter a valid email address.");
    return;
  }

  _clearError(root);
  _setLoading(root, true);

  try {
    const { vToken } = await sendOtpCode(email);
    _currentVToken = vToken;
    _currentEmail = email;

    // Transition to verify step
    root.querySelector("#otp-step-email").style.display = "none";
    const verifyStep = root.querySelector("#otp-step-verify");
    verifyStep.style.display = "";
    verifyStep.classList.remove("login-transition");
    void verifyStep.offsetWidth;
    verifyStep.classList.add("login-transition");
    root.querySelector("#otp-sent-email").textContent = email;

    setTimeout(() => {
      const input = root.querySelector("#otp-code-input");
      if (input) input.focus();
    }, 150);
  } catch (err) {
    _showError(root, err.message || "Failed to send verification code. Please try again.");
  } finally {
    _setLoading(root, false);
  }
}

async function _handleVerifyOtp(root) {
  const codeInput = root.querySelector("#otp-code-input");
  const code = (codeInput?.value || "").trim();

  if (!code || code.length < 4) {
    _showError(root, "Please enter the verification code from your email.");
    return;
  }

  if (!_currentVToken) {
    _showError(root, "Your session expired. Please request a new code.");
    return;
  }

  _clearError(root);
  _setLoading(root, true);

  try {
    await loginWithOtp(_currentVToken, code);

    const token = await getCdcJwt("firstName,lastName,email");
    loginWithCdcToken(token);

    // Track login in CDC via backend (must complete before navigation)
    await trackLoginServer();

    // Navigate to app
    const returnUrl = getReturnUrl() || "/";
    window.location.href = returnUrl;
  } catch (err) {
    _showError(root, err.message || "Verification failed. Please check the code and try again.");
  } finally {
    _setLoading(root, false);
  }
}
