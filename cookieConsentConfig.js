window.SAP = window.SAP || {};

window.SAP.global = {
  trustArc: {
    domain: "btp-ai-bp.docs.sap",
    privacyPolicyLink: "https://btp-ai-bp.docs.sap/privacy",
    cookieLink: "https://btp-ai-bp.docs.sap/cookies"
  },
  isProd: true
};

// Add click event to document for cookie preference elements
function setupCookiePreferenceLinks() {
  document.addEventListener("click", function (e) {
    if (e.target.matches(".open-cookie-preferences") || e.target.closest(".open-cookie-preferences")) {
      e.preventDefault();
      window?.truste?.eu?.clickListener();
    }
  });
}

// Run when DOM is fully loaded
document.addEventListener("DOMContentLoaded", setupCookiePreferenceLinks);
