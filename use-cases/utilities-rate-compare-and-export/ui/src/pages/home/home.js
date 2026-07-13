// Redirect Home page to Email Agent
if (typeof window !== "undefined") {
  const redirectToEmailAgent = () => {
    try {
      if (window.pageRouter && typeof window.pageRouter.navigate === "function") {
        window.pageRouter.navigate("/utilities-rate-compare-and-export");
      } else {
        window.location.replace("/utilities-rate-compare-and-export");
      }
    } catch (e) {
      window.location.href = "/utilities-rate-compare-and-export";
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", redirectToEmailAgent, { once: true });
  } else {
    // Allow a microtask tick so the router can attach if navigating within app
    setTimeout(redirectToEmailAgent, 0);
  }
}
