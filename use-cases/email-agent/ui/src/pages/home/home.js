// Redirect Home page to Email Agent
if (typeof window !== "undefined") {
  const redirectToEmailAgent = () => {
    try {
      if (window.pageRouter && typeof window.pageRouter.navigate === "function") {
        window.pageRouter.navigate("/email-agent");
      } else {
        window.location.replace("/email-agent");
      }
    } catch (e) {
      window.location.href = "/email-agent";
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", redirectToEmailAgent, { once: true });
  } else {
    // Allow a microtask tick so the router can attach if navigating within app
    setTimeout(redirectToEmailAgent, 0);
  }
}
