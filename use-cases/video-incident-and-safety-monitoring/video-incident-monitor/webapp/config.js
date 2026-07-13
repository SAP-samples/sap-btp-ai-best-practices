/**
 * Runtime configuration for Video Incident & Safety Monitoring frontend.
 * Decides backend base URL based on environment:
 * - Local development: http://localhost:5000
 * - Cloud Foundry: defaults to video-incident-monitor-backend route
 * - Optional override via meta tag: <meta name="backend-base-url" content="https://...">
 */
(function () {
    function detectBackendBaseUrl() {
        try {
            var meta = document.querySelector('meta[name="backend-base-url"]');
            if (meta && meta.content) {
                return meta.content;
            }
        } catch (e) {
            // ignore
        }
        var host = window.location.hostname;
        var protocol = window.location.protocol || "https:";
        // If running on CF (cfapps domain), use the backend app route by default
        if (host.indexOf("cfapps") !== -1) {
            return protocol + "//video-incident-monitor-backend.cfapps.eu10-004.hana.ondemand.com";
        }
        // Local development
        return "http://localhost:5000";
    }

    window.AI_CORE_VIDEO_CONFIG = {
        backendBaseUrl: detectBackendBaseUrl(),
        // Optional API key read from meta tag: <meta name="backend-api-key" content="...">
        apiKey: (function () {
            try {
                // 1) URL param override (?backend-api-key=... or ?api_key=...)
                var params = new URLSearchParams(window.location.search);
                var k = params.get('backend-api-key') || params.get('api_key');
                if (k) {
                    try { sessionStorage.setItem('backend-api-key', k); } catch (e) { }
                    return k;
                }
                // 2) Session storage (persist across reloads)
                var stored = null;
                try { stored = sessionStorage.getItem('backend-api-key'); } catch (e) { }
                if (stored) { return stored; }
                // 3) Meta tag fallback
                var metaKey = document.querySelector('meta[name="backend-api-key"]');
                return metaKey && metaKey.content ? metaKey.content : null;
            } catch (e) {
                return null;
            }
        })()
    };

    // Optional: expose a helper for debugging
    console.log("[Video Incident Monitor] Backend Base URL:", window.AI_CORE_VIDEO_CONFIG.backendBaseUrl);
})();
