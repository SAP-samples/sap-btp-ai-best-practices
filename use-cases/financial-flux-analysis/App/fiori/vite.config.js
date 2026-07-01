import { defineConfig } from "vite";

export default defineConfig({
  root: "webapp",
  server: {
    port: 5173,
    open: true,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        configure: function (proxy) {
          proxy.on("error", function (err, req, res) {
            console.warn(
              "[vite proxy] Backend not reachable on localhost:8000 —",
              err.message
            );
            if (res && !res.headersSent) {
              res.writeHead(503, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  status: "error",
                  message:
                    "Backend not available. Start the backend server: cd backend && uvicorn api:app --reload --port 8000"
                })
              );
            }
          });
        }
      }
    }
  },
  build: {
    outDir: "../dist",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks: undefined
      }
    }
  }
});