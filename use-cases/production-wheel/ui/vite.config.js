import { defineConfig } from "vite";

export default defineConfig({
  preview: {
    allowedHosts: process.env.VITE_APP_HOST ? [process.env.VITE_APP_HOST] : [],
  },
  build: {
    rollupOptions: {
      output: {
        /** Keep heavy chart and UI component libraries independently cacheable. */
        manualChunks(id) {
          if (
            id.includes("node_modules/chart.js") ||
            id.includes("node_modules/@kurkle")
          )
            return "charts";
          if (id.includes("node_modules/@ui5")) return "ui5";
        },
      },
    },
  },
});
