import { defineConfig } from "vite";

export default defineConfig({
  server: {
    host: "0.0.0.0",
    port: Number(process.env.VITE_DEV_PORT || 5178),
  },
  preview: {
    port: Number(process.env.VITE_PREVIEW_PORT || 4178),
    allowedHosts: [process.env.VITE_APP_HOST].filter(Boolean),
  },
});
