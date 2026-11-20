import { defineConfig } from "vite";

export default defineConfig({
  worker: {
    format: "es"
  },
  preview: {
    allowedHosts: [process.env.VITE_APP_HOST]
  }
});
