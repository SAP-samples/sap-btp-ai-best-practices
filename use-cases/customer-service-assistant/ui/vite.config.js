import { defineConfig } from "vite";

export default defineConfig({
  preview: {
    allowedHosts: [process.env.VITE_APP_HOST]
  }
});
