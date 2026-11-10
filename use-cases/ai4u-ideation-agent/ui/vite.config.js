import { defineConfig } from "vite";

const allowedHosts = process.env.VITE_APP_HOST
  ? [process.env.VITE_APP_HOST]
  : undefined;

export default defineConfig({
  server: {
    allowedHosts
  },
  preview: {
    allowedHosts
  }
});
