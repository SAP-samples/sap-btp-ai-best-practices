/**
 * Vite configuration.
 *
 * `@vitejs/plugin-react` enables JSX/TSX transforms and React Fast Refresh in dev.
 * Production serves the static `dist/` output via the CF staticfile buildpack.
 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()]
});
