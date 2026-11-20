import { defineConfig } from "@playwright/test";

// Own port and output folder: e2e builds bake test-only settings into the
// bundle, so they must never overwrite ui/dist or clash with a local preview.
const port = Number(process.env.E2E_PORT || 4273);
const outDir = "dist-e2e";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  use: { baseURL: `http://127.0.0.1:${port}` },
  webServer: {
    command: `npm run build -- --outDir ${outDir} && npm run preview -- --outDir ${outDir} --host 127.0.0.1 --port ${port}`,
    port,
    reuseExistingServer: false,
    env: {
      ...process.env,
      VITE_API_BASE_URL: `http://127.0.0.1:${port}`,
      VITE_API_KEY: "e2e-test-key",
    },
  },
});
