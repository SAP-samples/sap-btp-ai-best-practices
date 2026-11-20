# Local UI build for Cloud Foundry

## Purpose

`deploy.sh` compiles the Vite UI on the developer machine before `cf push`.
Cloud Foundry receives `ui/dist` and serves it through the Staticfile buildpack,
so the UI app does not need Node.js or run Vite during staging or startup.

## Inputs and output

- Inputs: installed dependencies in `ui/node_modules`, credentials in
  `api/.env`, and the deployed `API_BASE_URL` in `manifest.yaml`.
- `deploy.sh` generates one API key, passes it to the API app as `api_key`, and
  passes the same value to Vite as `VITE_API_KEY`. Vite also receives
  `VITE_API_BASE_URL` from the manifest. These values are embedded in the
  browser bundle; the API key is browser-visible.
- Output: `ui/dist/index.html`, assets, and `ui/dist/Staticfile`. Vite copies
  `ui/public/Staticfile` into the output. `pushstate: enabled` lets direct
  requests to client routes such as `/ecns/123` load the SPA entry point.
- The PDF.js module worker is bundled through Vite's `?worker&url` import with
  `worker.format: "es"`. This emits a hashed `.js` asset that the Staticfile
  server serves with a JavaScript MIME type. A raw `.mjs?url` asset can receive
  `application/octet-stream` and fail Chrome's module MIME check. PDF parsing
  remains in a separate worker, loaded only when a drawing is viewed.

## Related files

- `deploy.sh`: builds locally, checks the output, then runs `cf push`.
- `manifest.yaml`: points the UI app to `ui/dist`, selects
  `staticfile_buildpack`, and resets any previously saved Node start command.
- `ui/public/Staticfile`: configures Staticfile routing.
- `ui/src/services/api.js`: reads the Vite values embedded at build time.

The key build step is:

```bash
VITE_API_BASE_URL="$UI_API_BASE_URL" VITE_API_KEY="$API_KEY" npm run build --prefix ui
```

## Run and verify

```bash
npm install --prefix ui
./deploy.sh
```

`./deploy.sh` is the user-run deployment command. For a local-only check, run
`bash -n deploy.sh` and `bash tests/test_deploy_ui_build.sh`; the latter uses
stub `npm` and `cf` commands and never contacts Cloud Foundry.

## E2E builds never touch `ui/dist`

The Playwright suite (`npm --prefix ui run test:e2e`) builds the UI with
test-only settings (`VITE_API_BASE_URL` pointing at the preview itself and the
key `e2e-test-key`). It writes that build to `ui/dist-e2e` (git-ignored, never
deployed) and serves it on port 4273 (override with `E2E_PORT`). Earlier it
wrote to `ui/dist` on port 4173, so running the e2e suite silently replaced the
local build: a running `vite preview` then called itself instead of the API and
showed `Unexpected token '<', "<!doctype "... is not valid JSON`. If you see
that error, rebuild with `npm --prefix ui run build`.
