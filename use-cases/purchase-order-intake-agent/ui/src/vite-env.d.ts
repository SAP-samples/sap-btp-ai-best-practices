/**
 * Typings for the environment variables exposed by Vite (`import.meta.env`).
 * Values come from `ui/.env` locally and from manifest.yaml `env:` on Cloud Foundry.
 */
interface ImportMetaEnv {
  /** Base URL of the FastAPI backend, e.g. http://127.0.0.1:8000 */
  readonly VITE_API_BASE_URL?: string;
  /** Shared API key sent in the `X-API-Key` header; must match API_KEY in api/.env */
  readonly VITE_API_KEY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
