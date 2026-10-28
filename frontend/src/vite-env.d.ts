/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "1" serves every request from the in-browser mock API (src/mocks). */
  readonly VITE_API_MOCK?: string;
  /** API base path, default "/api/v1" (proxied to Django in dev). */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
