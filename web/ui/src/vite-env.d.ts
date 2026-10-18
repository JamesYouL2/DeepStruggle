/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** vite.config.ts: the commit the page was built from, "1" for a dirty tree, the stats endpoint. */
  readonly VITE_TS_BUILD_COMMIT?: string;
  readonly VITE_TS_BUILD_DIRTY?: string;
  readonly VITE_TS_STATS_ENDPOINT?: string;
}
