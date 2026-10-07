/**
 * The models this browser has downloaded from Hugging Face, kept so that opening the page again
 * does not fetch the same file again (a model is ~12 MB; the browser's HTTP cache never hits,
 * because a `resolve` URL redirects to a freshly signed CDN address every time).
 *
 * A file is kept under its content -- the LFS sha256 Hugging Face reports for it -- not under its
 * URL: a link names a branch, and `main` today is not `main` tomorrow, so a cache keyed by URL
 * would quietly keep analysing with a model the repo has since replaced. The sha256 costs one
 * small `paths-info` request per load, and the downloaded bytes are hashed before they are kept,
 * so a file that changed between that request and the download is used but never stored under
 * the wrong name.
 *
 * The Cache API exists only in a secure context (https, localhost); elsewhere every load is a
 * download, as before.
 */

const CACHE_NAME = "ts-models-v1";
/** Each model is ~12 MB; the oldest stored go first beyond this many. */
const MAX_MODELS = 8;
/** Cache keys must be http(s) URLs; this one is never fetched. */
const KEY_PREFIX = "https://model-cache.invalid/sha256/";

function cacheApi(): CacheStorage | null {
  return typeof caches !== "undefined" && typeof crypto !== "undefined" && crypto.subtle ? caches : null;
}

/**
 * The LFS sha256 of a file in a Hugging Face model repo, or null if the repo does not say (a
 * file stored in git itself, or a missing one -- the download then reports the latter).
 */
export async function hfFileSha256(repo: string, revision: string, path: string): Promise<string | null> {
  // A form body keeps this a CORS "simple" request: no preflight.
  const res = await fetch(`https://huggingface.co/api/models/${repo}/paths-info/${encodeURIComponent(revision)}`, {
    method: "POST",
    body: new URLSearchParams({ paths: path, expand: "false" }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const entries: Array<{ path: string; lfs?: { oid?: string } }> = await res.json();
  const oid = entries.find((e) => e.path === path)?.lfs?.oid;
  return oid && /^[0-9a-f]{64}$/.test(oid) ? oid : null;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
  return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function cachedModel(sha256: string): Promise<Uint8Array | null> {
  const api = cacheApi();
  if (!api) return null;
  const hit = await api.match(KEY_PREFIX + sha256, { cacheName: CACHE_NAME });
  return hit ? new Uint8Array(await hit.arrayBuffer()) : null;
}

/** Keep `bytes` if they are the file whose sha256 is `sha256`; returns whether they were kept. */
export async function storeModel(sha256: string, bytes: Uint8Array): Promise<boolean> {
  const api = cacheApi();
  if (!api || (await sha256Hex(bytes)) !== sha256) return false;
  const cache = await api.open(CACHE_NAME);
  await cache.put(KEY_PREFIX + sha256, new Response(bytes as Uint8Array<ArrayBuffer>));
  // keys() lists in insertion order, so the front is the oldest download.
  const keys = await cache.keys();
  for (const k of keys.slice(0, Math.max(0, keys.length - MAX_MODELS))) await cache.delete(k);
  return true;
}

/** The number of models kept; 0 where there is no Cache API. Asking creates no cache. */
export async function cachedModelCount(): Promise<number> {
  const api = cacheApi();
  if (!api || !(await api.has(CACHE_NAME))) return 0;
  return (await (await api.open(CACHE_NAME)).keys()).length;
}

export async function clearModelCache(): Promise<void> {
  await cacheApi()?.delete(CACHE_NAME);
}
