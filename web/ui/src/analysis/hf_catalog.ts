/**
 * The Hugging Face model repo's catalogue: its .onnx files, newest first, and the default model a
 * link that names none opens (models.json, written by tools/publish_hf.py). Plain fetches, no
 * onnxruntime, so the leaderboard page can read the default without the model runtime;
 * analysis/model.ts re-exports all of it for the workbench.
 */

export function hfFileUrl(repo: string, revision: string, path: string): string {
  return `https://huggingface.co/${repo}/resolve/${encodeURIComponent(revision)}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/** Where the page takes its model from when a link names none: the model this repo's
 *  default.json names, or its newest upload if it has none. */
export const DEFAULT_HF_REPO = "mihaild/deepstruggle";
export const DEFAULT_HF_REVISION = "main";

export interface HfModelFile {
  path: string;
  /** ISO-8601 time of the commit that last changed the file; "" if the API did not say. */
  date: string;
}

/**
 * The .onnx files of a public Hugging Face model repo, newest upload first. `expand=true` makes
 * the tree API report each file's last commit; an expanded listing is paged, and the next page
 * is named by the `Link` header, which the API exposes to cross-origin pages.
 */
export async function listHfModels(repo: string, revision: string): Promise<HfModelFile[]> {
  checkRepoName(repo);
  // Without `expand` the tree comes in pages of 1,000 entries; with it (for each file's last
  // commit) in pages of 50 -- 42 requests and ~17 s for one repo of a few published runs. So the
  // tree is listed plain, and only the .onnx files' dates are asked for, 50 paths a request.
  const files: HfModelFile[] = [];
  let url: string | null =
    `https://huggingface.co/api/models/${repo}/tree/${encodeURIComponent(revision)}?recursive=true`;
  while (url) {
    const res: Response = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}${res.status === 401 || res.status === 404 ? " (private or missing repo?)" : ""}`);
    const entries: Array<{ type: string; path: string; lastCommit?: { date?: string } }> = await res.json();
    for (const e of entries) {
      if (e.type === "file" && e.path.endsWith(".onnx")) files.push({ path: e.path, date: e.lastCommit?.date ?? "" });
    }
    url = res.headers.get("Link")?.match(/<([^>]+)>;\s*rel="next"/)?.[1] ?? null;
  }
  const undated = files.filter(f => !f.date);
  for (let k = 0; k < undated.length; k += 50) {
    const body = new URLSearchParams({ expand: "true" });
    for (const f of undated.slice(k, k + 50)) body.append("paths", f.path);
    // A form body keeps this a CORS "simple" request: no preflight.
    const res = await fetch(`https://huggingface.co/api/models/${repo}/paths-info/${encodeURIComponent(revision)}`,
      { method: "POST", body });
    if (!res.ok) throw new Error(`paths-info: HTTP ${res.status}`);
    const info: Array<{ path: string; lastCommit?: { date?: string } }> = await res.json();
    const date = new Map(info.map(i => [i.path, i.lastCommit?.date ?? ""]));
    for (const f of undated.slice(k, k + 50)) f.date = date.get(f.path) ?? "";
  }
  return sortNewestFirst(files);
}

function checkRepoName(repo: string): void {
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) throw new Error("a Hugging Face repo must look like owner/name");
}

/** ISO-8601 times order as strings; files uploaded in one commit fall back to their path. */
function sortNewestFirst(files: HfModelFile[]): HfModelFile[] {
  return files.sort((a, b) => (a.date === b.date ? a.path.localeCompare(b.path) : a.date < b.date ? 1 : -1));
}

/** What the page needs from a repo: its .onnx files, newest first, and the default model. */
export interface HfCatalog {
  files: HfModelFile[];
  /** The model a link that names none opens; null when the repo names none (newest is used). */
  defaultModel: string | null;
}

/**
 * A repo's catalogue. One request when it has `models.json` (written by tools/publish_hf.py on
 * every publish: each .onnx with its date, and the default model); otherwise its tree is listed
 * (listHfModels) and its `default.json` read, the way repos published before the manifest are.
 */
export async function hfCatalog(repo: string, revision: string): Promise<HfCatalog> {
  checkRepoName(repo);
  const res = await fetch(hfFileUrl(repo, revision, "models.json"));
  if (res.status === 404) {
    const [files, defaultModel] = await Promise.all([listHfModels(repo, revision), fetchHfDefault(repo, revision)]);
    return { files, defaultModel };
  }
  if (!res.ok) throw new Error(`models.json: HTTP ${res.status}`);
  let doc: { schema?: unknown; default?: unknown; models?: unknown };
  try {
    doc = await res.json();
  } catch {
    throw new Error(`${repo}'s models.json is not JSON`);
  }
  if (doc.schema !== 1 || !Array.isArray(doc.models)) {
    throw new Error(`${repo}'s models.json is not schema 1 (tools/publish_hf.py index rebuilds it)`);
  }
  const files: HfModelFile[] = [];
  for (const m of doc.models as Array<{ path?: unknown; date?: unknown }>) {
    if (typeof m.path !== "string" || !m.path.endsWith(".onnx")) throw new Error(`${repo}'s models.json lists a non-.onnx entry`);
    files.push({ path: m.path, date: typeof m.date === "string" ? m.date : "" });
  }
  const def = doc.default;
  if (def !== null && def !== undefined && typeof def !== "string") throw new Error(`${repo}'s models.json: default must be a path or null`);
  return { files: sortNewestFirst(files), defaultModel: (def as string | null | undefined) ?? null };
}

export interface HfModelGroup {
  /** The directory, "" for the repo's top level. A published run is one directory. */
  dir: string;
  files: HfModelFile[];
}

/**
 * `files` (newest first, as listHfModels returns them) by directory: a run published whole is a
 * directory with an .onnx beside every snapshot, so a flat list of every file is unreadable after
 * two runs. Groups are ordered by their newest file and keep the newest-first order inside.
 */
export function groupHfModels(files: HfModelFile[]): HfModelGroup[] {
  const groups = new Map<string, HfModelFile[]>();
  for (const f of files) {
    const cut = f.path.lastIndexOf("/");
    const dir = cut < 0 ? "" : f.path.slice(0, cut);
    if (!groups.has(dir)) groups.set(dir, []);
    groups.get(dir)!.push(f);
  }
  return [...groups].map(([dir, fs]) => ({ dir, files: fs }));
}

/**
 * The model the repo itself names as its default: `default.json` at the top level,
 * `{"model": "<path>.onnx"}`. null when the repo has no such file -- the page then takes the newest
 * upload, which is what it did before repos had one.
 */
export async function fetchHfDefault(repo: string, revision: string): Promise<string | null> {
  const res = await fetch(hfFileUrl(repo, revision, "default.json"));
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`default.json: HTTP ${res.status}`);
  let doc: unknown;
  try {
    doc = await res.json();
  } catch {
    throw new Error(`${repo}'s default.json is not JSON`);
  }
  const model = (doc as { model?: unknown } | null)?.model;
  if (typeof model !== "string" || !model.endsWith(".onnx")) {
    throw new Error(`${repo}'s default.json must be {"model": "<path>.onnx"}`);
  }
  return model;
}
