/**
 * A policy/value network in the page: an ONNX export of a checkpoint (tools/export_onnx.py) run
 * by onnxruntime-web, from one of three sources --
 *
 *   local  the checkpoints tree the local server lists (`/api/local/models`); it exports a
 *          .pt to ONNX on first request and caches it, so any snapshot can be picked
 *   hf     a file in a Hugging Face model repo, fetched from huggingface.co/<repo>/resolve/...
 *          and kept in the browser under its sha256 (model_cache.ts), so reopening the page
 *          does not download it again
 *   file   an .onnx dropped onto the page
 *
 * The file carries its own description (metadata_props, see onnx_meta.ts). The observation
 * width is checked before anything runs: a model handed an observation of another width does not
 * fail, it reads fixed slices and misreads them, which is why the Python side refuses such a
 * checkpoint too (bindings.ts_env.check_obs_width). An engine fingerprint that differs from the
 * page's is shown, not refused: the model was exported next to another engine build, which is
 * worth knowing but not necessarily wrong.
 */
import * as ort from "onnxruntime-web/wasm";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import { onnxMetadata } from "./onnx_meta";
import { cachedModel, hfFileSha256, storeModel } from "./model_cache";

// The runtime is served from this build, not a CDN, and single-threaded: GitHub Pages cannot send
// the cross-origin-isolation headers threads need, and one position is well under a millisecond.
ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
ort.env.wasm.numThreads = 1;

export type ModelSource =
  | { kind: "local"; path: string }
  | { kind: "hf"; repo: string; revision: string; path: string }
  | { kind: "file"; name: string };

/** What a source is called in the address bar; `file` has no address and is not shareable. */
export function sourceToParam(s: ModelSource): string | null {
  if (s.kind === "local") return `local:${s.path}`;
  if (s.kind === "hf") return `hf:${s.repo}@${s.revision}:${s.path}`;
  return null;
}

export function sourceFromParam(v: string): ModelSource | null {
  if (v.startsWith("local:")) return { kind: "local", path: v.slice(6) };
  const hf = v.match(/^hf:([^@:]+)@([^:]+):(.+)$/);
  if (hf) return { kind: "hf", repo: hf[1], revision: hf[2], path: hf[3] };
  // A bare run/snapshot path, as links from the server-side workbench wrote them.
  if (v && !v.includes(":")) return { kind: "local", path: v };
  return null;
}

export function sourceLabel(s: ModelSource): string {
  if (s.kind === "local") return s.path;
  if (s.kind === "hf") return `${s.repo}/${s.path}`;
  return s.name;
}

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

export interface ModelMeta {
  label: string;
  mergedInfluence: boolean;
  obsSize: number;
  actionSize: number;
  engineFingerprint: string;
  checkpoint: string;
  raw: Record<string, string>;
}

async function download(source: ModelSource, url: string): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`could not fetch ${sourceLabel(source)}: HTTP ${res.status} ${detail.slice(0, 300)}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

export interface Forward {
  logits: Float32Array;   // rows * actionSize
  vWin: Float32Array;     // rows
  vVp: Float32Array;      // rows
}

export class Model {
  private constructor(
    readonly source: ModelSource,
    readonly meta: ModelMeta,
    private readonly session: ort.InferenceSession,
    /** The bytes came from the browser's model cache rather than the network. */
    readonly fromCache: boolean,
  ) {}

  /**
   * Build from the file's bytes. `obsSize`/`actionSize` are the engine's; a model for another
   * width is refused here, with the reason, before it can misread a single observation.
   */
  static async fromBytes(source: ModelSource, bytes: Uint8Array, obsSize: number, actionSize: number,
                         fromCache = false): Promise<Model> {
    let raw: Record<string, string>;
    try {
      raw = onnxMetadata(bytes);
    } catch (e) {
      throw new Error(`${sourceLabel(source)} is not an ONNX model (${e instanceof Error ? e.message : e})`);
    }
    if (raw["ts.format"] !== "ts-onnx-v1") {
      throw new Error(`${sourceLabel(source)} was not exported by tools/export_onnx.py (no ts.format metadata)`);
    }
    const meta: ModelMeta = {
      label: raw["ts.label"] || sourceLabel(source),
      mergedInfluence: raw["ts.merged_influence"] === "true",
      obsSize: Number(raw["ts.obs_size"]),
      actionSize: Number(raw["ts.action_size"]),
      engineFingerprint: raw["ts.engine_fingerprint"] || "",
      checkpoint: raw["ts.checkpoint"] || "",
      raw,
    };
    if (meta.obsSize !== obsSize || meta.actionSize !== actionSize) {
      throw new Error(`${meta.label} reads ${meta.obsSize} observation floats and ${meta.actionSize} actions; `
        + `this engine has ${obsSize} and ${actionSize}. It was trained on another layout and would misread every position.`);
    }
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    return new Model(source, meta, session, fromCache);
  }

  static async fetch(source: ModelSource, obsSize: number, actionSize: number, fileBytes?: Uint8Array): Promise<Model> {
    if (source.kind === "file") {
      if (!fileBytes) throw new Error("a dropped model needs its bytes");
      return Model.fromBytes(source, fileBytes, obsSize, actionSize);
    }
    if (source.kind === "local") {
      // The local server exports and keeps the ONNX itself, and serves it from this machine.
      const url = `/api/local/models/onnx?path=${encodeURIComponent(source.path)}`;
      return Model.fromBytes(source, await download(source, url), obsSize, actionSize);
    }
    // The cache only saves a download: if Hugging Face will not say what the file is, or the
    // browser will not keep it, the model is downloaded and used all the same.
    const sha256 = await hfFileSha256(source.repo, source.revision, source.path).catch(() => null);
    const hit = sha256 ? await cachedModel(sha256).catch(() => null) : null;
    if (hit) return Model.fromBytes(source, hit, obsSize, actionSize, true);
    const bytes = await download(source, hfFileUrl(source.repo, source.revision, source.path));
    const model = await Model.fromBytes(source, bytes, obsSize, actionSize);
    // Only a model that loaded is kept: one refused here would be refused again from the cache.
    if (sha256) await storeModel(sha256, bytes).catch(() => false);
    return model;
  }

  /** One forward over `rows` positions: observations and masks row-major. */
  async run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward> {
    const out = await this.session.run({
      obs: new ort.Tensor("float32", obs, [rows, this.meta.obsSize]),
      mask: new ort.Tensor("uint8", masks, [rows, this.meta.actionSize]),
    });
    return {
      logits: out.logits.data as Float32Array,
      vWin: out.v_win.data as Float32Array,
      vVp: out.v_vp.data as Float32Array,
    };
  }
}
