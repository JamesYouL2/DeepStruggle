/**
 * Runs the rollout root for the page: in a worker with its own engine and model when the browser
 * has workers (rollout_worker.ts), else inline on the page's own engine. One search at a time; a
 * newer request supersedes an older one, whose answer is then dropped.
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model } from "./model";
import { RolloutSpec, SearchResult, rolloutRoot } from "./rollout";
import type { OutMsg } from "./rollout_worker";

export class RolloutRunner {
  private worker: Worker | null = null;
  private workerReady: Promise<void> | null = null;
  private workerModelKey = "";
  private next = 1;
  private pending = new Map<number, { resolve: (r: SearchResult | null) => void; reject: (e: Error) => void }>();

  constructor(private readonly engine: WasmEngine, private readonly base: string = import.meta.env.BASE_URL) {}

  /** Give the worker this model (its bytes travel once per model). */
  setModel(model: Model | null, key: string): void {
    this.failAll(new Error("the model changed"));
    this.workerModelKey = "";
    if (!model) return;
    if (typeof Worker === "undefined") return;
    try {
      if (!this.worker) {
        this.worker = new Worker(new URL("./rollout_worker.ts", import.meta.url), { type: "module" });
        this.worker.onmessage = (ev: MessageEvent<OutMsg>) => this.onMessage(ev.data);
        this.worker.onerror = (ev) => this.failAll(new Error(ev.message || "the search worker failed"));
      }
      const w = this.worker;
      this.workerModelKey = key;
      this.workerReady = new Promise<void>((resolve, reject) => {
        this.loadWaiters = { resolve, reject };
        // A copy: the page keeps using its own bytes-backed session.
        w.postMessage({ type: "load", base: this.base, bytes: model.bytes.slice(), source: model.source,
                        obsSize: model.meta.obsSize, actionSize: model.meta.actionSize });
      });
      this.workerReady.catch(() => { this.workerModelKey = ""; });
    } catch {
      this.worker = null;
      this.workerReady = null;
    }
  }

  private loadWaiters: { resolve: () => void; reject: (e: Error) => void } | null = null;

  /**
   * The root's decision for the position `state` (the engine's raw bytes) under `spec`. Inline
   * when there is no worker for this model: then the page's engine runs it between two renders.
   */
  async search(state: Uint8Array, model: Model, key: string, spec: RolloutSpec, seed: number): Promise<SearchResult | null> {
    if (this.worker && this.workerReady && this.workerModelKey === key) {
      await this.workerReady;
      const id = this.next++;
      return new Promise<SearchResult | null>((resolve, reject) => {
        this.pending.set(id, { resolve, reject });
        this.worker!.postMessage({ type: "search", id, state, spec, seed });
      });
    }
    // Inline: on the page's engine, which holds the live position. The batch copies the position
    // on add and never touches it again, so the board is intact when this returns.
    const keep = this.engine.snapshot();
    try {
      this.engine.restore(state);
      return await rolloutRoot(this.engine, model, spec, seed);
    } finally {
      this.engine.restore(keep);
    }
  }

  private onMessage(m: OutMsg): void {
    if (m.type === "loaded") {
      this.loadWaiters?.resolve();
      this.loadWaiters = null;
      return;
    }
    if (m.type === "error" && m.id === undefined) {
      this.loadWaiters?.reject(new Error(m.message));
      this.loadWaiters = null;
      this.failAll(new Error(m.message));
      return;
    }
    const p = m.id !== undefined ? this.pending.get(m.id) : undefined;
    if (!p || m.id === undefined) return;
    this.pending.delete(m.id);
    if (m.type === "result") p.resolve(m.result);
    else p.reject(new Error(m.message));
  }

  private failAll(e: Error): void {
    for (const p of this.pending.values()) p.reject(e);
    this.pending.clear();
  }
}
