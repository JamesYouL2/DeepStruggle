/**
 * Runs the rollout root for the page: in a worker with its own engine and model when the browser
 * has workers (rollout_worker.ts), else inline on the page's own engine. One search at a time, on
 * either path: a search awaits the model between its batch steps, and two at once would share one
 * engine and one rollout batch. A request still waiting when a newer one arrives is superseded --
 * it rejects without running, as the page has moved on from it; one already running finishes.
 *
 * A belief root (`spec.belief`) needs the game's moves: before each search the searching engine's
 * tracker is brought up to the session's moves -- only the new ones when what it saw is still the
 * game's beginning, else reset and replayed (an undo, a loaded position, a new worker). A replay
 * restores the record of the current turn exactly; the evidence earlier searches gathered about the
 * opponent's style (bindings/belief.hpp) starts again from there.
 */
import { WasmEngine } from "../engine/wasm_engine";
import type { Move } from "../game/session";
import { Model } from "./model";
import { RolloutSpec, SearchResult, rolloutRoot } from "./rollout";
import type { OutMsg } from "./rollout_worker";

export class RolloutRunner {
  private worker: Worker | null = null;
  private workerReady: Promise<void> | null = null;
  private workerModelKey = "";
  private next = 1;
  private pending = new Map<number, { resolve: (r: SearchResult | null) => void; reject: (e: Error) => void }>();
  /** Settles when the search before the newest request has finished. */
  private queue: Promise<void> = Promise.resolve();
  private latest = 0;
  /** The moves the searching engine's belief tracker has observed, and its hands (0: no tracker). */
  private beliefSeen: Move[] = [];
  private beliefHands = 0;
  private beliefOnWorker = false;

  constructor(private readonly engine: WasmEngine, private readonly base: string = import.meta.env.BASE_URL) {}

  /** Give the worker this model (its bytes travel once per model). */
  setModel(model: Model | null, key: string): void {
    this.failAll(new Error("the model changed"));
    this.workerModelKey = "";
    this.beliefSeen = [];
    this.beliefHands = 0;
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
                        actionSize: model.meta.actionSize });
      });
      this.workerReady.catch(() => { this.workerModelKey = ""; });
    } catch {
      this.worker = null;
      this.workerReady = null;
    }
  }

  private loadWaiters: { resolve: () => void; reject: (e: Error) => void } | null = null;

  /**
   * The root's decision for the position `state` (the engine's raw bytes) under `spec`, after the
   * search before it has finished. Rejects without searching if a newer request arrives first.
   */
  async search(state: Uint8Array, model: Model, key: string, spec: RolloutSpec, seed: number,
               moves: Move[] = []): Promise<SearchResult | null> {
    const ticket = ++this.latest;
    const before = this.queue;
    let done!: () => void;
    this.queue = new Promise<void>(resolve => { done = resolve; });
    try {
      await before;
      if (ticket !== this.latest) throw new Error("superseded by a newer search");
      return await this.searchNow(state, model, key, spec, seed, moves);
    } finally {
      done();
    }
  }

  /**
   * The moves the searching engine's tracker has not yet seen, and whether it must start over: what
   * it saw is still the game's beginning only if the game is at least as long and its last seen move
   * is the same object (an undo followed by new moves replaces it).
   */
  private beliefDelta(moves: Move[], hands: number, onWorker: boolean): { reset: boolean; add: Move[] } {
    const seen = this.beliefSeen;
    const prefix = hands === this.beliefHands && onWorker === this.beliefOnWorker && seen.length <= moves.length
      && (seen.length === 0 || moves[seen.length - 1] === seen[seen.length - 1]);
    const add = prefix ? moves.slice(seen.length) : moves.slice();
    this.beliefSeen = moves.slice();
    this.beliefHands = hands;
    this.beliefOnWorker = onWorker;
    return { reset: !prefix, add };
  }

  /** One search, now. Inline when there is no worker for this model: then the page's engine runs it between two renders. */
  private async searchNow(state: Uint8Array, model: Model, key: string, spec: RolloutSpec, seed: number,
                          moves: Move[]): Promise<SearchResult | null> {
    if (this.worker && this.workerReady && this.workerModelKey === key) {
      await this.workerReady;
      if (spec.belief > 0) {
        const { reset, add } = this.beliefDelta(moves, spec.belief, true);
        this.worker.postMessage({ type: "belief", reset, hands: spec.belief, moves: add });
      }
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
      if (spec.belief > 0) {
        const { reset, add } = this.beliefDelta(moves, spec.belief, false);
        if (reset) this.engine.beliefReset(spec.belief, 0);
        for (const m of add) {
          this.engine.restore(m.before);
          this.engine.beliefObserve(m.action);
        }
      }
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
