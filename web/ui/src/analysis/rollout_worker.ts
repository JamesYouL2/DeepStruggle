/**
 * The rollout root off the main thread: this worker holds its own engine instance and its own
 * onnxruntime session on the same model, takes a position as raw state bytes, and answers with
 * the decision. The page stays responsive while a decision's forwards run (rollout_runner.ts).
 *
 *   { type: "load", base, bytes, source, obsSize, actionSize }  -> { type: "loaded" } | { type: "error" }
 *   { type: "search", id, state, spec, seed }                   -> { type: "result", id, result } | { type: "error", id }
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model, ModelSource } from "./model";
import { RolloutSpec, SearchResult, rolloutRoot } from "./rollout";

interface LoadMsg { type: "load"; base: string; bytes: Uint8Array; source: ModelSource; obsSize: number; actionSize: number }
interface SearchMsg { type: "search"; id: number; state: Uint8Array; spec: RolloutSpec; seed: number }
type InMsg = LoadMsg | SearchMsg;
export type OutMsg =
  | { type: "loaded"; fingerprint: string }
  | { type: "result"; id: number; result: SearchResult | null }
  | { type: "error"; id?: number; message: string };

let engine: WasmEngine | null = null;
let model: Model | null = null;

const post = (m: OutMsg) => (self as unknown as Worker).postMessage(m);

/**
 * One message at a time. A search awaits the model between its batch steps, and the engine and its
 * rollout batch are this worker's only ones: a second search started meanwhile would restore its
 * own position into them under the first.
 */
let queue: Promise<void> = Promise.resolve();
self.onmessage = (ev: MessageEvent<InMsg>) => {
  queue = queue.then(() => handle(ev.data));
};

async function handle(msg: InMsg): Promise<void> {
  try {
    if (msg.type === "load") {
      engine = await WasmEngine.load(msg.base);
      model = await Model.fromBytes(msg.source, msg.bytes, msg.obsSize, msg.actionSize);
      post({ type: "loaded", fingerprint: engine.fingerprint });
      return;
    }
    if (!engine || !model) throw new Error("the search worker has no model loaded");
    engine.restore(msg.state);
    const result = await rolloutRoot(engine, model, msg.spec, msg.seed);
    post({ type: "result", id: msg.id, result });
  } catch (e) {
    post({ type: "error", id: msg.type === "search" ? msg.id : undefined, message: e instanceof Error ? e.message : String(e) });
  }
}
