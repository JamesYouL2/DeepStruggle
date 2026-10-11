/**
 * The rollout root in the page: the network's top moves, each played out by the network in sampled
 * worlds, then a cautious choice (ai/search/rollout_root.py's docstring). Everything but the
 * network calls is the engine's C++ (bindings/rollout_batch.hpp, exported by the WebAssembly
 * build): the candidates, the worlds, the playouts with their boundary count and suicide guard,
 * the leaves, and the decision with its statistics. This file is the loop that feeds
 * onnxruntime's logits and values in -- the same loop the Python searcher runs with torch.
 *
 * The spec is the leaderboard's: `rollout:k:worlds:horizon:rule` (`rollout` alone is the measured
 * player, 4:16:4:z2). The playouts are honest -- the candidates come from the real position, the
 * values from sampled worlds -- so the search never reads the other side's hand.
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model } from "./model";

export interface RolloutSpec {
  k: number;
  worlds: number;
  horizon: number;
  rule: string;
}

/** The measured player (research/log/E7_search_depth_and_value.md section 4). */
export const DEFAULT_ROLLOUT: RolloutSpec = { k: 4, worlds: 16, horizon: 4, rule: "z2" };

/** "rollout" or "rollout:k:worlds:horizon:rule", an omitted field the default's; null otherwise. */
export function parseRolloutParam(s: string | null): RolloutSpec | null {
  if (!s) return null;
  const parts = s.split(":");
  if (parts[0] !== "rollout" || parts.length > 5) return null;
  const num = (i: number, d: number, min: number) => {
    if (parts.length <= i || parts[i] === "") return d;
    const v = Number(parts[i]);
    return Number.isInteger(v) && v >= min ? v : NaN;
  };
  const spec: RolloutSpec = {
    k: num(1, DEFAULT_ROLLOUT.k, 1), worlds: num(2, DEFAULT_ROLLOUT.worlds, 1),
    horizon: num(3, DEFAULT_ROLLOUT.horizon, 0), rule: parts.length > 4 && parts[4] ? parts[4] : DEFAULT_ROLLOUT.rule,
  };
  if ([spec.k, spec.worlds, spec.horizon].some(v => Number.isNaN(v))) return null;
  if (!/^(argmax|z(\d+(\.\d+)?)?|kl(\d+(\.\d+)?)?)$/.test(spec.rule)) return null;
  return spec;
}

export function rolloutParam(spec: RolloutSpec): string {
  const d = DEFAULT_ROLLOUT;
  if (spec.k === d.k && spec.worlds === d.worlds && spec.horizon === d.horizon && spec.rule === d.rule) return "rollout";
  return `rollout:${spec.k}:${spec.worlds}:${spec.horizon}:${spec.rule}`;
}

/** What the root decided at one position, and the evidence (ts.rollout_decide's record). */
export interface SearchResult {
  kind: "rollout";
  spec: RolloutSpec;
  /** The candidates, the network's own move first, as flat actions. */
  candidates: number[];
  pick: number;
  worlds: number;
  q: number[];
  lead: number[];
  lead_se: number[];
  /** Network rows spent, the root included, and the wall time. */
  rows: number;
  ms: number;
}

/**
 * Run the root on the position `engine` holds. `engine` must not be the one drawing the board
 * while this runs (the batch copies the position, but the forwards interleave with everything
 * else) -- the page gives the worker its own; a page without workers runs it inline between moves.
 * Returns null when no decision is open or the network has a single legal move.
 */
export async function rolloutRoot(engine: WasmEngine, model: Model, spec: RolloutSpec, seed: number): Promise<SearchResult | null> {
  if (model.meta.mergedInfluence) {
    throw new Error("the rollout root plays in the E4 action view; this model decides in the merged (E4.1) view");
  }
  if (engine.isTerminal()) return null;
  const decider = engine.decisionPlayer();
  if (decider === 0) return null;
  const t0 = performance.now();
  const act = engine.actionSize;
  // The root: the network's logits at the real position, from the decider's side.
  const mask = engine.mask(false);
  const root = await model.run(engine.observation(decider), mask, 1);
  let rows = 1;
  const logits = root.logits.slice(0, act);
  const candidates = engine.rolloutCandidates(logits, mask, spec.k);
  if (candidates.length < 2) return null;

  engine.rolloutReset(spec.horizon, model.obsFeatures, seed);
  const width = engine.rolloutObsWidth();
  if (width !== model.meta.obsSize) {
    throw new Error(`the batch writes ${width} observation floats for this feature set; the model reads ${model.meta.obsSize}`);
  }
  engine.rolloutAdd(candidates, spec.worlds);
  for (let ply = 0; !engine.rolloutAllDone(); ply++) {
    if (ply > 400) throw new Error(`${engine.rolloutLive()} playouts still running after 400 plies`);
    const live = engine.rolloutFillLive();
    const out = await model.run(live.obs, live.masks, live.rows);
    rows += live.rows;
    engine.rolloutStep(out.logits, live.rows);
  }
  if (engine.rolloutRefused()) throw new Error(`the engine refused a playout action in ${engine.rolloutRefused()} rows`);
  const leaves = engine.rolloutFillLeaves();
  if (leaves.rows > 0) {
    const out = await model.run(leaves.obs, leaves.masks, leaves.rows);
    rows += leaves.rows;
    engine.rolloutSetLeafValues(out.vWin.slice(0, leaves.rows));
  }
  const d = engine.rolloutDecide(0, candidates, logits, mask, spec.rule);
  return { kind: "rollout", spec, candidates, pick: d.pick, worlds: d.worlds, q: d.q, lead: d.lead, lead_se: d.lead_se,
           rows, ms: Math.round(performance.now() - t0) };
}
