/**
 * The rollout root in the page: the network's top moves, each played out by the network in sampled
 * worlds, then a cautious choice (ai/search/rollout_root.py's docstring). Everything but the
 * network calls is the engine's C++ (bindings/rollout_batch.hpp, exported by the WebAssembly
 * build): the candidates, the worlds, the playouts with their boundary count and suicide guard,
 * the leaves, and the decision with its statistics. This file is the loop that feeds
 * onnxruntime's logits and values in -- the same loop the Python searcher runs with torch.
 *
 * The spec is the leaderboard's: `rollout:k:worlds:horizon:rule[:stop][:leaf][:keyed][:belief[N]]`
 * (`rollout` alone is the best player there, 4:32:4:z2 with stop=earlier, leaf=own). `keyed` ties
 * every draw to its situation (the leaderboard's chance=keyed); `belief` deals the worlds from the
 * policy-likelihood belief over the opponent's hand (bindings/belief.hpp, N hands, 2,048 by default --
 * the leaderboard's belief=N), the same C++ as Python's. The playouts are honest -- the candidates come
 * from the real position, the values from sampled worlds -- so the search never reads the other side's
 * hand; the belief reads only the moves both players saw.
 */
import { WasmEngine } from "../engine/wasm_engine";
import { Model } from "./model";

export interface RolloutSpec {
  k: number;
  worlds: number;
  horizon: number;
  rule: string;
  /** Where the playouts stop (the engine's rollout::Stop, by name). */
  stop: string;
  /** Whose side a leaf is read from (the engine's rollout::Leaf, by name). */
  leaf: string;
  /** Each draw tied to its situation (bindings/keyed_chance.hpp). */
  keyed: boolean;
  /** Hands the belief weighs (bindings/belief.hpp); 0 = uniform redeals. */
  belief: number;
}

/** The engine's codes, in the order of ai/search/rollout_root.py's STOPS and LEAF_SIDES. */
export const STOPS = ["boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal"] as const;
export const LEAVES = ["mover", "decider", "own"] as const;

/** The best player on the leaderboard: rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own). */
export const DEFAULT_ROLLOUT: RolloutSpec = { k: 4, worlds: 32, horizon: 4, rule: "z2", stop: "earlier", leaf: "own",
                                             keyed: false, belief: 0 };
/** ai/search/rollout_root.py's `belief` with no count. */
export const DEFAULT_BELIEF_HANDS = 2048;

/** "rollout" or "rollout:k:worlds:horizon:rule[:stop][:leaf][:keyed][:belief[N]]", an omitted field the
 *  default's; null otherwise. */
export function parseRolloutParam(s: string | null): RolloutSpec | null {
  if (!s) return null;
  const parts = s.split(":");
  if (parts[0] !== "rollout" || parts.length > 9) return null;
  const num = (i: number, d: number, min: number) => {
    if (parts.length <= i || parts[i] === "") return d;
    const v = Number(parts[i]);
    return Number.isInteger(v) && v >= min ? v : NaN;
  };
  const spec: RolloutSpec = {
    k: num(1, DEFAULT_ROLLOUT.k, 1), worlds: num(2, DEFAULT_ROLLOUT.worlds, 1),
    horizon: num(3, DEFAULT_ROLLOUT.horizon, 0), rule: parts.length > 4 && parts[4] ? parts[4] : DEFAULT_ROLLOUT.rule,
    stop: DEFAULT_ROLLOUT.stop, leaf: DEFAULT_ROLLOUT.leaf, keyed: false, belief: 0,
  };
  // The trailing fields are told apart by value, as the Python spec does.
  for (const p of parts.slice(5)) {
    if (!p) continue;
    const b = /^belief(\d*)$/.exec(p);
    if ((STOPS as readonly string[]).includes(p)) spec.stop = p;
    else if ((LEAVES as readonly string[]).includes(p)) spec.leaf = p;
    else if (p === "keyed") spec.keyed = true;
    else if (b) spec.belief = b[1] ? Number(b[1]) : DEFAULT_BELIEF_HANDS;
    else return null;
  }
  if ([spec.k, spec.worlds, spec.horizon].some(v => Number.isNaN(v))) return null;
  if (!/^(argmax|z(\d+(\.\d+)?)?|kl(\d+(\.\d+)?)?)$/.test(spec.rule)) return null;
  return spec;
}

export function rolloutParam(spec: RolloutSpec): string {
  const d = DEFAULT_ROLLOUT;
  if (spec.k === d.k && spec.worlds === d.worlds && spec.horizon === d.horizon && spec.rule === d.rule
      && spec.stop === d.stop && spec.leaf === d.leaf && !spec.keyed && !spec.belief) return "rollout";
  return `rollout:${spec.k}:${spec.worlds}:${spec.horizon}:${spec.rule}:${spec.stop}:${spec.leaf}`
    + (spec.keyed ? ":keyed" : "") + (spec.belief ? `:belief${spec.belief}` : "");
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
  /** With a belief: the temperature weights it settled on for this opponent (bindings/belief.hpp). */
  temperature_weights?: number[];
}

/** Hands evaluated per network call while the belief weighs them. */
const BELIEF_CHUNK = 256;

/**
 * Weigh the belief's hands for the position `engine` holds (prepared): every recorded decision of the
 * opponent's under every hand, through the network, the move's log-probability at each temperature
 * handed back to the C++. Returns the network rows spent.
 */
async function weighBelief(engine: WasmEngine, model: Model, width: number): Promise<number> {
  const D = engine.beliefDecisions();
  if (D === 0) return 0;
  const temps = engine.beliefTemperatures();
  const N = engine.beliefParticles();
  const act = engine.actionSize;
  const lp = new Float32Array(temps.length * D * N);
  for (let d = 0; d < D; d++) {
    const a = engine.beliefAction(d);
    for (let lo = 0; lo < N; lo += BELIEF_CHUNK) {
      const n = Math.min(BELIEF_CHUNK, N - lo);
      const rows = engine.beliefRows(d, lo, n, model.obsFeatures, width);
      const out = await model.run(rows.obs, rows.masks, n);
      for (let k = 0; k < n; k++) {
        const base = k * act;
        for (let t = 0; t < temps.length; t++) {
          // The move's masked log-softmax at temperature t, as ai/search/belief.device_log_probs.
          let mx = -Infinity;
          for (let i = 0; i < act; i++) if (rows.masks[base + i]) mx = Math.max(mx, out.logits[base + i] / temps[t]);
          let s = 0;
          for (let i = 0; i < act; i++) if (rows.masks[base + i]) s += Math.exp(out.logits[base + i] / temps[t] - mx);
          const v = rows.masks[base + a] ? out.logits[base + a] / temps[t] - mx - Math.log(s) : -Infinity;
          lp[(t * D + d) * N + lo + k] = Number.isFinite(v) ? v : -1e9;
        }
      }
    }
  }
  engine.beliefSetLogLikelihoods(lp);
  return D * N;
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

  engine.rolloutReset(spec.horizon, model.obsFeatures, seed, true, true,
                      STOPS.indexOf(spec.stop as typeof STOPS[number]), LEAVES.indexOf(spec.leaf as typeof LEAVES[number]),
                      spec.keyed);
  const width = engine.rolloutObsWidth();
  if (width !== model.meta.obsSize) {
    throw new Error(`the batch writes ${width} observation floats for this feature set; the model reads ${model.meta.obsSize}`);
  }
  let temperatureWeights: number[] | undefined;
  if (spec.belief > 0 && engine.beliefParticles() === spec.belief && engine.beliefPrepare()) {
    rows += await weighBelief(engine, model, width);
    temperatureWeights = engine.beliefTemperatureWeights();
    engine.rolloutAddBelief(candidates, spec.worlds);
  } else {
    if (spec.belief > 0 && engine.beliefParticles() !== spec.belief) {
      throw new Error(`the belief tracker holds ${engine.beliefParticles()} hands; the spec asks for ${spec.belief}`);
    }
    engine.rolloutAdd(candidates, spec.worlds);
  }
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
           rows, ms: Math.round(performance.now() - t0), temperature_weights: temperatureWeights };
}
