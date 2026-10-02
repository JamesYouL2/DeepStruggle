/**
 * Paired playouts in the page: how a few candidate moves at the open decision fare when the model
 * plays the game out from each, and the win-rate difference against the model's own choice.
 *
 * The method of the fork's playout probes (ai/eval/branch_oracle.py, ai/eval/choice_oracle.py),
 * run for ONE position:
 *
 * - **Paired.** Pair k samples one world -- the cards the mover cannot see are redealt between the
 *   opponent's hand and the deck (ai/search/dmcts.py `determinize`, including its forgetting of an
 *   opponent's chosen-but-unrevealed headline) and the engine's dice seed is reset -- and EVERY
 *   candidate is played from that same world with the same dice. The candidates differ only in the
 *   move, so a difference between them is measured with far less noise than from independent games.
 * - **No suicide in the mover's round.** From the candidate on, while the mover is still deciding,
 *   an option that loses the game on the spot is never taken while another exists: a forced event's
 *   inner choices are often ones the model never met in training (a Star Wars retrieval of a DEFCON
 *   card at DEFCON 2). The guard ends at the first decision by the other side.
 * - **The model plays both sides**, greedily (its most likely move), in its own action view.
 *
 * So a result is "how this move fares as the model follows it up": a move whose value lies in a
 * plan the model does not have reads worse than it is.
 *
 * Games run side by side -- every (pair, candidate) game is a GameState snapshot, swapped into the
 * page's single engine for each step -- so the network is called on batches. Decisions with one
 * legal move are taken without calling it. The engine is borrowed: the caller's position is
 * restored before `run` returns, and the caller must not act on the engine while it runs.
 *
 * One difference from the Python probes: the page's engine API does not expose
 * Engine::auto_advance_step, so forced decisions are stepped here one by one (free: no network
 * call). The games are the same games; the numbers are not bit-identical to the CLI's because the
 * redeal draws from a different random stream.
 */
import { WasmEngine, PLAYER_US } from "../engine/wasm_engine";
import { Forward } from "./model";

/** The model surface the runner needs; `Model` satisfies it, and tests inject a stub. */
export interface PlayoutModel {
  run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward>;
}

export interface PlayoutConfig {
  /** Pairs per candidate. */
  pairs: number;
  /** Pairs played per wave; results are reported after each wave. */
  wave: number;
  /** Positions per network call at most. */
  batch: number;
  /** Seed of the redeals and dice: the same seed replays the same games. */
  seed: number;
  /** A game still running after this many decisions is scored as a draw and counted. */
  maxDecisions: number;
}

export const DEFAULT_PLAYOUTS: PlayoutConfig = { pairs: 32, wave: 8, batch: 256, seed: 20261002, maxDecisions: 3000 };

export interface CandidateResult {
  idx: number;
  /** Mean score for the mover (1 win, ½ draw, 0 loss) over the pairs played. */
  score: number;
  /** This candidate minus the reference candidate (the first), paired by world; 0 for the reference. */
  diff: number;
  /** Standard error of `diff` over pairs; NaN below two pairs. */
  diffSe: number;
}

export interface PlayoutProgress {
  pairs: number;
  games: number;
  ms: number;
  /** Games stopped at `maxDecisions` (scored ½). Should stay 0. */
  capped: number;
  rows: CandidateResult[];
  done: boolean;
}

/** CardLocation (engine/include/ts/types.hpp) as the save JSON carries it. */
const LOC_DRAW_DECK = 1;
const LOC_HAND_US_UNKNOWN = 2;
const LOC_HAND_USSR_UNKNOWN = 4;
const PHASE_HEADLINE = 1;
const CARD_IDS_END = 110;

interface SaveShape {
  card_locations: number[];
  rng_state: number;
  current_phase: number;
  headline_stage: number;
  headline_us_card: number;
  headline_ussr_card: number;
  [key: string]: unknown;
}

/** Deterministic 32-bit RNG (mulberry32), so a run is reproducible from its seed. */
export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = (seed >>> 0) || 0x9e3779b9;
  }
  next(): number {
    this.s = (this.s + 0x6d2b79f5) >>> 0;
    let t = this.s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  /** A value that survives a JSON number round trip exactly (below 2^53). */
  int53(): number {
    return Math.floor(this.next() * 4294967296) * 2097152 + Math.floor(this.next() * 2097152);
  }
  shuffle<T>(a: T[]): void {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      const t = a[i];
      a[i] = a[j];
      a[j] = t;
    }
  }
}

/**
 * dmcts.py's determinize on the engine's current position, from `me`'s side: the opponent's unseen
 * cards and the deck reshuffled with counts kept, every card `me` has seen left where it is, an
 * opponent headline chosen but not yet revealed forgotten, and a fresh dice seed. Through the save
 * JSON (the page engine's state-editing API): `rng_state` loses low bits in the round trip, which
 * is harmless because it is replaced by the world's own seed.
 */
export function sampleWorld(engine: WasmEngine, me: number, rng: Rng): Uint8Array {
  const save = JSON.parse(engine.saveJson()) as SaveShape;
  const oppUnknown = me === PLAYER_US ? LOC_HAND_USSR_UNKNOWN : LOC_HAND_US_UNKNOWN;
  if (save.current_phase === PHASE_HEADLINE && save.headline_stage === 0) {
    if (me !== PLAYER_US && save.headline_us_card) save.headline_us_card = 0;
    else if (me === PLAYER_US && save.headline_ussr_card) save.headline_ussr_card = 0;
  }
  const locs = save.card_locations;
  const hand: number[] = [];
  const deck: number[] = [];
  for (let cid = 1; cid <= CARD_IDS_END; cid++) {
    if (locs[cid] === oppUnknown) hand.push(cid);
    else if (locs[cid] === LOC_DRAW_DECK) deck.push(cid);
  }
  const pool = hand.concat(deck);
  rng.shuffle(pool);
  for (let i = 0; i < pool.length; i++) locs[pool[i]] = i < hand.length ? oppUnknown : LOC_DRAW_DECK;
  save.rng_state = rng.int53();
  engine.loadSaveJson(JSON.stringify(save));
  return engine.snapshot();
}

/**
 * Apply a flat action in the model's action view on the engine's current position, as
 * GameSession.applyFlat does (a composed E4.1 "Ops -> Influence" action is its two E4 steps), but
 * unlogged. Returns the refusal reason, or null.
 */
export function applyFlat(engine: WasmEngine, idx: number, merged: boolean): string | null {
  if (merged && engine.isMergedInfluenceAction(idx)) {
    const L = engine.layout;
    const before = engine.snapshot();
    const why = engine.step(engine.decodeFlat(L.ops_influence_index));
    if (why) return why;
    if (engine.isTerminal()) return null;
    const second = idx === L.ops_influence_index ? L.confirm_done_index : idx;
    const why2 = engine.step(engine.decodeFlat(second));
    if (why2) engine.restore(before);
    return why2;
  }
  return engine.step(engine.decodeFlat(idx));
}

interface Game {
  state: Uint8Array;
  pair: number;
  cand: number;
  /** +1 US / -1 USSR: whose result this game scores. */
  mover: number;
  guard: boolean;
  decisions: number;
  score: number | null;
  capped: boolean;
}

/**
 * Let the page run (paint progress, take a Stop click, refuse a move made meanwhile). An ONNX
 * Runtime Web forward on one thread resolves without leaving the microtask queue, so without this a
 * whole run is one task: the page freezes and every key pressed during it lands after it ends.
 * Time-sliced, so the cost is a few per cent of a run rather than a timer per network call.
 */
const SLICE_MS = 30;
let lastYield = 0;
async function yieldToPage(): Promise<void> {
  const now = Date.now();
  if (now - lastYield < SLICE_MS) return;
  await new Promise<void>(resolve => setTimeout(resolve, 0));
  lastYield = Date.now();
}

export class PairedPlayouts {
  constructor(
    private readonly engine: WasmEngine,
    private readonly model: PlayoutModel,
    private readonly merged: boolean,
    private readonly config: PlayoutConfig = DEFAULT_PLAYOUTS,
  ) {}

  /**
   * Play every candidate (flat actions, legal at the engine's current decision; the first is the
   * reference the others are compared with) `config.pairs` times. `onProgress` is called after each
   * wave; `shouldStop` is polled between steps and ends the run early (the result then covers the
   * waves completed). The engine is handed back exactly as it was.
   */
  async run(candidates: number[], onProgress?: (p: PlayoutProgress) => void,
            shouldStop: () => boolean = () => false): Promise<PlayoutProgress> {
    const engine = this.engine;
    const cfg = this.config;
    const t0 = Date.now();
    const real = engine.snapshot();
    const scores: number[][] = candidates.map(() => []);   // [cand][pair]
    let capped = 0;
    const result = (done: boolean): PlayoutProgress => ({
      pairs: Math.min(...scores.map(s => s.length)), games: scores.reduce((n, s) => n + s.length, 0),
      ms: Date.now() - t0, capped, rows: summarize(candidates, scores), done,
    });
    try {
      if (engine.isTerminal() || candidates.length === 0) return result(true);
      const mover = engine.decisionPlayer();
      if (mover === 0) throw new Error("no decision is open");
      const mask = engine.mask(this.merged);
      for (const c of candidates) {
        if (!(c >= 0 && c < mask.length && mask[c])) throw new Error(`candidate ${c} is not legal here`);
      }
      const rng = new Rng(cfg.seed);
      for (let lo = 0; lo < cfg.pairs; lo += cfg.wave) {
        if (shouldStop()) break;
        const games: Game[] = [];
        for (let k = lo; k < Math.min(cfg.pairs, lo + cfg.wave); k++) {
          engine.restore(real);
          const world = sampleWorld(engine, mover, rng);
          for (let ci = 0; ci < candidates.length; ci++) {
            engine.restore(world);
            const why = applyFlat(engine, candidates[ci], this.merged);
            if (why) throw new Error(`candidate ${candidates[ci]} was refused in a sampled world: ${why}`);
            games.push({ state: engine.snapshot(), pair: k, cand: ci, mover, guard: true, decisions: 0,
                         score: null, capped: false });
          }
        }
        const finished = await this.playOut(games, shouldStop);
        if (!finished) break;
        for (const g of games) {
          if (g.capped) capped++;
          scores[g.cand].push(g.score ?? 0.5);
        }
        onProgress?.(result(lo + cfg.wave >= cfg.pairs));
      }
      return result(true);
    } finally {
      engine.restore(real);
    }
  }

  /** Step every game to the end, batching the network calls. False if stopped. */
  private async playOut(games: Game[], shouldStop: () => boolean): Promise<boolean> {
    const engine = this.engine;
    const obsSize = engine.obsSize;
    const actionSize = engine.actionSize;
    for (;;) {
      await yieldToPage();
      if (shouldStop()) return false;
      const waiting: { g: Game; obs: Float32Array; mask: Uint8Array }[] = [];
      for (const g of games) {
        if (g.score !== null) continue;
        engine.restore(g.state);
        // Advance through what needs no network: the end, and decisions with one legal move.
        for (;;) {
          if (engine.isTerminal()) {
            const u = engine.terminalUtility();
            const mine = g.mover === PLAYER_US ? u : -u;
            g.score = mine > 0 ? 1 : mine < 0 ? 0 : 0.5;
            break;
          }
          if (g.decisions >= this.config.maxDecisions) {
            g.score = 0.5;
            g.capped = true;
            break;
          }
          const who = engine.decisionPlayer();
          let mask = engine.mask(this.merged);
          const legal: number[] = [];
          for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
          if (who === 0 || legal.length === 0) {
            g.score = 0.5;
            g.capped = true;
            break;
          }
          if (g.guard && who !== g.mover) g.guard = false;
          g.decisions++;
          if (legal.length === 1) {
            applyFlat(engine, legal[0], this.merged);
            continue;
          }
          if (g.guard) mask = this.safeMask(mask, legal, g.mover);
          waiting.push({ g, obs: engine.observation(who), mask });
          g.state = engine.snapshot();
          break;
        }
      }
      if (waiting.length === 0) return true;
      for (let lo = 0; lo < waiting.length; lo += this.config.batch) {
        const part = waiting.slice(lo, lo + this.config.batch);
        const obs = new Float32Array(part.length * obsSize);
        const masks = new Uint8Array(part.length * actionSize);
        part.forEach((w, i) => {
          obs.set(w.obs, i * obsSize);
          masks.set(w.mask, i * actionSize);
        });
        const out = await this.model.run(obs, masks, part.length);
        await yieldToPage();
        part.forEach((w, i) => {
          let best = -1;
          let bestLogit = -Infinity;
          for (let a = 0; a < actionSize; a++) {
            if (!w.mask[a]) continue;
            const l = out.logits[i * actionSize + a];
            if (best < 0 || l > bestLogit) {
              best = a;
              bestLogit = l;
            }
          }
          engine.restore(w.g.state);
          const why = applyFlat(engine, best, this.merged);
          if (why) throw new Error(`the engine refused the model's move ${best}: ${why}`);
          w.g.state = engine.snapshot();
        });
      }
    }
  }

  /** The mask without the options that lose on the spot for `mover`, unless every option does. */
  private safeMask(mask: Uint8Array, legal: number[], mover: number): Uint8Array {
    const safe = legal.filter(a => !losesOnTheSpot(this.engine, a, this.merged, mover));
    if (safe.length === 0 || safe.length === legal.length) return mask;
    const out = new Uint8Array(mask.length);
    for (const a of safe) out[a] = 1;
    return out;
  }
}

/**
 * Does playing `idx` here end the game against `mover` at once (a battleground coup at DEFCON 2,
 * a DEFCON-lowering event)? Leaves the engine on the position it was called on.
 */
export function losesOnTheSpot(engine: WasmEngine, idx: number, merged: boolean, mover: number): boolean {
  const here = engine.snapshot();
  try {
    if (applyFlat(engine, idx, merged)) return false;
    if (!engine.isTerminal()) return false;
    const u = engine.terminalUtility();
    return (mover === PLAYER_US ? u : -u) < 0;
  } finally {
    engine.restore(here);
  }
}

/** Per candidate: mean score, and the paired difference against the first candidate. */
export function summarize(candidates: number[], scores: number[][]): CandidateResult[] {
  const ref = scores[0] ?? [];
  return candidates.map((idx, ci) => {
    const s = scores[ci];
    const n = Math.min(s.length, ref.length);
    const mean = s.length ? s.reduce((a, b) => a + b, 0) / s.length : NaN;
    if (ci === 0) return { idx, score: mean, diff: 0, diffSe: 0 };
    const d: number[] = [];
    for (let k = 0; k < n; k++) d.push(s[k] - ref[k]);
    const dm = d.length ? d.reduce((a, b) => a + b, 0) / d.length : NaN;
    let se = NaN;
    if (d.length > 1) {
      const v = d.reduce((a, x) => a + (x - dm) * (x - dm), 0) / (d.length - 1);
      se = Math.sqrt(v / d.length);
    }
    return { idx, score: mean, diff: dm, diffSe: se };
  });
}
