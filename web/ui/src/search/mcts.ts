/**
 * Determinized MCTS on the open decision, in the page: a diagnostic and a teacher, never the
 * product. The product is the no-search network; this answers two questions interactively --
 * what would a search do here, and where does search disagree with the raw net. 64 simulations
 * is where the strength curve flattens: a larger budget buys nothing above it and doubles the
 * wait.
 *
 * The port of `ai/search/batched_mcts.py` run for ONE position, plus `ai/search/dmcts.py`'s
 * determinize. It plays what `search:<model.onnx>:64:determinize` plays (tools/lib/player_agent.py
 * §search):
 *
 *   BatchedMCTSConfig(simulations=64, temperature=0.0, auto_advance=True, advance_root=False,
 *                     determinize=True) -- one sampled world per search, 64 simulations on one
 *                     tree, root visits summed (here: just read), the tree never reusing a
 *                     subtree across decisions. No Dirichlet noise (dirichlet_frac 0).
 *
 * Like that searcher, the root is searched EXACTLY as handed over (advance_root=False): the
 * caller's position is never settled under the tree. Only the searcher's own hypothetical
 * children advance past what nobody controls.
 *
 * Two deliberate deviations from the Python, both in `settle` (bindings/settle.py):
 *
 * 1. Children settle CHANCE nodes only (die rolls), not `Engine::auto_advance_step` (FORCED).
 *    The C++ auto-advance is a hand-written state machine over specific events (Suez, Muslim
 *    Revolution, ...) that the page's engine API does not expose, and the engine is not ours to
 *    change here. So the tree holds nodes the CLI searcher skips -- forced decisions stay in it
 *    as single-action edges. The tree is still a tree of real positions and backups are still
 *    correct; the leaf evaluations happen one decision earlier than the CLI searcher's, which is
 *    what a disagreement between the two is most likely to be.
 * 2. There is one world per search (`determinizations` = 1), because that is what the CLI
 *    searcher does: BatchedMCTS determinizes once per search, not once per simulation.
 *    Splitting 64 simulations over 8 worlds would play 8-sim trees, far shallower than the
 *    budget this player is defined by.
 *
 * Sign convention matches pimcts.py: every value is US-perspective. `v_win` arrives from the
 * mover's perspective and is negated for the USSR on the way in.
 */
import { WasmEngine, PLAYER_US } from "../engine/wasm_engine";
import { Forward } from "../analysis/model";

// The model surface the search needs -- `analysis/model.ts`'s `Model` satisfies it structurally,
// and tests inject a stub instead of an ONNX session.
export interface SearchModel {
  run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward>;
}

export interface SearchConfig {
  /** Total simulations across worlds. 64 is the knee of the sweep's curve. */
  simulations: number;
  /** Resample the hidden cards before the search, so the tree never reads the opponent's hand. */
  determinize: boolean;
  /** PUCT exploration constant (PIMCTSConfig.c_puct). */
  cPuct: number;
  seed: number;
  /**
   * First-play urgency (BatchedMCTSConfig.fpu_reduction): an unvisited move is valued at its
   * node's value less this, from the mover's side. 0 (absent) is the original rule.
   */
  fpuReduction?: number;
  /**
   * Choose the move by a noise-free Gumbel root over this many candidates (gumbel_root.py,
   * BatchedMCTSConfig.gumbel_k with gumbel_scale 0); 0 (absent) is the PUCT root above.
   */
  gumbelK?: number;
}

/** What `search:<model.onnx>:64:determinize` plays. */
export const SEARCH_64: SearchConfig = { simulations: 64, determinize: true, cPuct: 1.5, seed: 12345 };

/**
 * What `search:<model>:64:determinize:all:gumbel_k=8:gumbel_scale=0:fpu_reduction=0.2` plays: a
 * Gumbel root over the network's 8 most probable moves, 64 simulations by sequential halving.
 * The workbench's search.
 */
export const GUMBEL_64: SearchConfig = {
  simulations: 64, determinize: true, cPuct: 1.5, seed: 12345, fpuReduction: 0.2, gumbelK: 8,
};

export interface SearchRow {
  idx: number;
  /** Root visits (PUCT), or the simulations spent on this candidate (Gumbel). */
  visits: number;
  /** Mean backed-up value, US perspective; NaN before a first visit. */
  qUs: number;
  prior: number;
  /** Gumbel only: one of the root's k candidates. */
  candidate?: boolean;
  /** Gumbel only: the halving phase that eliminated it (0-based); null for the move played. */
  droppedInPhase?: number | null;
  /** Gumbel only: sigma(completed Q) at its last ranking. */
  sigma?: number;
}

export interface SearchResult {
  /**
   * The move to play: most visits, then the better mean value for the mover, then the higher
   * prior (batched_mcts.py `_most_visited`'s visit tie-break). Null when the position has
   * nothing to search (no open decision, or no legal action).
   */
  action: number | null;
  rows: SearchRow[];
  simulations: number;
  /** Worlds sampled by the determinization (1 in the default configuration). */
  worlds: number;
  ms: number;
  /**
   * The sampled world made legal what the real state does not (the Cambridge Five names the
   * regions on the opponent's hidden scoring cards). The pick was replaced by the greedy policy
   * over the real mask -- "the search proposes and the true mask disposes" (player_agent.py).
   */
  worldMismatch: boolean;
  /** No search signal at all (terminal root, or a single legal action): the greedy policy played. */
  fellBack: boolean;
  /** Which root chose: the PUCT tree's visits, or a Gumbel root's sequential halving. */
  root?: "puct" | "gumbel";
  /** Gumbel only: the network's value at the real position, from the mover's side. */
  valueMover?: number;
  /** Gumbel only: +1 if the US decides, -1 the USSR. */
  mover?: number;
}

/** CardLocation (engine/include/ts/types.hpp) as the save JSON carries it: an int per card id. */
export const LOC_DRAW_DECK = 1;
export const LOC_HAND_US_UNKNOWN = 2;
export const LOC_HAND_US_KNOWN = 3;
export const LOC_HAND_USSR_UNKNOWN = 4;
export const LOC_HAND_USSR_KNOWN = 5;
const CARD_IDS_END = 110;

/** The fields of a `ts_save_v2` save (bindings/state_json.cpp) that this file touches. */
interface SaveShape {
  card_locations: number[];
  rng_state: number;
  [key: string]: unknown;
}

/** Deterministic 32-bit RNG (mulberry32): a search is reproducible from its seed. */
class Rng {
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

interface SNode {
  /** The engine state at this node (after the searcher's own settle), as raw GameState bytes. */
  state: Uint8Array;
  /** +1 US, -1 USSR (batched_mcts.py `mover`); 0 for a terminal node. */
  mover: number;
  terminal: boolean;
  /** Leaf estimate, US perspective. */
  valueUs: number;
  actions: number[];
  priors: number[];
  n: number[];
  w: number[];
  children: Map<number, SNode>;
  expanded: boolean;
}

export class PageSearcher {
  private readonly rng: Rng;

  /**
   * `merged` is the model's action view (its `ts.merged_influence` metadata): the mask, the
   * action indices and the composition of a composed "Ops -> Influence" action all follow it.
   */
  constructor(
    private readonly engine: WasmEngine,
    private readonly model: SearchModel,
    private readonly merged: boolean,
    private readonly config: SearchConfig = SEARCH_64,
  ) {
    this.rng = new Rng(config.seed);
  }

  /**
   * Search the position the engine holds NOW, and leave it exactly as it was. The engine is a
   * singleton shared with the session, so every lookup restores a node's bytes first and this
   * method restores the caller's position before returning -- the caller must not act while it
   * runs (the page refuses clicks for the duration).
   */
  async search(): Promise<SearchResult> {
    if ((this.config.gumbelK ?? 0) > 0) return this.gumbelSearch();
    const t0 = Date.now();
    const engine = this.engine;
    const real = engine.snapshot();
    const rows: SearchRow[] = [];
    const empty: SearchResult = {
      action: null, rows, simulations: 0, worlds: 0, ms: 0, worldMismatch: false, fellBack: true,
    };
    try {
      engine.restore(real);
      const realMask = engine.mask(this.merged);

      if (engine.isTerminal()) return empty;

      const world = this.config.determinize ? this.sampleWorld(real) : real;
      const root = this.makeNode(world);
      for (let i = 0; i < root.actions.length; i++) {
        rows.push({ idx: root.actions[i], visits: 0, qUs: NaN, prior: 0 });
      }
      if (root.terminal || root.actions.length === 0) {
        return { ...empty, action: await this.greedy(real), ms: Date.now() - t0 };
      }

      await this.expand(root);
      for (let i = 0; i < rows.length; i++) rows[i].prior = root.priors[i];

      for (let s = 0; s < this.config.simulations; s++) {
        const { path, leaf } = this.descend(root);
        if (!leaf.terminal && !leaf.expanded) await this.expand(leaf);
        this.backup(path, leaf.valueUs);
      }

      for (let i = 0; i < rows.length; i++) {
        rows[i].visits = root.n[i];
        rows[i].qUs = root.n[i] > 0 ? root.w[i] / root.n[i] : NaN;
      }
      rows.sort((a, b) => b.visits - a.visits || (b.qUs - a.qUs) || (b.prior - a.prior));

      let picked: number | null = root.actions[this.mostVisited(root)];
      let worldMismatch = false;
      engine.restore(real);
      if (!realMask[picked]) {
        worldMismatch = true;
        picked = await this.greedy(real);
      }
      return {
        action: picked, rows, simulations: this.config.simulations, worlds: this.config.determinize ? 1 : 0,
        ms: Date.now() - t0, worldMismatch, fellBack: false, root: "puct",
      };
    } finally {
      engine.restore(real);
    }
  }

  // ---- the Gumbel root (ai/search/gumbel_root.py) -----------------------------------------------

  /**
   * GumbelRoot.choose for one position, noise-free (gumbel_scale 0):
   *
   * 1. Candidates: the k legal moves with the largest logits, from the network on the REAL state.
   * 2. Sequential halving over ceil(log2 k) phases. Each phase samples one world from the mover's
   *    side; in it every surviving candidate is played and the position it leads to is searched
   *    afresh by the PUCT tree above, undeterminized, with an equal share of the budget
   *    (floor(simulations / (phases x survivors)), at least 1). Its value is the searched root's
   *    (value + sum of backups) / (1 + visits), from the mover's side, pooled across phases
   *    weighted by the share. The better half by logit + sigma(completed Q) survives; the last
   *    phase keeps one, which is played.
   *
   * Candidates in one phase share the world's die rolls (the Python steps every candidate from
   * the same rng_state); the searches below them reseed per edge as the PUCT tree always does.
   * The page's deviation from the CLI searcher (children settle die rolls only) applies here too.
   */
  private async gumbelSearch(): Promise<SearchResult> {
    const t0 = Date.now();
    const engine = this.engine;
    const real = engine.snapshot();
    const empty: SearchResult = {
      action: null, rows: [], simulations: 0, worlds: 0, ms: 0, worldMismatch: false, fellBack: true, root: "gumbel",
    };
    try {
      engine.restore(real);
      if (engine.isTerminal()) return empty;
      const realMask = engine.mask(this.merged);
      const legal: number[] = [];
      for (let i = 0; i < realMask.length; i++) if (realMask[i]) legal.push(i);
      if (legal.length === 0) return empty;
      const mover = this.actingPlayer();
      const out = await this.model.run(engine.observation(mover), realMask, 1);
      const logits = new Map<number, number>();
      for (const a of legal) logits.set(a, out.logits[a]);
      const valueMover = out.vWin[0];
      const prior = softmaxOver(legal, logits);

      // Stable sort: equal logits keep the engine's order, as Python's sorted() does.
      let alive = legal.slice().sort((a, b) => logits.get(b)! - logits.get(a)!)
        .slice(0, Math.max(1, this.config.gumbelK ?? 1));
      const candidates = new Set(alive);
      const n = new Map<number, number>(legal.map(a => [a, 0]));
      const w = new Map<number, number>(legal.map(a => [a, 0]));
      const dropped = new Map<number, number>();
      let lastSigma = new Map<number, number>();
      const phases = alive.length > 1 ? Math.max(1, Math.ceil(Math.log2(alive.length))) : 0;
      let spent = 0;

      for (let ph = 0; ph < phases; ph++) {
        const world = this.config.determinize ? this.sampleWorld(real) : this.reseeded(real);
        const per = Math.max(1, Math.floor(this.config.simulations / (phases * alive.length)));
        for (const a of alive) {
          engine.restore(world);
          if (!engine.mask(this.merged)[a]) continue;     // illegal in this world (Cambridge Five)
          const refused = this.stepFlat(a);
          if (refused) throw new Error(`search stepped an illegal action (${a}): ${refused}`);
          this.drainChance();
          const vUs = await this.subSearch(engine.snapshot(), per);
          n.set(a, n.get(a)! + per);
          w.set(a, w.get(a)! + (mover === PLAYER_US ? vUs : -vUs) * per);
          spent += per;
        }
        const q = new Map<number, number>();
        for (const a of legal) if (n.get(a)! > 0) q.set(a, w.get(a)! / n.get(a)!);
        const sig = sigmaCompleted(legal, logits, valueMover, n, q);
        const ranked = alive.slice().sort((a, b) => (logits.get(b)! + sig.get(b)!) - (logits.get(a)! + sig.get(a)!));
        alive = ph === phases - 1 ? ranked.slice(0, 1) : ranked.slice(0, Math.max(1, Math.ceil(ranked.length / 2)));
        for (const a of ranked.slice(alive.length)) dropped.set(a, ph);
        lastSigma = sig;
      }

      const picked = alive[0];
      const rows: SearchRow[] = legal.map(a => {
        const na = n.get(a)!;
        const qMover = na > 0 ? w.get(a)! / na : NaN;
        return {
          idx: a, visits: na, qUs: mover === PLAYER_US ? qMover : -qMover, prior: prior.get(a)!,
          candidate: candidates.has(a), droppedInPhase: candidates.has(a) ? (dropped.get(a) ?? null) : undefined,
          sigma: lastSigma.get(a),
        };
      });
      // The played move first, then by how far each candidate got, then its prior.
      const reach = (r: SearchRow) => !r.candidate ? -2 : r.droppedInPhase === null ? phases : (r.droppedInPhase ?? -1);
      rows.sort((x, y) => reach(y) - reach(x) || y.prior - x.prior);
      return {
        action: picked, rows, simulations: spent, worlds: phases, ms: Date.now() - t0,
        worldMismatch: false, fellBack: phases === 0, root: "gumbel", valueMover, mover,
      };
    } finally {
      engine.restore(real);
    }
  }

  /**
   * An ordinary PUCT search of `state` with `sims` simulations, undeterminized (the Gumbel root
   * already sampled the world): its root's (value + sum of backups) / (1 + visits), US side.
   */
  private async subSearch(state: Uint8Array, sims: number): Promise<number> {
    const root = this.makeNode(state);
    if (root.terminal || root.actions.length === 0) return root.valueUs;
    await this.expand(root);
    for (let s = 0; s < sims; s++) {
      const { path, leaf } = this.descend(root);
      if (!leaf.terminal && !leaf.expanded) await this.expand(leaf);
      this.backup(path, leaf.valueUs);
    }
    let nSum = 0;
    let wSum = 0;
    for (let i = 0; i < root.n.length; i++) {
      nSum += root.n[i];
      wSum += root.w[i];
    }
    return (root.valueUs + wSum) / (1 + nSum);
  }

  /** The real state with a fresh rng_state: the undeterminized root still rolls new dice. */
  private reseeded(real: Uint8Array): Uint8Array {
    this.engine.restore(real);
    this.reseedRng();
    return this.engine.snapshot();
  }

  // ---- the world ------------------------------------------------------------------------------

  /**
   * dmcts.py's determinize: a clone with the unseen cards reshuffled between the opponent's
   * hand and the draw deck. Counts are preserved exactly and every card we have SEEN stays
   * where it was, so the sampled world is consistent with everything the mover can observe.
   * Public because the sampled world is the searcher's core honesty property and the tests pin
   * it (tests/web/js/search_tiebreak.ts).
   *
   * Through the save JSON, because that is the page engine's only state-editing API -- no C++
   * change. The round trip is exact for every field that matters: `card_locations`
   * are small ints and `persistent_effects` holds 43 flags (rules/flags.json), well under the
   * 2^53 a JSON number keeps. `rng_state` IS a full 64-bit value and its low bits do not
   * survive `JSON.parse` -- which is harmless here, because the very next line overwrites it
   * with the world's own seed (as batched_mcts.py does: worlds differ in deals AND die rolls).
   */
  public sampleWorld(real: Uint8Array): Uint8Array {
    const engine = this.engine;
    engine.restore(real);
    const me = this.actingPlayer();
    const oppUnknown = me === PLAYER_US ? LOC_HAND_USSR_UNKNOWN : LOC_HAND_US_UNKNOWN;
    const save = JSON.parse(engine.saveJson()) as SaveShape;
    const locs = save.card_locations;
    const hand: number[] = [];
    const deck: number[] = [];
    for (let cid = 1; cid <= CARD_IDS_END; cid++) {
      if (locs[cid] === oppUnknown) hand.push(cid);
      else if (locs[cid] === LOC_DRAW_DECK) deck.push(cid);
    }
    const pool = hand.concat(deck);
    if (pool.length > 0) {
      this.rng.shuffle(pool);
      for (let i = 0; i < pool.length; i++) {
        locs[pool[i]] = i < hand.length ? oppUnknown : LOC_DRAW_DECK;
      }
    }
    save.rng_state = this.rng.int53();
    engine.loadSaveJson(JSON.stringify(save));
    return engine.snapshot();
  }

  /** The player to move (pimcts.py `acting_player`): the decision's owner, else the phasing one. */
  private actingPlayer(): number {
    const p = this.engine.decisionPlayer();
    if (p !== 0) return p;
    const ph = this.engine.display().phasing_player;
    return ph === "US" ? PLAYER_US : ph === "USSR" ? -PLAYER_US : 0;
  }

  // ---- the tree -------------------------------------------------------------------------------

  private makeNode(state: Uint8Array): SNode {
    const engine = this.engine;
    engine.restore(state);
    const base: SNode = {
      state, mover: 0, terminal: true, valueUs: 0, actions: [], priors: [],
      n: [], w: [], children: new Map(), expanded: true,
    };
    if (engine.isTerminal()) {
      base.valueUs = engine.terminalUtility();
      return base;
    }
    const mask = engine.mask(this.merged);
    const actions: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) actions.push(i);
    base.mover = this.actingPlayer();
    if (actions.length === 0) {
      base.valueUs = 0;
      return base;
    }
    base.terminal = false;
    base.expanded = false;
    base.actions = actions;
    base.priors = new Array(actions.length).fill(1 / actions.length);
    base.n = new Array(actions.length).fill(0);
    base.w = new Array(actions.length).fill(0);
    return base;
  }

  /** Fill a node's priors and leaf value from the network (pimcts.py `_evaluate`). */
  private async expand(node: SNode): Promise<void> {
    if (node.expanded) return;
    const engine = this.engine;
    engine.restore(node.state);
    const obs = engine.observation(node.mover);
    const mask = engine.mask(this.merged);
    const out = await this.model.run(obs, mask, 1);
    // Softmax over the legal actions only, at temperature 1, renormalised over them.
    const legal = node.actions;
    let maxLogit = -Infinity;
    for (const i of legal) maxLogit = Math.max(maxLogit, out.logits[i]);
    let z = 0;
    const p = legal.map(i => {
      const e = Math.exp(out.logits[i] - maxLogit);
      z += e;
      return e;
    });
    node.priors = z > 1e-12 ? p.map(e => e / z) : legal.map(() => 1 / legal.length);
    node.valueUs = node.mover === PLAYER_US ? out.vWin[0] : -out.vWin[0];
    node.expanded = true;
  }

  /** PUCT, read from the mover's side of a US-perspective value (batched_mcts.py `_select`). */
  private select(node: SNode): number {
    let total = 0;
    for (const x of node.n) total += x;
    const sqrtTotal = Math.sqrt(total > 1 ? total : 1);
    const c = this.config.cPuct;
    const fpu = this.config.fpuReduction ?? 0;
    const usMoves = node.mover === PLAYER_US;
    let bestI = 0;
    let bestV = -Infinity;
    for (let i = 0; i < node.n.length; i++) {
      const ni = node.n[i];
      let q = ni > 0 ? node.w[i] / ni : node.valueUs;
      if (!usMoves) q = -q;
      if (ni === 0) q -= fpu;
      const v = q + c * node.priors[i] * sqrtTotal / (1 + ni);
      if (v > bestV) {
        bestV = v;
        bestI = i;
      }
    }
    return bestI;
  }

  /** Walk to a leaf: the path taken and the leaf reached (possibly unexpanded). */
  private descend(root: SNode): { path: Array<[SNode, number]>; leaf: SNode } {
    const path: Array<[SNode, number]> = [];
    let node = root;
    for (;;) {
      if (node.terminal || !node.expanded) return { path, leaf: node };
      const idx = this.select(node);
      path.push([node, idx]);
      const action = node.actions[idx];
      let child = node.children.get(action);
      if (!child) {
        child = this.makeNode(this.stepState(node.state, action));
        node.children.set(action, child);
        return { path, leaf: child };
      }
      node = child;
    }
  }

  private backup(path: Array<[SNode, number]>, valueUs: number): void {
    for (const [node, idx] of path) {
      node.n[idx] += 1;
      node.w[idx] += valueUs;
    }
  }

  /**
   * The move to play: most visits, then the better mean value for the mover, then the higher
   * prior. Visit counts tie often at small budgets and resolving a tie by list position played
   * whichever move the engine happened to enumerate first -- a 2-simulation search played that
   * way scored below its own prior. tests/training/test_search_tiebreak.py pins it on the Python
   * side; tests/web/js/search_tiebreak.ts pins it here.
   */
  private mostVisited(root: SNode): number {
    const usMoves = root.mover === PLAYER_US;
    let bestI = 0;
    let bestN = -Infinity;
    let bestQ = -Infinity;
    let bestP = -Infinity;
    for (let i = 0; i < root.actions.length; i++) {
      const ni = root.n[i];
      let qi = ni > 0 ? root.w[i] / ni : -Infinity;
      if (ni > 0 && !usMoves) qi = -qi;
      const pi = root.priors[i];
      if (ni > bestN || (ni === bestN && (qi > bestQ || (qi === bestQ && pi > bestP)))) {
        bestN = ni;
        bestQ = qi;
        bestP = pi;
        bestI = i;
      }
    }
    return bestI;
  }

  // ---- stepping the engine --------------------------------------------------------------------

  /** The parent's state stepped by one flat action and settled -- the child's bytes. */
  private stepState(parent: Uint8Array, action: number): Uint8Array {
    const engine = this.engine;
    engine.restore(parent);
    // Re-seed so sibling edges sample different die outcomes; without this every simulation
    // from one parent replays the same fixed roll (batched_mcts.py `_descend`).
    this.reseedRng();
    const refused = this.stepFlat(action);
    if (refused) throw new Error(`search stepped an illegal action (${action}): ${refused}`);
    this.drainChance();
    return engine.snapshot();
  }

  /** One flat action in the model's action view (Engine::step_flat's merged composition). */
  private stepFlat(action: number): string | null {
    const engine = this.engine;
    if (this.merged && engine.isMergedInfluenceAction(action)) {
      const L = engine.layout;
      const refused = engine.step(engine.decodeFlat(L.ops_influence_index), 0);
      if (refused) return refused;
      if (engine.isTerminal()) return null;
      const second = action === L.ops_influence_index ? L.confirm_done_index : action;
      return engine.step(engine.decodeFlat(second), 0);
    }
    return engine.step(engine.decodeFlat(action), 0);
  }

  /** Resolve die rolls and nothing else (settle.py, SettleMode.CHANCE -- see the header). */
  private drainChance(): void {
    const engine = this.engine;
    const ROLL_DIE = engine.layout.decision_types["ROLL_DIE"] ?? 7;
    let guard = 0;
    while (!engine.isTerminal() && engine.decisionPlayer() === 0 && engine.decisionType() === ROLL_DIE) {
      if (++guard > 256) throw new Error("the chance drain did not terminate");
      engine.step({ decision_type: ROLL_DIE, primary_id: 0, secondary_id: 0, flags: 0 }, 0);
    }
  }

  /** Put a fresh rng_state into the state on the engine (the save JSON is the only handle). */
  private reseedRng(): void {
    const engine = this.engine;
    const save = JSON.parse(engine.saveJson()) as SaveShape;
    save.rng_state = this.rng.int53();
    engine.loadSaveJson(JSON.stringify(save));
  }

  /** The model's argmax over the REAL mask -- what plays when search cannot (player_agent.py). */
  private async greedy(real: Uint8Array): Promise<number | null> {
    const engine = this.engine;
    engine.restore(real);
    const mask = engine.mask(this.merged);
    const legal: number[] = [];
    for (let i = 0; i < mask.length; i++) if (mask[i]) legal.push(i);
    if (legal.length === 0) return null;
    const obs = engine.observation(this.actingPlayer());
    const out = await this.model.run(obs, mask, 1);
    let best = legal[0];
    for (const i of legal) if (out.logits[i] > out.logits[best]) best = i;
    return best;
  }
}

// ---- sigma(completed Q) (gumbel_root.py `sigma_completed`, mctx's completed-by-mix-value) ------

/** mctx's constants: sigma = (c_visit + max visits) x c_scale x rescaled completed Q. */
export const C_VISIT = 50;
export const C_SCALE = 0.1;

/** Softmax of the logits over `legal`. */
export function softmaxOver(legal: number[], logits: Map<number, number>): Map<number, number> {
  let mx = -Infinity;
  for (const a of legal) mx = Math.max(mx, logits.get(a)!);
  const z = new Map<number, number>();
  let sum = 0;
  for (const a of legal) {
    const e = Math.exp(logits.get(a)! - mx);
    z.set(a, e);
    sum += e;
  }
  for (const a of legal) z.set(a, z.get(a)! / sum);
  return z;
}

/**
 * sigma(completed Q) per legal move: unvisited moves take the mixed value (the network's value
 * blended with the prior-weighted mean Q of the visited moves, by total visits), completed Q is
 * rescaled to [0, 1] over the legal moves, then scaled by (c_visit + max visits) x c_scale.
 * Values are from the mover's side. The same arithmetic as the Python, line for line.
 */
export function sigmaCompleted(
  legal: number[], logits: Map<number, number>, valueMover: number,
  n: Map<number, number>, q: Map<number, number>,
): Map<number, number> {
  const prior = softmaxOver(legal, logits);
  let total = 0;
  for (const a of legal) total += n.get(a) ?? 0;
  const visited = legal.filter(a => (n.get(a) ?? 0) > 0);
  let vMix = valueMover;
  if (visited.length > 0) {
    let pz = 0;
    for (const a of visited) pz += prior.get(a)!;
    let meanQ = 0;
    if (pz > 0) {
      for (const a of visited) meanQ += prior.get(a)! * q.get(a)!;
      meanQ /= pz;
    } else {
      for (const a of visited) meanQ += q.get(a)!;
      meanQ /= visited.length;
    }
    vMix = (valueMover + total * meanQ) / (1 + total);
  }
  const cq = new Map<number, number>();
  for (const a of legal) cq.set(a, (n.get(a) ?? 0) > 0 ? q.get(a)! : vMix);
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of legal) {
    lo = Math.min(lo, cq.get(a)!);
    hi = Math.max(hi, cq.get(a)!);
  }
  const scale = hi - lo > 1e-8 ? hi - lo : 1;
  let maxN = 0;
  for (const a of legal) maxN = Math.max(maxN, n.get(a) ?? 0);
  const out = new Map<number, number>();
  for (const a of legal) out.set(a, (C_VISIT + maxN) * C_SCALE * (cq.get(a)! - lo) / scale);
  return out;
}
