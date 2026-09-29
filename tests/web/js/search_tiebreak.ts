// Bundled by tests/web/test_page_search.py (esbuild) and run with node: the page's 64-sim
// searcher (web/ui/src/search/mcts.ts) against the properties that make it the same player as
// `search:<model.onnx>:64:determinize`, with a stub network so every value is known.
//   node search_tiebreak.mjs <web/ui/public/>
//
// Pinned here (the Python side is tests/training/test_search_tiebreak.py):
//   - one simulation plays the network's argmax, not whatever the engine lists first;
//   - a visit tie breaks by mover-mean value, then prior -- again not by list position;
//   - the determinization preserves every count and every card the mover has seen;
//   - the search hands the engine singleton back exactly as it found it;
//   - the pick is legal in the REAL state's mask.
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { PageSearcher, SEARCH_64, LOC_DRAW_DECK } from "../../../web/ui/src/search/mcts";

const [publicDir] = process.argv.slice(2);
const engine = await WasmEngine.load(pathToFileURL(publicDir).href);
const ACT = engine.actionSize;

// A stub network: a known prior per action index and a constant value, so every search outcome
// is computable here. Constant v_win makes every mean value equal, which is exactly the
// condition under which the old tie-break fell through to the engine's order.
const logit = (i: number) => ((i * 37) % 101) / 100 - 0.5;
const stub = {
  async run(_obs: Float32Array, masks: Uint8Array, rows: number) {
    const logits = new Float32Array(rows * ACT);
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < ACT; i++) logits[r * ACT + i] = masks[r * ACT + i] ? logit(i) : -1e9;
    }
    return { logits, vWin: new Float32Array(rows).fill(0.5), vVp: new Float32Array(rows).fill(0) };
  },
};

const legalHere = (): number[] => {
  const mask = engine.mask(false);
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
};
const argmaxLogit = (legal: number[]): number => {
  let best = legal[0];
  for (const i of legal) if (logit(i) > logit(best)) best = i;
  return best;
};
const sameBytes = (a: Uint8Array, b: Uint8Array): boolean => {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
};

engine.newGame(42);
const real = engine.snapshot();
const legal = legalHere();
const argmax = argmaxLogit(legal);

// The fixture must actually exercise the bug: the argmax cannot be the engine's first listing.
const fixtureDiscriminates = legal.length > 1 && argmax !== legal[0];

// ---- 1. one simulation plays the network's argmax ---------------------------------------------
const one = await new PageSearcher(engine, stub, false, { ...SEARCH_64, determinize: false, simulations: 1 }).search();
const oneSimArgmax = one.action === argmax;

// ---- 2. a visit tie breaks by value, then prior -- not list order ------------------------------
const two = await new PageSearcher(engine, stub, false, { ...SEARCH_64, determinize: false, simulations: 2 }).search();
const expectedPick = (): number | null => {
  let best: { idx: number; n: number; q: number; p: number } | null = null;
  for (const r of two.rows) {
    const q = Number.isFinite(r.qUs) ? r.qUs : -Infinity;
    const cand = { idx: r.idx, n: r.visits, q, p: r.prior };
    if (!best || cand.n > best.n || (cand.n === best.n && (cand.q > best.q || (cand.q === best.q && cand.p > best.p)))) {
      best = cand;
    }
  }
  return best ? best.idx : null;
};
const twoSimTieBreak = two.action !== null && two.action === expectedPick();
const visitsSum = two.rows.reduce((s, r) => s + r.visits, 0) === two.simulations && two.action !== null;

// ---- 3. the determinization preserves counts and everything seen ------------------------------
engine.restore(real);
const saveBefore = JSON.parse(engine.saveJson());
const me = engine.decisionPlayer();
const oppUnknown = me === 1 ? 4 : 2; // HAND_USSR_UNKNOWN : HAND_US_UNKNOWN (types.hpp)
const poolOf = (locs: number[]) => {
  const hand: number[] = [];
  const deck: number[] = [];
  for (let cid = 1; cid <= 110; cid++) {
    if (locs[cid] === oppUnknown) hand.push(cid);
    else if (locs[cid] === LOC_DRAW_DECK) deck.push(cid);
  }
  return { hand, deck };
};
const searcher = new PageSearcher(engine, stub, false, SEARCH_64);
const world = searcher.sampleWorld(real);
engine.restore(world);
const saveWorld = JSON.parse(engine.saveJson());
engine.restore(real);

const before = poolOf(saveBefore.card_locations);
const after = poolOf(saveWorld.card_locations);
const worldCounts = before.hand.length === after.hand.length && before.deck.length === after.deck.length;
// Every card outside the hidden pool keeps its location: our hand, both KNOWN (seen) cards,
// the discard, the removed pile, ongoing events, peeked cards -- the sampled world contradicts
// nothing the mover has observed.
let seenUntouched = true;
for (let cid = 1; cid <= 110; cid++) {
  const inPool = (before.hand.includes(cid) || before.deck.includes(cid));
  if (!inPool && saveBefore.card_locations[cid] !== saveWorld.card_locations[cid]) seenUntouched = false;
}
// Sampled worlds differ from one another (the shuffle is real).
let worldsDiffer = false;
for (let k = 0; k < 8 && !worldsDiffer; k++) {
  engine.restore(searcher.sampleWorld(real));
  const w = JSON.parse(engine.saveJson());
  if (JSON.stringify(w.card_locations) !== JSON.stringify(saveWorld.card_locations)) worldsDiffer = true;
}
engine.restore(real);

// ---- 4. the engine is handed back untouched, and the pick is legal here -----------------------
const full = await new PageSearcher(engine, stub, false, SEARCH_64).search();
const engineRestored = sameBytes(engine.snapshot(), real);
const pickLegal = full.action !== null && legal.includes(full.action);
const visitsSum64 = full.rows.reduce((s, r) => s + r.visits, 0) === full.simulations;

// ---- 5. it still works off the opening: step a few moves, then search -------------------------
const ROLL_DIE = engine.layout.decision_types["ROLL_DIE"] ?? 7;
let steps = 0;
for (let k = 0; k < 3; k++) {
  const mask = engine.mask(false);
  let first = -1;
  for (let i = 0; i < mask.length; i++) if (mask[i]) { first = i; break; }
  if (first < 0 || engine.isTerminal()) break;
  const refused = engine.step(engine.decodeFlat(first), 0);
  if (refused) throw new Error(`fixture step ${k} refused: ${refused}`);
  steps++;
  let guard = 0;
  while (!engine.isTerminal() && engine.decisionPlayer() === 0 && engine.decisionType() === ROLL_DIE) {
    if (++guard > 64) throw new Error("chance drain did not terminate in the fixture");
    engine.step({ decision_type: ROLL_DIE, primary_id: 0, secondary_id: 0, flags: 0 }, 0);
  }
}
const afterSteps = engine.snapshot();
const later = await new PageSearcher(engine, stub, false, SEARCH_64).search();
const laterLegal = later.action !== null && legalHere().includes(later.action);
const laterRestored = sameBytes(engine.snapshot(), afterSteps);

console.log(JSON.stringify({
  fixtureDiscriminates,
  oneSimArgmax,
  twoSimTieBreak,
  visitsSum: visitsSum && visitsSum64,
  worldCounts,
  seenUntouched,
  worldsDiffer,
  engineRestored,
  pickLegal,
  secondPosition: steps > 0 && laterLegal && laterRestored,
}));
