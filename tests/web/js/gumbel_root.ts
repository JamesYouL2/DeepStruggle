// Bundled by tests/web/test_page_gumbel.py (esbuild) and run with node: the page's Gumbel root
// (web/ui/src/search/mcts.ts, GUMBEL_64) against the properties that make it the player
// `search:<model>:64:determinize:all:gumbel_k=8:gumbel_scale=0:fpu_reduction=0.2` is
// (ai/search/gumbel_root.py), with a stub network so every value is known.
//   node gumbel_root.mjs <web/ui/public/> <sigma cases JSON from Python>
//
// Pinned here:
//   - sigma(completed Q) is gumbel_root.py's, number for number (the cases come from Python);
//   - the candidates are the k largest logits, and k = 1 plays the argmax without searching;
//   - sequential halving: 8 -> 4 -> 2 -> 1 over three rounds, within the budget;
//   - equal values everywhere leave the logits to decide: the argmax is played;
//   - the move played maximises logit + sigma among the last round's pair;
//   - first-play urgency: a large reduction keeps a 2-simulation tree on its first child;
//   - the engine singleton is handed back byte-identical, and the pick is legal in the real state.
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { PageSearcher, GUMBEL_64, SEARCH_64, sigmaCompleted } from "../../../web/ui/src/search/mcts";

const [publicDir, casesJson] = process.argv.slice(2);
const engine = await WasmEngine.load(pathToFileURL(publicDir).href);
const ACT = engine.actionSize;

// ---- 1. sigma(completed Q) against the Python's ----------------------------------------------
interface SigmaCase {
  logits: Record<string, number>; value: number; n: Record<string, number>; q: Record<string, number>;
  expected: Record<string, number>;
}
const cases = JSON.parse(casesJson) as SigmaCase[];
let sigmaMaxErr = 0;
for (const c of cases) {
  const legal = Object.keys(c.logits).map(Number);
  const toMap = (o: Record<string, number>) => new Map(Object.entries(o).map(([k, v]) => [Number(k), v]));
  const got = sigmaCompleted(legal, toMap(c.logits), c.value, toMap(c.n), toMap(c.q));
  for (const a of legal) sigmaMaxErr = Math.max(sigmaMaxErr, Math.abs(got.get(a)! - c.expected[String(a)]));
}
const sigmaMatchesPython = cases.length > 0 && sigmaMaxErr < 1e-9;

// ---- stubs ------------------------------------------------------------------------------------
const logit = (i: number) => ((i * 37) % 101) / 100 - 0.5;
const makeStub = (value: (obs: Float32Array) => number) => ({
  async run(obs: Float32Array, masks: Uint8Array, rows: number) {
    const logits = new Float32Array(rows * ACT);
    for (let r = 0; r < rows; r++) {
      for (let i = 0; i < ACT; i++) logits[r * ACT + i] = masks[r * ACT + i] ? logit(i) : -1e9;
    }
    return { logits, vWin: new Float32Array(rows).fill(value(obs)), vVp: new Float32Array(rows).fill(0) };
  },
});
const flat = makeStub(() => 0.5);
// A value that depends on the position, so candidates' values differ: a hash of the observation.
const hashed = makeStub(obs => {
  let h = 2166136261;
  for (let i = 0; i < obs.length; i++) h = Math.imul(h ^ Math.round(obs[i] * 1000), 16777619);
  return ((h >>> 0) % 2001) / 1000 - 1;
});

const legalHere = (): number[] => {
  const mask = engine.mask(false);
  const out: number[] = [];
  for (let i = 0; i < mask.length; i++) if (mask[i]) out.push(i);
  return out;
};
const sameBytes = (a: Uint8Array, b: Uint8Array): boolean => a.length === b.length && a.every((x, i) => x === b[i]);

engine.newGame(42);
const real = engine.snapshot();
const legal = legalHere();
const byLogit = legal.slice().sort((a, b) => logit(b) - logit(a));
const argmax = byLogit[0];
const fixtureHasEight = legal.length >= 8 && argmax !== legal[0];

// ---- 2. candidates, k = 1, halving, budget ---------------------------------------------------
const k1 = await new PageSearcher(engine, flat, false, { ...GUMBEL_64, gumbelK: 1 }).search();
const kOnePlaysArgmax = k1.action === argmax && k1.simulations === 0 && k1.fellBack;

const g = await new PageSearcher(engine, hashed, false, GUMBEL_64).search();
const cands = g.rows.filter(r => r.candidate).map(r => r.idx).sort((a, b) => a - b);
const candidatesAreTopK = JSON.stringify(cands) === JSON.stringify(byLogit.slice(0, 8).sort((a, b) => a - b));
const outIn = (ph: number) => g.rows.filter(r => r.candidate && r.droppedInPhase === ph).length;
const halving = g.worlds === 3 && outIn(0) === 4 && outIn(1) === 2 && outIn(2) === 1
  && g.rows.filter(r => r.droppedInPhase === null).length === 1
  && g.rows.find(r => r.droppedInPhase === null)!.idx === g.action;
const withinBudget = g.simulations > 0 && g.simulations <= GUMBEL_64.simulations;
// Simulations per candidate: floor(64 / (3 x survivors)) in each round it took part in.
const perRound = [Math.floor(64 / 24), Math.floor(64 / 12), Math.floor(64 / 6)];
const spentRight = g.rows.filter(r => r.candidate).every(r => {
  const rounds = r.droppedInPhase === null ? 3 : (r.droppedInPhase as number) + 1;
  return r.visits === perRound.slice(0, rounds).reduce((s, x) => s + x, 0);
});

// The last round: the two who reached it, ranked by logit + sigma (log prior orders as logit).
const finalists = g.rows.filter(r => r.candidate && (r.droppedInPhase === null || r.droppedInPhase === 2));
const score = (r: (typeof g.rows)[number]) => Math.log(r.prior) + (r.sigma ?? 0);
const pickMaximisesScore = finalists.length === 2 && finalists.every(r => score(r) <= score(finalists.find(f => f.idx === g.action)!) + 1e-9);

// ---- 3. equal values: the logits decide -------------------------------------------------------
const eq = await new PageSearcher(engine, flat, false, GUMBEL_64).search();
const equalValuesPlayArgmax = eq.action === argmax;

// ---- 4. first-play urgency in the PUCT tree --------------------------------------------------
const fpu = await new PageSearcher(engine, flat, false, { ...SEARCH_64, determinize: false, simulations: 2, fpuReduction: 10 }).search();
const fpuKeepsFirstChild = fpu.rows.filter(r => r.visits > 0).length === 1 && fpu.rows.find(r => r.idx === argmax)!.visits === 2;

// ---- 5. the engine handed back, the pick legal; and off the opening ---------------------------
const engineRestored = sameBytes(engine.snapshot(), real);
const pickLegal = g.action !== null && legal.includes(g.action);
const ROLL_DIE = engine.layout.decision_types["ROLL_DIE"] ?? 7;
for (let s = 0; s < 3; s++) {
  const l = legalHere();
  if (l.length === 0 || engine.isTerminal()) break;
  const refused = engine.step(engine.decodeFlat(l[0]), 0);
  if (refused) throw new Error(`fixture step refused: ${refused}`);
  let guard = 0;
  while (!engine.isTerminal() && engine.decisionPlayer() === 0 && engine.decisionType() === ROLL_DIE) {
    if (++guard > 64) throw new Error("chance drain did not terminate in the fixture");
    engine.step({ decision_type: ROLL_DIE, primary_id: 0, secondary_id: 0, flags: 0 }, 0);
  }
}
const afterSteps = engine.snapshot();
const later = await new PageSearcher(engine, hashed, false, GUMBEL_64).search();
const secondPosition = later.action !== null && legalHere().includes(later.action) && sameBytes(engine.snapshot(), afterSteps);

console.log(JSON.stringify({
  sigmaMatchesPython, sigmaMaxErr, fixtureHasEight, kOnePlaysArgmax, candidatesAreTopK, halving,
  withinBudget, spentRight, pickMaximisesScore, equalValuesPlayArgmax, fpuKeepsFirstChild,
  engineRestored, pickLegal, secondPosition,
}));
