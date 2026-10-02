// Bundled by tests/web/test_page_playouts.py (esbuild) and run with node: the page's paired
// playouts (web/ui/src/analysis/playouts.ts) on positions the Python side built.
//   node page_playouts.mjs <web/ui/public/> <case.json>
// A stub network stands in for a model: its logits are a pure function of the observation, so two
// games in the same position always play the same move whatever batch they sit in.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { WasmEngine, PLAYER_US } from "../../../web/ui/src/engine/wasm_engine";
import { PairedPlayouts, Rng, losesOnTheSpot, sampleWorld } from "../../../web/ui/src/analysis/playouts";
import { Forward } from "../../../web/ui/src/analysis/model";

const [publicDir, casePath] = process.argv.slice(2);
const engine = await WasmEngine.load(pathToFileURL(publicDir).href);
const c = JSON.parse(readFileSync(casePath, "utf8")) as {
  mid: string; headline: string; coup: string; coup_bg: number; coup_non_bg: number;
};

const stub = {
  async run(obs: Float32Array, masks: Uint8Array, rows: number): Promise<Forward> {
    const n = engine.obsSize;
    const A = engine.actionSize;
    const logits = new Float32Array(rows * A);
    for (let r = 0; r < rows; r++) {
      let h = 2166136261;
      for (let i = 0; i < n; i += 7) h = Math.imul(h ^ Math.round(obs[r * n + i] * 1000), 16777619);
      for (let a = 0; a < A; a++) logits[r * A + a] = ((Math.imul(h ^ a, 2654435761) >>> 0) % 1000) / 1000;
    }
    return { logits, vWin: new Float32Array(rows), vVp: new Float32Array(rows) };
  },
};

type Save = { card_locations: number[]; headline_us_card: number; headline_ussr_card: number };
const legalOf = (m: Uint8Array) => [...m.keys()].filter(i => m[i]);
const bytesEqual = (a: Uint8Array, b: Uint8Array) => a.length === b.length && a.every((x, i) => x === b[i]);

// 1. The sampled world keeps every count and every card the mover has seen.
engine.loadSaveJson(c.mid);
const me = engine.decisionPlayer();
const before = JSON.parse(engine.saveJson()) as Save;
sampleWorld(engine, me, new Rng(7));
const after = JSON.parse(engine.saveJson()) as Save;
const count = (s: Save, loc: number) => s.card_locations.filter(x => x === loc).length;
const oppUnknown = me === PLAYER_US ? 4 : 2;
let seenKept = true;
for (let cid = 1; cid <= 110; cid++) {
  const was = before.card_locations[cid];
  if (was !== oppUnknown && was !== 1 && after.card_locations[cid] !== was) seenKept = false;
}
const world = {
  counts_kept: [1, 2, 3, 4, 5, 6, 7].every(l => count(before, l) === count(after, l)),
  seen_kept: seenKept,
  shuffled: JSON.stringify(before.card_locations) !== JSON.stringify(after.card_locations),
};

// 2. An opponent headline chosen but not revealed is forgotten.
engine.loadSaveJson(c.headline);
const hlMe = engine.decisionPlayer();
const hlBefore = JSON.parse(engine.saveJson()) as Save;
sampleWorld(engine, hlMe, new Rng(3));
const hlAfter = JSON.parse(engine.saveJson()) as Save;
const headline = {
  had_choice: (hlMe === PLAYER_US ? hlBefore.headline_ussr_card : hlBefore.headline_us_card) !== 0,
  forgotten: (hlMe === PLAYER_US ? hlAfter.headline_ussr_card : hlAfter.headline_us_card) === 0,
};

// 3. The no-suicide check: a battleground coup at DEFCON 2 loses on the spot, a non-battleground one does not.
engine.loadSaveJson(c.coup);
const coupMover = engine.decisionPlayer();
const coupBytes = engine.snapshot();
const suicide = {
  bg_loses: losesOnTheSpot(engine, c.coup_bg, false, coupMover),
  non_bg_safe: !losesOnTheSpot(engine, c.coup_non_bg, false, coupMover),
  engine_back: bytesEqual(coupBytes, engine.snapshot()),
};

// 4. Paired playouts: the engine handed back byte-identical, the same move twice pairs to exactly 0,
//    and the same seed replays the same games.
engine.loadSaveJson(c.mid);
const real = engine.snapshot();
const legal = legalOf(engine.mask(false));
const cfg = { pairs: 6, wave: 4, batch: 64, seed: 11, maxDecisions: 4000 };
const progress: number[] = [];
const runner = new PairedPlayouts(engine, stub, false, cfg);
const res = await runner.run([legal[0], legal[0], legal[1]], p => progress.push(p.pairs));
const again = await new PairedPlayouts(engine, stub, false, cfg).run([legal[0], legal[0], legal[1]]);
let stopped = 0;
const early = await new PairedPlayouts(engine, stub, false, { ...cfg, pairs: 12 }).run(
  [legal[0], legal[1]], undefined, () => ++stopped > 30);
const playouts = {
  engine_back: bytesEqual(real, engine.snapshot()),
  pairs: res.pairs, games: res.games, capped: res.capped,
  same_move_diff: res.rows[1].diff, same_move_se: res.rows[1].diffSe,
  scores_in_range: res.rows.every(r => r.score >= 0 && r.score <= 1),
  reproducible: JSON.stringify(res.rows) === JSON.stringify(again.rows),
  progress,
  early_pairs: early.pairs,
};

console.log(JSON.stringify({ world, headline, suicide, playouts }));
