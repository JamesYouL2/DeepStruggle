// Driven by tests/web/test_wasm_engine.py: runs the WebAssembly engine in Node and prints what
// the Python side compares against the native build.
//   node wasm_parity.mjs <ts_engine.mjs> selftest <games>
//   node wasm_parity.mjs <ts_engine.mjs> walk <actions.json>
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [modulePath, mode, arg] = process.argv.slice(2);
const createModule = (await import(pathToFileURL(modulePath).href)).default;
const M = await createModule();

if (mode === "selftest") {
  const stepsPtr = M._malloc(4);
  const digest = M._ts_selftest(Number(arg), stepsPtr) >>> 0;
  console.log(JSON.stringify({ digest, steps: M.HEAPU32[stepsPtr >> 2], fingerprint: M.UTF8ToString(M._ts_fingerprint()) }));
  process.exit(0);
}

// rollout: {seed, actions, k, worlds, horizon, batch_seed} -- walk to the position, then the
// rollout batch (bindings/rollout_batch.hpp) on its first k legal moves under a fixed synthetic
// policy and critic, so the Python module's batch can be compared row for row.
if (mode === "rollout") {
  const spec = JSON.parse(readFileSync(arg, "utf8"));
  M._ts_new_game(spec.seed >>> 0, 0);
  for (const [t, p, s, f] of spec.actions) {
    if (!M._ts_step(t, p, s, f, 0)) {
      console.log(JSON.stringify({ error: `refused: ${M.UTF8ToString(M._ts_last_error())}` }));
      process.exit(0);
    }
  }
  const ACT = M._ts_action_size();
  const mask = new Uint8Array(M.HEAPU8.buffer, M._ts_mask(0), ACT).slice();
  const cands = [];
  for (let a = 0; a < ACT && cands.length < spec.k; a++) if (mask[a]) cands.push(a);
  const candPtr = M._malloc(4 * cands.length);
  M.HEAP32.set(cands, candPtr >> 2);
  const stop = { boundaries: 0, turn: 1, later: 2, earlier: 3, predeal: 4, earlier_predeal: 5 }[spec.stop ?? "boundaries"];
  const leaf = { mover: 0, decider: 1, own: 2 }[spec.leaf ?? "mover"];
  if (!M._ts_rollout_reset(spec.horizon, 1, 1, 0, spec.batch_seed >>> 0, Math.floor(spec.batch_seed / 4294967296) >>> 0, stop, leaf)) {
    console.log(JSON.stringify({ error: M.UTF8ToString(M._ts_last_error()) }));
    process.exit(0);
  }
  const rows = M._ts_rollout_add(candPtr, cands.length, spec.worlds);
  const W = M._ts_rollout_obs_width();
  const hash = createHash("sha256");
  let plies = 0;
  // The synthetic policy: logit[a] = ((a * 2654435761 + ply * 40503 + 7) mod 1000) / 1000.
  while (!M._ts_rollout_all_done()) {
    const n = M._ts_rollout_fill_live();
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_obs(), n * W * 4).slice());
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_masks(), n * ACT).slice());
    const lgPtr = M._malloc(4 * n * ACT);
    const lg = new Float32Array(M.HEAPF32.buffer, lgPtr, n * ACT);
    for (let j = 0; j < n; j++) {
      for (let a = 0; a < ACT; a++) lg[j * ACT + a] = (((a * 2654435761) % 4294967296 + plies * 40503 + 7) % 1000) / 1000;
    }
    M._ts_rollout_step(lgPtr);
    M._free(lgPtr);
    plies += 1;
    if (plies > 2000) break;
  }
  const leaves = M._ts_rollout_fill_leaves();
  hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_rollout_buf_obs(), leaves * W * 4).slice());
  const vPtr = M._malloc(4 * Math.max(1, leaves));
  const lv = new Float32Array(M.HEAPF32.buffer, vPtr, Math.max(1, leaves));
  for (let j = 0; j < leaves; j++) lv[j] = ((j * 31) % 17) / 17 - 0.5;
  M._ts_rollout_set_leaf_values(vPtr);
  const values = Array.from(new Float32Array(M.HEAPF32.buffer, M._ts_rollout_values(), rows));
  const table = [];
  for (let r = 0; r < rows; r++) {
    table.push([M._ts_rollout_row_position(r), M._ts_rollout_row_world(r), M._ts_rollout_row_candidate(r),
                M._ts_rollout_row_done(r), M._ts_rollout_row_terminal(r)]);
  }
  // The decision under each rule, from synthetic root logits: logit[a] = ((a * 7919) mod 97) / 97.
  const rootPtr = M._malloc(4 * ACT), maskPtr = M._malloc(ACT);
  const root = new Float32Array(M.HEAPF32.buffer, rootPtr, ACT);
  for (let a = 0; a < ACT; a++) root[a] = ((a * 7919) % 97) / 97;
  M.HEAPU8.set(mask, maskPtr);
  const topPtr = M._malloc(4 * 3);
  const nTop = M._ts_rollout_candidates(rootPtr, maskPtr, 3, topPtr);
  const top = Array.from(M.HEAP32.subarray(topPtr >> 2, (topPtr >> 2) + nTop));
  const decisions = {};
  for (const rule of ["argmax", "z2", "kl0.05"]) {
    const rulePtr = M.stringToNewUTF8(rule);
    decisions[rule] = JSON.parse(M.UTF8ToString(M._ts_rollout_decide(0, candPtr, cands.length, rootPtr, maskPtr, rulePtr)));
    M._free(rulePtr);
  }
  console.log(JSON.stringify({ candidates: cands, rows, worlds_used: M._ts_rollout_worlds_used(0), plies, leaves,
                               refused: M._ts_rollout_refused(), digest: hash.digest("hex"), values, table,
                               top, decisions }));
  process.exit(0);
}

// walk: {seed, actions: [[type, primary, secondary, flags], ...]}
const { seed, actions } = JSON.parse(readFileSync(arg, "utf8"));
const OBS = M._ts_obs_size();
const ACT = M._ts_action_size();
const hash = createHash("sha256");
const enc = new TextEncoder();
const record = () => {
  hash.update(enc.encode(M.UTF8ToString(M._ts_display_json())));
  hash.update(enc.encode(M.UTF8ToString(M._ts_save_json())));
  for (const p of [1, -1]) {
    hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_observation(p), OBS * 4).slice());
  }
  hash.update(new Uint8Array(M.HEAPU8.buffer, M._ts_mask(0), ACT).slice());
};
M._ts_new_game(seed >>> 0, 0);
record();
let steps = 0;
for (const [t, p, s, f] of actions) {
  if (!M._ts_step(t, p, s, f, 0)) {
    console.log(JSON.stringify({ error: `refused at step ${steps}: ${M.UTF8ToString(M._ts_last_error())}` }));
    process.exit(0);
  }
  record();
  steps += 1;
}
console.log(JSON.stringify({ digest: hash.digest("hex"), steps, ending: M._ts_is_terminal() ? M.UTF8ToString(M._ts_ending_reason()) : null }));
