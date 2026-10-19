// Bundled by tests/web/test_event_tracks.py: the action log of a headline whose two events move VP,
// under node.
//   node event_tracks.mjs <public dir> <cases.json>
// Each case is a saved position at the headline, the two cards to commit, and the clicks after
// them (`then`: [decision_type, primary_id, forced die]). Each is opened in the page's GameSession
// on the WebAssembly engine, the clicks are made, and the last log entry and the steps' event
// records come back for the Python side to check.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";

interface Case { save: string; us: number; ussr: number; then?: [number, number, number][] }

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const cases: Record<string, Case> = JSON.parse(readFileSync(process.argv[3], "utf8"));
const out: Record<string, unknown> = {};
for (const [name, c] of Object.entries(cases)) {
  const s = new GameSession(engine);
  s.loadPosition(c.save, name);
  for (let k = 0; k < 2; k++) {
    const card = engine.decisionPlayer() > 0 ? c.us : c.ussr;
    const why = s.apply({ decision_type: 1, primary_id: card, secondary_id: 0, flags: 0 });
    if (why) throw new Error(`${name}: ${why}`);
  }
  for (const [type, primary, die] of c.then ?? []) {
    const why = s.apply({ decision_type: type, primary_id: primary, secondary_id: 0, flags: 0 }, die);
    if (why) throw new Error(`${name}: ${why}`);
  }
  const last = s.logs[s.logs.length - 1];
  out[name] = { text: last.text, details: last.details ?? [], event_tracks: s.steps.map(st => st.event_tracks ?? []) };
}
console.log(JSON.stringify(out));
