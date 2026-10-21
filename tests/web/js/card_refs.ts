// Bundled by tests/web/test_card_refs.py: the cards a log line names, made hoverable
// (game/card_refs.ts), under node.
//   node card_refs.mjs <public dir> <seed> <cases.json>
// Plays two turns of a game in the page's GameSession on the WebAssembly engine, by a fixed
// choice among the legal actions, groups its log as the page does, and reports every line and
// heading next to what linkCardRefs makes of it -- and the same for the hand-written cases.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { WasmEngine } from "../../../web/ui/src/engine/wasm_engine";
import { GameSession } from "../../../web/ui/src/game/session";
import { groupLog } from "../../../web/ui/src/game/log_view";
import { CardNames, linkCardRefs } from "../../../web/ui/src/game/card_refs";
import { CARDS_METADATA } from "../../../web/ui/src/metadata";

const engine = await WasmEngine.load(pathToFileURL(process.argv[2]).href);
const s = new GameSession(engine);
s.newGame(Number(process.argv[3]));
let x = Number(process.argv[3]) >>> 0;
// A move that ends the game is taken back and another chosen, as log_view.ts does.
while (!engine.isTerminal() && engine.display().turn <= 2) {
  for (let tries = 0; tries < 50; tries++) {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    const legal: number[] = [];
    engine.mask(false).forEach((v, i) => { if (v) legal.push(i); });
    const why = s.applyFlat(legal[x % legal.length], false);
    if (why) throw new Error(why);
    if (!engine.isTerminal() || tries === 49) break;
    s.undo(false);
  }
}

// The names main.ts gives it: the engine's and the rules'.
const names = new Map<number, string[]>();
const ids = new Map<string, number>();
for (const c of CARDS_METADATA) {
  const all = [...new Set([engine.cardName(c.id), c.name].filter(n => n))];
  names.set(c.id, all);
  for (const n of all) ids.set(n, c.id);
}
const cards: CardNames = { namesOf: id => names.get(id) ?? [], idOf: n => ids.get(n) };

const lines = groupLog(s.logs).flatMap(e => [e.title, ...e.lines]);
const cases = JSON.parse(readFileSync(process.argv[4], "utf8")) as string[];
const out = (text: string) => ({ text, html: linkCardRefs(text, cards) });
console.log(JSON.stringify({
  game: lines.map(out),
  cases: cases.map(out),
  names: Object.fromEntries([...names].map(([id, n]) => [id, n])),
}));
