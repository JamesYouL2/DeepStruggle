/**
 * The roll tray: the latest action round's dice on the map, as faces with their modifiers, total,
 * target and outcome -- the coup, realignment and war overlays of other Twilight Struggle tables,
 * in one place. The rolls come from `state.recent_rolls` (game/rolls.ts).
 */
import { RecentRoll } from "./types";

type Side = "US" | "USSR";

const PIPS: Record<number, Array<[number, number]>> = {
  1: [[2, 2]],
  2: [[1, 1], [3, 3]],
  3: [[1, 1], [2, 2], [3, 3]],
  4: [[1, 1], [3, 1], [1, 3], [3, 3]],
  5: [[1, 1], [3, 1], [2, 2], [1, 3], [3, 3]],
  6: [[1, 1], [3, 1], [1, 2], [3, 2], [1, 3], [3, 3]],
};

/** A die face in a side's colour, as inline SVG; a value outside 1..6 shows as the number. */
export function dieSvg(value: number, side: Side | string): string {
  const cls = side === "US" ? "die-us" : side === "USSR" ? "die-ussr" : "die-neutral";
  const pips = PIPS[value];
  const body = pips
    ? pips.map(([x, y]) => `<circle cx="${x * 5}" cy="${y * 5}" r="1.7"/>`).join("")
    : `<text x="10" y="13.6" text-anchor="middle">${value}</text>`;
  return `<svg class="die ${cls}" viewBox="0 0 20 20" width="22" height="22" role="img" aria-label="${side} rolls ${value}">`
    + `<rect x="0.75" y="0.75" width="18.5" height="18.5" rx="4"/>${body}</svg>`;
}

const signed = (v: number): string => (v >= 0 ? `+${v}` : `−${-v}`);
const sideTag = (s: string): string => `<span class="ct-${s.toLowerCase()}">${s}</span>`;
const other = (s: string): Side => (s === "US" ? "USSR" : "US");
const esc = (s: string): string => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]!));

/**
 * A die, its modifier and the total: "[4] +3 = 7". Unmodified, just the die -- unless it is one
 * side of a contest, where both totals are shown so they can be compared ("[3] +0 = 3").
 */
function term(roll: number, mod: number, side: string, contest = false): string {
  const shown = mod || contest;
  return dieSvg(roll, side) + (shown ? `<span class="roll-mod">${signed(mod)}</span><span class="roll-eq">=</span><b>${roll + mod}</b>` : "");
}

interface RollView {
  kind: string;
  where: string;
  dice: string;
  result: string;
  ok: boolean;
}

/** What a roll was, how it was made and how it came out. */
export function describeRoll(r: RecentRoll): RollView {
  const roller = r.roller ?? "NONE";
  const r1 = r.roll1 ?? 0, m1 = r.mod1 ?? 0, r2 = r.roll2 ?? 0, m2 = r.mod2 ?? 0;
  const net = r.net_delta ?? 0;
  const card = r.card_name ?? "";
  const country = r.country_name ?? "";
  switch (r.type) {
    case "COUP":
      return {
        kind: "Coup", where: country, ok: !!r.success,
        dice: `${term(r1, m1, roller)}<span class="roll-vs">vs</span><b>${m2}</b>`,
        result: r.success ? `${sideTag(roller)} succeeds by ${net}` : "fails",
      };
    case "REALIGNMENT": {
      const us = r1 + m1, ussr = r2 + m2;
      const winner = us > ussr ? "US" : ussr > us ? "USSR" : "";
      return {
        kind: "Realignment", where: country, ok: winner === roller,
        dice: `${term(r1, m1, "US", true)}<span class="roll-vs">vs</span>${term(r2, m2, "USSR", true)}`,
        result: !winner ? "tie, nothing removed"
          : net ? `${sideTag(winner)} removes ${net} ${sideTag(other(winner))}`
          : `${sideTag(winner)} wins, no ${sideTag(other(winner))} to remove`,
      };
    }
    case "SPACE_RACE":
      return {
        kind: "Space Race", where: `box ${r.country_id ?? "?"}`, ok: !!r.success,
        dice: `${dieSvg(r1, roller)}<span class="roll-vs">needs ≤</span><b>${m1}</b>`,
        result: r.success ? `${sideTag(roller)} advances` : "fails",
      };
    case "WAR_EVENT":
      return {
        kind: card || "War", where: country, ok: !!r.success,
        dice: `${term(r1, m1, roller)}<span class="roll-vs">needs ≥</span><b>${m2}</b>`,
        result: r.success ? `${sideTag(roller)} wins, +${net} VP` : "fails",
      };
    case "OLYMPIC_GAMES": {
      // The engine records the sponsor's roll with its +2 already in (roll1 = die + 2, mod1 = 2).
      const winner = r.success ? roller : other(roller);
      return {
        kind: "Olympic Games", where: `${roller} sponsors`, ok: !!r.success,
        dice: `${term(r1 - m1, m1, roller, true)}<span class="roll-vs">vs</span>${term(r2, m2, other(roller), true)}`,
        result: `${sideTag(winner)} +2 VP`,
      };
    }
    case "SUMMIT": {
      const winner = net > 0 ? "US" : net < 0 ? "USSR" : "";
      return {
        kind: "Summit", where: "", ok: !!r.success,
        dice: `${term(r1, m1, "US", true)}<span class="roll-vs">vs</span>${term(r2, m2, "USSR", true)}`,
        result: winner ? `${sideTag(winner)} +2 VP, may move DEFCON` : "tie",
      };
    }
    case "TRAP_ESCAPE":
      return {
        kind: card || "Trap", where: `${roller} escapes?`, ok: !!r.success,
        dice: `${dieSvg(r1, roller)}<span class="roll-vs">needs ≤</span><b>${m2 || 4}</b>`,
        result: r.success ? "escapes" : "stays trapped",
      };
    default:
      return { kind: r.type, where: country, ok: !!r.success, dice: dieSvg(r1, roller), result: "" };
  }
}

/** Fill the tray with `rolls`, or hide it when there are none. */
export function renderRollTray(el: HTMLElement, rolls: RecentRoll[]): void {
  el.classList.toggle("hidden", rolls.length === 0);
  el.innerHTML = rolls.map(r => {
    const v = describeRoll(r);
    return `<div class="roll-row ${v.ok ? "roll-ok" : "roll-fail"}" data-type="${esc(r.type)}">`
      + `<div class="roll-head"><span class="roll-kind">${esc(v.kind)}</span>`
      + (v.where ? `<span class="roll-where">${esc(v.where)}</span>` : "")
      + `<span class="roll-when">${esc(r.when)}</span></div>`
      + `<div class="roll-body">${v.dice}<span class="roll-result"><span class="roll-arrow">→</span>${v.result}</span></div>`
      + `</div>`;
  }).join("");
}
