/**
 * A region's scoring, side by side: the US's summary on the left, the USSR's on the right, the
 * rows between them -- standing, battlegrounds, the other countries, those next to the enemy
 * superpower, the total -- as the Playdek app's scoring screen lays it out. Shown on hovering a
 * region's badge and, pinned in the modal, on clicking (or tapping) it.
 *
 * Every number is the engine's (WasmEngine.regionScores): what the scoring counted and what it
 * awards. Nothing here adds up a score.
 */
import type { RegionScoreAudit } from "./map_view";
import type { RegionalStatus, RegionSideScore } from "./types";
import { escapeHtml } from "./html";

const STANDING_NAME: Record<RegionalStatus, string> = {
  NONE: "None", PRESENCE: "Presence", DOMINATION: "Domination", CONTROL: "Control",
};

/** The same, as a map badge has room for. */
export const STANDING_SHORT: Record<RegionalStatus, string> = {
  NONE: "—", PRESENCE: "Pres", DOMINATION: "Dom", CONTROL: "Ctrl",
};

const title = (s: string): string => s.toLowerCase().replace(/\b\w/g, c => c.toUpperCase());

export type NetSide = "us" | "ussr" | "even";

/** "+3 USSR", "+2 US", "even", or who wins outright -- in full, and as short as a map badge needs. */
export function regionNet(a: RegionScoreAudit): { text: string; short: string; side: NetSide } {
  if (a.wins === "US") return { text: "US wins the game", short: "US WINS", side: "us" };
  if (a.wins === "USSR") return { text: "USSR wins the game", short: "USSR WINS", side: "ussr" };
  const vp = (text: string, side: NetSide) => ({ text, short: text, side });
  if (a.netVpUs > 0) return vp(`+${a.netVpUs} US`, "us");
  if (a.netVpUs < 0) return vp(`+${-a.netVpUs} USSR`, "ussr");
  return vp("even", "even");
}

function row(label: string, us: string, ussr: string, cls = ""): string {
  return `<div class="rs-row ${cls}"><span class="rs-us">${us}</span><span class="rs-label">${label}</span><span class="rs-ussr">${ussr}</span></div>`;
}

/** A count and the VP it brings: "2 · +2". */
const counted = (n: number, vp: number): string => `${n}<small>${vp ? ` · +${vp}` : ""}</small>`;

/** A standing and what it alone is worth: "Domination · 7". */
function standing(s: RegionSideScore): string {
  const name = STANDING_NAME[s.status ?? "NONE"];
  return `${name}<small>${s.base_vp ? ` · ${s.base_vp}` : ""}</small>`;
}

export function regionSummaryHtml(a: RegionScoreAudit): string {
  const net = regionNet(a);
  let rows: string;
  if (a.southeastAsia) {
    // Southeast Asia Scoring has no standings: each country is worth its VP (Thailand 2).
    const worth = (s: RegionSideScore, vp: number) => s.controlled.filter(c => c.vp === vp).length;
    rows = row("Countries (1 VP)", String(worth(a.us, 1)), String(worth(a.ussr, 1)))
      + row("Thailand (2 VP)", worth(a.us, 2) ? "✓" : "—", worth(a.ussr, 2) ? "✓" : "—");
  } else {
    const other = (s: RegionSideScore) => String(s.countries - s.battlegrounds);
    rows = row("Standing", standing(a.us), standing(a.ussr), "rs-standing")
      + row(`Battlegrounds <small>of ${a.totalBattlegrounds}</small>`,
            counted(a.us.battlegrounds, a.us.battlegrounds), counted(a.ussr.battlegrounds, a.ussr.battlegrounds))
      + row("Other countries", other(a.us), other(a.ussr))
      + row("Next to the enemy", counted(a.us.adjacent, a.us.adjacent), counted(a.ussr.adjacent, a.ussr.adjacent));
  }
  rows += row("Total", String(a.us.vp), String(a.ussr.vp), "rs-total");
  const list = (names: string[]): string => escapeHtml(names.length ? names.join(", ") : "none");
  const uncounted = a.shuttleDiplomacy
    ? `<div class="rs-foot">Shuttle Diplomacy: one USSR battleground is not counted in this scoring</div>`
    : "";
  return `
    <div class="rs" style="--rs-accent: ${escapeHtml(a.color)}" data-region="${escapeHtml(a.regionName)}">
      <div class="rs-head"><span class="rs-title">${escapeHtml(title(a.regionName))} Scoring</span><span class="rs-net rs-${net.side}">${net.text}</span></div>
      <div class="rs-sides"><span class="rs-us">US</span><span></span><span class="rs-ussr">USSR</span></div>
      ${rows}
      <div class="rs-list"><span class="rs-us">US controls</span> ${list(a.usControlled)}</div>
      <div class="rs-list"><span class="rs-ussr">USSR controls</span> ${list(a.ussrControlled)}</div>
      ${uncounted}
      <div class="rs-foot">${escapeHtml(a.scoringCardText)}</div>
      <div class="rs-foot">${escapeHtml(a.scoringCardName)}: ${escapeHtml(a.scoringCard.where)}</div>
    </div>`;
}
