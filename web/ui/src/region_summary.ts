/**
 * A region's scoring, side by side: the US's summary on the left, the USSR's on the right, the
 * rows between them -- standing, battlegrounds, the other countries, those next to the enemy
 * superpower, the total -- as the Playdek app's scoring screen lays it out. Shown on hovering a
 * region's badge and, pinned in the modal, on clicking (or tapping) it.
 */
import type { RegionScoreAudit } from "./map_view";

const STANDING_NAME: Record<RegionScoreAudit["usStatus"], string> = {
  NONE: "None", PRESENCE: "Presence", DOMINATION: "Domination", CONTROL: "Control",
};

/** The same, as a map badge has room for. */
export const STANDING_SHORT: Record<RegionScoreAudit["usStatus"], string> = {
  NONE: "\u2014", PRESENCE: "Pres", DOMINATION: "Dom", CONTROL: "Ctrl",
};

/** The card's own terms, per region id (Europe's Control wins the game outright). */
const TERMS: Record<number, string> = {
  0: "Presence 3 · Domination 7 · Control wins the game",
  1: "Presence 3 · Domination 7 · Control 9",
  2: "Presence 3 · Domination 5 · Control 7",
  3: "Presence 1 · Domination 4 · Control 6",
  4: "Presence 1 · Domination 3 · Control 5",
  5: "Presence 2 · Domination 5 · Control 6",
};

const title = (s: string): string => s.toLowerCase().replace(/\b\w/g, c => c.toUpperCase());

export type NetSide = "us" | "ussr" | "even";

/** "+3 USSR", "+2 US", "even", or who wins outright -- in full, and as short as a map badge needs. */
export function regionNet(a: RegionScoreAudit): { text: string; short: string; side: NetSide } {
  if (a.isInstantWinUs) return { text: "US wins the game", short: "US WINS", side: "us" };
  if (a.isInstantWinUssr) return { text: "USSR wins the game", short: "USSR WINS", side: "ussr" };
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

export function regionSummaryHtml(a: RegionScoreAudit): string {
  const net = regionNet(a);
  const se = a.regionId === 6;
  let rows: string;
  if (se) {
    rows = row("Countries (1 VP)", String(a.usCountries - a.usBattlegrounds), String(a.ussrCountries - a.ussrBattlegrounds))
      + row("Thailand (2 VP)", a.usBattlegrounds ? "✓" : "—", a.ussrBattlegrounds ? "✓" : "—");
  } else {
    const usOther = a.usCountries - a.usBattlegrounds, ussrOther = a.ussrCountries - a.ussrBattlegrounds;
    rows = row("Standing",
      `${STANDING_NAME[a.usStatus]}<small>${a.usBaseVp ? ` · ${a.usBaseVp}` : ""}</small>`,
      `${STANDING_NAME[a.ussrStatus]}<small>${a.ussrBaseVp ? ` · ${a.ussrBaseVp}` : ""}</small>`, "rs-standing")
      + row(`Battlegrounds <small>of ${a.totalBattlegrounds}</small>`, counted(a.usBattlegrounds, a.usBattlegrounds), counted(a.ussrBattlegrounds, a.ussrBattlegrounds))
      + row("Other countries", String(usOther), String(ussrOther))
      + row("Next to the enemy", counted(a.usSuperpowerAdjacent, a.usSuperpowerAdjacent), counted(a.ussrSuperpowerAdjacent, a.ussrSuperpowerAdjacent));
  }
  rows += row("Total", String(a.usTotalVp), String(a.ussrTotalVp), "rs-total");
  const list = (names: string[]): string => (names.length ? names.join(", ") : "none");
  return `
    <div class="rs" style="--rs-accent: ${a.color}" data-region="${a.regionName}">
      <div class="rs-head"><span class="rs-title">${title(a.regionName)} Scoring</span><span class="rs-net rs-${net.side}">${net.text}</span></div>
      <div class="rs-sides"><span class="rs-us">US</span><span></span><span class="rs-ussr">USSR</span></div>
      ${rows}
      <div class="rs-list"><span class="rs-us">US controls</span> ${list(a.usControlled)}</div>
      <div class="rs-list"><span class="rs-ussr">USSR controls</span> ${list(a.ussrControlled)}</div>
      <div class="rs-foot">${se ? "1 VP a country, Thailand 2; scored once, then removed" : `${TERMS[a.regionId] ?? ""}; +1 a battleground, +1 a country next to the enemy superpower`}</div>
      <div class="rs-foot">${a.scoringCardName}: ${a.scoringCard.where}</div>
    </div>`;
}
