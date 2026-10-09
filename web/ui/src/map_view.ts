import { GameState, LastTurn, LastTurnCountry, MapMetadata } from "./types";
import { MAP_METADATA } from "./metadata";

export interface RegionScoreAudit {
  regionId: number;
  regionName: string;
  badgePos: [number, number]; // [x, y]
  color: string;
  scoringCardName: string;
  /** Where the region's scoring card is, which says whether it can still score this cycle. */
  scoringCard: ScoringCardWhereabouts;
  totalBattlegrounds: number;
  usCountries: number;
  ussrCountries: number;
  usBattlegrounds: number;
  ussrBattlegrounds: number;
  usSuperpowerAdjacent: number;
  ussrSuperpowerAdjacent: number;
  usStatus: "NONE" | "PRESENCE" | "DOMINATION" | "CONTROL";
  ussrStatus: "NONE" | "PRESENCE" | "DOMINATION" | "CONTROL";
  usBaseVp: number;
  ussrBaseVp: number;
  usTotalVp: number;
  ussrTotalVp: number;
  netVpUs: number;
  isInstantWinUs?: boolean;
  isInstantWinUssr?: boolean;
  details: string[];
}

/**
 * A scoring card is `live` while it can still be played before the next reshuffle -- in the draw
 * deck or a hand (or face down in a headline, or being looked at by an event) -- and `out` once
 * discarded or removed. `later` is a Mid War scoring card before the Mid War deck comes in.
 */
export interface ScoringCardWhereabouts {
  status: "live" | "out" | "later";
  /** The engine's location, in words. */
  where: string;
}

const SCORING_CARD_WHERE: Record<string, ScoringCardWhereabouts> = {
  DRAW_DECK: { status: "live", where: "in the draw deck" },
  HAND_US_UNKNOWN: { status: "live", where: "in the US hand" },
  HAND_US_KNOWN: { status: "live", where: "in the US hand (the USSR knows)" },
  HAND_USSR_UNKNOWN: { status: "live", where: "in the USSR hand" },
  HAND_USSR_KNOWN: { status: "live", where: "in the USSR hand (the US knows)" },
  HEADLINE_COMMITTED: { status: "live", where: "headlined, not yet resolved" },
  PEEKED_TEMP: { status: "live", where: "being looked at by an event" },
  DISCARD_PILE: { status: "out", where: "discarded -- back only after a reshuffle" },
  REMOVED_FROM_GAME: { status: "out", where: "removed from the game -- never scores again" },
  ONGOING_EVENT: { status: "out", where: "in play as an ongoing event" },
  UNAVAILABLE: { status: "later", where: "not in the game yet -- comes in with its war's deck" },
  // A player's view (game/view.ts): the deck and the other side's unseen cards are one pile.
  UNSEEN: { status: "live", where: "in the draw deck or the other side's hand" },
};

export function scoringCardWhereabouts(state: GameState, cardId: number): ScoringCardWhereabouts {
  const loc = state.card_locations?.[String(cardId)] ?? "";
  return SCORING_CARD_WHERE[loc] ?? { status: "later", where: loc ? loc.toLowerCase() : "unknown" };
}

const SCORING_CARD_TAG: Record<ScoringCardWhereabouts["status"], { text: string; fill: string; stroke: string; ink: string }> = {
  live: { text: "LIVE", fill: "rgba(22, 101, 52, 0.95)", stroke: "#4ADE80", ink: "#DCFCE7" },
  out: { text: "OUT", fill: "rgba(51, 65, 85, 0.95)", stroke: "#64748B", ink: "#94A3B8" },
  later: { text: "NOT YET", fill: "rgba(15, 23, 42, 0.90)", stroke: "#475569", ink: "#64748B" },
};

export class MapView {
  private svg: SVGSVGElement;
  private mapData: MapMetadata | null = null;
  private lastState: GameState | null = null;
  private onCountryClick: (countryId: number) => void;
  private tooltipEl: HTMLElement | null = null;

  // Pan and zoom state
  private viewBox = { x: 0, y: 0, w: 1000, h: 650 };
  private isPanning = false;
  private startPoint = { x: 0, y: 0 };

  constructor(svgElement: SVGSVGElement, onCountryClick: (countryId: number) => void) {
    this.svg = svgElement;
    this.onCountryClick = onCountryClick;
    this.tooltipEl = document.getElementById("card-tooltip");
    this.setupPanAndZoom();
    this.updateViewBox();
    this.fetchMetadata();
  }

  public fetchMetadata() {
    if (this.mapData) return;
    // Bundled from rules/map.json (metadata.ts): the page needs no server to draw the board.
    this.mapData = MAP_METADATA;
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  public setMapData(data: MapMetadata) {
    this.mapData = data;
    if (this.lastState) {
      this.render(this.lastState);
    }
  }

  private updateViewBox() {
    this.svg.setAttribute(
      "viewBox",
      `${this.viewBox.x} ${this.viewBox.y} ${this.viewBox.w} ${this.viewBox.h}`
    );
  }

  public zoom(factor: number) {
    const cx = this.viewBox.x + this.viewBox.w / 2;
    const cy = this.viewBox.y + this.viewBox.h / 2;
    this.viewBox.w *= factor;
    this.viewBox.h *= factor;
    this.viewBox.x = cx - this.viewBox.w / 2;
    this.viewBox.y = cy - this.viewBox.h / 2;
    this.updateViewBox();
  }

  public resetView() {
    this.viewBox = { x: 0, y: 0, w: 1000, h: 650 };
    this.updateViewBox();
  }

  private setupPanAndZoom() {
    this.svg.addEventListener("mousedown", (e) => {
      if ((e.target as HTMLElement).closest(".svg-country-node") || (e.target as HTMLElement).closest(".svg-region-badge")) return;
      this.isPanning = true;
      this.startPoint = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener("mousemove", (e) => {
      if (!this.isPanning) return;
      const dx = ((e.clientX - this.startPoint.x) / this.svg.clientWidth) * this.viewBox.w;
      const dy = ((e.clientY - this.startPoint.y) / this.svg.clientHeight) * this.viewBox.h;
      this.viewBox.x -= dx;
      this.viewBox.y -= dy;
      this.startPoint = { x: e.clientX, y: e.clientY };
      this.updateViewBox();
    });

    window.addEventListener("mouseup", () => {
      this.isPanning = false;
    });

    this.svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const zoomFactor = e.deltaY > 0 ? 1.1 : 0.9;
      this.zoom(zoomFactor);
    }, { passive: false });
  }

  public calculateRegionalScoring(state: GameState): RegionScoreAudit[] {
    if (!this.mapData) return [];

    const countries = this.mapData.countries || [];
    const flags = new Set(state.flags || []);

    const formosanActive = flags.has("FORMOSAN_RESOLUTION_ACTIVE");
    const shuttleActive = flags.has("SHUTTLE_DIPLOMACY_ACTIVE");

    const regionConfigs: Array<{
      id: number;
      name: string;
      cardName: string;
      cardId: number;
      color: string;
      pos: [number, number];
      baseVps: { PRESENCE: number; DOMINATION: number; CONTROL: number };
    }> = [
      { id: 0, name: "EUROPE", cardName: "Europe Scoring (#2)", cardId: 2, color: "#3B82F6", pos: [440, 24], baseVps: { PRESENCE: 3, DOMINATION: 7, CONTROL: 0 } },
      { id: 1, name: "ASIA", cardName: "Asia Scoring (#1)", cardId: 1, color: "#EA580C", pos: [830, 180], baseVps: { PRESENCE: 3, DOMINATION: 7, CONTROL: 9 } },
      { id: 2, name: "MIDDLE EAST", cardName: "Middle East Scoring (#3)", cardId: 3, color: "#0284C7", pos: [630, 230], baseVps: { PRESENCE: 3, DOMINATION: 5, CONTROL: 7 } },
      { id: 3, name: "AFRICA", cardName: "Africa Scoring (#79)", cardId: 79, color: "#D97706", pos: [425, 535], baseVps: { PRESENCE: 1, DOMINATION: 4, CONTROL: 6 } },
      { id: 4, name: "CENTRAL AMERICA", cardName: "Central America Scoring (#37)", cardId: 37, color: "#16A34A", pos: [115, 260], baseVps: { PRESENCE: 1, DOMINATION: 3, CONTROL: 5 } },
      { id: 5, name: "SOUTH AMERICA", cardName: "South America Scoring (#81)", cardId: 81, color: "#059669", pos: [232, 465], baseVps: { PRESENCE: 2, DOMINATION: 5, CONTROL: 6 } },
    ];

    const audits: RegionScoreAudit[] = [];

    // 1. Standard 6 Regions
    regionConfigs.forEach(cfg => {
      const regCountries = countries.filter(c => {
        const cRegionName = c.region.toUpperCase();
        return cRegionName === cfg.name || (cfg.id === 0 && cRegionName === "EUROPE") || (cfg.id === 1 && cRegionName === "ASIA") || (cfg.id === 2 && cRegionName.includes("MIDDLE")) || (cfg.id === 3 && cRegionName === "AFRICA") || (cfg.id === 4 && cRegionName.includes("CENTRAL")) || (cfg.id === 5 && cRegionName.includes("SOUTH"));
      });

      let taiwanIsBg = false;
      if (cfg.id === 1 && formosanActive) {
        const twState = state.countries ? state.countries["Taiwan"] : null;
        if (twState && twState.controlled_by === "US") {
          taiwanIsBg = true;
        }
      }

      let totalBg = regCountries.filter(c => c.battleground).length + (taiwanIsBg ? 1 : 0);

      let usCountries = 0;
      let ussrCountries = 0;
      let usBg = 0;
      let ussrBg = 0;
      let usSpAdj = 0;
      let ussrSpAdj = 0;

      const usControlledList: string[] = [];
      const ussrControlledList: string[] = [];

      regCountries.forEach(c => {
        const cState = state.countries ? state.countries[c.name] : null;
        const ctrl = cState ? cState.controlled_by : "NONE";
        const isBg = c.battleground || (c.name === "Taiwan" && taiwanIsBg);

        if (ctrl === "US") {
          usCountries++;
          if (isBg) usBg++;
          if (c.superpower_adjacent === "USSR") usSpAdj++;
          usControlledList.push(`${c.name}${isBg ? " ★" : ""}`);
        } else if (ctrl === "USSR") {
          ussrCountries++;
          if (isBg) ussrBg++;
          if (c.superpower_adjacent === "USA") ussrSpAdj++;
          ussrControlledList.push(`${c.name}${isBg ? " ★" : ""}`);
        }
      });

      let effectiveUssrBg = ussrBg;
      if (shuttleActive && (cfg.id === 1 || cfg.id === 2) && effectiveUssrBg > 0) {
        effectiveUssrBg--;
      }

      const usNonBg = Math.max(0, usCountries - usBg);
      const ussrNonBg = Math.max(0, ussrCountries - effectiveUssrBg);

      // Evaluate US Status
      let usStatus: "NONE" | "PRESENCE" | "DOMINATION" | "CONTROL" = "NONE";
      if (usCountries > ussrCountries && usBg === totalBg) {
        usStatus = "CONTROL";
      } else if (usCountries > ussrCountries && usBg > effectiveUssrBg && usBg >= 1 && usNonBg >= 1) {
        usStatus = "DOMINATION";
      } else if (usCountries >= 1) {
        usStatus = "PRESENCE";
      }

      // Evaluate USSR Status
      let ussrStatus: "NONE" | "PRESENCE" | "DOMINATION" | "CONTROL" = "NONE";
      if (ussrCountries > usCountries && effectiveUssrBg === totalBg) {
        ussrStatus = "CONTROL";
      } else if (ussrCountries > usCountries && effectiveUssrBg > usBg && effectiveUssrBg >= 1 && ussrNonBg >= 1) {
        ussrStatus = "DOMINATION";
      } else if (ussrCountries >= 1) {
        ussrStatus = "PRESENCE";
      }

      const usBase = usStatus !== "NONE" ? cfg.baseVps[usStatus] : 0;
      const ussrBase = ussrStatus !== "NONE" ? cfg.baseVps[ussrStatus] : 0;

      let isInstantWinUs = false;
      let isInstantWinUssr = false;
      let usTotal = 0;
      let ussrTotal = 0;
      let netVp = 0;

      if (cfg.id === 0) { // Europe
        if (usStatus === "CONTROL") {
          isInstantWinUs = true;
          usTotal = 20;
          netVp = 20;
        } else if (ussrStatus === "CONTROL") {
          isInstantWinUssr = true;
          ussrTotal = 20;
          netVp = -20;
        } else {
          usTotal = usBase + usBg + usSpAdj;
          ussrTotal = ussrBase + ussrBg + ussrSpAdj;
          netVp = usTotal - ussrTotal;
        }
      } else {
        usTotal = usBase + usBg + usSpAdj;
        ussrTotal = ussrBase + (shuttleActive && (cfg.id === 1 || cfg.id === 2) ? effectiveUssrBg : ussrBg) + ussrSpAdj;
        netVp = usTotal - ussrTotal;
      }

      const details: string[] = [
        `📊 ${cfg.name} SCORING BREAKDOWN (${cfg.cardName})`,
        `• Total Regional Battlegrounds: ${totalBg}`,
        `• US Controlled (${usCountries} total, ${usBg} BG): ${usControlledList.length ? usControlledList.join(", ") : "None"}`,
        `• USSR Controlled (${ussrCountries} total, ${ussrBg} BG): ${ussrControlledList.length ? ussrControlledList.join(", ") : "None"}`,
        `• US Standing: ${usStatus} -> Base: ${usBase} VP + ${usBg} BGs${usSpAdj > 0 ? ` + ${usSpAdj} Superpower Adj` : ""} = ${usTotal} VP`,
        `• USSR Standing: ${ussrStatus} -> Base: ${ussrBase} VP + ${ussrBg} BGs${ussrSpAdj > 0 ? ` + ${ussrSpAdj} Superpower Adj` : ""} = ${ussrTotal} VP`,
        `• Current Net VP for US: ${netVp > 0 ? "+" + netVp : netVp} VP ${isInstantWinUs ? "(US INSTANT WIN)" : (isInstantWinUssr ? "(USSR INSTANT WIN)" : "")}`
      ];

      audits.push({
        regionId: cfg.id,
        regionName: cfg.name,
        badgePos: cfg.pos,
        color: cfg.color,
        scoringCardName: cfg.cardName,
        scoringCard: scoringCardWhereabouts(state, cfg.cardId),
        totalBattlegrounds: totalBg,
        usCountries,
        ussrCountries,
        usBattlegrounds: usBg,
        ussrBattlegrounds: ussrBg,
        usSuperpowerAdjacent: usSpAdj,
        ussrSuperpowerAdjacent: ussrSpAdj,
        usStatus,
        ussrStatus,
        usBaseVp: usBase,
        ussrBaseVp: ussrBase,
        usTotalVp: usTotal,
        ussrTotalVp: ussrTotal,
        netVpUs: netVp,
        isInstantWinUs,
        isInstantWinUssr,
        details
      });
    });

    // 2. Southeast Asia Scoring (Card #38)
    const seCountryNames = ["Burma", "Laos/Cambodia", "Vietnam", "Malaysia", "Indonesia", "Philippines"];
    let seUsVp = 0;
    let seUssrVp = 0;
    const seUsList: string[] = [];
    const seUssrList: string[] = [];

    seCountryNames.forEach(cName => {
      const cState = state.countries ? state.countries[cName] : null;
      if (cState?.controlled_by === "US") {
        seUsVp += 1;
        seUsList.push(`${cName} (1 VP)`);
      } else if (cState?.controlled_by === "USSR") {
        seUssrVp += 1;
        seUssrList.push(`${cName} (1 VP)`);
      }
    });

    const thaiState = state.countries ? state.countries["Thailand"] : null;
    if (thaiState?.controlled_by === "US") {
      seUsVp += 2;
      seUsList.push("★ Thailand (2 VP)");
    } else if (thaiState?.controlled_by === "USSR") {
      seUssrVp += 2;
      seUssrList.push("★ Thailand (2 VP)");
    }

    const seNetVp = seUsVp - seUssrVp;
    audits.push({
      regionId: 6,
      regionName: "SOUTHEAST ASIA",
      badgePos: [875, 485],
      color: "#F59E0B",
      scoringCardName: "Southeast Asia Scoring (#38)",
      scoringCard: scoringCardWhereabouts(state, 38),
      totalBattlegrounds: 1,
      usCountries: seUsList.length,
      ussrCountries: seUssrList.length,
      usBattlegrounds: thaiState?.controlled_by === "US" ? 1 : 0,
      ussrBattlegrounds: thaiState?.controlled_by === "USSR" ? 1 : 0,
      usSuperpowerAdjacent: 0,
      ussrSuperpowerAdjacent: 0,
      usStatus: seUsVp > 0 ? "PRESENCE" : "NONE",
      ussrStatus: seUssrVp > 0 ? "PRESENCE" : "NONE",
      usBaseVp: seUsVp,
      ussrBaseVp: seUssrVp,
      usTotalVp: seUsVp,
      ussrTotalVp: seUssrVp,
      netVpUs: seNetVp,
      details: [
        "📊 SOUTHEAST ASIA SCORING BREAKDOWN (Card #38)",
        "• 1 VP each for Burma, Laos/Cambodia, Vietnam, Malaysia, Indonesia, Philippines. 2 VP for Thailand.",
        `• US Controlled: ${seUsList.length ? seUsList.join(", ") : "None"} (${seUsVp} VP)`,
        `• USSR Controlled: ${seUssrList.length ? seUssrList.join(", ") : "None"} (${seUssrVp} VP)`,
        `• Current Net VP for US: ${seNetVp > 0 ? "+" + seNetVp : seNetVp} VP`
      ]
    });

    return audits;
  }

  public render(state: GameState) {
    this.lastState = state;
    if (!this.mapData) {
      this.fetchMetadata();
      return;
    }

    const countries = this.mapData.countries || [];
    const countryMap = new Map<string, any>();
    countries.forEach(c => countryMap.set(c.name, c));

    this.svg.innerHTML = "";

    // 1. Connection Adjacency Lines
    const linesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    linesGroup.setAttribute("id", "adjacency-lines");
    const drawnEdges = new Set<string>();

    const superpowers = [
      { id: "USA", label: "UNITED STATES", pos: [85, 230], color: "#1E3A8A", stroke: "#3B82F6" },
      { id: "USSR", label: "SOVIET UNION", pos: [635, 120], color: "#7F1D1D", stroke: "#EF4444" }
    ];

    // Country-to-Country Edges
    countries.forEach(c => {
      (c.neighbours || []).forEach(nName => {
        const n = countryMap.get(nName);
        if (!n) return;
        const edgeKey = [Math.min(c.id, n.id), Math.max(c.id, n.id)].join("-");
        if (drawnEdges.has(edgeKey)) return;
        drawnEdges.add(edgeKey);

        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", c.pos[0].toString());
        line.setAttribute("y1", c.pos[1].toString());
        line.setAttribute("x2", n.pos[0].toString());
        line.setAttribute("y2", n.pos[1].toString());
        const isInterRegion = (c.region !== n.region);
        line.setAttribute("stroke", isInterRegion ? "#DC2626" : "#475569");
        line.setAttribute("stroke-width", isInterRegion ? "1.6" : "1.2");
        line.setAttribute("stroke-dasharray", isInterRegion ? "3,3" : "2,2");
        linesGroup.appendChild(line);
      });

      // Superpower Adjacency Edges
      if (c.superpower_adjacent) {
        const sp = superpowers.find(s => s.id === c.superpower_adjacent);
        if (sp) {
          if (c.name === "Japan") {
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("d", `M ${c.pos[0]} ${c.pos[1]} C 960 210, 960 500, 480 600 S 70 320, ${sp.pos[0]} ${sp.pos[1]}`);
            path.setAttribute("stroke", "#3B82F6");
            path.setAttribute("stroke-width", "2.0");
            path.setAttribute("stroke-dasharray", "4,4");
            path.setAttribute("fill", "none");
            path.setAttribute("opacity", "0.8");
            linesGroup.appendChild(path);
          } else {
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", c.pos[0].toString());
            line.setAttribute("y1", c.pos[1].toString());
            line.setAttribute("x2", sp.pos[0].toString());
            line.setAttribute("y2", sp.pos[1].toString());
            line.setAttribute("stroke", sp.stroke);
            line.setAttribute("stroke-width", "2.0");
            line.setAttribute("stroke-dasharray", "4,4");
            line.setAttribute("opacity", "0.8");
            linesGroup.appendChild(line);
          }
        }
      }
    });
    this.svg.appendChild(linesGroup);

    // 2. Superpower Nodes
    const spGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    spGroup.setAttribute("id", "superpower-nodes");

    superpowers.forEach(sp => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const w = 90, h = 24;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", (sp.pos[0] - w/2).toString());
      rect.setAttribute("y", (sp.pos[1] - h/2).toString());
      rect.setAttribute("width", w.toString());
      rect.setAttribute("height", h.toString());
      rect.setAttribute("rx", "4");
      rect.setAttribute("fill", sp.color);
      rect.setAttribute("stroke", sp.stroke);
      rect.setAttribute("stroke-width", "2.0");
      g.appendChild(rect);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("x", sp.pos[0].toString());
      text.setAttribute("y", (sp.pos[1] + 3.5).toString());
      text.setAttribute("text-anchor", "middle");
      text.setAttribute("fill", "#FFFFFF");
      text.setAttribute("font-size", "10");
      text.setAttribute("font-weight", "900");
      text.setAttribute("letter-spacing", "1");
      text.textContent = sp.label;
      g.appendChild(text);
      spGroup.appendChild(g);
    });
    this.svg.appendChild(spGroup);

    // 3. Regional Scoring Overlay Badges (Showing current net points for US)
    this.renderRegionalScoringBadges(state);

    // 4. Country Nodes
    const legalNodes = new Set(
      state.legal_actions && state.legal_actions.decision_type === 5
        ? (state.legal_actions.valid_ids || []).filter((id: number) => id < 84)
        : []
    );

    const nodesGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodesGroup.setAttribute("id", "country-nodes");
    // A player's view: the countries the other side's latest turn changed or targeted.
    const lastTurn = new Map((state.last_turn?.countries || []).map(lc => [lc.id, lc]));
    this.renderLastTurnSummary(state);

    countries.forEach(c => {
      const stateC = state.countries ? state.countries[c.name] : null;
      const usInf = stateC ? stateC.us_influence : 0;
      const ussrInf = stateC ? stateC.ussr_influence : 0;
      const controlledBy = stateC ? stateC.controlled_by : "NONE";
      const isLegal = legalNodes.has(c.id);

      const boxW = 34;
      const boxH = 19;
      const x = c.pos[0];
      const y = c.pos[1];

      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", `svg-country-node ${isLegal ? "legal-target" : ""}`);
      g.setAttribute("data-id", c.id.toString());
      g.setAttribute("data-name", c.name);
      g.setAttribute("data-control", controlledBy);

      // Card Background
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", "country-card-bg");
      rect.setAttribute("x", (x - boxW/2).toString());
      rect.setAttribute("y", (y - boxH/2).toString());
      rect.setAttribute("width", boxW.toString());
      rect.setAttribute("height", boxH.toString());
      rect.setAttribute("rx", "2.5");

      // Base border depending on Battleground & Control
      let borderColor = c.battleground ? "#A855F7" : "#334155";
      let strokeWidth = c.battleground ? "1.8" : "1.0";

      if (controlledBy === "US") {
        borderColor = "#3B82F6";
        strokeWidth = "2.4";
      } else if (controlledBy === "USSR") {
        borderColor = "#EF4444";
        strokeWidth = "2.4";
      }

      // Control is carried by the fill, not only the border: a country that is a legal target has
      // its border overridden by the pulsing highlight (style.css), which is exactly when it
      // matters most who holds it.
      const fill = controlledBy === "US" ? "#1E40AF"
        : controlledBy === "USSR" ? "#991B1B"
        : c.battleground ? "#1E1B4B" : "#0F172A";
      rect.setAttribute("fill", fill);
      rect.setAttribute("stroke", borderColor);
      rect.setAttribute("stroke-width", strokeWidth);
      g.appendChild(rect);

      // Battleground Purple Top Stripe
      if (c.battleground) {
        const bgBanner = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        bgBanner.setAttribute("x", (x - boxW/2 + 0.8).toString());
        bgBanner.setAttribute("y", (y - boxH/2 + 0.8).toString());
        bgBanner.setAttribute("width", (boxW - 1.6).toString());
        bgBanner.setAttribute("height", "3.0");
        bgBanner.setAttribute("rx", "1.2");
        bgBanner.setAttribute("fill", "#7C3AED");
        g.appendChild(bgBanner);
      }

      // Stability Badge (Upper Right Corner)
      const badgeW = 6.2, badgeH = 5.2;
      const badgeX = x + boxW/2 - badgeW - 0.8;
      const badgeY = y - boxH/2 + 0.8;
      const stabRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      stabRect.setAttribute("x", badgeX.toString());
      stabRect.setAttribute("y", badgeY.toString());
      stabRect.setAttribute("width", badgeW.toString());
      stabRect.setAttribute("height", badgeH.toString());
      stabRect.setAttribute("rx", "1.2");
      stabRect.setAttribute("fill", c.battleground ? "#5B21B6" : "#1E293B");
      stabRect.setAttribute("stroke", c.battleground ? "#A855F7" : "#475569");
      stabRect.setAttribute("stroke-width", "0.6");
      g.appendChild(stabRect);

      const stabText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      stabText.setAttribute("x", (badgeX + badgeW/2).toString());
      stabText.setAttribute("y", (badgeY + badgeH/2 + 1.2).toString());
      stabText.setAttribute("text-anchor", "middle");
      stabText.setAttribute("fill", "#FFFFFF");
      stabText.setAttribute("font-size", "3.6");
      stabText.setAttribute("font-weight", "bold");
      stabText.textContent = c.stability.toString();
      g.appendChild(stabText);

      // Country Name Text
      const nameText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      nameText.setAttribute("x", (x - boxW/2 + 1.6).toString());
      nameText.setAttribute("y", (y - boxH/2 + (c.battleground ? 7.2 : 5.4)).toString());
      nameText.setAttribute("fill", c.battleground ? "#FAF5FF" : "#F1F5F9");
      nameText.setAttribute("font-size", c.name.length > 12 ? "2.8" : "3.2");
      nameText.setAttribute("font-weight", "bold");
      nameText.textContent = c.name.length > 13 ? c.name.substring(0, 11) + ".." : c.name;
      g.appendChild(nameText);

      // Influence Counters (Bottom Bar)
      // USSR (Red) Box on Left
      const infBoxW = 8.8, infBoxH = 5.6;
      const ussrBoxX = x - boxW/2 + 1.6;
      const ussrBoxY = y + boxH/2 - infBoxH - 1.2;
      // The controller's own box is outlined in white: the influence that holds the country.
      const ussrRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      ussrRect.setAttribute("class", "inf-box inf-ussr");
      ussrRect.setAttribute("x", ussrBoxX.toString());
      ussrRect.setAttribute("y", ussrBoxY.toString());
      ussrRect.setAttribute("width", infBoxW.toString());
      ussrRect.setAttribute("height", infBoxH.toString());
      ussrRect.setAttribute("rx", "1.0");
      ussrRect.setAttribute("fill", ussrInf > 0 ? "#DC2626" : "#0F172A");
      ussrRect.setAttribute("stroke", controlledBy === "USSR" ? "#FFFFFF" : "#EF4444");
      ussrRect.setAttribute("stroke-width", controlledBy === "USSR" ? "1.1" : ussrInf > 0 ? "1.0" : "0.6");
      g.appendChild(ussrRect);

      const ussrText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      ussrText.setAttribute("x", (ussrBoxX + infBoxW/2).toString());
      ussrText.setAttribute("y", (ussrBoxY + infBoxH/2 + 1.2).toString());
      ussrText.setAttribute("text-anchor", "middle");
      ussrText.setAttribute("fill", "#FFFFFF");
      ussrText.setAttribute("font-size", "3.4");
      ussrText.setAttribute("font-weight", "bold");
      ussrText.textContent = ussrInf.toString();
      g.appendChild(ussrText);

      // US (Blue) Box on Right of bottom
      const usBoxX = ussrBoxX + infBoxW + 2.0;
      const usBoxY = ussrBoxY;
      const usRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      usRect.setAttribute("class", "inf-box inf-us");
      usRect.setAttribute("x", usBoxX.toString());
      usRect.setAttribute("y", usBoxY.toString());
      usRect.setAttribute("width", infBoxW.toString());
      usRect.setAttribute("height", infBoxH.toString());
      usRect.setAttribute("rx", "1.0");
      usRect.setAttribute("fill", usInf > 0 ? "#2563EB" : "#0F172A");
      usRect.setAttribute("stroke", controlledBy === "US" ? "#FFFFFF" : "#3B82F6");
      usRect.setAttribute("stroke-width", controlledBy === "US" ? "1.1" : usInf > 0 ? "1.0" : "0.6");
      g.appendChild(usRect);

      const usText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      usText.setAttribute("x", (usBoxX + infBoxW/2).toString());
      usText.setAttribute("y", (usBoxY + infBoxH/2 + 1.2).toString());
      usText.setAttribute("text-anchor", "middle");
      usText.setAttribute("fill", "#FFFFFF");
      usText.setAttribute("font-size", "3.4");
      usText.setAttribute("font-weight", "bold");
      usText.textContent = usInf.toString();
      g.appendChild(usText);

      const lc = lastTurn.get(c.id);
      if (lc && state.last_turn) this.markLastTurn(g, lc, state.last_turn, x, y, boxW, boxH, ussrBoxX, usBoxX, infBoxW);

      // Click Event Handler
      g.addEventListener("click", () => {
        this.onCountryClick(c.id);
      });

      nodesGroup.appendChild(g);
    });

    this.svg.appendChild(nodesGroup);
  }

  /**
   * Mark a country the other side's latest turn touched: a dashed ring around it, the signed
   * change over each influence box, and the change in its hover title.
   */
  private markLastTurn(g: SVGGElement, lc: LastTurnCountry, lt: LastTurn, x: number, y: number,
                       boxW: number, boxH: number, ussrBoxX: number, usBoxX: number, infBoxW: number) {
    const NS = "http://www.w3.org/2000/svg";
    g.classList.add("last-turn");
    const ring = document.createElementNS(NS, "rect");
    ring.setAttribute("class", "last-turn-ring");
    ring.setAttribute("x", (x - boxW/2 - 2.2).toString());
    ring.setAttribute("y", (y - boxH/2 - 2.2).toString());
    ring.setAttribute("width", (boxW + 4.4).toString());
    ring.setAttribute("height", (boxH + 4.4).toString());
    ring.setAttribute("rx", "3.5");
    g.insertBefore(ring, g.firstChild);
    const chip = (delta: number, boxX: number, side: "us" | "ussr") => {
      if (!delta) return;
      const t = document.createElementNS(NS, "text");
      t.setAttribute("class", `last-turn-delta ${side}`);
      t.setAttribute("x", (boxX + infBoxW/2).toString());
      t.setAttribute("y", (y + boxH/2 + 7.4).toString());
      t.setAttribute("text-anchor", "middle");
      t.textContent = delta > 0 ? `+${delta}` : `\u2212${-delta}`;
      g.appendChild(t);
    };
    chip(lc.ussr, ussrBoxX, "ussr");
    chip(lc.us, usBoxX, "us");
    const title = document.createElementNS(NS, "title");
    title.textContent = `${lt.player}'s last turn (${lt.label}): ${lc.name} ${describeLastTurnCountry(lc)}`;
    g.appendChild(title);
  }

  /** The line under the map naming what the other side's latest turn did. */
  private renderLastTurnSummary(state: GameState) {
    const el = document.getElementById("map-last-turn");
    if (!el) return;
    const lt = state.last_turn;
    if (!lt) {
      el.classList.add("hidden");
      el.textContent = "";
      return;
    }
    el.classList.remove("hidden");
    const what = lt.countries.length
      ? lt.countries.map(lc => `${lc.name} ${describeLastTurnCountry(lc)}`).join(" · ")
      : "no country changed";
    el.textContent = `${lt.player}'s last turn (${lt.label}): ${what}`;
  }

  private renderRegionalScoringBadges(state: GameState) {
    const audits = this.calculateRegionalScoring(state);
    if (!audits.length) return;

    const overlayGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
    overlayGroup.setAttribute("id", "regional-scoring-overlay");

    audits.forEach(audit => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "svg-region-badge");
      g.setAttribute("data-region", audit.regionName);
      g.style.cursor = "pointer";

      const x = audit.badgePos[0];
      const y = audit.badgePos[1];
      const badgeW = 76;
      const badgeH = 22;

      // Card Background
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", (x - badgeW/2).toString());
      rect.setAttribute("y", (y - badgeH/2).toString());
      rect.setAttribute("width", badgeW.toString());
      rect.setAttribute("height", badgeH.toString());
      rect.setAttribute("rx", "3.5");
      rect.setAttribute("fill", "rgba(15, 23, 42, 0.90)");
      rect.setAttribute("stroke", audit.color);
      rect.setAttribute("stroke-width", "1.2");
      rect.setAttribute("filter", "drop-shadow(0px 2px 4px rgba(0,0,0,0.5))");
      g.appendChild(rect);

      // Region Title text
      const titleText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      titleText.setAttribute("x", (x - badgeW/2 + 4).toString());
      titleText.setAttribute("y", (y - badgeH/2 + 6.5).toString());
      titleText.setAttribute("fill", "#94A3B8");
      titleText.setAttribute("font-size", "3.6");
      titleText.setAttribute("font-weight", "800");
      titleText.setAttribute("letter-spacing", "0.5");
      titleText.textContent = audit.regionName;
      g.appendChild(titleText);

      // Value Pill / Net VP for US
      let vpText = `${audit.netVpUs > 0 ? "+" + audit.netVpUs : audit.netVpUs} VP`;
      let vpColor = "#94A3B8";
      if (audit.isInstantWinUs) {
        vpText = "★ US WIN";
        vpColor = "#3B82F6";
      } else if (audit.isInstantWinUssr) {
        vpText = "★ USSR WIN";
        vpColor = "#EF4444";
      } else if (audit.netVpUs > 0) {
        vpText = `+${audit.netVpUs} (US)`;
        vpColor = "#60A5FA";
      } else if (audit.netVpUs < 0) {
        vpText = `${audit.netVpUs} (USSR)`;
        vpColor = "#F87171";
      } else {
        vpText = "0 VP (Tie)";
        vpColor = "#CBD5E1";
      }

      const valText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      valText.setAttribute("x", (x + badgeW/2 - 4).toString());
      valText.setAttribute("y", (y - badgeH/2 + 6.5).toString());
      valText.setAttribute("text-anchor", "end");
      valText.setAttribute("fill", vpColor);
      valText.setAttribute("font-size", "4.0");
      valText.setAttribute("font-weight", "900");
      valText.textContent = vpText;
      g.appendChild(valText);

      // Sub-row: US standing vs USSR standing
      const subText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      subText.setAttribute("x", x.toString());
      subText.setAttribute("y", (y + badgeH/2 - 3.5).toString());
      subText.setAttribute("text-anchor", "middle");
      subText.setAttribute("fill", "#E2E8F0");
      subText.setAttribute("font-size", "3.2");
      subText.setAttribute("font-weight", "600");
      subText.textContent = `US: ${audit.usStatus.substring(0,4)} (${audit.usTotalVp}) | USSR: ${audit.ussrStatus.substring(0,4)} (${audit.ussrTotalVp})`;
      g.appendChild(subText);

      // Middle row: can the region's scoring card still come this cycle? Inside the badge,
      // because country nodes are drawn over the badges and would hide anything sticking out.
      const tag = SCORING_CARD_TAG[audit.scoringCard.status];
      const tagW = tag.text.length * 2.4 + 4;
      const tagH = 5.2;
      const tagX = x + badgeW / 2 - tagW - 3;
      const tagY = y - tagH / 2 - 0.4;
      const tagG = document.createElementNS("http://www.w3.org/2000/svg", "g");
      tagG.setAttribute("class", "svg-region-card-tag");
      tagG.setAttribute("data-status", audit.scoringCard.status);
      const tagLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tagLabel.setAttribute("x", (x - badgeW / 2 + 4).toString());
      tagLabel.setAttribute("y", (tagY + tagH / 2 + 1.0).toString());
      tagLabel.setAttribute("fill", "#64748B");
      tagLabel.setAttribute("font-size", "2.8");
      tagLabel.setAttribute("font-weight", "700");
      tagLabel.setAttribute("letter-spacing", "0.4");
      tagLabel.textContent = "SCORING CARD";
      tagG.appendChild(tagLabel);
      const tagRect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      tagRect.setAttribute("x", tagX.toString());
      tagRect.setAttribute("y", tagY.toString());
      tagRect.setAttribute("width", tagW.toString());
      tagRect.setAttribute("height", tagH.toString());
      tagRect.setAttribute("rx", "1.6");
      tagRect.setAttribute("fill", tag.fill);
      tagRect.setAttribute("stroke", tag.stroke);
      tagRect.setAttribute("stroke-width", "0.6");
      if (audit.scoringCard.status === "later") tagRect.setAttribute("stroke-dasharray", "1,0.8");
      tagG.appendChild(tagRect);
      const tagText = document.createElementNS("http://www.w3.org/2000/svg", "text");
      tagText.setAttribute("x", (tagX + tagW / 2).toString());
      tagText.setAttribute("y", (tagY + tagH / 2 + 1.2).toString());
      tagText.setAttribute("text-anchor", "middle");
      tagText.setAttribute("fill", tag.ink);
      tagText.setAttribute("font-size", "3.4");
      tagText.setAttribute("font-weight", "800");
      tagText.setAttribute("letter-spacing", "0.3");
      tagText.textContent = tag.text;
      tagG.appendChild(tagText);
      g.appendChild(tagG);

      // Hover / Tooltip listeners
      g.addEventListener("mouseenter", (e) => this.showRegionTooltip(e, audit));
      g.addEventListener("mouseleave", () => this.hideRegionTooltip());

      overlayGroup.appendChild(g);
    });

    this.svg.appendChild(overlayGroup);
  }

  private showRegionTooltip(e: MouseEvent, audit: RegionScoreAudit) {
    if (!this.tooltipEl) return;

    this.tooltipEl.innerHTML = `
      <div style="font-weight: bold; font-size: 13px; color: #F8FAFC; margin-bottom: 4px;">
        ${audit.scoringCardName}
      </div>
      <div style="font-size: 11px; color: ${SCORING_CARD_TAG[audit.scoringCard.status].stroke}; margin-bottom: 4px;">
        Scoring card ${audit.scoringCard.where}
      </div>
      <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">
        CURRENT US NET VALUE: <strong style="color: ${audit.netVpUs > 0 ? '#60A5FA' : (audit.netVpUs < 0 ? '#F87171' : '#CBD5E1')}; font-size: 12px;">${audit.netVpUs > 0 ? '+' + audit.netVpUs : audit.netVpUs} VP</strong>
      </div>
      <div style="font-size: 11px; color: #E2E8F0; line-height: 1.4; display: flex; flex-direction: column; gap: 3px;">
        ${audit.details.map(d => `<div>${d}</div>`).join("")}
      </div>
    `;

    this.tooltipEl.classList.remove("hidden");
    const x = Math.min(window.innerWidth - 340, e.clientX + 15);
    const y = Math.min(window.innerHeight - 250, e.clientY + 15);
    this.tooltipEl.style.left = `${x}px`;
    this.tooltipEl.style.top = `${y}px`;
  }

  private hideRegionTooltip() {
    if (!this.tooltipEl) return;
    this.tooltipEl.classList.add("hidden");
  }
}

/** "USSR +2, US −1", or what a target that moved nothing was: "targeted, no change". */
export function describeLastTurnCountry(lc: LastTurnCountry): string {
  const signed = (v: number) => (v > 0 ? `+${v}` : `\u2212${-v}`);
  const parts: string[] = [];
  if (lc.ussr) parts.push(`USSR ${signed(lc.ussr)}`);
  if (lc.us) parts.push(`US ${signed(lc.us)}`);
  return parts.length ? parts.join(", ") : "targeted, no change";
}
