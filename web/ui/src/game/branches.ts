/**
 * What an event's branch choice (CHOOSE_BRANCH) means, for the HUD's buttons and the action log.
 *
 * The engine asks for a branch in six events (engine/src/events/*.cpp, card_dispatcher.cpp), and the
 * branch is a number whose meaning is the event's: Chernobyl's is a region, Summit's and How I
 * Learned to Stop Worrying's the DEFCON level to set (Summit offers the current level and one either
 * side -- choosing the current one leaves DEFCON alone). The page used to show "Option 3" for those.
 */
import { GameState } from "../types";

export interface BranchMeta {
  /** The button's title. */
  title: string;
  /** The button's description. */
  desc: string;
  /** The log's verb phrase: "<player> <verb> (<event>)". */
  verb: string;
}

/** The engine's `Region` order, which is Chernobyl's branch. */
export const REGIONS = ["Europe", "Asia", "Middle East", "Africa", "Central America", "South America"];

const WARSAW_PACT = 16, OLYMPIC_GAMES = 20, SUMMIT = 45, HOW_I_LEARNED = 46, CHERNOBYL = 94, WARGAMES = 100;

const FIXED: Record<number, Record<number, BranchMeta>> = {
  [WARSAW_PACT]: {
    0: { title: "Remove US influence", desc: "Remove all US influence from up to 4 countries in Eastern Europe.", verb: "removes US influence from Eastern Europe" },
    1: { title: "Add USSR influence", desc: "Add 5 USSR influence in Eastern Europe (at most 2 per country).", verb: "adds USSR influence in Eastern Europe" },
  },
  [OLYMPIC_GAMES]: {
    0: { title: "Participate", desc: "Both roll a die, the sponsor +2; the higher roll gains 2 VP.", verb: "participates in the Olympic Games" },
    1: { title: "Boycott", desc: "DEFCON degrades by 1 and the sponsor conducts 4 Ops.", verb: "boycotts the Olympic Games" },
  },
  [WARGAMES]: {
    0: { title: "End the game", desc: "Give the opponent 6 VP, then the game ends and is scored as it stands.", verb: "ends the game, giving the opponent 6 VP" },
    1: { title: "Continue", desc: "Do not end the game.", verb: "declines to end the game" },
  },
};

function defconLevel(level: number, current: number): BranchMeta {
  const how = level === current ? "unchanged" : level > current ? "improve" : "degrade";
  const desc = level === 1 ? "DEFCON 1 ends the game: the phasing player loses."
    : level === current ? `Leave DEFCON at ${current}.`
    : `${level > current ? "Improve" : "Degrade"} DEFCON from ${current} to ${level}.`;
  return { title: `Set DEFCON to ${level} (${how})`, desc, verb: `sets DEFCON to ${level} (${how})` };
}

/** The meaning of branch `branch` of `card`'s event in `state`, or null for an event it does not know. */
export function branchMeta(card: number, branch: number, state: GameState): BranchMeta | null {
  if (card === CHERNOBYL) {
    const region = REGIONS[branch] ?? `region ${branch}`;
    return { title: region, desc: `The USSR may not place influence in ${region} with Ops for the rest of the turn.`,
             verb: `designates ${region}` };
  }
  if (card === SUMMIT || card === HOW_I_LEARNED) return defconLevel(branch, state.defcon);
  return FIXED[card]?.[branch] ?? null;
}
