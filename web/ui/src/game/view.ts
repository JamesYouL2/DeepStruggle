/**
 * Whose eyes the board is drawn for: the developer (everything) or one side, playing against the
 * model.
 *
 * A player's view is a redaction of the display state, made once, before any view renders it, so
 * no panel can show the other side's private information by forgetting to hide it:
 *
 * - the other side's hand keeps only the cards the engine says this side has seen
 *   (`HAND_<side>_KNOWN`); the rest are counted in `hidden_cards`, not named;
 * - those unseen cards and the draw deck share one location, `UNSEEN` -- "the draw deck or the
 *   other hand" -- which is all a player can know about any of them;
 * - a headline the other side has chosen but not yet revealed is not shown (unless this side
 *   holds the space race's Man in Space perk, which makes the other side reveal first);
 * - while the other side is deciding, its legal actions are withheld (they are its hand, for a
 *   card choice) and `view_hidden_decision` is set;
 * - the action log drops the lines that name a card going into, or out of, the other side's
 *   unseen hand, and the other side's still face-down headline is logged without the card.
 *
 * The engine itself, the model and the undo history keep the full state: only what is drawn changes.
 */
import { ActionLogItem, GameState } from "../types";

export type ViewMode = "DEV" | "US" | "USSR";

/** The location a player's view gives a card in the draw deck or the other side's unseen hand. */
export const UNSEEN = "UNSEEN";

export function viewFromParam(s: string | null): ViewMode {
  const v = (s || "").toUpperCase();
  return v === "US" || v === "USSR" ? v : "DEV";
}

export function viewToParam(v: ViewMode): string | null {
  return v === "DEV" ? null : v.toLowerCase();
}

export function opponentOf(v: ViewMode): "US" | "USSR" | null {
  return v === "US" ? "USSR" : v === "USSR" ? "US" : null;
}

const MOVED = /moved: (\S+) -> (\S+)/;

/**
 * The log as `view` may read it. `faceDownHeadline`: the other side has chosen a headline that is
 * still face down, so its latest headline line is not shown with the card (earlier headlines have
 * been revealed since, and their lines stay as they are).
 */
export function redactLogs(logs: ActionLogItem[], view: ViewMode, faceDownHeadline = false): ActionLogItem[] {
  const opp = opponentOf(view);
  if (!opp) return logs;
  const unseen = `HAND_${opp}_UNKNOWN`;
  const commit = `${opp} commits Headline Card:`;
  let faceDown = -1;
  if (faceDownHeadline) {
    for (let i = logs.length - 1; i >= 0; i--) {
      if (logs[i].player === opp && logs[i].text.startsWith(commit)) { faceDown = i; break; }
    }
  }
  return logs.map((log, i) => {
    let text = log.text;
    if (i === faceDown) text = `${opp} commits a Headline Card (face down until both are revealed)`;
    const details = (log.details || []).filter(d => {
      const m = MOVED.exec(d);
      if (!m) return true;
      const [, from, to] = m;
      if (from === unseen || to === unseen) return false;
      // Cards the other side draws to choose from (Our Man in Tehran and the like) are its own.
      if (log.player === opp && (from === "PEEKED_TEMP" || to === "PEEKED_TEMP")) return false;
      return true;
    });
    return text === log.text && details.length === (log.details || []).length ? log : { ...log, text, details };
  });
}

/** The display state as `view` may see it (the state itself for the developer). */
export function redactState(s: GameState, view: ViewMode): GameState {
  const opp = opponentOf(view);
  if (!opp) return { ...s, view };
  const r: GameState = structuredClone(s);
  r.view = view;
  const locs = (s.card_locations || {}) as Record<string, string>;
  const known = `HAND_${opp}_KNOWN`;
  const unseen = `HAND_${opp}_UNKNOWN`;
  const oppDeciding = s.decision_context?.decision_player === opp;

  // Hands: the other side's unseen cards are only a count.
  const keep = (id: number) => locs[String(id)] === known;
  const oppHand = s.hands?.[opp] || [];
  r.hands[opp] = oppHand.filter(keep);
  const cardsKey = `${opp}_cards` as "US_cards" | "USSR_cards";
  if (s.hands?.[cardsKey]) r.hands[cardsKey] = s.hands[cardsKey]!.filter(c => keep(c.id));
  const hidden = oppHand.length - r.hands[opp].length;
  r.hidden_cards = { US: 0, USSR: 0, [opp]: hidden } as { US: number; USSR: number };

  // Locations: the deck and the other hand's unseen cards are one pile; so are the other side's
  // drawn-to-choose cards while it chooses.
  const rl: Record<string, string> = {};
  for (const [id, loc] of Object.entries(locs)) {
    rl[id] = loc === unseen || loc === "DRAW_DECK" || (loc === "PEEKED_TEMP" && oppDeciding) ? UNSEEN : loc;
  }
  r.card_locations = rl;
  r.unseen_count = (s.draw_deck_count || 0) + hidden;

  // A headline chosen but still in the hand is face down -- unless this side holds the space
  // race's Man in Space perk (box 4, the other side not there), which makes the other side
  // reveal its headline first.
  const hlKey = opp === "US" ? "headline_us_card" : "headline_ussr_card";
  const hl = s[hlKey];
  const me = opp === "US" ? "USSR" : "US";
  const perk = (s.space?.[me] ?? 0) >= 4 && (s.space?.[opp] ?? 0) < 4;
  const faceDownHeadline = !!hl && !perk && (locs[String(hl)] || "").startsWith(`HAND_${opp}`);
  if (faceDownHeadline) r[hlKey] = 0;

  // The other side's options, while it decides, are not ours to see.
  if (oppDeciding && !s.is_terminal) {
    r.view_hidden_decision = true;
    r.legal_actions = { ...s.legal_actions, valid_ids: [], valid_action_labels: {} };
  }
  if (s.action_logs) r.action_logs = redactLogs(s.action_logs, view, faceDownHeadline);
  return r;
}
