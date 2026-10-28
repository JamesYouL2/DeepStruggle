/**
 * The cards a log line names, made hoverable: each mention is wrapped in a
 * `<span class="log-card" data-card-id>` that main.ts gives the card's tooltip (its Ops, side and
 * text, as the hand tabs show it). The log says a card as `Name (#id)`, `'Name' (#id, n Ops)` (the
 * headline reveal), `Card #id (Name)` (a card move), or by name alone in a deal ("US draws 3
 * cards: A, B, C"). Lines are the redacted log (game/view.ts), so nothing named here is hidden.
 */

import { escapeHtml } from "../html";

const REF = /Card #(\d{1,3}) \([^)]*\)|\(#(\d{1,3})(?:, [^)]*)?\)/g;
const DRAWS = /^(\s*• (?:US|USSR) draws \d+ cards?: )(.+)$/;

export interface CardNames {
  /** The names a card goes by in the log (the engine's and the rules'); empty for no such card. */
  namesOf(id: number): readonly string[];
  /** The card a name is, if it is one. */
  idOf(name: string): number | undefined;
}

/** `inner` is text, escaped here. */
function span(id: number, inner: string): string {
  return `<span class="log-card" data-card-id="${id}">${escapeHtml(inner)}</span>`;
}

/**
 * `text` as HTML: every card it names wrapped for hovering, and all of it escaped -- a log line can
 * come from a replay file (its descriptions), which is anyone's text.
 */
export function linkCardRefs(text: string, cards: CardNames): string {
  const deal = DRAWS.exec(text);
  if (deal) {
    return escapeHtml(deal[1]) + deal[2].split(", ").map(n => {
      const id = cards.idOf(n);
      return id === undefined ? escapeHtml(n) : span(id, n);
    }).join(", ");
  }
  let out = "";
  let last = 0;
  for (const m of text.matchAll(REF)) {
    const at = m.index ?? 0;
    const id = parseInt(m[1] ?? m[2], 10);
    const names = cards.namesOf(id);
    if (names.length === 0) continue;
    // `(#id)` follows the card's name: take the name in, quoted or not.
    let start = at;
    if (m[2] !== undefined) {
      const before = text.slice(last, at);
      const form = names.flatMap(n => [`${n} `, `'${n}' `]).find(f => before.endsWith(f));
      if (form) start = at - form.length;
    }
    const end = at + m[0].length;
    out += escapeHtml(text.slice(last, start)) + span(id, text.slice(start, end));
    last = end;
  }
  return out + escapeHtml(text.slice(last));
}
