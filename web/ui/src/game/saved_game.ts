/**
 * The game on the board, kept in the browser (localStorage) after every move, so a page reload
 * picks it up with its history -- its seed, steps and tags (game/record.ts) -- instead of opening
 * the `pos=` link as a new game from a loaded position, which the stats database could no longer
 * re-drive from the seed. Per browser, best effort: storage may be off or full, and then a
 * reload behaves as it always did.
 */
import { GameRecordData } from "./record";

const KEY = "ts.workbench.game.v1";

export interface SavedGame {
  record: GameRecordData;
  submitted: boolean;
}

export function storeGame(game: SavedGame): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify({ v: 1, ...game }));
  } catch {
    // No storage: a reload opens the link's position, as before.
  }
}

export function loadStoredGame(): SavedGame | null {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return null;
    const v = JSON.parse(raw);
    if (v?.v !== 1 || !v.record || !Array.isArray(v.record.steps)) return null;
    return { record: v.record as GameRecordData, submitted: v.submitted === true };
  } catch {
    return null;
  }
}
