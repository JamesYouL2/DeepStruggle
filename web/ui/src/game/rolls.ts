/**
 * The die rolls of the latest action round that rolled, for the map's roll tray: a coup, a
 * realignment, a Space Race attempt, a war, the Olympics, the Summit, a trap escape.
 *
 * The engine keeps only the roll made by the step just taken (`die_roll` is cleared at every
 * step), so a roll followed by more steps -- the other side's whole turn played at once -- is gone
 * from the current state. The rolls are read from the steps' snapshots instead, and shown only
 * while they are recent: from the half action round they were made in and the next one (the
 * other side's reply), after which they are history and belong to the log.
 *
 * Everything here is public: dice are rolled in the open.
 */
import { DieRollState, GameState, RecentRoll } from "../types";
import { ReplayStep } from "../replay_controls";

const MAX_ROLLS = 4;

/** A half action round as a number: the headline is 1, then USSR AR1 = 2, US AR1 = 3, ... */
function halfRound(turn: number, phase: string, ar: number, phasing: string): number {
  if (phase === "HEADLINE" || phase === "SETUP") return turn * 100 + 1;
  return turn * 100 + ar * 2 + (phasing === "US" ? 1 : 0);
}

function rolled(step: ReplayStep): boolean {
  const t = step.state_snapshot?.die_roll?.type;
  return !!t && t !== "NONE";
}

/**
 * The rolls of the latest half action round with any, oldest first (at most four), when that
 * round is `now`'s or the one before it; otherwise none. `steps[0..end)` are the steps taken so
 * far, `start` the state before the first.
 */
export function recentRolls(start: GameState | null, steps: ReplayStep[], end: number, now: GameState): RecentRoll[] {
  const before = (k: number): GameState | null => (k === 0 ? start : steps[k - 1].state_snapshot);
  const ordOf = (k: number): number => {
    const b = before(k);
    const phasing = b?.phasing_player ?? steps[k].state_snapshot.die_roll?.roller ?? "NONE";
    return halfRound(steps[k].turn, steps[k].phase, steps[k].ar, phasing);
  };
  let k = Math.min(end, steps.length) - 1;
  while (k >= 0 && !rolled(steps[k])) k--;
  if (k < 0) return [];
  const ord = ordOf(k);
  const nowOrd = halfRound(now.turn, now.current_phase_name ?? "", now.action_round, now.phasing_player);
  if (nowOrd - ord > 1) return [];

  const out: RecentRoll[] = [];
  for (; k >= 0 && out.length < MAX_ROLLS; k--) {
    if (!rolled(steps[k])) continue;
    if (ordOf(k) !== ord) break;
    const s = steps[k];
    const phase = s.phase === "HEADLINE" ? "Headline" : `AR${s.ar}`;
    out.push({ ...(s.state_snapshot.die_roll as DieRollState), when: `T${s.turn} ${phase}` });
  }
  return out.reverse();
}
