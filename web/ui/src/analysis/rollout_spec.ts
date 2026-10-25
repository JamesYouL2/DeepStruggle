/**
 * The rollout root's spec and its `search=` grammar, apart from the search itself (rollout.ts, which
 * needs the engine and the model): the leaderboard page reads specs too, without either.
 *
 * `rollout:k:worlds:horizon:rule[:stop][:leaf][:keyed|keyed2][:aligned][:box4][:belief[N]]` is the
 * leaderboard's rollout player: `keyed` is chance=keyed, `keyed2` chance=keyed2, `belief` belief=N
 * (2,048 hands with no count). tools/lib/player_spec.py's `workbench_search` writes a leaderboard player in it, every field
 * spelled out, as rolloutParam does.
 */

export interface RolloutSpec {
  k: number;
  worlds: number;
  horizon: number;
  rule: string;
  /** Where the playouts stop (the engine's rollout::Stop, by name). */
  stop: string;
  /** Whose side a leaf is read from (the engine's rollout::Leaf, by name). */
  leaf: string;
  /** Each draw tied to its situation (bindings/keyed_chance.hpp). */
  keyed: boolean;
  /** Keyed chance with the situation's player fields narrowed to a byte (keyed_chance.hpp's keyed2):
   *  `keyed` alone keeps the original packing, in which a USSR side hides the turn. It implies keyed, so a
   *  parsed spec never sets both: at most one of the two is true. */
  keyed2: boolean;
  /** Every candidate's horizon counted from the end of the mover's own action round, the headline one boundary. */
  aligned: boolean;
  /** The opponent's headline revealed by Space Race box 4 kept in every world. */
  box4: boolean;
  /** Hands the belief weighs (bindings/belief.hpp); 0 = uniform redeals. */
  belief: number;
}

/** The engine's codes, in the order of ai/search/rollout_root.py's STOPS and LEAF_SIDES. */
export const STOPS = ["boundaries", "turn", "later", "earlier", "predeal", "earlier_predeal"] as const;
export const LEAVES = ["mover", "decider", "own"] as const;

/** The best 32-world player on the leaderboard (owner, 2026-10-21): the corrected keyed root,
 *  rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed,aligned=true,box4=true). */
export const DEFAULT_ROLLOUT: RolloutSpec = { k: 4, worlds: 32, horizon: 4, rule: "z2", stop: "earlier", leaf: "own",
                                             keyed: true, keyed2: false, aligned: true, box4: true, belief: 0 };
/** ai/search/rollout_root.py's `belief` with no count. */
export const DEFAULT_BELIEF_HANDS = 2048;

/** "rollout" or "rollout:k:worlds:horizon:rule[:stop][:leaf][:keyed|keyed2][:aligned][:box4][:belief[N]]", an omitted field the
 *  built-in default's; null otherwise. "rollout" alone is DEFAULT_ROLLOUT, the default when this page was
 *  built -- links written before the default moved to the Hugging Face repo keep their meaning. */
export function parseRolloutParam(s: string | null): RolloutSpec | null {
  if (!s) return null;
  const parts = s.split(":");
  if (parts[0] !== "rollout" || parts.length > 11) return null;
  // "rollout" alone is the default player; a spelled-out spec says every switch it turns on, so one
  // written before a switch existed keeps meaning what it meant.
  if (parts.length === 1) return { ...DEFAULT_ROLLOUT };
  const num = (i: number, d: number, min: number) => {
    if (parts.length <= i || parts[i] === "") return d;
    const v = Number(parts[i]);
    return Number.isInteger(v) && v >= min ? v : NaN;
  };
  const spec: RolloutSpec = {
    k: num(1, DEFAULT_ROLLOUT.k, 1), worlds: num(2, DEFAULT_ROLLOUT.worlds, 1),
    horizon: num(3, DEFAULT_ROLLOUT.horizon, 0), rule: parts.length > 4 && parts[4] ? parts[4] : DEFAULT_ROLLOUT.rule,
    stop: DEFAULT_ROLLOUT.stop, leaf: DEFAULT_ROLLOUT.leaf, keyed: false, keyed2: false, aligned: false, box4: false, belief: 0,
  };
  // The trailing fields are told apart by value, as the Python spec does.
  for (const p of parts.slice(5)) {
    if (!p) continue;
    const b = /^belief(\d*)$/.exec(p);
    if ((STOPS as readonly string[]).includes(p)) spec.stop = p;
    else if ((LEAVES as readonly string[]).includes(p)) spec.leaf = p;
    else if (p === "keyed") spec.keyed = true;
    else if (p === "keyed2") spec.keyed2 = true;
    else if (p === "aligned") spec.aligned = true;
    else if (p === "box4") spec.box4 = true;
    else if (b) spec.belief = b[1] ? Number(b[1]) : DEFAULT_BELIEF_HANDS;
    else return null;
  }
  // keyed2 is keyed chance with its own packing, so `:keyed:keyed2` is the one search keyed2 spells alone.
  if (spec.keyed2) spec.keyed = false;
  if ([spec.k, spec.worlds, spec.horizon].some(v => Number.isNaN(v))) return null;
  if (!/^(argmax|z(\d+(\.\d+)?)?|kl(\d+(\.\d+)?)?)$/.test(spec.rule)) return null;
  return spec;
}

/** The spec spelled out, every field: a link then means the same search whatever the default
 *  becomes (the default is the Hugging Face repo's, and changes there). */
export function rolloutParam(spec: RolloutSpec): string {
  return `rollout:${spec.k}:${spec.worlds}:${spec.horizon}:${spec.rule}:${spec.stop}:${spec.leaf}`
    + (spec.keyed2 ? ":keyed2" : spec.keyed ? ":keyed" : "") + (spec.aligned ? ":aligned" : "") + (spec.box4 ? ":box4" : "")
    + (spec.belief ? `:belief${spec.belief}` : "");
}

/** The same search: every field equal. */
export function specEqual(a: RolloutSpec, b: RolloutSpec): boolean {
  return a.k === b.k && a.worlds === b.worlds && a.horizon === b.horizon && a.rule === b.rule && a.stop === b.stop
    && a.leaf === b.leaf && a.keyed === b.keyed && a.keyed2 === b.keyed2 && a.aligned === b.aligned && a.box4 === b.box4 && a.belief === b.belief;
}

/** A `search=` value: "policy" (null, the policy's argmax) or a rollout spec; undefined when it is neither. */
export function parseSearchParam(s: string): RolloutSpec | null | undefined {
  if (s === "policy") return null;
  return parseRolloutParam(s) ?? undefined;
}
