/**
 * The engine's own names and numbers, as the WebAssembly build exports them
 * (`ts_action_layout_json`, bindings/wasm/ts_engine_wasm.cpp): the flat action space's offsets, the
 * enums (DecisionType, Resolution, OpMode, Region, the rollout root's stop rules and leaf sides),
 * the card ids the page words on its own, and the Space Race track. The page keeps no copy of any of
 * them -- a second copy is what drifted (a DecisionType table that stopped at 6, play-mode labels
 * left on a retired enum).
 *
 * Every WasmEngine sets it when it loads, so a module with no engine of its own (the HUD, the log's
 * branch names) reads the same layout as the engine on the board.
 */

export interface SpaceBox {
  min_ops: number;
  max_roll: number;
  vp_first: number;
  vp_second: number;
}

export interface ActionLayout {
  size: number;
  offsets: {
    card: number; play_mode: number; op_mode: number; roll_die: number;
    node: number; branch: number; defcon_value: number; region: number;
  };
  confirm_done_index: number;
  ops_influence_index: number;
  /** DecisionType by name. */
  decision_types: Record<string, number>;
  /** Resolution (how a card is played) by name. */
  resolutions: Record<string, number>;
  /** OpMode by name. */
  op_modes: Record<string, number>;
  /** Region by name. */
  regions: Record<string, number>;
  /** ts::card_ids by name, for the cards the page words on its own. */
  card_ids: Record<string, number>;
  /** rollout::Stop and rollout::Leaf names in enum order: ts_rollout_reset takes the index. */
  rollout_stops: string[];
  rollout_leaves: string[];
  /** SpaceRace::get_box_info for boxes 0..8. */
  space_boxes: SpaceBox[];
}

let current: ActionLayout | null = null;

export function setEngineLayout(layout: ActionLayout): void {
  current = layout;
}

/** The layout of the engine the page loaded; null until one has loaded. */
export function engineLayout(): ActionLayout | null {
  return current;
}

/** An enum's names in value order: `namesByValue(L.resolutions)[2]` is "OPS_INFLUENCE". */
export function namesByValue(e: Record<string, number>): string[] {
  const out: string[] = [];
  for (const [name, v] of Object.entries(e)) out[v] = name;
  return out;
}

/** "MIDDLE_EAST" -> "Middle East". */
export function titleCase(name: string): string {
  return name.toLowerCase().split("_").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
}

/** A card id the engine exports by name; throws for a name it does not (a page newer than its engine). */
export function cardId(layout: ActionLayout, name: string): number {
  const id = layout.card_ids[name];
  if (id === undefined) throw new Error(`the engine exports no card id ${name}`);
  return id;
}
