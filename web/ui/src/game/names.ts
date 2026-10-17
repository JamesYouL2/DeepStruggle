/**
 * Readable names for flat action indices -- the port of `ActionEncoder.get_action_name`
 * (bindings/action_encoder.py), with the offsets taken from the engine's own layout rather than
 * copied, and names for the two P17 heads (DEFCON value, region) that the Python version leaves
 * as "UnknownAction".
 */
import { ActionLayout, WasmEngine } from "../engine/wasm_engine";
import { GameState } from "../types";
import { branchMeta, REGIONS } from "./branches";

const MODES = ["EVENT", "SPACE", "OPS_INFLUENCE", "OPS_COUP", "OPS_REALIGN"];
const OP_MODES = ["INFLUENCE", "COUP", "REALIGN"];

/** `state`, the position the action is taken in, names an event's branch by what it means
 * (game/branches.ts); without it a branch is a number, a region and a DEFCON level still named. */
export function actionName(engine: WasmEngine, idx: number, state?: GameState): string {
  const L: ActionLayout = engine.layout;
  const o = L.offsets;
  if (idx === L.confirm_done_index) return "CONFIRM_DONE / PASS";
  if (idx < o.play_mode) return `SelectCard #${idx + 1} (${engine.cardName(idx + 1)})`;
  if (idx < o.roll_die) {
    const mode = idx - o.play_mode;
    const name = MODES[mode] ?? String(mode);
    // The last three slots are also the deferred Ops choice; without a node to ask, say both.
    if (idx >= o.op_mode) return `Resolution: ${name} / OpMode: ${OP_MODES[idx - o.op_mode]}`;
    return `Resolution: ${name}`;
  }
  if (idx === o.roll_die) return "RollDie";
  if (idx < o.node) return `Unassigned #${idx}`;
  if (idx < o.branch) return `PointNode #${idx - o.node} (${engine.countryName(idx - o.node)})`;
  const card = state?.decision_context?.resolving_card ?? 0;
  if (idx < L.confirm_done_index) {
    const meta = state ? branchMeta(card, idx - o.branch, state) : null;
    return meta ? meta.title : `Branch #${idx - o.branch}`;
  }
  if (idx >= o.defcon_value && idx < o.region) {
    const level = idx - o.defcon_value + 1;
    const meta = state ? branchMeta(card, level, state) : null;
    return meta ? meta.title : `Set DEFCON to ${level}`;
  }
  if (idx >= o.region && idx < L.size) return REGIONS[idx - o.region] ?? `Region #${idx - o.region}`;
  return `UnknownAction #${idx}`;
}
