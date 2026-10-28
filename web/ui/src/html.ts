/**
 * Text into HTML. Anything the page did not write itself -- a replay file's descriptions, players
 * and trace, a model's names, a log line -- goes through this before it reaches innerHTML: a
 * dropped or served replay is a file anyone could have written.
 *
 * `&`, `<` and `"` are what can end text or a double-quoted attribute (every attribute this page
 * writes is double-quoted), so those are escaped; `>` and `'` are left as they read, so an escaped
 * log line ("HAND_US -> DISCARD_PILE", "'NATO' (#21)") is the same text.
 */
const ENTITIES: Record<string, string> = { "&": "&amp;", "<": "&lt;", '"': "&quot;" };

export function escapeHtml(s: unknown): string {
  return String(s ?? "").replace(/[&<"]/g, c => ENTITIES[c]);
}

/** A number for a template (a replay's numeric fields are only typed, not checked): NaN for anything else. */
export function num(v: unknown): number {
  return typeof v === "number" ? v : Number.NaN;
}
