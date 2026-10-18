/**
 * The game-over dialog (index.html #result-dialog): the result, how the game was played -- the
 * record's summary (game/record.ts) in words -- and the offer to add it to the stats database
 * with the player's Playdek login and rating, which estimate the models' strength on Playdek's
 * scale (tools/game_stats.py). Offered once when a game ends, and again from the Result button.
 * Nothing is sent unless the player presses Add; the record can also be downloaded.
 *
 * The login and rating are remembered in this browser (localStorage) to prefill the next game.
 */
import { RecordSummary, Submission } from "./game/record";
import { STATS_ENDPOINT } from "./build_info";

const PLAYDEK_KEY = "ts.stats.playdek.v1";

export interface ResultView {
  winner: string;
  margin: number;
  endTurn: number;
  reason: string;
  summary: RecordSummary;
  /** The model's name and how it searched, for the auto-played side ("" if none). */
  model: string;
  submitted: boolean;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

/** The summary's departures from a straight game against one model, in words. */
export function summaryNotes(s: RecordSummary): string[] {
  const notes: string[] = [];
  if (s.model_side === null) notes.push("The model did not play a side.");
  if (s.model_side === "mixed") notes.push("The model played both sides.");
  if (s.human_moves_for_model) notes.push(`You made ${plural(s.human_moves_for_model, "move")} for the model's side.`);
  if (s.dev_view_moves) notes.push(`${plural(s.dev_view_moves, "move")} of yours ${s.dev_view_moves === 1 ? "was" : "were"} made in the Developer view, which shows the model's hand.`);
  if (s.analysis_moves) notes.push(`${plural(s.analysis_moves, "move")} of yours ${s.analysis_moves === 1 ? "was" : "were"} made with the model's analysis on screen.`);
  if (s.suggested_moves) notes.push(`${plural(s.suggested_moves, "move")} of yours played the model's suggestion.`);
  if (s.models.includes("")) notes.push("Some of the model side's moves were made with no model loaded.");
  if (s.models.length > 1) notes.push("The model changed during the game.");
  if (s.searches.length > 1) notes.push("The model's search setting changed during the game.");
  if (s.forced_dice) notes.push(`${plural(s.forced_dice, "die", "dice")} ${s.forced_dice === 1 ? "was" : "were"} forced.`);
  if (s.debug_overrides) notes.push(`${plural(s.debug_overrides, "debug override")}.`);
  if (s.loaded_position) notes.push("The game started from a loaded position.");
  return notes;
}

interface Playdek {
  login: string;
  rating: number | null;
}

function loadPlaydek(): Playdek {
  try {
    const v = JSON.parse(window.localStorage.getItem(PLAYDEK_KEY) ?? "null");
    if (v && typeof v.login === "string") return { login: v.login, rating: typeof v.rating === "number" ? v.rating : null };
  } catch {
    // nothing remembered
  }
  return { login: "", rating: null };
}

function storePlaydek(p: Playdek): void {
  try {
    window.localStorage.setItem(PLAYDEK_KEY, JSON.stringify(p));
  } catch {
    // not remembered; asked again next time
  }
}

export class ResultDialog {
  private readonly el = document.getElementById("result-dialog")!;
  private readonly form = document.getElementById("result-form") as HTMLFormElement;
  private readonly login = document.getElementById("result-login") as HTMLInputElement;
  private readonly rating = document.getElementById("result-rating") as HTMLInputElement;
  private readonly note = document.getElementById("result-note") as HTMLTextAreaElement;
  private readonly submit = document.getElementById("result-submit") as HTMLButtonElement;
  private readonly status = document.getElementById("result-status")!;
  private sending = false;

  /**
   * `makeSubmission` builds the game's record with the form's Playdek fields and note;
   * `onSubmitted` is told once the database has it.
   */
  constructor(
    private readonly makeSubmission: (playdek: { login: string; rating: number }, note: string) => Promise<Submission>,
    private readonly onSubmitted: () => void,
  ) {
    document.getElementById("result-dialog-close")?.addEventListener("click", () => this.close());
    document.getElementById("result-skip")?.addEventListener("click", () => this.close());
    this.el.addEventListener("click", (ev) => { if (ev.target === this.el) this.close(); });
    this.el.addEventListener("keydown", (ev) => { if (ev.key === "Escape") this.close(); });
    this.form.addEventListener("submit", (ev) => {
      ev.preventDefault();
      void this.send();
    });
    document.getElementById("result-download")?.addEventListener("click", () => void this.download());
  }

  get isOpen(): boolean {
    return !this.el.classList.contains("hidden");
  }

  open(v: ResultView): void {
    const s = v.summary;
    const outcome = document.getElementById("result-dialog-outcome")!;
    const verdict = v.winner === "DRAW" ? "Draw" : `${v.winner} wins`;
    const you = s.human_side === null ? "" : v.winner === "DRAW" ? "" : v.winner === s.human_side ? " -- you won" : " -- you lost";
    outcome.innerHTML = `<b>${verdict}</b>${you}. VP ${v.margin > 0 ? "+" : ""}${v.margin}, turn ${v.endTurn}`
      + (v.reason ? ` (${escapeHtml(v.reason)})` : "") + ".";
    const facts = document.getElementById("result-dialog-facts")!;
    facts.innerHTML = [
      s.human_side ? `<dt>You played</dt><dd>${s.human_side}</dd>` : "",
      v.model ? `<dt>Against</dt><dd>${escapeHtml(v.model)}</dd>` : "",
      s.undos ? `<dt>Taken back</dt><dd>${plural(s.undos, "move")}</dd>` : "",
    ].join("");
    const notes = summaryNotes(s);
    const list = document.getElementById("result-dialog-notes")!;
    list.innerHTML = notes.length
      ? `<p class="result-notes-intro">Recorded with these notes, so the game can be filtered:</p><ul>${notes.map(n => `<li>${escapeHtml(n)}</li>`).join("")}</ul>`
      : `<p class="result-notes-intro">A straight game against one model: recorded as clean.</p>`;

    const p = loadPlaydek();
    this.login.value = p.login;
    this.rating.value = p.rating === null ? "" : String(p.rating);
    this.note.value = "";
    this.setStatus("", "");
    this.sending = false;
    if (v.submitted) {
      this.submit.disabled = true;
      this.setStatus("This game is in the database already.", "ok");
    } else if (!STATS_ENDPOINT) {
      this.submit.disabled = true;
      this.setStatus("This build has no stats database (TS_STATS_ENDPOINT): download the record instead.", "");
    } else {
      this.submit.disabled = false;
    }
    this.el.classList.remove("hidden");
    (p.login ? this.rating : this.login).focus();
  }

  close(): void {
    this.el.classList.add("hidden");
  }

  private setStatus(text: string, cls: "" | "ok" | "error"): void {
    this.status.textContent = text;
    this.status.className = `result-status ${cls}`;
  }

  /** The form's Playdek fields, or null (with the reason shown) when they are not filled in. */
  private playdek(require: boolean): { login: string; rating: number } | null {
    const login = this.login.value.trim();
    const rating = Number(this.rating.value);
    const ratingOk = this.rating.value.trim() !== "" && Number.isFinite(rating) && rating >= 1 && rating <= 5000;
    if (require && (!login || !ratingOk)) {
      this.setStatus(!login ? "Your Playdek login, please." : "Your Playdek rating, please (1-5000).", "error");
      (!login ? this.login : this.rating).focus();
      return null;
    }
    return { login, rating: ratingOk ? Math.round(rating) : 0 };
  }

  private async send(): Promise<void> {
    if (this.sending || !STATS_ENDPOINT) return;
    const p = this.playdek(true);
    if (!p) return;
    storePlaydek(p);
    this.sending = true;
    this.submit.disabled = true;
    this.setStatus("Sending…", "");
    try {
      const body = await this.makeSubmission(p, this.note.value.trim());
      const res = await fetch(`${STATS_ENDPOINT.replace(/\/+$/, "")}/games`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const reply = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(reply?.error ?? `HTTP ${res.status}`);
      this.onSubmitted();
      this.setStatus(reply?.stored === false ? "This game is in the database already." : "Added. Thank you!", "ok");
    } catch (e) {
      this.submit.disabled = false;
      this.setStatus(`Not added: ${e instanceof Error ? e.message : String(e)}`, "error");
    } finally {
      this.sending = false;
    }
  }

  /** The record as the database would get it, as a file. */
  private async download(): Promise<void> {
    const p = this.playdek(false)!;
    try {
      const body = await this.makeSubmission(p, this.note.value.trim());
      const url = URL.createObjectURL(new Blob([JSON.stringify(body, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = `ts-game-${body.id}.json`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) {
      this.setStatus(`Could not make the record: ${e instanceof Error ? e.message : String(e)}`, "error");
    }
  }
}
