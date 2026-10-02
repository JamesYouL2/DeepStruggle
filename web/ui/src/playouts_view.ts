/**
 * The "Paired playouts" panel: pick a few of the model's candidate moves at the open decision, play
 * each out many times from the same sampled worlds and dice (analysis/playouts.ts), and see the
 * win-rate difference against the model's own choice as the pairs come in.
 *
 * The app lends it the engine and the loaded model through `Host`; while a run is going the app
 * refuses moves (`busy`), since the runner swaps game states through the page's single engine.
 */
import { AnalysisChoice, LiveAnalysis } from "./analysis_view";
import { DEFAULT_PLAYOUTS, PairedPlayouts, PlayoutModel, PlayoutProgress } from "./analysis/playouts";
import { WasmEngine } from "./engine/wasm_engine";

export interface PlayoutHost {
  engine(): WasmEngine | null;
  model(): (PlayoutModel & { meta: { mergedInfluence: boolean } }) | null;
  analysis(): LiveAnalysis | undefined;
  /** Called when a run starts and ends, so the app can refuse moves meanwhile. */
  setBusy(busy: boolean): void;
}

const LISTED = 6;
const DEFAULT_CHECKED = 3;

function esc(s: string): string {
  return s.replace(/[&<>"]/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[ch] as string));
}

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

export class PlayoutPanel {
  private readonly body = el<HTMLDivElement>("playouts-body");
  private readonly runButton = el<HTMLButtonElement>("btn-playouts-run");
  private readonly pairsSelect = el<HTMLSelectElement>("playouts-pairs");
  private running = false;
  private stopRequested = false;
  /** Which decision the candidate list and any result belong to (the analysis object). */
  private shownFor: LiveAnalysis | undefined;
  private checked = new Set<number>();
  private last: { progress: PlayoutProgress; names: Map<number, string> } | null = null;

  constructor(private readonly host: PlayoutHost) {
    this.runButton.addEventListener("click", () => (this.running ? this.stop() : this.start()));
    this.body.addEventListener("change", ev => {
      const box = ev.target as HTMLInputElement;
      if (!box.matches("input[data-flat-idx]")) return;
      const idx = parseInt(box.getAttribute("data-flat-idx") || "", 10);
      if (box.checked) this.checked.add(idx);
      else this.checked.delete(idx);
      this.renderButton();
    });
  }

  get busy(): boolean {
    return this.running;
  }

  /** A new position or readout: reset the candidate list (a result belongs to its decision). */
  render(): void {
    if (this.running) return;
    const a = this.host.analysis();
    if (a !== this.shownFor) {
      this.shownFor = a;
      this.last = null;
      this.checked = new Set(this.choices(a).slice(0, DEFAULT_CHECKED).map(c => c.idx));
    }
    this.paint();
  }

  private choices(a: LiveAnalysis | undefined): AnalysisChoice[] {
    if (!a || !a.policy || a.choices.length < 2) return [];
    return [...a.choices].sort((x, y) => y.p - x.p).slice(0, LISTED);
  }

  private renderButton(): void {
    const a = this.host.analysis();
    const ok = !!a && !!this.host.model() && this.choices(a).length >= 2;
    this.runButton.textContent = this.running ? "Stop" : "Run";
    this.runButton.disabled = !this.running && (!ok || this.checked.size < 2);
    this.pairsSelect.disabled = this.running;
  }

  private paint(): void {
    this.renderButton();
    const a = this.host.analysis();
    const choices = this.choices(a);
    if (!this.host.model()) {
      this.body.innerHTML = `<div class="trace-empty">Load a model to play candidate moves out.</div>`;
      return;
    }
    if (choices.length < 2) {
      this.body.innerHTML = `<div class="trace-empty">No decision with two or more moves is open.</div>`;
      return;
    }
    const fav = a!.policy!.argmax_idx;
    const res = new Map(this.last?.progress.rows.map(r => [r.idx, r]) ?? []);
    const ref = this.last?.progress.rows[0]?.idx;
    const rows = choices.map(c => {
      const r = res.get(c.idx);
      const isRef = c.idx === ref;
      let cells = `<span class="playouts-score"></span><span class="playouts-diff"></span>`;
      if (r) {
        const diff = isRef ? "reference" : Number.isNaN(r.diffSe)
          ? `${(100 * r.diff).toFixed(1)}`
          : `${r.diff >= 0 ? "+" : "−"}${Math.abs(100 * r.diff).toFixed(1)} ± ${(100 * r.diffSe).toFixed(1)}`;
        const cls = isRef || Number.isNaN(r.diffSe) ? "" : r.diff - 2 * r.diffSe > 0 ? " better" : r.diff + 2 * r.diffSe < 0 ? " worse" : "";
        cells = `<span class="playouts-score">${(100 * r.score).toFixed(1)}%</span><span class="playouts-diff${cls}">${diff}</span>`;
      }
      return `
        <label class="playouts-row${c.idx === fav ? " favourite" : ""}">
          <input type="checkbox" data-flat-idx="${c.idx}" ${this.checked.has(c.idx) ? "checked" : ""} ${this.running ? "disabled" : ""}>
          <span class="playouts-name">${c.idx === fav ? "★ " : ""}${esc(c.name ?? "#" + c.idx)}</span>
          <span class="playouts-p">${(100 * c.p).toFixed(0)}%</span>
          ${cells}
        </label>`;
    });
    const p = this.last?.progress;
    const status = p
      ? `${p.pairs} pairs × ${p.rows.length} moves · ${p.games} games · ${(p.ms / 1000).toFixed(1)} s`
        + (p.capped ? ` · ${p.capped} games stopped unfinished` : "") + (p.done ? "" : " · running…")
      : "";
    this.body.innerHTML = `
      <div class="playouts-head"><span>move</span><span>model</span><span>win %</span><span>− ★ (pts)</span></div>
      ${rows.join("")}
      <div class="trace-note">${status ? esc(status) + "<br>" : ""}The mover's win rate when the model plays the game
      out after each move; each pair is the same redeal of the unseen cards and the same dice for every move. ±
      is one standard error of the difference. Green or red only beyond two. A move whose value lies in a plan
      the model does not have reads worse than it is.</div>`;
  }

  private async start(): Promise<void> {
    const engine = this.host.engine();
    const model = this.host.model();
    const a = this.host.analysis();
    if (!engine || !model || !a || this.running) return;
    const fav = a.policy?.argmax_idx;
    const picked = this.choices(a).filter(c => this.checked.has(c.idx)).map(c => c.idx);
    // The model's own choice is the reference when it is among the picks; otherwise the likeliest pick.
    const candidates = fav !== undefined && picked.includes(fav) ? [fav, ...picked.filter(i => i !== fav)] : picked;
    if (candidates.length < 2) return;
    const names = new Map(a.choices.map(c => [c.idx, c.name ?? "#" + c.idx]));
    const pairs = parseInt(this.pairsSelect.value, 10) || DEFAULT_PLAYOUTS.pairs;
    this.running = true;
    this.stopRequested = false;
    this.host.setBusy(true);
    this.last = null;
    this.paint();
    try {
      const runner = new PairedPlayouts(engine, model, model.meta.mergedInfluence, { ...DEFAULT_PLAYOUTS, pairs });
      const done = await runner.run(candidates, progress => {
        this.last = { progress, names };
        this.paint();
      }, () => this.stopRequested);
      this.last = { progress: done, names };
    } catch (e) {
      this.body.innerHTML = `<div class="analysis-error">Playouts failed: ${esc(e instanceof Error ? e.message : String(e))}</div>`;
      this.running = false;
      this.host.setBusy(false);
      this.renderButton();
      return;
    }
    this.running = false;
    this.host.setBusy(false);
    this.paint();
  }

  private stop(): void {
    this.stopRequested = true;
  }
}
