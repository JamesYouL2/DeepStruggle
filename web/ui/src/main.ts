/**
 * The workbench: watch replays, test the engine by playing it, and play with a model.
 *
 * Everything runs in the page. The engine is the WebAssembly build of engine/ + bindings/
 * (engine/wasm_engine.ts), the game -- stepping, dice, undo, the action log -- is
 * game/session.ts, and models run as ONNX in onnxruntime-web (analysis/). A server is optional:
 * the local one (web/server/main.py) only lists this machine's checkpoints and replays, and the
 * same page works from GitHub Pages with models from Hugging Face or a dropped file.
 */
import { GameState, MicroAction } from "./types";
import { MapView } from "./map_view";
import { TracksView } from "./tracks_view";
import { CardsView } from "./cards_view";
import { ActionHud } from "./action_hud";
import { ReplayControls, ReplayStep } from "./replay_controls";
import { DebugPanel } from "./debug_panel";
import { decorateChoices, policyChipHtml, renderTracePanel, renderValueRibbon, setActionSpace } from "./trace_view";
import { AnalysisPanel, AutoSide, LiveAnalysis, ModelPick } from "./analysis_view";
import { WasmEngine } from "./engine/wasm_engine";
import { GameSession } from "./game/session";
import { decodePosition, encodePosition } from "./game/position";
import { Model, sourceFromParam, sourceToParam } from "./analysis/model";
import { analyze } from "./analysis/readout";
import { DEFAULT_ROLLOUT, RolloutSpec, parseRolloutParam, rolloutParam } from "./analysis/rollout";
import { RolloutRunner } from "./analysis/rollout_runner";
import { ViewMode, opponentOf, redactState, viewFromParam, viewToParam } from "./game/view";
import { lastTurnOf } from "./game/last_turn";
import { recentRolls } from "./game/rolls";
import { describeEventTracks } from "./game/describe";
import { MoveHow, buildSubmission, sha256Hex, summarize } from "./game/record";
import { loadStoredGame, storeGame } from "./game/saved_game";
import { ResultDialog } from "./result_dialog";
import { BUILD_COMMIT, BUILD_DIRTY } from "./build_info";

/** Pause before an auto-played move, long enough to see each one land. */
const AUTO_PLAY_DELAY_MS = 350;

export class TSApp {
  private state: GameState | null = null;
  /** `state` as the selected view may see it (game/view.ts) -- what every panel renders. */
  private shown: GameState | null = null;
  /** Whose eyes the board is drawn for: the developer, or one side playing the model. */
  private view: ViewMode = "DEV";
  private isReplayMode: boolean = false;
  private replaySteps: ReplayStep[] = [];
  private replayCurrentStep: number = 0;
  private isVpFilterActive: boolean = false;

  private mapView: MapView;
  private tracksView: TracksView;
  private cardsView: CardsView;
  private actionHud: ActionHud;
  public replayControls: ReplayControls;
  private debugPanel: DebugPanel;
  private analysisPanel: AnalysisPanel;

  private engine: WasmEngine | null = null;
  private session: GameSession | null = null;
  private model: Model | null = null;
  private modelKey = "";
  private modelLoads = 0;
  /**
   * Analysis was turned off on purpose, by the user or by the link. The link then says
   * `model=off`, so that opening it does not load the default model instead.
   */
  private analysisOff = false;

  /** The live position, its readout and its link token, kept while a replay is on screen. */
  private liveState: GameState | null = null;
  private liveAnalysis: LiveAnalysis | undefined = undefined;
  private positionToken: string | null = null;
  /** Bumped on every change of position; async work for an older one is dropped. */
  private version = 0;

  private urlPosition: string | null = null;
  private urlModel: string | null = null;
  private urlAuto: AutoSide = "";
  /** The link named a view: the page then does not ask for the play mode. */
  private urlView = false;
  private autoPlayTimer: number | null = null;
  /**
   * How the model picks its move (#search-mode-select, `search=` in the URL): null is the policy's
   * argmax from one forward; a spec is the rollout root (analysis/rollout.ts), run by `runner` --
   * in a worker with its own engine and model where the browser has workers, else inline.
   */
  private searchSpec: RolloutSpec | null = null;
  /** The link named a search (`search=`): a fresh open then keeps it rather than offering the rollout root. */
  private urlSearch = false;
  private runner: RolloutRunner | null = null;
  private searchRuns = 0;
  /** sha256 of the loaded model's file: the model's identity in the game's record. */
  private modelSha = "";
  private resultDialog: ResultDialog;
  /** The board showed a finished game at the last refresh: the dialog opens when a game ends. */
  private wasOver = false;

  constructor() {
    this.parseQueryParams();

    const tracksContainer = document.getElementById("global-tracks")!;
    const mapSvg = document.getElementById("ts-map-svg") as unknown as SVGSVGElement;

    this.tracksView = new TracksView(tracksContainer);
    this.mapView = new MapView(mapSvg, (countryId: number) => this.handleCountryClick(countryId));
    this.cardsView = new CardsView((cardId: number) => this.handleCardClick(cardId));
    this.actionHud = new ActionHud((action: MicroAction) => this.sendAction(action));
    this.debugPanel = new DebugPanel((override: any) => this.sendDebugOverride(override));

    this.replayControls = new ReplayControls((replayState: GameState, stepIndex: number, allSteps: ReplayStep[]) => {
      // A replay's snapshots are the full state: watching one is the developer's view.
      if (this.view !== "DEV") this.setView("DEV");
      this.setReplayMode(true);
      this.state = replayState;
      this.replaySteps = allSteps;
      this.replayCurrentStep = stepIndex;
      this.renderState();
      this.renderTrace();
    });
    this.replayControls.liveReplay = () => this.session?.replay() ?? null;

    this.analysisPanel = new AnalysisPanel(
      (pick: ModelPick | null) => this.loadModel(pick),
      (flatIdx: number) => this.sendFlatAction(flatIdx),
      () => {
        this.syncUrl();
        this.ensureSearch();   // the new auto side's decision may not have been searched
        this.maybeAutoPlay();
      },
    );
    this.analysisPanel.setAutoSide(this.urlAuto);
    this.analysisPanel.show(!this.isReplayMode);
    this.actionHud.onRerender = () => this.redecorate();
    this.resultDialog = new ResultDialog(
      (playdek, note) => this.submission(playdek, note),
      () => {
        if (!this.session) return;
        this.session.submitted = true;
        this.storeGame();
      },
    );

    this.setReadoutShown(this.view === "DEV");
    this.setupAnalysisToggle();
    this.setupViewSelect();
    this.setupSearchSelect();
    this.setupGlobalControls();
    this.setupBottomResizer();
    this.setupFileDrop();
    this.boot();
  }

  private badge(text: string, cls: string, title = ""): void {
    const el = document.getElementById("connection-status")!;
    el.textContent = text;
    el.className = `status-badge ${cls}`;
    el.title = title;
  }

  /**
   * Load the engine, start a game -- or open the position a link names -- and the link's model.
   */
  private async boot() {
    try {
      this.engine = await WasmEngine.load();
    } catch (e) {
      this.badge("NO ENGINE", "disconnected", String(e));
      window.alert(String(e));
      return;
    }
    setActionSpace(this.engine.layout);
    this.runner = new RolloutRunner(this.engine);
    this.session = new GameSession(this.engine);
    if (!this.urlPosition || !(await this.resumeStoredGame(this.urlPosition))) {
      this.session.newGame(Math.floor(Math.random() * 1_000_000));
      if (this.urlPosition) {
        try {
          this.session.loadPosition(await decodePosition(this.urlPosition), this.urlPosition);
        } catch (e) {
          window.alert(`Could not open the position in the link -- starting a new game instead.\n\n${e}`);
        }
      }
    }
    this.checkEngineFreshness();
    this.refresh();
    const src = this.urlModel && this.urlModel !== "off" ? sourceFromParam(this.urlModel) : null;
    if (this.urlModel === "off") {
      this.analysisOff = true;
      this.analysisPanel.showSource(null);
    } else if (src && src.kind !== "file") {
      this.analysisPanel.showSource(src);
      this.loadModel({ source: src });
    } else if (!this.isReplayMode) {
      this.loadDefaultModel();
    }
    // A fresh open with nothing decided by the link: ask how to play. A link that names a
    // position, a replay or a view has decided already.
    if (!this.urlPosition && !this.isReplayMode && !this.urlView) this.askMode("open");
  }

  /**
   * A reload: the link names the position this browser's stored game (game/saved_game.ts) is at,
   * so that game is played over again from its seed -- history, undo and record intact -- rather
   * than opened as a new game from a loaded position. False, with the board to be set up
   * afresh, when it is another position or the stored game no longer plays to it.
   */
  private async resumeStoredGame(token: string): Promise<boolean> {
    const stored = loadStoredGame();
    if (!stored || !this.session || !this.engine) return false;
    try {
      const target = await decodePosition(token);
      const start = stored.record.start_position ? await decodePosition(stored.record.start_position) : null;
      this.session.replayRecord(stored.record, start, stored.submitted);
      if (this.engine.saveJson() === target) {
        // A game that had ended has had its dialog: reopening the page does not offer it again.
        this.wasOver = !!this.session.result;
        return true;
      }
    } catch (e) {
      console.warn("The stored game did not play back:", e);
    }
    return false;
  }

  /** This browser's copy of the game on the board, for a reload (game/saved_game.ts). */
  private storeGame() {
    if (!this.session) return;
    storeGame({ record: this.session.record(), submitted: this.session.submitted });
  }

  /** The settings a move is made under, for the game's record: called just before each move. */
  private noteSettings() {
    if (!this.session) return;
    const m = this.model;
    this.session.setSettings({
      view: this.view,
      auto: this.analysisPanel.autoSide,
      model: m ? this.modelSha : "",
      search: this.searchSpec ? rolloutParam(this.searchSpec) : "policy",
    }, m ? { label: m.meta.label, source: sourceToParam(m.source) ?? "", checkpoint: m.meta.checkpoint } : undefined);
  }

  /** How a move by the side to move is made: who made it, and whether the readout was on screen. */
  private moveHow(source: MoveHow["source"]): MoveHow {
    if (source === "auto" || !this.engine || !this.model) return { source, analysis: false };
    const p = this.engine.decisionPlayer();
    const mover = p > 0 ? "US" : p < 0 ? "USSR" : null;
    return { source, analysis: mover !== null && this.analysisPanel.showsReadoutFor(mover) };
  }

  /** The game's record for the stats database, with the result dialog's Playdek fields. */
  private submission(playdek: { login: string; rating: number }, note: string) {
    const s = this.session;
    if (!s || !s.result || !this.engine) return Promise.reject(new Error("the game is not over"));
    return buildSubmission(s.record(), s.result, this.engine.saveJson(),
      { commit: BUILD_COMMIT, dirty: BUILD_DIRTY, engine_fingerprint: this.engine.fingerprint }, playdek, note);
  }

  /** The result dialog for the game on the board, which must be over. */
  private openResult() {
    const s = this.session;
    if (!s || !s.result) return;
    const summary = summarize(s.record(), true);
    const model = summary.models.length === 1 ? s.models[summary.models[0]] : undefined;
    const search = summary.searches.length === 1 ? summary.searches[0] : "";
    this.resultDialog.open({
      winner: s.result.winner, margin: s.result.margin, endTurn: s.result.end_turn, reason: s.result.reason,
      summary, submitted: s.submitted,
      model: model ? `${model.label}${search ? ` (${search === "policy" ? "policy" : `rollout root ${search.replace(/^rollout:/, "")}`})` : ""}` : "",
    });
  }

  /** After a change of position: the Result button, and the dialog when the game has just ended. */
  private offerResult() {
    const s = this.session;
    const over = !!s?.result && !this.isReplayMode;
    document.getElementById("btn-game-result")?.classList.toggle("hidden", !over);
    // Once per ending: a refresh of the same finished board does not reopen it, an undo and a new
    // ending does.
    const ended = over && !this.wasOver;
    this.wasOver = over;
    if (ended) this.openResult();
  }

  /**
   * A link that names no model gets the default Hugging Face repo's default model (its
   * default.json, else its newest upload). Dropped
   * -- result and error alike -- if a model was picked while the repo was being listed.
   */
  private async loadDefaultModel() {
    const before = this.modelLoads;
    try {
      const source = await this.analysisPanel.defaultHf();
      if (this.modelLoads === before) this.loadModel({ source });
    } catch (e) {
      if (this.modelLoads === before) this.analysisPanel.setError(e instanceof Error ? e.message : String(e));
    }
  }

  /** Against the local server's sources: a page built from other sources plays other rules. */
  private async checkEngineFreshness() {
    const fp = this.engine!.fingerprint;
    try {
      const res = await fetch("/api/local/info");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const info = await res.json();
      if (info.engine_fingerprint !== fp) {
        this.badge("ENGINE STALE", "stale",
          `The page's engine (${fp.slice(0, 12)}) is not what the sources here build (${String(info.engine_fingerprint).slice(0, 12)}). `
          + "Rebuild it with tools/scripts/build_web.sh.");
        return;
      }
      this.badge(`ENGINE ${fp.slice(0, 8)}`, "connected", `engine fingerprint ${fp} -- matches the local sources`);
    } catch {
      this.badge(`ENGINE ${fp.slice(0, 8)}`, "connected", `engine fingerprint ${fp} (no local server to compare with)`);
    }
  }

  private parseQueryParams() {
    const params = new URLSearchParams(window.location.search);
    if (params.has("replay")) {
      this.isReplayMode = true;
    }
    this.urlPosition = params.get("pos");
    this.urlModel = params.get("model");
    const auto = (params.get("auto") || "").toUpperCase();
    this.urlAuto = auto === "US" || auto === "USSR" ? auto : "";
    // A replay link shows everything; a player's view cannot watch one.
    this.urlView = params.has("view");
    this.view = this.isReplayMode ? "DEV" : viewFromParam(params.get("view"));
    this.searchSpec = parseRolloutParam(params.get("search"));
    this.urlSearch = params.has("search");
  }

  /**
   * The play-mode question (#mode-dialog): on a fresh open, and before every New Game. The game
   * on the board stays live underneath, so a link, a test or an impatient player can go on without
   * answering; Escape and the close button keep the current mode. On "new", the choice also deals
   * the new game -- cancelling deals nothing.
   *
   * It also asks how the model plays when you play a side (#mode-opponent): the rollout root, its
   * strongest player, or the policy -- the search select's two settings. A fresh open offers the
   * rollout root unless the link named a search; New Game offers the setting in force. Choosing a
   * side sets it; the Developer choice leaves the search select as it is, and cancelling sets nothing.
   */
  private askMode(reason: "open" | "new") {
    const dialog = document.getElementById("mode-dialog");
    if (!dialog) return;
    const note = document.getElementById("mode-dialog-note");
    if (note) {
      note.textContent = reason === "new"
        ? "A fresh deal follows your choice. Close to keep the game on the board."
        : "You can change this any time in the Play mode select above.";
    }
    const rolloutRadio = document.getElementById("mode-opponent-rollout") as HTMLInputElement | null;
    const policyRadio = document.getElementById("mode-opponent-policy") as HTMLInputElement | null;
    const offerRollout = this.searchSpec !== null || (reason === "open" && !this.urlSearch);
    if (rolloutRadio) rolloutRadio.checked = offerRollout;
    if (policyRadio) policyRadio.checked = !offerRollout;
    // The spec choosing it keeps: the one in force, else the measured one.
    const rolloutSpec = this.searchSpec ?? DEFAULT_ROLLOUT;
    const specText = document.getElementById("mode-opponent-rollout-spec");
    if (specText) {
      specText.textContent = `its top ${rolloutSpec.k} move${rolloutSpec.k === 1 ? "" : "s"}, each played out in `
        + `${rolloutSpec.worlds} sampled world${rolloutSpec.worlds === 1 ? "" : "s"}`;
    }
    dialog.classList.remove("hidden");
    const choose = (mode: ViewMode | null) => {
      dialog.classList.add("hidden");
      let searchChanged = false;
      if ((mode === "US" || mode === "USSR") && rolloutRadio) {
        const spec = rolloutRadio.checked ? rolloutSpec : null;
        // Set without searching: the deal or the view below reads the position once, for both.
        if (spec !== this.searchSpec) {
          this.setSearch(spec, false);
          searchChanged = true;
        }
      }
      for (const b of buttons) b.removeEventListener("click", onClick);
      document.getElementById("mode-dialog-close")?.removeEventListener("click", onClose);
      dialog.removeEventListener("click", onBackdrop);
      window.removeEventListener("keydown", onKey);
      if (mode === null) return;
      if (reason === "new") this.newGame(mode);
      else if (mode !== this.view) this.setView(mode);
      else if (searchChanged) this.reconsider();
    };
    const buttons = Array.from(dialog.querySelectorAll<HTMLButtonElement>(".mode-choice"));
    const onClick = (ev: Event) => {
      const m = (ev.currentTarget as HTMLElement).getAttribute("data-mode") as ViewMode;
      choose(m);
    };
    const onClose = () => choose(null);
    const onBackdrop = (ev: Event) => { if (ev.target === dialog) choose(null); };
    const onKey = (ev: KeyboardEvent) => { if (ev.key === "Escape") choose(null); };
    for (const b of buttons) b.addEventListener("click", onClick);
    document.getElementById("mode-dialog-close")?.addEventListener("click", onClose);
    dialog.addEventListener("click", onBackdrop);
    window.addEventListener("keydown", onKey);
    const current = this.view === "US" ? "mode-choice-us" : this.view === "USSR" ? "mode-choice-ussr" : "mode-choice-dev";
    const focus = document.getElementById(current) as HTMLButtonElement | null;
    focus?.focus();
  }

  /** A fresh deal in `mode`. The link then names the new position. */
  private newGame(mode: ViewMode) {
    if (!this.session) return;
    if (mode !== this.view) this.setView(mode, false);
    this.session.newGame(Math.floor(Math.random() * 1_000_000));
    if (this.isReplayMode) this.setReplayMode(false);
    renderTracePanel([], 0);
    this.refresh();
  }

  /**
   * The search select (#search-mode-select): the policy's argmax, or the rollout root at the
   * default spec (the leaderboard's best; another comes in through `search=rollout:k:worlds:horizon:rule[:stop][:leaf]` in the
   * URL and is kept while the select says rollout). A change re-reads the position on screen.
   */
  private setupSearchSelect() {
    const sel = document.getElementById("search-mode-select") as HTMLSelectElement | null;
    if (!sel) return;
    sel.value = this.searchSpec ? "rollout" : "policy";
    sel.addEventListener("change", () => {
      this.setSearch(sel.value === "rollout" ? (this.searchSpec ?? DEFAULT_ROLLOUT) : null);
    });
  }

  /**
   * Set how the model picks its move (null: the policy), from the select or the play-mode
   * question. The readout itself does not depend on it: the position's readout loses its search
   * and, with `apply`, is searched again if it needs to be. Without, the caller does that.
   */
  private setSearch(spec: RolloutSpec | null, apply = true) {
    this.searchSpec = spec;
    const sel = document.getElementById("search-mode-select") as HTMLSelectElement | null;
    if (sel) sel.value = spec ? "rollout" : "policy";
    this.searchRuns++;          // a search still running was asked for under the old setting
    // A new object: a search still running for the old one checks it is the readout on screen.
    const a = this.liveAnalysis;
    if (a) this.liveAnalysis = { ...a, search: undefined, search_pending: undefined, search_error: undefined };
    this.syncUrl();
    if (apply) this.reconsider();
  }

  /**
   * The readout on screen is unchanged, but what is shown of it, what is searched or who
   * auto-plays may not be: redraw it, search it if it now needs a search, and re-arm auto-play.
   */
  private reconsider() {
    if (!this.isReplayMode) {
      this.analysisPanel.render(this.liveAnalysis, this.shown);
      this.analysisPanel.decorate(this.shown);
    }
    this.ensureSearch();
    this.maybeAutoPlay();
  }

  /**
   * A decision is searched when auto-play will play it, or when the readout on screen shows it --
   * nothing else would use the seconds a search takes.
   */
  private searchWanted(decider: "US" | "USSR"): boolean {
    return decider === this.analysisPanel.autoSide || this.analysisPanel.showsReadoutFor(decider);
  }

  /** Search the readout on screen if the setting asks for a search, it needs one, and has none yet. */
  private ensureSearch() {
    const a = this.liveAnalysis;
    if (!a || !a.decision_player || !this.searchSpec || this.isReplayMode) return;
    if (a.search || a.search_pending || a.search_error) return;
    if (!this.searchWanted(a.decision_player)) return;
    void this.runSearch(a, this.version);
  }

  /**
   * The search's decision for the position the readout `a` describes, into `a` and the panel.
   * Dropped if the board, the model or the search setting moved on meanwhile; a failure is shown
   * in the panel and auto-play falls back to the policy.
   */
  private async runSearch(a: LiveAnalysis, version: number) {
    const spec = this.searchSpec;
    const model = this.model;
    if (!spec || !model || !this.runner || !this.engine || !a.decision_player) return;
    const run = ++this.searchRuns;
    const key = this.modelKey;
    a.search_pending = true;
    if (!this.isReplayMode) this.analysisPanel.render(a, this.shown);
    // Still wanted only while `a` is the readout on screen and the setting is the one it searched
    // for. The position alone is not enough: changing the search select re-reads the same position
    // into a new readout without a search, and a stale answer landing after it used to draw its
    // section back into the panel over the policy's.
    const current = () => version === this.version && key === this.modelKey && run === this.searchRuns
      && a === this.liveAnalysis && spec === this.searchSpec;
    try {
      const result = await this.runner.search(this.engine.snapshot(), model, key, spec, this.seedFor(version));
      if (!current()) return;
      a.search_pending = false;
      a.search = result ?? undefined;
    } catch (e) {
      if (!current()) return;
      a.search_pending = false;
      a.search_error = e instanceof Error ? e.message : String(e);
    }
    if (!this.isReplayMode) {
      this.analysisPanel.render(a, this.shown);
      this.analysisPanel.decorate(this.shown);
    }
    this.maybeAutoPlay();
  }

  /** The search's seed for a position: from the game's step, so a reopened link searches alike. */
  private seedFor(_version: number): number {
    const step = this.liveState?.step_index ?? 0;
    return ((step + 1) * 1_000_003) >>> 0;
  }

  /** The view selector: developer (everything) or one side against the model. */
  private setupViewSelect() {
    const sel = document.getElementById("view-select") as HTMLSelectElement | null;
    sel?.addEventListener("change", () => this.setView(sel.value as ViewMode));
    this.setView(this.view, false);
  }

  /**
   * Switch whose eyes the board is drawn for. A player's view locks auto-play to the other side
   * (the model plays it), never shows the readout of the model's own decisions (analysis_view.ts),
   * and turns off the tools that show the full state -- Debug Tools and Replay Mode -- leaving
   * replay mode if it was on. A change of view also sets Show analysis to that view's default:
   * on for the developer, off for a player, whom the readout would give hints.
   */
  private setView(v: ViewMode, rerender = true) {
    const changed = v !== this.view;
    this.view = v;
    const sel = document.getElementById("view-select") as HTMLSelectElement | null;
    if (sel) sel.value = v;
    const opp = opponentOf(v);
    this.analysisPanel.setHiddenSide(opp);
    // A view's own default; the toggle overrides it until the view changes again.
    if (changed) this.setReadoutShown(opp === null);
    document.body.classList.toggle("player-view", opp !== null);
    for (const id of ["btn-toggle-debug", "btn-toggle-replay"]) {
      const b = document.getElementById(id) as HTMLButtonElement | null;
      if (!b) continue;
      b.disabled = opp !== null;
      b.title = opp ? "Shows the full state -- switch to the Developer view to use it" : "";
    }
    if (!rerender) return;
    if (opp && this.isReplayMode) {
      this.setReplayMode(false);
      this.state = this.liveState;
      renderTracePanel([], 0);
    }
    this.renderState();
    this.syncUrl();
    // The readout is the same in every view, but which decisions are searched is not: search the
    // one on screen if the new view needs it, keeping any search already made or running. And
    // re-arm auto-play now: a move armed for the old auto side must not fire for the player.
    this.ensureSearch();
    this.maybeAutoPlay();
  }

  /** Show analysis (#analysis-visible-toggle): the readout and its hints on the board, or not. */
  private setReadoutShown(on: boolean) {
    this.analysisPanel.setReadoutShown(on);
    const chk = document.getElementById("analysis-visible-toggle") as HTMLInputElement | null;
    if (chk) chk.checked = on;
  }

  private setupAnalysisToggle() {
    const chk = document.getElementById("analysis-visible-toggle") as HTMLInputElement | null;
    chk?.addEventListener("change", () => {
      this.setReadoutShown(chk.checked);
      this.reconsider();
    });
  }

  /** A player's view, and the decision on the board is the other side's (the model's). */
  private hiddenSideToMove(): boolean {
    const opp = opponentOf(this.view);
    if (!opp || !this.engine) return false;
    const p = this.engine.decisionPlayer();
    return (p > 0 ? "US" : p < 0 ? "USSR" : "") === opp;
  }

  /**
   * Keep the address bar naming the board on screen -- the position itself, the model and the
   * auto-play side -- so it can be copied and shared at any moment. `replaceState`, never
   * `pushState`: every move would otherwise add a history entry, and Back would walk the game
   * move by move instead of leaving the page. A dropped model file has no address and is left out.
   */
  private syncUrl() {
    if (this.isReplayMode || !this.positionToken) return;
    const params = new URLSearchParams(window.location.search);
    params.delete("game_id");
    params.delete("role");
    const model = this.model ? sourceToParam(this.model.source) : null;
    if (model) params.set("model", model);
    else if (this.analysisOff) params.set("model", "off");
    else params.delete("model");
    const auto = this.analysisPanel.autoSide;
    if (auto) params.set("auto", auto.toLowerCase()); else params.delete("auto");
    const view = viewToParam(this.view);
    if (view) params.set("view", view); else params.delete("view");
    if (this.searchSpec) params.set("search", rolloutParam(this.searchSpec)); else params.delete("search");
    params.set("pos", this.positionToken);
    const next = `${window.location.pathname}?${params.toString()}${window.location.hash}`;
    if (next !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(window.history.state, "", next);
    }
  }

  /**
   * The position changed (a move, an undo, a new game, a loaded link): redraw it, then -- off the
   * critical path -- its link token, the model's readout, and auto-play.
   */
  private refresh() {
    if (!this.session) return;
    const version = ++this.version;
    this.liveState = this.session.state();
    this.liveAnalysis = undefined;
    if (!this.isReplayMode) {
      this.state = this.liveState;
      this.renderState();
    }
    this.storeGame();
    this.offerResult();
    encodePosition(this.engine!.saveJson()).then(token => {
      if (version !== this.version) return;
      this.positionToken = token;
      this.liveState!.position = token;
      this.syncUrl();
    });
    this.runAnalysis(version);
  }

  private async runAnalysis(version: number) {
    const model = this.model;
    if (!model || !this.engine) {
      this.maybeAutoPlay();
      return;
    }
    const key = this.modelKey;
    try {
      const a = await analyze(this.engine, model, key);
      if (version !== this.version || key !== this.modelKey) return;   // the board moved on
      this.liveAnalysis = a;
      if (!this.isReplayMode) {
        this.analysisPanel.render(a, this.shown);
        this.analysisPanel.decorate(this.shown);
      }
      // Only a decision that auto-play plays or the readout shows is searched (searchWanted):
      // another would spend seconds on a recommendation nobody sees.
      if (this.searchSpec && a.decision_player && !this.isReplayMode && this.searchWanted(a.decision_player)) {
        // The search answers later and auto-play waits for it (favourite() is null meanwhile).
        void this.runSearch(a, version);
        return;
      }
      this.maybeAutoPlay();
    } catch (e) {
      if (key === this.modelKey) this.analysisPanel.setError(`The model failed on this position: ${e}`);
    }
  }

  private async loadModel(pick: ModelPick | null) {
    const load = ++this.modelLoads;
    this.analysisOff = !pick;
    this.liveAnalysis = undefined;
    if (!pick) {
      this.model = null;
      this.modelKey = "";
      this.modelSha = "";
      this.runner?.setModel(null, "");
      this.analysisPanel.setOff();
      this.renderState();
      this.syncUrl();
      return;
    }
    this.analysisPanel.setLoading(pick.source);
    try {
      const m = await Model.fetch(pick.source, this.engine!.obsSize, this.engine!.actionSize, pick.bytes);
      // The file's own hash names the model in a game's record: a revision like `main` moves.
      const sha = await sha256Hex(m.bytes).catch(() => "");
      if (load !== this.modelLoads) return;   // another pick superseded this one
      this.model = m;
      this.modelSha = sha;
      this.modelKey = `m${load}`;
      this.runner?.setModel(m, this.modelKey);
      const fp = m.meta.engineFingerprint;
      this.analysisPanel.setLoaded({
        key: this.modelKey,
        label: m.meta.label,
        merged: m.meta.mergedInfluence,
        checkpoint: m.meta.checkpoint,
        fromCache: m.fromCache,
        engineWarning: fp && fp !== this.engine!.fingerprint
          ? `Exported next to another engine build (${fp.slice(0, 8)}; this page runs ${this.engine!.fingerprint.slice(0, 8)}).`
          : undefined,
      });
    } catch (e) {
      if (load !== this.modelLoads) return;
      this.model = null;
      this.modelKey = "";
      this.modelSha = "";
      this.analysisPanel.setError(e instanceof Error ? e.message : String(e));
    }
    this.syncUrl();
    this.runAnalysis(this.version);
  }

  /** Drop an .onnx anywhere to analyse with it, or a .tslog.json to watch it. */
  private setupFileDrop() {
    window.addEventListener("dragover", (e) => {
      e.preventDefault();
      document.body.classList.add("drop-target");
    });
    window.addEventListener("dragleave", () => document.body.classList.remove("drop-target"));
    window.addEventListener("drop", async (e) => {
      e.preventDefault();
      document.body.classList.remove("drop-target");
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      if (f.name.endsWith(".onnx")) {
        this.analysisPanel.dropFile(f);
      } else if (f.name.endsWith(".json")) {
        try {
          // A replay shows everything: watching one is the developer's view.
          if (this.view !== "DEV") this.setView("DEV");
          this.replayControls.loadReplayData(JSON.parse(await f.text()));
        } catch (err) {
          window.alert(`Not a replay: ${err}`);
        }
      }
    });
  }

  private setReplayMode(on: boolean) {
    this.isReplayMode = on;
    this.replayControls.watching = on;
    const replayBtn = document.getElementById("btn-toggle-replay");
    if (replayBtn) {
      replayBtn.textContent = on ? "Live Mode" : "Replay Mode";
      replayBtn.className = on ? "btn btn-warning btn-sm" : "btn btn-primary btn-sm";
    }
    this.analysisPanel?.show(!on);
  }

  private renderState() {
    if (!this.state) return;
    // Every panel draws the view's redaction, never the full state (game/view.ts).
    const shown = redactState(this.state, this.view);
    // A player's view marks what the other side's latest turn did to the map.
    const opp = opponentOf(this.view);
    if (opp && !this.isReplayMode && this.session) shown.last_turn = lastTurnOf(this.session.start, this.session.steps, opp);
    // The dice of the latest action round, while recent (the engine keeps only the last step's).
    if (this.isReplayMode) shown.recent_rolls = recentRolls(null, this.replaySteps, this.replayCurrentStep + 1, shown);
    else if (this.session) shown.recent_rolls = recentRolls(this.session.start, this.session.steps, this.session.steps.length, shown);
    this.shown = shown;
    this.tracksView.render(shown);
    this.mapView.render(shown);
    this.cardsView.render(shown);
    this.actionHud.render(shown);
    this.renderLogStream();
    if (!this.isReplayMode) {
      this.analysisPanel.render(this.liveAnalysis, shown);
      this.analysisPanel.decorate(shown);
    }
  }

  /** Re-apply the probability badges after part of the board re-rendered on its own. */
  private redecorate() {
    if (this.isReplayMode) {
      const next = this.replaySteps[this.replayCurrentStep + 1];
      decorateChoices(next?.policy, this.shown);
    } else {
      this.analysisPanel.decorate(this.shown);
    }
  }

  private getFilteredIndices(
    totalCount: number,
    isVpChanging: (index: number) => boolean
  ): { targetIndices: Set<number>; includedIndices: number[] } {
    const targetIndices = new Set<number>();
    for (let i = 0; i < totalCount; i++) {
      if (isVpChanging(i)) {
        targetIndices.add(i);
      }
    }

    if (!this.isVpFilterActive) {
      const all: number[] = [];
      for (let i = 0; i < totalCount; i++) all.push(i);
      return { targetIndices, includedIndices: all };
    }

    const includedSet = new Set<number>();
    targetIndices.forEach(targetIdx => {
      for (let offset = -2; offset <= 2; offset++) {
        const idx = targetIdx + offset;
        if (idx >= 0 && idx < totalCount) {
          includedSet.add(idx);
        }
      }
    });

    const sorted = Array.from(includedSet).sort((a, b) => a - b);
    return { targetIndices, includedIndices: sorted };
  }

  /**
   * The replay-only views of the model's trace.
   *
   * The board on screen is the position *after* the current step, which is the node the NEXT
   * step was decided at -- so the probabilities painted on the cards, buttons and map come from
   * step N+1, while the critic numbers describe the position itself and come from step N.
   */
  private renderTrace() {
    const ribbon = document.getElementById("rep-value-ribbon");
    if (ribbon) {
      renderValueRibbon(ribbon, this.replaySteps, this.replayCurrentStep,
                        (idx: number) => this.replayControls.goToStep(idx));
    }
    const next = this.replaySteps[this.replayCurrentStep + 1];
    renderTracePanel(this.replaySteps, this.replayCurrentStep);
    decorateChoices(next?.policy, this.state);
  }

  private renderLogStream() {
    const logContainer = document.getElementById("action-log-stream");
    const logCountBadge = document.getElementById("log-count");
    if (!logContainer) return;

    if (this.isReplayMode && this.replaySteps && this.replaySteps.length > 0) {
      const isVpChangingReplay = (idx: number): boolean => {
        const s = this.replaySteps[idx];
        const prev = idx > 0 ? this.replaySteps[idx - 1] : null;
        const vpDiff = prev ? (s.state_snapshot?.victory_points !== prev.state_snapshot?.victory_points) : false;
        const descHasVp = Boolean(s.description && (s.description.includes("Victory Points") || s.description.includes(" VP")));
        // Two Events of one step can move VP and cancel: the snapshots then show no change.
        const eventVp = (s.event_tracks ?? s.state_snapshot?.event_tracks ?? []).some(t => t.vp_delta !== 0);
        return Boolean(vpDiff || descHasVp || eventVp);
      };

      const { targetIndices, includedIndices } = this.getFilteredIndices(this.replaySteps.length, isVpChangingReplay);

      if (logCountBadge) {
        logCountBadge.textContent = this.isVpFilterActive
          ? `${includedIndices.length} filtered (${targetIndices.size} VP events)`
          : `${this.replaySteps.length} steps (Current: #${this.replayCurrentStep + 1})`;
      }

      logContainer.innerHTML = "";
      if (includedIndices.length === 0) {
        logContainer.innerHTML = `<div class="log-empty-msg">No VP-affecting events found in replay.</div>`;
        return;
      }

      includedIndices.forEach((idx, k) => {
        const step = this.replaySteps[idx];
        const isCurrent = idx === this.replayCurrentStep;
        const isVpTarget = targetIndices.has(idx);

        if (k > 0 && idx > includedIndices[k - 1] + 1) {
          const gap = idx - includedIndices[k - 1] - 1;
          const sep = document.createElement("div");
          sep.className = "log-context-separator";
          sep.textContent = `••• ${gap} step${gap > 1 ? "s" : ""} skipped •••`;
          logContainer.appendChild(sep);
        }

        const item = document.createElement("div");
        item.className = `log-item ${isCurrent ? "active-replay-step" : ""} ${isVpTarget ? "vp-target-event" : ""}`;

        const playerClass = (step.player || "SYSTEM").toUpperCase();

        // Format die roll details if available in snapshot
        let dieRollHtml = "";
        const dieRoll = step.state_snapshot?.die_roll;
        if (dieRoll && dieRoll.type && dieRoll.type !== "NONE") {
          const rType = dieRoll.type;
          const rPlayer = dieRoll.roller || step.player;
          const cName = dieRoll.country_name || (dieRoll.country_id !== undefined && dieRoll.country_id < 84 ? `Country #${dieRoll.country_id}` : "");
          const r1 = dieRoll.roll1 ?? 0;
          const tot1 = dieRoll.total1 ?? r1;
          const r2 = dieRoll.roll2 ?? 0;
          const tot2 = dieRoll.total2 ?? r2;
          const succ = dieRoll.success;
          const net = dieRoll.net_delta ?? 0;

          let detailText = "";
          if (rType === "COUP") {
            const defTarget = dieRoll.mod2 ?? 0;
            const res = succ ? `Net +${net} Inf (Success)` : "Coup Failed";
            detailText = `🎲 Coup in ${cName}: ${rPlayer} rolls ${r1} (Total ${tot1}) vs ${defTarget} Def -> ${res}`;
          } else if (rType === "REALIGNMENT") {
            detailText = `🎲 Realignment in ${cName}: US rolls ${r1} (Tot ${tot1}), USSR rolls ${r2} (Tot ${tot2}) -> Net ${net}`;
          } else if (rType === "SPACE_RACE") {
            const status = succ ? "SUCCESS (Advanced)" : "FAILED";
            detailText = `🎲 Space Race Attempt: ${rPlayer} rolls ${r1} -> ${status}`;
          } else if (rType === "WAR_EVENT") {
            const cStr = dieRoll.card_name || "War Event";
            const targetStr = cName ? ` in ${cName}` : "";
            const res = succ ? "Success (Target Captured & VP)" : "Failed";
            detailText = `🎲 ${cStr}${targetStr}: ${rPlayer} rolls ${r1} (Tot ${tot1}) -> ${res}`;
          } else {
            detailText = `🎲 ${rType} Roll: ${rPlayer} rolls ${r1}`;
          }
          dieRollHtml = `<div class="log-detail-line log-die-line">${detailText}</div>`;
        }
        const prevSnapshot = idx > 0 ? this.replaySteps[idx - 1].state_snapshot : undefined;
        const eventHtml = prevSnapshot && step.state_snapshot
          ? describeEventTracks(prevSnapshot, step.state_snapshot, step.event_tracks ?? step.state_snapshot.event_tracks)
              .map(line => `<div class="log-detail-line log-vp-line">${line.trim()}</div>`).join("")
          : "";

        item.innerHTML = `
          <div>
            <div style="display: flex; gap: 6px; align-items: baseline; width: 100%;">
              <span class="log-index" style="font-size: 10px; color: var(--text-dim); min-width: 28px;">#${idx + 1}</span>
              <span class="log-step">[T${step.turn} AR${step.ar}]</span>
              <span class="log-player ${playerClass}">${playerClass}:</span>
              <span class="log-text" style="flex: 1;">${step.description || `Action type=${step.action?.decision_type}`}</span>
              ${policyChipHtml(this.replaySteps, idx)}
              ${isVpTarget ? '<span class="log-vp-badge">🎯 VP CHANGE</span>' : ''}
            </div>
            ${eventHtml}
            ${dieRollHtml}
          </div>
        `;

        item.addEventListener("click", () => {
          this.replayControls.goToStep(idx);
        });

        logContainer.appendChild(item);
      });

      // Smooth scroll active step into view
      requestAnimationFrame(() => {
        const activeEl = logContainer.querySelector(".active-replay-step") as HTMLElement;
        if (activeEl) {
          activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }
      });
      return;
    }

    if (!this.shown?.action_logs) return;

    // The view's log (game/view.ts redacts the other side's private lines).
    const logs = this.shown.action_logs;
    const isVpChangingLive = (idx: number) => {
      const log = logs[idx];
      const hasVpDelta = log.vp_delta !== undefined && log.vp_delta !== 0;
      const hasVpDetail = (log.details || []).some(d => d.includes("Victory Points:") || d.includes(" VP"));
      const hasVpText = log.text.includes("Victory Points:") || log.text.includes(" VP");
      return hasVpDelta || hasVpDetail || hasVpText;
    };

    const { targetIndices, includedIndices } = this.getFilteredIndices(logs.length, isVpChangingLive);

    if (logCountBadge) {
      logCountBadge.textContent = this.isVpFilterActive
        ? `${includedIndices.length} filtered (${targetIndices.size} VP events)`
        : `${logs.length} events`;
    }

    logContainer.innerHTML = "";
    if (includedIndices.length === 0) {
      logContainer.innerHTML = `<div class="log-empty-msg">No VP-affecting events recorded yet.</div>`;
      return;
    }

    includedIndices.forEach((idx, k) => {
      const log = logs[idx];
      const isVpTarget = targetIndices.has(idx);

      if (k > 0 && idx > includedIndices[k - 1] + 1) {
        const gap = idx - includedIndices[k - 1] - 1;
        const sep = document.createElement("div");
        sep.className = "log-context-separator";
        sep.textContent = `••• ${gap} step${gap > 1 ? "s" : ""} skipped •••`;
        logContainer.appendChild(sep);
      }

      const item = document.createElement("div");
      item.className = `log-item ${isVpTarget ? "vp-target-event" : ""}`;

      let detailsHtml = "";
      if (log.details && log.details.length > 0) {
        detailsHtml = log.details.map(d => {
          let lineClass = "log-detail-line";
          if (d.includes("🎲")) lineClass += " log-die-line";
          if (d.includes("Victory Points:")) lineClass += " log-vp-line";
          return `<div class="${lineClass}">${d}</div>`;
        }).join("");
      }

      let vpDeltaBadge = "";
      if (log.vp_delta !== undefined && log.vp_delta !== 0) {
        const deltaClass = log.vp_delta > 0 ? "us" : "ussr";
        const sign = log.vp_delta > 0 ? `+${log.vp_delta}` : `${log.vp_delta}`;
        vpDeltaBadge = `<span class="vp-delta-pill ${deltaClass}">${sign} VP</span>`;
      }

      item.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
          <div>
            <span class="log-step">[T${log.turn} AR${log.ar}]</span>
            <span class="log-player ${log.player}">${log.player}:</span>
            <span class="log-main-text">${log.text}</span>
          </div>
          ${vpDeltaBadge}
        </div>
        ${detailsHtml}
      `;
      logContainer.appendChild(item);
    });

    // Ensure auto-scrolling to the latest bottom line
    requestAnimationFrame(() => {
      logContainer.scrollTop = logContainer.scrollHeight;
    });
  }

  private handleCountryClick(countryId: number) {
    if (this.isReplayMode || this.hiddenSideToMove()) return;
    if (!this.state || !this.state.legal_actions) return;

    // If active decision is POINT_NODE (5), send action
    if (this.state.decision_context.decision_type === 5) {
      if (this.state.legal_actions.valid_ids.includes(countryId)) {
        const roll = this.actionHud ? this.actionHud.selectedDieRoll : 0;
        this.sendAction({
          decision_type: 5,
          primary_id: countryId,
          secondary_id: roll,
          flags: 0
        });
      }
    }
  }

  private handleCardClick(cardId: number) {
    if (this.isReplayMode || this.hiddenSideToMove()) return;
    if (!this.state || !this.state.legal_actions) return;

    if (this.state.decision_context.decision_type === 1) {
      if (this.state.legal_actions.valid_ids.includes(cardId) || this.state.legal_actions.valid_ids.length === 0) {
        this.sendAction({
          decision_type: 1,
          primary_id: cardId,
          secondary_id: 0,
          flags: 0
        });
      }
    }
  }

  /** A refused action changes nothing; say so without interrupting play. */
  private refused(why: string) {
    console.warn("Engine refused:", why);
    const el = document.getElementById("connection-status");
    if (!el) return;
    const prev = { text: el.textContent, cls: el.className, title: el.title };
    this.badge("REFUSED", "disconnected", why);
    window.setTimeout(() => {
      el.textContent = prev.text;
      el.className = prev.cls;
      el.title = prev.title;
    }, 1500);
  }

  /**
   * A click. `secondary_id` carries the HUD's manual die (0 = roll, 1..6 = force), the
   * workbench's affordance for testing the engine, as it did on the server.
   */
  private sendAction(action: MicroAction) {
    if (this.isReplayMode || !this.session) return;
    if (this.hiddenSideToMove()) {
      this.refused(`it is ${opponentOf(this.view)}'s move -- the model plays that side in this view`);
      return;
    }
    this.noteSettings();
    const forcedDie = action.secondary_id ?? 0;
    const why = this.session.apply(action, forcedDie, this.moveHow("human"));
    if (why) {
      this.refused(why);
      return;
    }
    this.afterOwnMove(forcedDie);
    this.refresh();
  }

  /**
   * A forced die is for one roll: once a move of yours has rolled with it, the selector is back on
   * Auto -- otherwise one click on a face would force every roll after it, unnoticed.
   */
  private afterOwnMove(forcedDie: number) {
    if (forcedDie > 0 && this.session?.tags[this.session.tags.length - 1]?.rolled) this.actionHud.selectedDieRoll = 0;
  }

  /**
   * Play a flat action in the loaded model's own action view (the favourite button, a row of the
   * panel's list, or auto-play). A composed E4.1 action is applied as its two E4 steps.
   */
  private sendFlatAction(flatIdx: number, forcedDie: number = this.actionHud.selectedDieRoll, auto = false) {
    if (this.isReplayMode || !this.session) return;
    // In a player's view only auto-play moves for the other side.
    if (!auto && this.hiddenSideToMove()) {
      this.refused(`it is ${opponentOf(this.view)}'s move -- the model plays that side in this view`);
      return;
    }
    this.noteSettings();
    const how = this.moveHow(auto ? "auto" : "suggested");
    const why = this.session.applyFlat(flatIdx, this.model?.meta.mergedInfluence ?? false, forcedDie, how);
    if (why) {
      this.refused(why);
      return;
    }
    if (!auto) this.afterOwnMove(forcedDie);
    this.refresh();
  }

  /**
   * Auto-play: when the side to move is the auto-play side, play the model's favourite after a
   * short pause (so a person can follow the moves). Re-armed on every readout; a pending move is
   * dropped if the position changes before it fires.
   */
  private maybeAutoPlay() {
    if (this.autoPlayTimer !== null) {
      window.clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (this.isReplayMode) return;
    const idx = this.analysisPanel.autoPlayMove();
    if (idx === null) return;
    const version = this.version;
    this.autoPlayTimer = window.setTimeout(() => {
      this.autoPlayTimer = null;
      if (this.isReplayMode || version !== this.version) return;
      // The view or the auto side may have changed since it was armed.
      if (this.analysisPanel.autoPlayMove() !== idx) return;
      // Always the engine's own die: the manual die selector is for the moves you make.
      this.sendFlatAction(idx, 0, true);
    }, AUTO_PLAY_DELAY_MS);
  }

  /**
   * Take back the last move. With auto-play on, keep taking back until the decision is yours
   * again -- otherwise auto-play would replay the move just taken back, and Cancel could never
   * get past it.
   */
  private cancelAction() {
    if (this.isReplayMode || !this.session || !this.engine) return;
    if (this.autoPlayTimer !== null) {
      window.clearTimeout(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
    if (!this.session.undo()) return;
    const auto = this.analysisPanel.autoSide;
    const side = () => (this.engine!.decisionPlayer() > 0 ? "US" : this.engine!.decisionPlayer() < 0 ? "USSR" : "");
    while (auto && side() === auto && this.session.canUndo) {
      this.session.undo();
    }
    this.refresh();
  }

  private sendDebugOverride(override: any) {
    if (!this.session) return;
    let why: string | null = null;
    if (override.op === "set_country") why = this.session.setCountry(override.country_id, override.us, override.ussr);
    else if (override.op === "set_defcon") why = this.session.setDefcon(override.defcon);
    else if (override.op === "set_vp") why = this.session.setVp(override.vp);
    if (why) {
      this.refused(why);
      return;
    }
    this.refresh();
  }
  private setupBottomResizer() {
    const resizer = document.getElementById("bottom-resizer");
    const bottomPanel = document.getElementById("bottom-panel");
    if (!resizer || !bottomPanel) return;

    let isResizing = false;
    let startY = 0;
    let startHeight = 0;

    resizer.addEventListener("mousedown", (e) => {
      isResizing = true;
      startY = e.clientY;
      startHeight = bottomPanel.offsetHeight;
      resizer.classList.add("active");
      document.body.style.cursor = "ns-resize";
      e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
      if (!isResizing) return;
      const deltaY = startY - e.clientY;
      const newHeight = Math.max(100, Math.min(window.innerHeight * 0.65, startHeight + deltaY));
      bottomPanel.style.height = `${newHeight}px`;
    });

    window.addEventListener("mouseup", () => {
      if (isResizing) {
        isResizing = false;
        resizer.classList.remove("active");
        document.body.style.cursor = "";
      }
    });
  }

  private setupGlobalControls() {
    // Undo / Cancel Action button
    document.getElementById("btn-undo-action")?.addEventListener("click", () => {
      this.cancelAction();
    });

    // Global keyboard shortcuts (Ctrl+Z / Cmd+Z / Escape for undo). None while the play-mode
    // question or the result dialog is open: their Escape closes them, and must not also take
    // back a move.
    window.addEventListener("keydown", (e) => {
      const question = document.getElementById("mode-dialog");
      if (question && !question.classList.contains("hidden")) return;
      if (this.resultDialog.isOpen) {
        if (e.key === "Escape") this.resultDialog.close();
        return;
      }
      const t = e.target as HTMLElement | null;
      const typing = !!t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !typing) {
        e.preventDefault();
        this.cancelAction();
      } else if (e.key === "Escape" && !typing) {
        this.cancelAction();
      } else if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        if (!typing && !this.isReplayMode) {
          e.preventDefault();
          this.analysisPanel.playFavourite();
        }
      }
    });

    // Zoom buttons
    document.getElementById("btn-zoom-in")?.addEventListener("click", () => this.mapView.zoom(0.85));
    document.getElementById("btn-zoom-out")?.addEventListener("click", () => this.mapView.zoom(1.15));
    document.getElementById("btn-zoom-reset")?.addEventListener("click", () => this.mapView.resetView());
    document.querySelectorAll<HTMLButtonElement>("#map-region-jump [data-region]").forEach(b =>
      b.addEventListener("click", () => this.mapView.zoomToRegion(b.dataset.region ?? "")));

    // New Game: ask how to play, then a fresh seed in that mode (askMode -> newGame).
    document.getElementById("btn-new-game")?.addEventListener("click", () => {
      if (!this.session) return;
      this.askMode("new");
    });
    document.getElementById("btn-game-result")?.addEventListener("click", () => this.openResult());

    // The model chooser: the source controls, dropped down from the technical bar by Change,
    // closed by Change again, a click elsewhere, or Escape.
    const chooser = document.getElementById("model-chooser");
    const chooseBtn = document.getElementById("btn-model-choose");
    if (chooser && chooseBtn) {
      const setOpen = (open: boolean) => {
        chooser.classList.toggle("hidden", !open);
        chooseBtn.setAttribute("aria-expanded", String(open));
        chooseBtn.textContent = open ? "▴" : "▾";
      };
      chooseBtn.addEventListener("click", () => setOpen(chooser.classList.contains("hidden")));
      document.addEventListener("click", (e) => {
        const t = e.target as Node;
        if (!chooser.classList.contains("hidden") && !chooser.contains(t) && !chooseBtn.contains(t)) setOpen(false);
      });
      window.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !chooser.classList.contains("hidden")) setOpen(false);
      });
    }

    // Debug Tools button
    document.getElementById("btn-toggle-debug")?.addEventListener("click", () => {
      if (this.view !== "DEV") return;   // the inspector shows the full state
      if (this.state && !this.isReplayMode) this.debugPanel.openDebugModal(this.state);
    });

    // Replay Mode Toggle button
    const replayBtn = document.getElementById("btn-toggle-replay")!;
    replayBtn.addEventListener("click", () => {
      if (this.view !== "DEV") return;   // a replay's snapshots are the full state
      this.setReplayMode(!this.isReplayMode);
      if (!this.isReplayMode && this.liveState) {
        // Back to the live game: the board still shows the replay's last position.
        this.state = this.liveState;
        renderTracePanel([], 0);
        this.renderState();
        if (this.liveAnalysis) {
          this.analysisPanel.render(this.liveAnalysis, this.shown);
          this.analysisPanel.decorate(this.shown);
        }
        this.syncUrl();
        this.ensureSearch();   // no search runs in replay mode
        this.maybeAutoPlay();
        return;
      }
      this.renderLogStream();
    });

    // VP Filter Checkbox
    const vpFilterCheckbox = document.getElementById("chk-filter-vp") as HTMLInputElement;
    if (vpFilterCheckbox) {
      vpFilterCheckbox.addEventListener("change", (e) => {
        this.isVpFilterActive = (e.target as HTMLInputElement).checked;
        this.renderLogStream();
      });
    }

    // Discard & Removed pile modals
    document.getElementById("btn-view-discard")?.addEventListener("click", () => {
      if (this.state) this.cardsView.showPileModal("Discard Pile", this.state.discard_pile || []);
    });
    document.getElementById("btn-view-removed")?.addEventListener("click", () => {
      if (this.state) this.cardsView.showPileModal("Removed from Game Pile", this.state.removed_pile || []);
    });

    // Modal Close button & ESC key
    const closeModal = () => {
      document.getElementById("modal-container")?.classList.add("hidden");
    };
    document.getElementById("modal-close")?.addEventListener("click", closeModal);
    document.getElementById("modal-container")?.addEventListener("click", (e) => {
      if ((e.target as HTMLElement).id === "modal-container") {
        closeModal();
      }
    });
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeModal();
      }
    });
  }
}

// Boot application
window.addEventListener("DOMContentLoaded", () => {
  (window as any).__wb = new TSApp();
});
