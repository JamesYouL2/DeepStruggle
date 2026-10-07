# Frontend & Web Workbench Developer Guide (`web/ui/`)

The interactive web client and development workbench for **Twilight Struggle**, built with
**Vite + TypeScript + vanilla CSS + SVG**. It runs entirely in the browser -- the engine as
WebAssembly, models as ONNX -- so `dist/` works from GitHub Pages as well as from the local server
(`web/server/main.py`), which adds this machine's checkpoints and replays.

---

## 1. Architecture & Component Structure

```
web/ui/
├── index.html                  # Workbench layout (Top Bar; cards | SVG Map | HUD rail; Bottom Drawer)
├── package.json                # Dependencies and build scripts
├── vite.config.ts              # base path (TS_WEB_BASE, for Pages), /api proxy to the local server
├── tsconfig.json
├── src/
│   ├── main.ts                 # TSApp coordinator: boot, refresh(), links, auto-play, Action Stream
│   ├── engine/wasm_engine.ts   # the WebAssembly engine (public/engine/, built by build_web.sh)
│   ├── game/                   # session.ts (stepping, undo, log, export), describe.ts (log text),
│   │                           # names.ts (flat action names), position.ts (link tokens)
│   ├── analysis/               # model.ts (ONNX sources + onnxruntime-web), onnx_meta.ts,
│   │                           # readout.ts (policy + critic for the position on screen)
│   ├── metadata.ts             # rules/map.json and rules/cards.json, bundled
│   ├── map_view.ts             # SVG Deluxe Map renderer (84 countries, lines, influence badges, pan/zoom)
│   ├── cards_view.ts           # Hand tabs + Card Explorer (left column), Active Continuous Effects panel (EFFECT_INFO_MAP)
│   ├── tracks_view.ts          # Turn, Action Round, DEFCON, Mil Ops, Space Race, and Victory Points tracks
│   ├── action_hud.ts           # Decision HUD, human-readable branch catalog (CARD_BRANCHES), and dice selector
│   ├── replay_controls.ts      # Replay timeline scrubber, playback engine, and server replay picker
│   ├── trace_view.ts           # Replay trace: value ribbon, log chips, readout panel, probability badges;
│   │                           # the critic display (which reading is calibrated) shared with analysis_view
│   ├── analysis_view.ts        # Live model analysis panel: model picker, critic readings, top choices, badges
│   ├── debug_panel.ts          # State inspector and engine override tools
│   ├── style.css               # Theme tokens, dark mode palette, animations, layout styles
│   └── types.ts                # TypeScript interface mirrors of engine GameState and MicroAction
├── public/engine/              # ts_engine.mjs + .wasm, not committed (tools/scripts/build_web.sh)
└── dist/                       # Production bundle, not committed (local server, or GitHub Pages)
```

---

## 2. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Frontend Documentation Synchronized**:
> When adding new UI components, modifying SVG styling, expanding effect catalogs, adding branch mappings, or altering API metadata consumption, you **MUST** update this file and root [`AGENTS.md`](../../AGENTS.md).

---

## 3. Key Frontend Features & Capabilities

1. **Interactive SVG Deluxe Map (`map_view.ts`)**:
   - Renders all 84 countries with regional color coding, battleground badges, stability indicators, and US/USSR influence counts.
   - **Control is the card's fill**: blue for a US-controlled country, red for a USSR one, the
     controller's influence box outlined in white (`data-control` on the node). Not the border
     alone, because a legal target's border is replaced by the pulsing highlight.
   - Highlights valid country targets with dynamic glowing borders when a `POINT_NODE` decision is active.
   - Each region's scoring badge has a **scoring-card row** (`scoringCardWhereabouts`, from
     `state.card_locations`): `LIVE` while the card is in the draw deck or a hand (or headlined,
     or peeked), `OUT` once discarded or removed, `NOT YET` for a Mid War card before Mid War;
     the hover tooltip says where exactly. It sits inside the badge because country nodes are
     drawn over the badges.
   - A model's probability for a country (`trace_view.ts` `svgBadge`) is a small badge in the
     corner right of the influence boxes, placed from `rect.inf-us`, so it never covers a
     number. The HTML badge's CSS is scoped to `span.trace-choice-badge`: unscoped, it reached the
     SVG text and drew it at 10 map units.
   - **This turn's effects** (`#map-turn-effects`, `CardsView.renderTurnEffects`) run across the
     top of the map: every flag whose `EFFECT_INFO_MAP` duration is `TURN`, the card's text on
     hover. Permanent and conditional effects, and the Space Race attempt flags, stay in the
     right rail's effects panel.
   - Mouse pan and zoom, with metadata loaded autonomously and re-rendered on arrival.
2. **Layout: cards on the left, the board, the HUD rail on the right (`index.html`, `style.css`)**:
   - `.main-content` is a three-column grid: `#cards-sidebar` (the hands panel -- USSR / US /
     Piles / All tabs, the switch unchanged) the full height of the board on the left, the map in
     the middle, and the right rail with the Decision HUD, model analysis / replay readout and
     active effects. Below 1280px the side columns narrow; below 960px it becomes one scrolling
     column: map, HUD rail, cards.
   - The *All* tab lists each card's name and where it is -- no era/side line and no hover
     description (the hand tabs keep their tooltip); a click opens the card's text.
3. **Active Continuous Effects Panel (`cards_view.ts`)**:
   - `EFFECT_INFO_MAP` describes every persistent continuous-effect and state bit (e.g. *Containment*, *Brezhnev Doctrine*, *Red Scare/Purge*, *Quagmire*, *Bear Trap*, *Flower Power*, *Willy Brandt*, *NORAD*, *NATO*, Space Race bonuses) with side tags (`US`, `USSR`, `Both`, `Neutral`), duration badges (`Turn Only`, `Permanent`, `Conditional`, `Space Perk`), and rule summaries. Add an entry here whenever the engine gains an effect bit.
4. **Descriptive Branch Choice HUD (`action_hud.ts`)**:
   - `CARD_BRANCHES` translates numeric branch IDs into readable options (e.g. *Branch 0: Award 2 VP / Branch 1: Conduct 4 Ops* for Olympic Games; *Branch 0: Remove all US Influence / Branch 1: Add 5 USSR Influence* for Warsaw Pact).
   - An event's card choice (`SELECT_CARD` with a `resolving_card`) lists every offered card as a
     button in the HUD -- name, ops, side and where it is -- because those cards are mostly in no
     hand tab: Our Man in Tehran's five drawn cards (`PEEKED_TEMP`, "Drawn"), Star Wars' and SALT
     Negotiations' discard pile. The pass button appears whenever the mask offers card 0, labelled
     for what it means there (Done / Take no card / No card can be chosen).
     `tests/web/test_e2e_card_choices.py` plays all three from the US hand.
5. **Deluxe Space Race Track & Modal Inspector (`tracks_view.ts`)**:
   - Renders the 9-step ladder (0..8) in the top status bar with US and USSR marker tokens and turn attempt counters (`Attempts: used/max`).
   - Clicking the widget opens the full modal: all 9 boxes, required Ops, success rolls, 1st/2nd VP awards, and special ongoing privileges.
6. **Interactive Action Stream & Replay Timeline (`replay_controls.ts`, `main.ts`)**:
   - Timeline scrubber with play/pause, step forward/backward, and server replay loading.
   - The bottom Action Stream lists recorded game events with active-step highlighting (`.active-replay-step`) and click-to-scrub navigation.
7. **The game runs in the page (`engine/`, `game/`, `main.ts`)**:
   - `engine/wasm_engine.ts` loads the WebAssembly build of `engine/` + `bindings/`
     (`bindings/wasm/ts_engine_wasm.cpp`, built into `public/engine/` by
     `tools/scripts/build_web.sh`). Display state, saves, rules and observations are the same C++
     as the Python stack; `tests/web/test_wasm_engine.py` holds the two builds to bit-identical
     games. **After any engine change, rebuild the page** -- it runs its own copy of the engine.
     The header badge shows the page engine's fingerprint and turns **ENGINE STALE** when the
     local server's sources have moved on (`/api/local/info`).
   - `game/session.ts` is the game: apply a click (a MicroAction; `secondary_id` carries the HUD's
     manual die) or a model's flat action, resolve dice, undo (raw GameState snapshots; the log and
     the replay are restored with it), debug overrides (undoable), load a shared position, export
     the game as a `.tslog.json` (*Export Replay*). A composed E4.1 action is applied as its two E4
     steps. `game/describe.ts` writes the action log -- a line-for-line port of the old Python
     session, pinned by a golden (`tests/web/test_describe_golden.py`).
   - `refresh()` in `main.ts` is the one place a change of position lands: render, then (async,
     versioned so a stale result is dropped) the link token, the model's readout, and auto-play.
8. **Live model analysis (`analysis/`, `analysis_view.ts`)**:
   - Models are ONNX exports of checkpoints (`tools/export_onnx.py`) run by onnxruntime-web,
     single-threaded (GitHub Pages cannot send the cross-origin-isolation headers threads need),
     from three sources: **local** checkpoints (`/api/local/models`; the server exports on first
     use), a **Hugging Face** model repo (`owner/repo`, listed from the public tree API), or a
     dropped **file**. The file carries its description in ONNX metadata (`analysis/onnx_meta.ts`):
     a model whose observation width differs from the engine's is refused, one exported next to
     another engine build is flagged.
   - **The default model.** A link with no `model` loads the newest `.onnx` in
     `DEFAULT_HF_REPO` (`mihaild/deepstruggle@main`, `analysis/model.ts`): `listHfModels` asks the
     tree API with `expand=true`, which gives each file's last commit date, follows the `Link`
     pages, and sorts newest first. The default is dropped if the user picks a model while it is
     listing. Turning analysis off writes `model=off`, so that link does not load the default.
     The E2E browsers launch with `HF_BLOCKED` (huggingface.co does not resolve); tests that need
     the repo serve it with `page.route`.
   - `analysis/readout.ts` is the port of the old server readout: one batched forward (the
     decider's observation with its real mask, and both sides' observations for the critic),
     softmax over the legal actions at temperature 1, the argmax as the favourite.
     `tests/web/test_e2e_workbench.py` checks it against `read_policy` / `read_critic` in Python.
   - **Which critic reading is believed.** The value head is trained only on the observation of
     the side to move, so of the two readings (`v_win_us` from the US observation, `v_win_ussr`
     from the USSR one) only the decider's is calibrated; the other is untrained extrapolation
     and can be far off. `trace_view.ts` `valueSidesHtml` shows, for each side, just its own
     P(victory) = (1 + v_win)/2 and expected final VP margin (`v_vp` × `VP_LIMIT` = 20), with the
     decider's card lit and the other dimmed -- no labels, bar or raw table. Live,
     the decider is `analysis.decision_player` (the readout is on the position itself); in a
     replay it is `replayDecider()`: the critic is read *after* the step's action, so the decider
     is the **next** step's `player` (the last step falls back to its snapshot's decision
     context). A terminal position has no decider -- its snapshot's stale decision context is
     ignored -- so both readings are dimmed and the result is shown below them. The replay's
     value ribbon and the log rows' `ΔP` chip plot the calibrated reading as P(US wins); the
     untrained one is only the ribbon's faint dashed line. `tests/web/test_value_readings.py`
     pins the rules on the TypeScript itself; the E2E tests check them on screen.
   - Every legal action's probability is painted on the card, HUD button or country that sends
     it, matched by the MicroAction each choice carries. For an E4.1 (merged-influence) model the
     composed placements go on the countries and the influence button carries their sum. The
     favourite is marked ★ (the replay's played move stays ◀).
   - *★ Play favourite* / key `F` / clicking a row in the panel plays a flat action in the model's
     own action view. **Auto-play** (none / USSR / US, `auto=` in the URL): whenever the chosen
     side is to move, `maybeAutoPlay()` plays the model's favourite after `AUTO_PLAY_DELAY_MS`,
     with the engine's own die, and drops the move if the position changed meanwhile. With
     auto-play on, *Cancel* keeps undoing until the decision is the other side's again.
     `ActionHud.onRerender` re-applies badges after the HUD redraws itself (the die selector).
   - **The address bar is the share link.** `syncUrl()` writes `pos` (the engine's save JSON,
     zlib, base64url -- `game/position.ts`, interchangeable with Python's zlib), `model`
     (`local:` / `hf:` source, or `off`; a dropped file has no address) and `auto`, with
     `history.replaceState`, never `pushState`, so moves do not pile up in Back. A reload of the
     same link keeps the game's history; *New Game* starts afresh in the page.
   - A generic `.hidden { display: none }` rule backs every mode-specific panel.
9. **Bundled metadata (`metadata.ts`)**:
   - `rules/map.json` and `rules/cards.json` are imported at build time, so the page needs no
     server to draw the board. The flat action layout comes from the engine
     (`WasmEngine.layout`), never from a hand-kept copy.

---

## 4. Development & Build Commands

```bash
tools/scripts/build_web.sh      # from the repo root: the WebAssembly engine, then npm run build
npm install                     # install dependencies
npm run dev                     # Vite dev server, proxying /api to the local server on port 8000
npm run build                   # production bundle into dist/ (needs public/engine/ built first)
```

---

## 5. Verification & Regression Testing

`public/engine/` and `dist/` are not committed, so build them before running the web suite
(`tools/scripts/build_web.sh`). Invoke pytest as a module, never via `.venv/bin/pytest`: that
console script carries an absolute shebang and breaks if the venv is moved. The browser tests need
a Playwright Chromium (`.venv/bin/python -m playwright install chromium`).

```bash
tools/scripts/build_web.sh
PYTHONPATH=.:build/release .venv/bin/python -m pytest -q tests/web
```

| test | holds |
|:---|:---|
| `test_wasm_engine.py` | the WebAssembly engine is the native one: 200 whole games, and display JSON / save JSON / observations / masks at every step of games driven by clicks |
| `test_describe_golden.py` | the TypeScript action log writes what the Python session wrote (1,098 steps); after an engine change refuses a recorded action, re-record with `fixtures/make_describe_golden.py` |
| `test_e2e_board_display.py` | in Chromium: this turn's effects on top of the map (not permanent ones, not Space Race attempts), control as the country's fill (a highlighted target included) |
| `test_position_tokens.py` | a `pos=` token means the same position to the page and to Python's zlib |
| `test_local_server.py` | the local server lists and exports checkpoints, serves replays, and nothing outside its trees |
| `test_e2e_workbench.py` | in Chromium: a game played by the page, the model readout against Python's `read_policy`/`read_critic`, the live critic lighting the side to move, country probabilities clear of the influence, the cards column left of the map (and stacked when narrow), auto-play + undo, links, a dropped `.onnx`, debug overrides, replay export, and the page on a static server with no API (GitHub Pages) |
| `test_e2e_replay_trace.py`, `test_e2e_space_race.py` | replay trace views (the readout lights the next step's player); the header tracks and the Space Race widget |
| `test_value_readings.py` | `trace_view.ts` under node: the decider of each replay position, the calibrated P(US wins), VP ×20, the terminal case, the `ΔP` chip |
| `test_web_workbench.py` | the bundled rules metadata, the page's DOM, replay snapshots |

The TypeScript under test in node (`tests/web/js/*.ts`) is bundled with the esbuild that ships
with Vite (`tests/web/js_runner.py`).
