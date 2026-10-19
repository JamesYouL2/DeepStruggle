/**
 * The action log: what an applied action did, in words, and what it changed.
 *
 * A line-for-line port of the Python session's `describe_action_and_deltas` (web/server/session.py
 * before the workbench moved into the browser), pinned by tests/web/test_describe_golden.py
 * against lines that function produced. Names come from the engine, not from rules/*.json, so
 * the text is the engine's own.
 *
 * Iteration order matters for a multi-line diff: countries are walked by id and cards 1..110, as
 * the Python dicts were built; the JSON the page reads has its keys sorted by name instead.
 */
import { DieRollState, EventTrackState, GameState, MicroAction } from "../types";
import { CARDS_METADATA } from "../metadata";
import { branchMeta } from "./branches";
import { WasmEngine } from "../engine/wasm_engine";

const DT = { SELECT_CARD: 1, SELECT_PLAY_MODE: 2, SELECT_OP_MODE: 4, POINT_NODE: 5, CHOOSE_BRANCH: 6 };
/** Grain Sales to Soviets: its node offers the drawn card back (a CONFIRM_DONE) for its own 2 Ops. */
const GRAIN_SALES = 67;
/** The cards whose card choices the log words by name (see eventCardChoice), and Space Walk's id. */
const STAR_WARS = 85, SALT = 43, UN_INTERVENTION = 32, MISSILE_ENVY = 49, FIVE_YEAR_PLAN = 5, SPACE_WALK = 250;
const TERRORISM = 92;
const SIDE = new Map(CARDS_METADATA.map(c => [c.id, c.side]));

function isConfirmDone(a: Required<MicroAction>): boolean {
  return (a.flags & 0x80) !== 0 || a.primary_id === 255;
}

function signed(v: number): string {
  return v > 0 ? `+${v}` : `${v}`;
}

function vpStr(vp: number): string {
  return vp > 0 ? `+${vp} (US)` : vp < 0 ? `${vp} (USSR)` : "0 (Tie)";
}

function countriesById(s: GameState) {
  return Object.values(s.countries || {}).sort((a, b) => a.id - b.id);
}

export function describeActionAndDeltas(
  engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
): string[] {
  const d = action.decision_type;
  const primary = action.primary_id;
  const secondary = action.secondary_id;
  const flags = action.flags;
  const ctx = before.decision_context;
  const p = ctx?.decision_player ?? "NONE";
  const logs: string[] = [];

  // 1. What was done
  if (d === DT.SELECT_CARD) {
    const resolving = ctx?.resolving_card ?? 0;
    const cardName = primary >= 1 && primary <= 110 ? engine.cardName(primary) : `#${primary}`;
    const phase = before.current_phase_name ?? "";
    if (resolving !== 0 && resolving !== SPACE_WALK) {
      logs.push(eventCardChoice(engine, before, after, action, p, resolving));
    } else if (resolving === SPACE_WALK) {
      if (isConfirmDone(action) || primary === 0) {
        logs.push(`${p} passes Eagle / Bear Has Landed (Box 6) turn-end discard opportunity.`);
      } else {
        logs.push(`${p} discards ${cardName} (#${primary}) via Eagle / Bear Has Landed (Box 6 privilege).`);
      }
    } else if (phase === "HEADLINE") {
      const usBefore = before.headline_us_card ?? 0;
      const ussrBefore = before.headline_ussr_card ?? 0;
      const isSecond = (p === "US" && ussrBefore > 0) || (p === "USSR" && usBefore > 0);
      logs.push(`${p} commits Headline Card: ${cardName} (#${primary})`);
      if (isSecond) {
        const usCard = p === "US" ? primary : usBefore;
        const ussrCard = p === "USSR" ? primary : ussrBefore;
        const usName = engine.cardName(usCard);
        const ussrName = engine.cardName(ussrCard);
        const usOps = engine.cardOps(usCard);
        const ussrOps = engine.cardOps(ussrCard);
        const firstP = usOps >= ussrOps ? "US" : "USSR";
        const secondP = firstP === "US" ? "USSR" : "US";
        const firstName = firstP === "US" ? usName : ussrName;
        const secondName = firstP === "US" ? ussrName : usName;
        logs.push(`  ★ Headlines Simultaneously Revealed: US plays '${usName}' (#${usCard}, ${usOps} Ops) vs USSR plays '${ussrName}' (#${ussrCard}, ${ussrOps} Ops)`);
        logs.push(`  ★ Resolution Order: 1st ${firstP} '${firstName}' (${Math.max(usOps, ussrOps)} Ops) -> 2nd ${secondP} '${secondName}' (${Math.min(usOps, ussrOps)} Ops)`);
      }
    } else {
      logs.push(`${p} plays Card: ${cardName} (#${primary})`);
    }
  } else if (d === DT.POINT_NODE) {
    if (isConfirmDone(action) || primary === 255 || (primary === 0 && (flags & 128))) {
      logs.push(`${p} confirms / finishes point node selections.`);
    } else {
      const cName = primary < 84 ? engine.countryName(primary) : `Node #${primary}`;
      const resolving = ctx?.resolving_card ?? 0;
      const phase = before.current_phase_name ?? "";
      if (phase === "SETUP") {
        logs.push(`${p} places 1 Influence in ${cName} during Setup.`);
      } else if (resolving > 0) {
        logs.push(`${p} targets ${cName} for Event: ${engine.cardName(resolving)} (#${resolving})`);
      } else {
        const opMode = (ctx as { op_mode?: number } | undefined)?.op_mode ?? 0;
        if (opMode === 1) logs.push(`${p} attempts Coup in ${cName}`);
        else if (opMode === 2) logs.push(`${p} conducts Realignment in ${cName}`);
        else logs.push(`${p} places Influence in ${cName}`);
      }
    }
  } else if (d === DT.SELECT_PLAY_MODE) {
    // P17: one resolution node. On an opponent card EVENT reads as event-first and any OPS_* as
    // ops-first, which is what the retired CHOOSE_TIMING_BRANCH used to say.
    const modes: Record<number, string> = {
      0: "EVENT", 1: "SPACE RACE", 2: "OPERATIONS - INFLUENCE PLACEMENT",
      3: "OPERATIONS - COUP ATTEMPT", 4: "OPERATIONS - REALIGNMENT",
    };
    if (isConfirmDone(action) && (ctx?.resolving_card ?? 0) === GRAIN_SALES) {
      // The decline at Grain Sales' node: the drawn card goes back, Grain Sales' own Ops are taken.
      const drawn = ctx?.pending_op_card ?? 0;
      logs.push(`${p} returns ${engine.cardName(drawn)} (#${drawn}) to the USSR hand and conducts Grain Sales' 2 Ops`);
    } else if (primary === 1) {
      const rollInfo = secondary > 0 ? ` (Input Roll: ${secondary})` : "";
      logs.push(`${p} attempts Space Race with pending card${rollInfo}`);
    } else {
      logs.push(`${p} selects play mode: ${modes[primary] ?? String(primary)}`);
    }
  } else if (d === DT.SELECT_OP_MODE) {
    const opModes: Record<number, string> = { 0: "INFLUENCE PLACEMENT", 1: "COUP ATTEMPT", 2: "REALIGNMENT" };
    logs.push(`${p} chooses Op mode: ${opModes[primary] ?? String(primary)}`);
  } else if (d === DT.CHOOSE_BRANCH) {
    const card = ctx?.resolving_card ?? 0;
    const meta = isConfirmDone(action) ? null : branchMeta(card, primary, before);
    if (isConfirmDone(action)) logs.push(`${p} confirms / passes option.`);
    else if (meta) logs.push(`${p} ${meta.verb} (${engine.cardName(card)})`);
    else logs.push(`${p} chooses option: ${primary}`);
  } else {
    logs.push(`${p} action: type=${d} primary=${primary} secondary=${secondary} flags=${flags}`);
  }

  logs.push(...describeDeltas(engine, before, after, p, d === DT.SELECT_CARD ? primary : 0));
  return logs;
}

/**
 * What changed from `before` to `after`: influence, the tracks, effects, cards that moved, and the
 * die roll `after` records. The deltas half of `describeActionAndDeltas`; on its own it is how the
 * session logs the changes a step made after the action itself (the turn's end).
 */
export function describeDeltas(engine: WasmEngine, before: GameState, after: GameState, p: string,
                               chosen = 0): string[] {
  const logs: string[] = [];
  // 2. Influence
  const oldCountries = before.countries || {};
  for (const c of countriesById(after)) {
    const old = oldCountries[c.name];
    if (!old) continue;
    const usDiff = c.us_influence - old.us_influence;
    const ussrDiff = c.ussr_influence - old.ussr_influence;
    if (usDiff !== 0 || ussrDiff !== 0) {
      const parts: string[] = [];
      if (usDiff !== 0) parts.push(`US: ${old.us_influence} -> ${c.us_influence} (${signed(usDiff)})`);
      if (ussrDiff !== 0) parts.push(`USSR: ${old.ussr_influence} -> ${c.ussr_influence} (${signed(ussrDiff)})`);
      logs.push(`  • ${c.name} Influence: ` + parts.join(", "));
    }
  }

  // 3. Tracks
  if (before.defcon !== after.defcon) logs.push(`  • DEFCON: ${before.defcon} -> ${after.defcon}`);
  if (before.victory_points !== after.victory_points) {
    const vb = before.victory_points ?? 0;
    const va = after.victory_points ?? 0;
    const delta = va - vb;
    logs.push(`  • Victory Points: ${vpStr(vb)} -> ${vpStr(va)} (${delta > 0 ? `+${delta} VP` : `${delta} VP`})`);
  }
  // The Python session read `us_mil_ops`/`ussr_mil_ops`, keys the display state never had (it
  // carries `mil_ops: {US, USSR}`), so a MilOps change was never logged. Fixed here; the golden
  // test drops these lines when comparing with that function's output.
  for (const side of ["US", "USSR"] as const) {
    const b = before.mil_ops?.[side];
    const a = after.mil_ops?.[side];
    if (b !== a) logs.push(`  • ${side} MilOps: ${b} -> ${a}`);
  }

  // 3.5 Effect flags
  const oldFlags = new Set(before.flags || []);
  const newFlags = new Set(after.flags || []);
  for (const f of [...newFlags].filter(x => !oldFlags.has(x)).sort()) logs.push(`  • Effect Activated: ${f}`);
  for (const f of [...oldFlags].filter(x => !newFlags.has(x)).sort()) logs.push(`  • Effect Cancelled: ${f}`);

  // 4. Cards that moved
  const oldLocs = before.card_locations || {};
  const newLocs = after.card_locations || {};
  for (let cid = 1; cid <= 110; cid++) {
    const o = oldLocs[String(cid)];
    const n = newLocs[String(cid)];
    if (n === undefined) continue;
    if (o && o !== n) logs.push(`  • Card #${cid} (${engine.cardName(cid)}) moved: ${o} -> ${n}`);
  }

  // 4.5 The random discards events make, said as discards; Missile Envy's exchange, as one
  logs.push(...randomDiscards(engine, before, after, chosen));
  logs.push(...missileEnvyExchange(engine, before, after));

  // 5. The die roll, if one happened
  logs.push(...describeRoll(engine, before, after.die_roll, p));
  return logs;
}

/**
 * The dice line for one roll, as the display state writes a roll (`die_roll`); `before` is the
 * position the roll was made from (a coup's defence reads its stability). Nothing for no roll.
 */
/**
 * What each Event of a step did to VP and DEFCON, when the step's before/after cannot say it. The
 * second headline commit resolves both headlines: Nuclear Test Ban's +1 VP for the US and Pershing
 * II's +1 VP for the USSR net to no change, and the Tracks lines then say nothing at all. A lone
 * Event that accounts for the whole change is already told by those lines, so it adds nothing.
 */
export function describeEventTracks(before: GameState, after: GameState,
                                    tracks: EventTrackState[] | undefined): string[] {
  if (!tracks || tracks.length === 0) return [];
  const vp = tracks.reduce((sum, t) => sum + t.vp_delta, 0);
  const defcon = tracks.reduce((sum, t) => sum + t.defcon_delta, 0);
  const netVp = (after.victory_points ?? 0) - (before.victory_points ?? 0);
  const netDefcon = (after.defcon ?? 0) - (before.defcon ?? 0);
  if (tracks.length === 1 && vp === netVp && defcon === netDefcon) return [];
  return tracks.map(t => {
    const parts: string[] = [];
    if (t.vp_delta !== 0) parts.push(`${t.vp_delta > 0 ? "US" : "USSR"} +${Math.abs(t.vp_delta)} VP`);
    if (t.defcon_delta !== 0) parts.push(`DEFCON ${signed(t.defcon_delta)}`);
    return `  • ${t.card_name || `Card #${t.card_id}`} (#${t.card_id}): ${parts.join(", ")}`;
  });
}

export function describeRoll(engine: WasmEngine, before: GameState, roll: DieRollState | undefined, p: string): string[] {
  const logs: string[] = [];
  const r = roll || ({} as DieRollState);
  const type = r.type ?? "NONE";
  const cname = () => r.country_name || ((r.country_id ?? 255) < 84 ? engine.countryName(r.country_id ?? 0) : "");
  const sgn = (v: number) => (v >= 0 ? "+" : "");
  if (type === "COUP") {
    const c = cname();
    const stab = before.countries?.[c]?.stability ?? 1;
    const roller = r.roller || p;
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const def = stab * 2;
    const net = r.net_delta ?? 0;
    const cardPrefix = r.card_name ? ` (${r.card_name})` : "";
    const status = r.success ? `Net +${net} Influence (Coup Succeeded)` : "Coup Failed (Roll + Ops <= 2x Stability)";
    logs.push(`  🎲 Coup in ${c}${cardPrefix}: ${roller} rolls ${roll1} (+${mod1} Ops = ${tot1}) vs ${def} Defense (2x Stability ${stab}) -> ${status}`);
  } else if (type === "REALIGNMENT") {
    const c = cname();
    const rus = r.roll1 ?? 0, mus = r.mod1 ?? 0, tus = r.total1 ?? rus + mus;
    const rr = r.roll2 ?? 0, mr = r.mod2 ?? 0, tr = r.total2 ?? rr + mr;
    const net = r.net_delta ?? 0;
    const res = tus > tr ? `US wins by +${tus - tr} (Removes ${net} USSR Influence)`
      : tr > tus ? `USSR wins by +${tr - tus} (Removes ${net} US Influence)` : "Tie (No Influence Removed)";
    logs.push(`  🎲 Realignment in ${c}: US rolls ${rus} (${sgn(mus)}${mus} mod = ${tus}), USSR rolls ${rr} (${sgn(mr)}${mr} mod = ${tr}) -> ${res}`);
  } else if (type === "SPACE_RACE") {
    const target = r.country_id ?? 0;
    const roll1 = r.roll1 ?? 0;
    const status = r.success ? `SUCCESS (Advanced to Box #${target})` : `FAILED (Roll ${roll1} > ${r.mod1 ?? 0} threshold)`;
    logs.push(`  🎲 Space Race Attempt (Target Box #${target}): ${r.roller || p} rolls ${roll1} -> ${status}`);
  } else if (type === "WAR_EVENT") {
    const card = r.card_name || `Card #${r.card_id}`;
    const c = cname();
    const roll1 = r.roll1 ?? 0, mod1 = r.mod1 ?? 0, tot1 = r.total1 ?? roll1 + mod1;
    const threshold = r.mod2 ?? 0;
    const target = c ? ` targeting ${c}` : "";
    const status = r.success ? `Success (${tot1} >= ${threshold} target: Target Captured & VP Awarded)` : `Roll Failed (${tot1} < ${threshold} target)`;
    logs.push(`  🎲 ${card} Roll${target}: ${r.roller || p} rolls ${roll1} (${sgn(mod1)}${mod1} mod = ${tot1}) -> ${status}`);
  } else if (type === "OLYMPIC_GAMES") {
    const sponsor = r.roller || p;
    // The engine records the sponsor's roll with its +2 already in (roll1 = die + 2, mod1 = 2), so
    // roll1 is the sponsor's total and total1 counts the +2 twice.
    const st = r.roll1 ?? 0, sr = st - (r.mod1 ?? 2);
    const or = r.roll2 ?? 0, ot = r.total2 ?? or;
    const opp = sponsor === "USSR" ? "US" : "USSR";
    const res = st > ot ? `Sponsor ${sponsor} wins by +${st - ot} (+2 VP)` : ot > st ? `Opponent ${opp} wins` : "Tie (No VP)";
    logs.push(`  🎲 Olympic Games Competition: Sponsor (${sponsor}) rolls ${sr} (+2 = ${st}), Opponent (${opp}) rolls ${or} -> ${res}`);
  } else if (type === "SUMMIT") {
    const ur = r.roll1 ?? 0, ud = r.mod1 ?? 0, ut = r.total1 ?? ur + ud;
    const sr = r.roll2 ?? 0, sd = r.mod2 ?? 0, stt = r.total2 ?? sr + sd;
    const diff = Math.abs(ut - stt);
    const res = ut > stt ? `US wins Summit by +${diff}` : stt > ut ? `USSR wins Summit by +${diff}` : "Tied Summit (No Effect)";
    logs.push(`  🎲 Summit Rolls: US rolls ${ur} (+${ud} Dom = ${ut}), USSR rolls ${sr} (+${sd} Dom = ${stt}) -> ${res}`);
  } else if (type === "TRAP_ESCAPE") {
    const status = r.success ? "Success! Discard canceled trap." : "Failed (Roll > 4, remains trapped)";
    logs.push(`  🎲 Trap Escape Roll: ${r.roller || p} rolls ${r.roll1 ?? 0} (Needed 1–4) -> ${status}`);
  }
  return logs;
}

/** Where a card was, for "from the US hand" and the like ("" where it says nothing). */
function from(loc: string | undefined): string {
  if (!loc) return "";
  if (loc.startsWith("HAND_US_")) return " from the US hand";
  if (loc.startsWith("HAND_USSR_")) return " from the USSR hand";
  if (loc === "DISCARD_PILE") return " from the discard pile";
  if (loc === "PEEKED_TEMP") return " from the cards drawn";
  return "";
}

/**
 * A card chosen while an event resolves (`resolving_card`), said as what happened to it -- never as
 * a play or a headline commit, which is what these choices were logged as: the card the USSR has the
 * US discard with Aldrich Ames read as the USSR's headline, with a "Headlines Simultaneously
 * Revealed" line for it.
 */
function eventCardChoice(engine: WasmEngine, before: GameState, after: GameState, action: Required<MicroAction>,
                         p: string, resolving: number): string {
  const event = engine.cardName(resolving);
  const id = action.primary_id;
  if (isConfirmDone(action) || id === 0) return `${p} chooses no (further) card (${event})`;
  const card = `${engine.cardName(id)} (#${id})`;
  const opp = p === "US" ? "USSR" : "US";
  const src = before.card_locations?.[String(id)];
  const dst = after.card_locations?.[String(id)];
  if (resolving === STAR_WARS) return `${p} takes ${card} from the discard pile and plays its event (${event})`;
  if (resolving === SALT) return `${p} takes ${card} back into hand from the discard pile (${event})`;
  if (resolving === UN_INTERVENTION) return `${p} plays ${card} with ${event}: its event is cancelled, ${p} uses its Ops`;
  if (resolving === MISSILE_ENVY) return `${p} gives ${card} to the ${opp} (${event})`;
  if (resolving === GRAIN_SALES && id === UN_INTERVENTION) {
    return `${p} plays UN Intervention on the card ${event} drew: its event is cancelled, ${p} uses its Ops`;
  }
  const verb = dst === "DISCARD_PILE" ? "discards" : dst === "REMOVED_FROM_GAME" ? "removes"
    : dst === "DRAW_DECK" ? "returns to the draw deck" : dst?.startsWith(`HAND_${p}_`) ? "takes"
    : dst?.startsWith(`HAND_${opp}_`) ? `gives the ${opp}` : "chooses";
  return `${p} ${verb} ${card}${from(src)} (${event})`;
}

/**
 * The events that discard at random -- Five Year Plan (the USSR discards one; a US event discarded
 * so fires) and Terrorism (the player's opponent discards one, the US two once Iranian Hostage
 * Crisis is in play) -- make no choice, so their discards were only card-move lines. Each gets its
 * own line here: "✂ USSR discards Decolonization (#30) at random (Five Year Plan)". Read where the
 * event resolves in this step: played for the event (its play mode), or leaving play here (a
 * headline, or an opponent's card played for Ops); the step's own played and headline cards leave
 * a hand too and are not the event's discards.
 */
function randomDiscards(engine: WasmEngine, before: GameState, after: GameState, chosen: number): string[] {
  const lines: string[] = [];
  const c = before.decision_context;
  const ownPlay = [chosen, c?.pending_op_card, before.headline_us_card, before.headline_ussr_card,
                   after.headline_us_card, after.headline_ussr_card];
  const events: Array<[number, string[]]> = [[FIVE_YEAR_PLAN, ["USSR"]], [TERRORISM, ["US", "USSR"]]];
  for (const [event, victims] of events) {
    const played = c?.decision_type_name === "SELECT_PLAY_MODE" && c.pending_op_card === event;
    const leaves = before.card_locations?.[String(event)] !== after.card_locations?.[String(event)];
    if (!played && !leaves) continue;
    for (let cid = 1; cid <= 110; cid++) {
      if (cid === event || ownPlay.includes(cid)) continue;
      const o = before.card_locations?.[String(cid)] ?? "";
      const n = after.card_locations?.[String(cid)] ?? "";
      const side = victims.find(v => o.startsWith(`HAND_${v}_`));
      if (!side || (n !== "DISCARD_PILE" && n !== "REMOVED_FROM_GAME")) continue;
      const fires = event === FIVE_YEAR_PLAN && SIDE.get(cid) === "us" ? " — a US event, so its event fires" : "";
      lines.push(`  ✂ ${side} discards ${engine.cardName(cid)} (#${cid}) at random (${engine.cardName(event)})${fires}`);
    }
  }
  return lines;
}

/**
 * Missile Envy's exchange: the player takes the opponent's highest-Ops card and hands Missile Envy
 * over. The taken card's event fires if it is the taker's own or both sides' event; an opponent's
 * event is played for its Ops instead, without the event. The giver must play Missile Envy for Ops in
 * its next action round. With one card of the highest Ops the exchange is no choice, so it was only
 * card-move lines (the taken card often straight to the discard pile, its event already played).
 * The engine marks the exchange: Missile Envy becomes the giver's forced card (`forced_card_id`,
 * `forced_card_player`) -- not merely Missile Envy reaching a hand, which a deal does too.
 */
function missileEnvyExchange(engine: WasmEngine, before: GameState, after: GameState): string[] {
  const forced = after.forced_card_id === MISSILE_ENVY && after.forced_card_player !== "NONE";
  const already = before.forced_card_id === MISSILE_ENVY && before.forced_card_player === after.forced_card_player;
  if (!forced || already) return [];
  const giver = after.forced_card_player, taker = giver === "US" ? "USSR" : "US";
  const headlines = [before.headline_us_card, before.headline_ussr_card, after.headline_us_card, after.headline_ussr_card];
  const lines: string[] = [];
  for (let cid = 1; cid <= 110; cid++) {
    if (cid === MISSILE_ENVY || headlines.includes(cid)) continue;
    const was = before.card_locations?.[String(cid)] ?? "";
    const now = after.card_locations?.[String(cid)] ?? "";
    if (!was.startsWith(`HAND_${giver}_`) || now.startsWith(`HAND_${giver}_`)) continue;
    const side = SIDE.get(cid);
    const how = side === giver.toLowerCase()
      ? `a ${giver} event, so the ${taker} uses its ${engine.cardOps(cid)} Ops and the event does not fire`
      : `${side === "neutral" ? "an event for both sides" : `a ${taker} event`}, so its event fires for the ${taker}`;
    lines.push(`  ⇄ The ${taker} takes ${engine.cardName(cid)} (#${cid}) from the ${giver} hand (Missile Envy) — ${how}`);
  }
  lines.push(`  ⇄ Missile Envy goes to the ${giver} hand: the ${giver} must play it for Ops in its next action round`);
  return lines;
}
