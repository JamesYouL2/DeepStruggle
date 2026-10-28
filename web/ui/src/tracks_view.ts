import { GameState } from './types';
import { engineLayout } from './engine/layout';
import { escapeHtml, num } from './html';

export interface SpaceBoxMetadata {
  box: number;
  name: string;
  minOps: number;
  maxRoll: number;
  vpFirst: number;
  vpSecond: number;
  ability: string;
  abilityShort: string;
}

/** The boxes' names and privileges, in words; what each box asks and pays is the engine's. */
const SPACE_RACE_TEXT: Array<Pick<SpaceBoxMetadata, "name" | "ability" | "abilityShort">> = [
  { name: "Start / Launchpad", ability: "None", abilityShort: "None" },
  { name: "Earth Satellite", ability: "None", abilityShort: "None" },
  { name: "Animal in Space", ability: "May attempt Space Race twice per turn", abilityShort: "2 Attempts / Turn" },
  { name: "Man in Space", ability: "None", abilityShort: "None" },
  { name: "Man in Earth Orbit", ability: "Opponent must select & reveal headline first", abilityShort: "Opponent Reveals Headline 1st" },
  { name: "Lunar Orbit", ability: "None", abilityShort: "None" },
  { name: "Eagle / Bear Has Landed", ability: "May discard 1 held card at end of turn", abilityShort: "Discard Held Card at Turn End" },
  { name: "Space Shuttle", ability: "None", abilityShort: "None" },
  { name: "Space Station", ability: "May play 8 Action Rounds per turn", abilityShort: "8th Action Round" },
];

/**
 * The track: each box's Ops, roll and VP as the engine plays them (SpaceRace::get_box_info, exported
 * as WasmEngine.layout.space_boxes), with its name and privilege. Empty until the engine has loaded.
 */
export function spaceRaceBoxes(): SpaceBoxMetadata[] {
  const boxes = engineLayout()?.space_boxes ?? [];
  return boxes.map((b, box) => ({
    box, minOps: b.min_ops, maxRoll: b.max_roll, vpFirst: b.vp_first, vpSecond: b.vp_second,
    ...(SPACE_RACE_TEXT[box] ?? { name: `Box ${box}`, ability: "None", abilityShort: "None" }),
  }));
}

/** A box's name, safe for any number a replay's state holds. */
function boxName(box: number): string {
  return escapeHtml(SPACE_RACE_TEXT[box]?.name ?? `Box ${box}`);
}

/** What the next box asks, as the app's "NEXT PROJECT: 2 OPS: Roll 1-3": a card of N Ops, a roll. */
function nextProject(at: number): string {
  const boxes = spaceRaceBoxes();
  if (!boxes.length) return "";
  const next = boxes[at + 1];
  if (!next) return `<span class="pill-next" title="The last box, the Space Station: the space race is over">done</span>`;
  return `<span class="pill-next" title="Box ${next.box}, ${next.name}: a card of ${next.minOps}+ Ops, then a roll of 1-${next.maxRoll}">`
    + `next ${next.minOps} Ops · 1–${next.maxRoll}</span>`;
}

export class TracksView {
  private container: HTMLElement;
  private currentState: GameState | null = null;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  public render(state: GameState) {
    this.currentState = state;
    // Numbers and names, as a replay snapshot may hold anything.
    const vp = num(state.victory_points);
    const defcon = num(state.defcon);
    const turn = num(state.turn);
    const ar = num(state.action_round);
    const phase = num(state.current_phase);
    const phasing = escapeHtml(state.phasing_player);

    const usMilOps = num(state.mil_ops?.US ?? 0);
    const ussrMilOps = num(state.mil_ops?.USSR ?? 0);
    const usSpace = num(state.space?.US ?? 0);
    const ussrSpace = num(state.space?.USSR ?? 0);

    const usTurnsUsed = num(state.space_turns_used?.US ?? 0);
    const ussrTurnsUsed = num(state.space_turns_used?.USSR ?? 0);
    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    // Needle pos in percentage: -20 (USSR) is 0%, 0 is 50%, +20 (US) is 100%
    const needlePercent = Math.max(0, Math.min(100, ((vp + 20) / 40) * 100));

    let vpLabel = 'TIE 0';
    let vpClass = 'tie';
    if (vp > 0) {
      vpLabel = `US +${vp}`;
      vpClass = 'us';
    } else if (vp < 0) {
      vpLabel = `USSR +${Math.abs(vp)}`;
      vpClass = 'ussr';
    }

    const eraName = turn <= 3 ? 'EARLY WAR' : (turn <= 7 ? 'MID WAR' : 'LATE WAR');

    this.container.innerHTML = `
      <!-- DEFCON Track -->
      <div class="track-group">
        <span class="track-label">DEFCON</span>
        <div class="defcon-scale">
          ${[5, 4, 3, 2, 1].map(lvl => `
            <div class="defcon-box ${defcon === lvl ? 'active' : ''}" data-level="${lvl}">
              ${lvl}
            </div>
          `).join('')}
        </div>
      </div>

      <!-- Victory Points Meter -->
      <div class="track-group">
        <span class="track-label">VICTORY POINTS</span>
        <div class="vp-meter-container">
          <div class="vp-bar">
            <div class="vp-fill-ussr" style="width: ${needlePercent}%;"></div>
            <div class="vp-center-line"></div>
            <div class="vp-needle" style="left: ${needlePercent}%;"></div>
          </div>
          <span class="vp-val-badge ${vpClass}">${vpLabel}</span>
        </div>
      </div>

      <!-- Turn & Era -->
      <div class="track-group">
        <span class="track-label">TURN & ERA</span>
        <div class="turn-ar-badge">
          <strong>T${turn}/10</strong>
          <span style="color: var(--text-dim);">•</span>
          <span>${eraName}</span>
        </div>
      </div>

      <!-- Action Round -->
      <div class="track-group">
        <span class="track-label">ACTION ROUND</span>
        <div class="turn-ar-badge">
          <strong>${(phase === 0 || state.current_phase_name === 'SETUP') ? 'SETUP' : ((phase === 1 || state.current_phase_name === 'HEADLINE') ? 'HEADLINE' : ((phase === 3 || state.current_phase_name === 'GAME_OVER') ? 'GAME OVER' : `AR ${ar}`))}</strong>
        </div>
      </div>

      <!-- Military Ops Track -->
      <div class="track-group">
        <span class="track-label">MILITARY OPS (REQ: ${defcon})</span>
        <div class="superpower-track">
          <span style="color: var(--ussr-red-light); font-weight: bold;">USSR:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= ussrMilOps ? 'filled-ussr' : ''}"></div>`).join('')}
          </div>
          <span style="color: var(--text-dim); margin: 0 4px;">|</span>
          <span style="color: var(--us-blue-light); font-weight: bold;">US:</span>
          <div class="milops-pips">
            ${[1, 2, 3, 4, 5].map(i => `<div class="pip ${i <= usMilOps ? 'filled-us' : ''}"></div>`).join('')}
          </div>
        </div>
      </div>

      <!-- Space Race Visual Track Widget -->
      <div class="track-group space-race-track-group">
        <span class="track-label">SPACE RACE</span>
        <div class="space-race-header-widget" id="space-race-widget" role="button" tabindex="0" title="Space Race: USSR at Box #${ussrSpace}, US at Box #${usSpace}. Click to open full Deluxe Space Track & Perks.">
          <!-- USSR Pill -->
          <div class="space-pill ussr" title="USSR Space: Box #${ussrSpace} (${boxName(ussrSpace)}) | Attempts Used: ${ussrTurnsUsed}/${maxUssrAttempts}">
            <span class="pill-dot ussr"></span>
            <span class="pill-side">USSR:</span>
            <span class="pill-box-num">#${ussrSpace}</span>
            <span class="pill-attempts">(${ussrTurnsUsed}/${maxUssrAttempts})</span>
            ${nextProject(ussrSpace)}
          </div>

          <!-- Mini Progress Ladder (Boxes 0 to 8) -->
          <div class="space-mini-ladder">
            ${[0, 1, 2, 3, 4, 5, 6, 7, 8].map(step => {
              const hasUs = usSpace === step;
              const hasUssr = ussrSpace === step;
              let occ = '';
              if (hasUs && hasUssr) occ = 'both';
              else if (hasUs) occ = 'us';
              else if (hasUssr) occ = 'ussr';

              return `
                <div class="mini-box ${occ ? 'active ' + occ : ''}" data-step="${step}" title="Box ${step}: ${boxName(step)}">
                  <span class="mini-box-num">${step}</span>
                  <div class="mini-box-tokens">
                    ${hasUssr ? '<div class="mini-token ussr" title="USSR at Box ' + step + '"></div>' : ''}
                    ${hasUs ? '<div class="mini-token us" title="US at Box ' + step + '"></div>' : ''}
                  </div>
                </div>
              `;
            }).join('')}
          </div>

          <!-- US Pill -->
          <div class="space-pill us" title="US Space: Box #${usSpace} (${boxName(usSpace)}) | Attempts Used: ${usTurnsUsed}/${maxUsAttempts}">
            <span class="pill-dot us"></span>
            <span class="pill-side">US:</span>
            <span class="pill-box-num">#${usSpace}</span>
            <span class="pill-attempts">(${usTurnsUsed}/${maxUsAttempts})</span>
            ${nextProject(usSpace)}
          </div>

          <!-- Expand Icon Button -->
          <span class="space-view-btn" title="View Space Race Details">
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
              <polyline points="15 3 21 3 21 9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
            </svg>
          </span>
        </div>
      </div>
    `;

    // Hook up click & keyboard access to view detailed space race modal
    const widget = document.getElementById('space-race-widget');
    if (widget) {
      widget.addEventListener('click', () => {
        if (this.currentState) this.showSpaceRaceModal(this.currentState);
      });
      widget.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          if (this.currentState) this.showSpaceRaceModal(this.currentState);
        }
      });
    }

    // Update active player badge in top-right
    const activeBadge = document.getElementById('active-player-badge');
    if (activeBadge) {
      const activePlayer = state.decision_context?.decision_player || phasing || 'NONE';
      activeBadge.textContent = `${activePlayer} TURN`;
      activeBadge.className = `player-badge ${activePlayer.toLowerCase()}`;
    }
  }

  public showSpaceRaceModal(state: GameState) {
    const modalContainer = document.getElementById('modal-container');
    const modalContent = modalContainer?.querySelector('.modal-content');
    const modalTitle = document.getElementById('modal-title');
    const modalBody = document.getElementById('modal-body');
    if (!modalContainer || !modalTitle || !modalBody) return;

    modalTitle.innerHTML = `🚀 <span>DELUXE SPACE RACE TRACK & PERKS</span>`;
    modalContent?.classList.add('modal-lg');

    const usSpace = num(state.space?.US ?? 0);
    const ussrSpace = num(state.space?.USSR ?? 0);
    const usTurnsUsed = num(state.space_turns_used?.US ?? 0);
    const ussrTurnsUsed = num(state.space_turns_used?.USSR ?? 0);

    const maxUsAttempts = (usSpace >= 2 && ussrSpace < 2) ? 2 : 1;
    const maxUssrAttempts = (ussrSpace >= 2 && usSpace < 2) ? 2 : 1;

    let leadText = "TIED ON SPACE RACE TRACK";
    let leadClass = "tie";
    if (usSpace > ussrSpace) {
      leadText = `🔵 US LEADS BY ${usSpace - ussrSpace} SPACES`;
      leadClass = "us";
    } else if (ussrSpace > usSpace) {
      leadText = `🔴 USSR LEADS BY ${ussrSpace - usSpace} SPACES`;
      leadClass = "ussr";
    }

    modalBody.innerHTML = `
      <div class="space-modal-container">
        <!-- Summary Header Dashboard -->
        <div class="space-modal-summary-header">
          <!-- USSR Card -->
          <div class="space-summary-card ussr-border">
            <div class="card-header-row">
              <h4 style="color: var(--ussr-red-light); margin: 0;">🔴 USSR Space Program</h4>
              <span class="badge ${ussrTurnsUsed >= maxUssrAttempts ? 'badge-danger' : 'badge-success'}">
                ${ussrTurnsUsed >= maxUssrAttempts ? 'Max Attempts Used' : 'Attempt Available'}
              </span>
            </div>
            <div class="summary-stat-row">
              <span>Position:</span>
              <strong>Box #${ussrSpace} — ${boxName(ussrSpace)}</strong>
            </div>
            <div class="summary-stat-row">
              <span>Attempts this Turn:</span>
              <strong>${ussrTurnsUsed} / ${maxUssrAttempts} max</strong>
            </div>
            <div class="summary-stat-row">
              <span>Active Superpower Perks:</span>
              <em class="perk-highlight">${this.getActivePerkSummary(ussrSpace, usSpace)}</em>
            </div>
          </div>

          <!-- Center Lead Status -->
          <div class="space-lead-indicator">
            <span class="lead-badge ${leadClass}">${leadText}</span>
            <div class="lead-subtext">Play card Ops ≥ Box requirement to attempt advance</div>
          </div>

          <!-- US Card -->
          <div class="space-summary-card us-border">
            <div class="card-header-row">
              <h4 style="color: var(--us-blue-light); margin: 0;">🔵 US Space Program</h4>
              <span class="badge ${usTurnsUsed >= maxUsAttempts ? 'badge-danger' : 'badge-success'}">
                ${usTurnsUsed >= maxUsAttempts ? 'Max Attempts Used' : 'Attempt Available'}
              </span>
            </div>
            <div class="summary-stat-row">
              <span>Position:</span>
              <strong>Box #${usSpace} — ${boxName(usSpace)}</strong>
            </div>
            <div class="summary-stat-row">
              <span>Attempts this Turn:</span>
              <strong>${usTurnsUsed} / ${maxUsAttempts} max</strong>
            </div>
            <div class="summary-stat-row">
              <span>Active Superpower Perks:</span>
              <em class="perk-highlight">${this.getActivePerkSummary(usSpace, ussrSpace)}</em>
            </div>
          </div>
        </div>

        <!-- 9 Milestone Boxes Grid -->
        <div class="space-ladder-grid">
          ${spaceRaceBoxes().map(box => {
            const isUsHere = usSpace === box.box;
            const isUssrHere = ussrSpace === box.box;
            let occupancyClass = "";
            if (isUsHere && isUssrHere) occupancyClass = "occupied-both";
            else if (isUsHere) occupancyClass = "occupied-us";
            else if (isUssrHere) occupancyClass = "occupied-ussr";

            const isPassedByUs = usSpace > box.box && box.box > 0;
            const isPassedByUssr = ussrSpace > box.box && box.box > 0;

            const isUsActivePerk = (box.box === 2 && usSpace >= 2 && ussrSpace < 2) ||
                                  (box.box === 4 && usSpace >= 4 && ussrSpace < 4) ||
                                  (box.box === 6 && usSpace >= 6 && ussrSpace < 6) ||
                                  (box.box === 8 && usSpace >= 8 && ussrSpace < 8);

            const isUssrActivePerk = (box.box === 2 && ussrSpace >= 2 && usSpace < 2) ||
                                    (box.box === 4 && ussrSpace >= 4 && usSpace < 4) ||
                                    (box.box === 6 && ussrSpace >= 6 && usSpace < 6) ||
                                    (box.box === 8 && ussrSpace >= 8 && usSpace < 8);

            return `
              <div class="space-box-card ${occupancyClass}" data-box="${box.box}">
                <div class="space-box-header">
                  <div class="box-title-group">
                    <span class="space-box-num">BOX #${box.box}</span>
                    <span class="space-box-name">${box.name}</span>
                  </div>
                  <div class="space-box-tokens">
                    ${isUssrHere ? '<span class="space-player-token ussr">🔴 USSR</span>' : ''}
                    ${isUsHere ? '<span class="space-player-token us">🔵 US</span>' : ''}
                  </div>
                </div>

                <div class="space-box-details">
                  <div class="detail-item">
                    <span class="detail-label">Required Ops:</span>
                    <strong class="detail-val">${box.minOps > 0 ? `≥ ${box.minOps} Ops` : 'None'}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">Roll to Advance:</span>
                    <strong class="detail-val">${box.maxRoll > 0 ? `🎲 1–${box.maxRoll}` : 'None'}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">1st Player VP:</span>
                    <strong class="detail-val vp-first">${box.vpFirst > 0 ? `+${box.vpFirst} VP` : '—'}</strong>
                  </div>
                  <div class="detail-item">
                    <span class="detail-label">2nd Player VP:</span>
                    <strong class="detail-val vp-second">${box.vpSecond > 0 ? `+${box.vpSecond} VP` : '—'}</strong>
                  </div>
                </div>

                ${box.ability !== 'None' ? `
                  <div class="space-box-ability ${(isUsActivePerk || isUssrActivePerk) ? 'active-privilege' : ''}">
                    <div class="ability-title">✨ Privilege (1st Player Only):</div>
                    <div class="ability-desc">${box.ability}</div>
                    ${isUssrActivePerk ? '<div class="perk-holder ussr">🔴 ACTIVE FOR USSR</div>' : ''}
                    ${isUsActivePerk ? '<div class="perk-holder us">🔵 ACTIVE FOR US</div>' : ''}
                    ${(!isUsActivePerk && !isUssrActivePerk && usSpace >= box.box && ussrSpace >= box.box && box.box > 0) ? '<div class="perk-holder cancelled">✕ CANCELLED (BOTH REACHED)</div>' : ''}
                  </div>
                ` : ''}

                <div class="space-box-footer">
                  <span class="${isUssrHere ? 'status-current ussr' : (isPassedByUssr ? 'status-passed' : 'status-pending')}">
                    USSR: ${isUssrHere ? '📍 HERE' : (isPassedByUssr ? '✓ Cleared' : 'Pending')}
                  </span>
                  <span class="${isUsHere ? 'status-current us' : (isPassedByUs ? 'status-passed' : 'status-pending')}">
                    US: ${isUsHere ? '📍 HERE' : (isPassedByUs ? '✓ Cleared' : 'Pending')}
                  </span>
                </div>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    modalContainer.classList.remove('hidden');
  }

  private getActivePerkSummary(myStep: number, oppStep: number): string {
    const perks: string[] = [];
    if (myStep >= 2 && oppStep < 2) perks.push("2 attempts/turn");
    if (myStep >= 4 && oppStep < 4) perks.push("Opponent reveals headline first");
    if (myStep >= 6 && oppStep < 6) perks.push("Discard held card at turn end");
    if (myStep >= 8 && oppStep < 8) perks.push("8th Action Round");
    return perks.length > 0 ? perks.join(", ") : "None";
  }
}
