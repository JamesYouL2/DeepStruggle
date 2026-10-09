/**
 * The Elo leaderboard page (leaderboard.html). Its data, leaderboard.json, is fitted from the
 * records in leaderboard/ by `tools/leaderboard.py fit` when the page is built
 * (tools/scripts/build_web.sh), so the page computes nothing: a view -- the main players, or the
 * main players plus some lineages -- is a filter over that one fit, and a rating is the same
 * number in every view. The view is kept in the link, so a filtered table can be shared.
 */

interface Seat { w: number; l: number; d: number }

interface Network {
  file: string;
  sha256: string;
  description: string;
  old_names: string[];
  /** The .pt's path in the Hugging Face repo once published; its .onnx sits beside it. */
  hf: string | null;
  /** The behaviour report, a path in the GitHub repository. */
  report: string | null;
}

interface Player {
  network: string | null;
  lineage: string | null;
  kind: string;
  inference: Record<string, unknown>;
  description: string;
}

interface Rating {
  player: string;
  elo: number;
  se: number;
  main: boolean;
  games: number;
  score: number;
  opponents: number;
  main_opponents: number;
}

interface Pair { a: string; b: string; a_as_us: Seat; a_as_ussr: Seat; records: string[] }

interface Epoch {
  description: string;
  engines: string[];
  anchor: string;
  anchor_elo: number;
  main: string[];
  seat_us: number;
  seat_us_se: number;
  unrated: string[];
  ratings: Rating[];
  pairs: Pair[];
}

interface LeaderboardData {
  schema: number;
  commit: string | null;
  hf_repo: string;
  github: string;
  networks: Record<string, Network>;
  players: Record<string, Player>;
  epochs: Record<string, Epoch>;
}

type View = "main" | "lineage" | "all";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

function esc(s: string): string {
  return s.replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

const KIND_LABEL: Record<string, string> = { policy: "", gumbel: "Gumbel", search: "PUCT", bot: "bot" };

class LeaderboardPage {
  private epoch = "";
  private view: View = "main";
  private lineages = new Set<string>();
  private selected: string | null = null;

  constructor(private readonly data: LeaderboardData) {
    const params = new URLSearchParams(location.search);
    const names = Object.keys(data.epochs);
    this.epoch = names.includes(params.get("epoch") ?? "") ? params.get("epoch")! : names[names.length - 1];
    const v = params.get("view");
    this.view = v === "lineage" || v === "all" ? v : "main";
    for (const l of (params.get("lineage") ?? "").split(",")) if (l) this.lineages.add(l);
    this.selected = params.get("player");

    const sel = $<HTMLSelectElement>("lb-epoch");
    sel.innerHTML = names.map(n => `<option value="${esc(n)}">${esc(n)}</option>`).join("");
    sel.value = this.epoch;
    sel.addEventListener("change", () => { this.epoch = sel.value; this.selected = null; this.render(); });
    for (const r of document.querySelectorAll<HTMLInputElement>("input[name=lb-view]")) {
      r.checked = r.value === this.view;
      r.addEventListener("change", () => { if (r.checked) { this.view = r.value as View; this.render(); } });
    }
    $("lb-lineages").addEventListener("change", (ev) => {
      const box = ev.target as HTMLInputElement;
      if (box.checked) this.lineages.add(box.value); else this.lineages.delete(box.value);
      this.render();
    });
    $("lb-body").addEventListener("click", (ev) => {
      if ((ev.target as HTMLElement).closest("a")) return;
      const row = (ev.target as HTMLElement).closest<HTMLElement>("tr[data-player]");
      if (!row) return;
      const p = row.dataset.player!;
      this.selected = this.selected === p ? null : p;
      this.render();
    });
    this.render();
  }

  private get ep(): Epoch { return this.data.epochs[this.epoch]; }

  private ratingsShown(): Rating[] {
    const ep = this.ep;
    if (this.view === "all") return ep.ratings;
    if (this.view === "main") return ep.ratings.filter(r => r.main);
    return ep.ratings.filter(r => r.main || this.lineages.has(this.data.players[r.player]?.lineage ?? ""));
  }

  private links(pid: string): string {
    const pl = this.data.players[pid];
    const net = pl?.network ? this.data.networks[pl.network] : undefined;
    if (!net) return "";
    const out: string[] = [];
    if (net.report) {
      out.push(`<a href="https://github.com/${esc(this.data.github)}/blob/${esc(this.data.commit ?? "main")}/${esc(net.report)}"
        title="Behaviour report of the network's greedy play">report</a>`);
    }
    if (net.hf) {
      const onnx = net.hf.replace(/\.pt$/, ".onnx");
      out.push(`<a href="https://huggingface.co/${esc(this.data.hf_repo)}/blob/main/${esc(net.hf)}" title="The weights on Hugging Face">weights</a>`);
      out.push(`<a href="./?model=${encodeURIComponent(`hf:${this.data.hf_repo}@main:${onnx}`)}"
        title="Open the network in the engine workbench">analyse</a>`);
    }
    return out.join("");
  }

  private renderLineages(): void {
    const box = $("lb-lineages");
    box.hidden = this.view !== "lineage";
    const all = new Set<string>();
    for (const r of this.ep.ratings) {
      const l = this.data.players[r.player]?.lineage;
      if (l) all.add(l);
    }
    box.innerHTML = [...all].sort().map(l =>
      `<label><input type="checkbox" value="${esc(l)}" ${this.lineages.has(l) ? "checked" : ""}> ${esc(l)}</label>`).join("")
      || `<span class="lb-dim">No lineages in this epoch.</span>`;
  }

  private renderTable(): void {
    const rows = this.ratingsShown();
    $("lb-body").innerHTML = rows.map((r, k) => {
      const pl = this.data.players[r.player];
      const kind = KIND_LABEL[pl?.kind ?? ""] ?? pl?.kind ?? "";
      const pct = r.games ? (100 * r.score / r.games).toFixed(1) + "%" : "";
      return `<tr data-player="${esc(r.player)}" class="${r.player === this.selected ? "selected" : ""}">
        <td class="num lb-dim">${k + 1}</td>
        <td><span class="lb-player">${esc(r.player)}</span>${r.main ? `<span class="lb-tag main">MAIN</span>` : ""}${kind ? `<span class="lb-tag kind">${esc(kind)}</span>` : ""}</td>
        <td class="num lb-elo">${r.elo.toFixed(0)}</td>
        <td class="num lb-dim lb-hide-narrow">${r.player === this.ep.anchor ? "anchor" : r.se.toFixed(0)}</td>
        <td class="num lb-hide-narrow">${r.games.toLocaleString("en-US")}</td>
        <td class="num">${pct}</td>
        <td class="lb-links lb-hide-narrow">${this.links(r.player)}</td>
      </tr>`;
    }).join("");
  }

  private renderDetail(): void {
    const box = $("lb-detail");
    const pid = this.selected;
    const rating = pid ? this.ep.ratings.find(r => r.player === pid) : undefined;
    if (!pid || !rating) { box.hidden = true; return; }
    box.hidden = false;
    const pl = this.data.players[pid];
    const net = pl?.network ? this.data.networks[pl.network] : undefined;
    const fmtSeat = (s: Seat) => {
      const n = s.w + s.l + s.d;
      return n ? `${(100 * (s.w + s.d / 2) / n).toFixed(1)}%` : "–";
    };
    const h2h = this.ep.pairs.filter(p => p.a === pid || p.b === pid).map(p => {
      // From this player's side: the pair is stored from `a`'s.
      const flip = p.b === pid;
      const opp = flip ? p.a : p.b;
      const asUs: Seat = flip ? { w: p.a_as_ussr.l, l: p.a_as_ussr.w, d: p.a_as_ussr.d } : p.a_as_us;
      const asUssr: Seat = flip ? { w: p.a_as_us.l, l: p.a_as_us.w, d: p.a_as_us.d } : p.a_as_ussr;
      const all: Seat = { w: asUs.w + asUssr.w, l: asUs.l + asUssr.l, d: asUs.d + asUssr.d };
      const oppRating = this.ep.ratings.find(r => r.player === opp);
      return { opp, all, asUs, asUssr, oppElo: oppRating?.elo };
    }).sort((x, y) => (y.oppElo ?? 0) - (x.oppElo ?? 0));
    const desc = [pl?.description, net?.description].filter(Boolean).map(s => esc(s!)).join(" ");
    const old = net?.old_names.length ? ` Formerly ${net.old_names.map(n => `<code>${esc(n)}</code>`).join(", ")}.` : "";
    const links = this.links(pid);
    box.innerHTML = `
      <h2>${esc(pid)}</h2>
      <p>${desc}${old}</p>
      ${links ? `<p class="lb-links">${links}</p>` : ""}
      <p>Elo <b>${rating.elo.toFixed(0)}</b>${pid === this.ep.anchor ? " (the anchor)" : ` ± ${rating.se.toFixed(0)}`} · ${rating.games.toLocaleString("en-US")} games against ${rating.opponents} opponents, ${rating.main_opponents} of them main players.</p>
      <div class="lb-table-wrap"><table class="lb-table">
        <thead><tr><th>Against</th><th class="num">Elo</th><th class="num">Games</th><th class="num">Score</th>
          <th class="num lb-hide-narrow">as US</th><th class="num lb-hide-narrow">as USSR</th></tr></thead>
        <tbody>${h2h.map(h => `<tr>
          <td><span class="lb-player">${esc(h.opp)}</span></td>
          <td class="num lb-dim">${h.oppElo === undefined ? "–" : h.oppElo.toFixed(0)}</td>
          <td class="num">${(h.all.w + h.all.l + h.all.d).toLocaleString("en-US")}</td>
          <td class="num">${fmtSeat(h.all)}</td>
          <td class="num us lb-hide-narrow">${fmtSeat(h.asUs)}</td>
          <td class="num ussr lb-hide-narrow">${fmtSeat(h.asUssr)}</td></tr>`).join("")}</tbody>
      </table></div>`;
  }

  private renderScale(): void {
    const ep = this.ep;
    $("lb-scale").innerHTML = `${esc(ep.description)} Anchor <code>${esc(ep.anchor)}</code> = ${ep.anchor_elo.toFixed(0)}.
      The main players are fitted on their games against each other; everyone else against them, held fixed,
      so a rating does not move with the view. The US seat is worth ${ep.seat_us >= 0 ? "+" : ""}${ep.seat_us.toFixed(0)} ± ${ep.seat_us_se.toFixed(0)} Elo.
      Click a row for its head-to-head results.`
      + (ep.unrated.length ? ` <span class="lb-dim">Unrated (no games that connect them to the main players): ${ep.unrated.map(esc).join(", ")}.</span>` : "");
    const d = this.data;
    const records = `https://github.com/${esc(d.github)}/blob/${esc(d.commit ?? "main")}/leaderboard/matches/${esc(this.epoch)}.jsonl`;
    $("lb-footer").innerHTML = `Fitted from <a href="${records}">the match records</a>`
      + (d.commit ? ` at commit <code>${esc(d.commit.slice(0, 10))}</code>` : "")
      + ` by <code>tools/leaderboard.py</code>. Weights: <a href="https://huggingface.co/${esc(d.hf_repo)}">${esc(d.hf_repo)}</a>.`;
  }

  private render(): void {
    this.renderLineages();
    this.renderScale();
    this.renderTable();
    this.renderDetail();
    const params = new URLSearchParams();
    params.set("epoch", this.epoch);
    if (this.view !== "main") params.set("view", this.view);
    if (this.view === "lineage" && this.lineages.size) params.set("lineage", [...this.lineages].sort().join(","));
    if (this.selected) params.set("player", this.selected);
    history.replaceState(null, "", `?${params}`);
  }
}

async function main(): Promise<void> {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}leaderboard.json`);
    if (!res.ok) throw new Error(`leaderboard.json: HTTP ${res.status} (build it with tools/scripts/build_web.sh)`);
    const data = await res.json() as LeaderboardData;
    if (data.schema !== 1) throw new Error(`leaderboard.json schema ${data.schema}; this page reads schema 1`);
    new LeaderboardPage(data);
  } catch (e) {
    const box = $("lb-error");
    box.hidden = false;
    box.textContent = `Could not load the leaderboard: ${e instanceof Error ? e.message : e}`;
  }
}

main();
