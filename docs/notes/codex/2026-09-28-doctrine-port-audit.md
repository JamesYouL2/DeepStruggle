# Doctrine port: correctness, strength evidence, and next work

Reviewed September 28, 2026 (America/Los_Angeles). This is a focused source audit with executable Python component probes and inspected, matching-revision CI evidence. It is not a complete rules certification or a newly run head-to-head tournament.

## Revisions

- New repository: JamesYouL2/DeepStruggle main `7ff9ca5f35fc49216494e2809aea2c189480cde4`.
- Existing tournament branch: `feat/parallel-tournament-onnx`, `109f488837894bbd70c075743bce36c0a9cd5188`; merge base is the reviewed main. Not merged at review time. Engine, Doctrine policy/evaluator/schedule are unchanged from main; DoctrineAgent adds reseeding for tournament shards.
- Original comparison bot: JamesYouL2/struggler main `0085ba06004fe1e84b2d416c483e91e3048b4756`.
- Historical Struggler tag v0.1.0: `50e5af5bbd32bb9fc3de15ee6a8a8384db8a9db9`. Not the proposed strength baseline.
- Remote main and tournament heads rechecked at the end; unchanged. Read active tournament code before recommending infrastructure.

## Main assessment

Doctrine is a new policy using the old evaluator, not a strength-preserving port of the entire old bot. Its six tests check a fixed-board evaluator value, weight coverage, one hidden-hand swap, legality through a turn, scoring deadlines, and one CIA survival scenario. Those do not establish equivalent decision quality.

The most useful immediate work is terminal-outcome ordering and shared-resource hand safety, followed by the VP-price reachability bug and scoring-schedule corrections. Then establish an original-bot baseline. A larger neural network or deeper MCTS is premature as the first response to these defects.

## Actual strength evidence

[Completed tournament run 36514768362](https://github.com/JamesYouL2/DeepStruggle/actions/runs/36514768362), pool job 109237231998, at tournament SHA `109f488`:

| Matchup | Doctrine record |
| --- | --- |
| E5-11-43@560M, both seats | 13 wins, 187 losses, 0 draws; 6.5% |
| Doctrine as US | 7–93 |
| Doctrine as USSR | 6–94 |

200 games, 100 paired seeds / 100 games per seat, 100 shards, all 16 parts pooled. Opponent artifact named `E5-11-43_560M.onnx`. Recorded engine fingerprint: `dd7544681737a72f14d32f942e7df76c88a5a42e2cf3bdc51cc066c333a363c4`. This is an existing CI measurement, not a tournament launched during this audit. The workflow's default temperature is 0.1. It is not an equal-search-budget experiment.

Doctrine's losses: 91 at 20 VP, 64 at DEFCON 1 (33 classified own decision, 31 opponent decision), 25 at final scoring, six Europe-control losses, one Wargames loss. Classification describes who caused DEFCON to fall; it does not by itself prove that the immediately preceding move had a safe alternative. The defects below are credible explanations to test, not established causes of particular logged losses.

The bot documentation also reports 40–0 against HeuristicBotV2 and 48–2 against HeuristicMCTS16. Those are repository claims, not independently reproduced in this audit. They illustrate why beating weak baselines does not validate the port. Perfect-information MCTS is a diagnostic opponent, not a fair hidden-information baseline.

**Original Struggler versus Doctrine remains unmeasured.** No ready tournament entrant for the unchanged original StrategicPlayer was found. `tests/differential/struggler_adapter.py` is engine-test scaffolding, not that entrant; the differential suite is explicitly disabled/broken. The inherited `docs/STRUGGLER_COMPARISON.md` describes alekpinel's repository and claims parity that the current disabled suite does not establish. Do not use it as evidence for JamesYouL2's current strategic bot, or compare win rates from different engines/checkpoints as though they were a head-to-head.

## Findings

### F1 — High: terminal results and ongoing board values have incompatible bounds

Location: `bot/doctrine/policy.py:DoctrinePolicy.static`, `_settle`, `_fatal_at_two`.

Terminal positions are replaced by `+/-40 * ctx.vp_price`, while nonterminal positions retain absolute board value plus VP and military terms. No bound ensures ongoing values fall between the terminal endpoints. Thus maximization can prefer losing immediately to continuing a sufficiently unfavorable game, or reject an immediate win for a sufficiently favorable ongoing board. `_fatal_at_two` additionally labels values below half the loss magnitude as fatal even when the simulated state is not terminal.

Executable component reproduction using the actual Python evaluator, actual parsed country metadata, actual `_prepare`, and a constructed turn-9 board: `vp_price=2.3237112639972755`; ongoing value `-663.84339944325`; certain-loss value `-92.94845055989101`. The ongoing board had USSR stability-sized influence everywhere, with US control restored at US-home-adjacent countries. This proves the value-domain defect; it is not a native-engine replay or evidence that this exact board occurred in the tournament.

Correction: represent terminal outcomes distinctly, or put all backed-up values on one bounded utility scale. Preserve well-defined expectation over chance; do not substitute infinities into probability arithmetic. Do not infer certain death from a heuristic score threshold. Acceptance: immediate wins outrank ongoing alternatives, certain losses rank below viable alternatives, and mixed chance outcomes remain finite and correctly ordered. Reproduce on native reachable saved positions before attributing any of the 33 own-decision DEFCON losses to this defect.

### F2 — High: the reduced DEFCON hand check reuses scarce disposal capacity

Location: `policy.py:_defcon_hazard`, `_fatal_at_two`, approximately lines 174–231. Comparison: original `src/struggler/bots/strategic/defcon.py`.

Each card is tested on a fresh clone of the same current position. The count of independently safe cards is then compared with rounds remaining. If two dangerous cards can each be spaced using the same one remaining attempt, both count as safe. Selecting a card does not advance the other cards' modeled space capacity, track, or other hand resources. This is not a whole-hand feasibility check.

Static counterexample shape: USSR holds Duck and Cover, Grain Sales, and a safe card, must fill three rounds, has no China card, one space attempt, and the US has a battleground coup that can cause DEFCON suicide. Each hazardous card has an individual space exit; both cannot use the one slot. An engine-level fixture should pin the exact board and disable any alternative disposal route. This shape was not executed natively here.

Correction: port a bounded joint survival search over remaining hand, rounds, space track/attempts, China and relevant event effects. Acceptance: two cards requiring the same last disposal slot cannot both count as safely disposable, and choosing a space play consumes the slot for the remainder. Reuse old hand-safety fixtures, including both seats and DEFCON 3 timing. Fix F1 first so a low board value does not masquerade as proof of death.

### F3 — Medium: one-Op VP pricing uses stale reachability

Location: `policy.py:_one_op_value`, approximately lines 272–287.

The loop adds influence, refreshes control/reach, evaluates, and removes influence without refreshing before the next candidate's reach test. The next iteration sees reach created by the previous hypothetical placement. This can cascade through increasing country indices.

Executable component probe with real map metadata and an empty board found US candidates such as United Kingdom, Norway, Sweden, West Germany and France, although the root US reach was only home-adjacent countries. Candidate valuation refreshes again, so the bug is specifically candidate eligibility / the derived best-Op price. It does not bypass the live engine mask or directly make the bot submit illegal placements.

Correction: compute candidate eligibility from the root snapshot or restore all derived fields after undo; use legal Ops eligibility for effect restrictions as well. Acceptance: independent root-snapshot enumeration and `_one_op_value` agree, and country iteration order cannot create extra candidates. This price feeds every VP reward and terminal penalty.

### F4 — Medium: scoring schedule drops draws during the reshuffle deal

Location: `bot/doctrine/schedule.py:urgency`, calculation of `after`.

The future recycled-deck draw sum starts at `reshuffle_in + 1`. Cards drawn from the recycled pile during the very deal that exhausts the old pile are omitted. Component probe: turn 9, one card left in the draw pile, Asia Scoring discarded. Asia mass is exactly `0.75`, final scoring alone. Yet the modeled next deal needs 14 cards, so after drawing the one remaining card it draws from the discard; Asia has nonzero probability of scoring on turn 10.

Correction: account for the unfilled portion of the exhaustion deal before adding subsequent deals. Acceptance: final-turn reshuffle fixtures give positive return probability to discarded scoring cards, with normalized mass and no double-counting of the old draw pile.

### F5 — Medium: known opponent cards inflate hidden-card probability

Location: `schedule.py:urgency`, `theirs = len(_hand(...))` and `p_opp = theirs / (theirs + pile)`.

The unseen pool should contain only unknown opponent cards plus the draw pile. The code counts known opponent cards too. Component probe at turn 10: eight known opponent cards, one unknown, ten draw-pile cards. For an unseen scoring card the implementation assigns `9/19 = 0.473684...` to the opponent, rather than `1/11 = 0.090909...`. Exact known scoring cards already have their own branch, so counting all known cards in the unknown pool is not justified.

Correction: use unknown-hand count in the unseen probability calculation and retain the known-card branch. Acceptance: progressively revealing unrelated opponent cards shrinks the unknown pool consistently; moving an unknown card between hidden hand and deck leaves the same information-set evaluation.

## Port coverage and next improvements

| Component | Current Doctrine | Consequence / next step |
| --- | --- | --- |
| Country/region evaluator and fitted weights | Ported; one golden-board parity check | Broaden parity to real shared positions and scoring modifiers |
| Action-round engine sandbox | Native simulation, structural choices searched, placements greedy | Useful foundation; verify boundary and chance behavior with native fixtures |
| Terminal ordering | Finite endpoints mixed with absolute ongoing value | Fix F1 before tuning |
| Whole-hand survival | Independent per-card tests | Restore joint resource accounting |
| Opponent reply | Not ported | First tactical A/B after safety and baseline validity |
| Hand/hold/Ask Not/Missile Envy values | Not ported | Lost card-option value; add bounded hand-aware comparisons |
| Future scoring schedule | Simplified; F4/F5 | Repair arithmetic before fitting weights |
| Space progression / durable effects | No general continuation term in `static` | Audit distinctions with unchanged board/VP, then test adding only missing values |
| Hidden state | One determinization per choice | Test several shared sampled worlds per candidate, after privacy tests |
| Multi-core / multi-runner tournaments and ONNX | Already on PR #5 branch | Integrate existing work; do not rebuild it |

Do not promise a 5-point gain from replies. Struggler's ledger contains both an older reply-off result of -5.6 points and a later -1.6-point result with an interval spanning zero. Those are evidence to prioritize an experiment, not transferable gains in Doctrine.

The new architecture makes diagnostic simulation, saved-position comparisons and batched experiments cheaper. It does not remove Python evaluator cost: Doctrine still constructs positions and repeatedly calls a Python board evaluator around native micro-steps. Profile end-to-end move selection after the correctness work. If evaluation dominates, a coarse native batch of candidate-state evaluations or incremental evaluation may be worthwhile; raw engine step speed is not bot speed. No language rewrite multiplier was measured here.

Learning is a later experiment: first compare frozen neural policy, corrected Doctrine, and the original bot. Distillation from Doctrine should wait until its errors are fixed and its teacher quality is measured. Existing PIMCTS is privileged; existing DMCTS still needs information-set invariance tests, including pool ordering and RNG, before calling its results fair.

## Establishing the old-bot comparison

1. Freeze the revisions above and a model file by content hash; retain engine and observation/action-schema provenance. The current ONNX loader checks widths and records an engine fingerprint but does not enforce semantic compatibility. The HF workflow pins a filename, not an immutable content hash.
2. Build a legacy StrategicPlayer entrant using entitled native state observations and explicit action translation. The old bot's internal sandbox can remain Python. Reject unsupported actions; never silently fall back. Because native play-mode choices are merged differently, this is meaningful adapter work, not a one-line registration.
3. Validate the adapter on shared saved positions before interpreting wins: both seats, opponent events, traps, headline timing, free Ops, China, scoring, and terminal responsibility. Compare complete action-round outcomes and legal action meaning, not just raw action IDs.
4. Run original versus Doctrine on the same authoritative rules engine, paired deals/both seats, then each versus the same frozen neural opponent. Start with a bounded pilot to expose integration faults; choose a larger sample for the desired detectable difference. A few dozen games do not settle a five-point change. Bootstrap by seed pair and reserve fresh confirmation seeds for selected changes.
5. Report win/draw/loss by seat, termination causes, illegal/rejected actions, unfinished games, wall time and actual search work. Keep present-configuration strength separate from an equal-time-budget experiment. Pin shard size because agent sampling streams depend on it.

No new tournament or training was launched, and no experiment arm was registered. The existing historical ledger was consulted to avoid presenting old reply experiments as new evidence.

## Verification and scope limits

Matching-main [CI run 36510278338](https://github.com/JamesYouL2/DeepStruggle/actions/runs/36510278338), backend job 109220638039, was inspected through its logs:

- 393 C++ tests passed.
- Native invariant fuzzer completed 2,000 games / 357,643 steps.
- Backend Python suites: 1,852 passed, 38 skipped.
- Whole-corpus conversion was skipped. Web and differential suites are outside that CI job.
- Type check reported 11 errors in `tools/scrape_card_strategies.py`; it is advisory (`continue-on-error`). Green CI does not mean zero typing errors.

Local native compilation was unavailable: no clang was installed, and package installation failed under the environment's permissions. The audit did not bypass the project's compiler requirement. Four finding categories were exercised with the exact Python functions plus fake state accessors and parsed C++ card/map metadata, not a compiled engine. F2 is source-traced only. Native regression reproductions and attribution of individual losses remain outstanding. No new full card survey, training-stack audit, or original-bot tournament was performed.

Earlier upstream findings in BUGS.md ENG-2 are marked fixed and matching-main CI exercises their tests; they are not repeated as open defects. TEST-1 remains open. This review adds port-specific findings rather than treating old upstream audit notes as current evidence.

Implementation unchanged. Publication authorized after the review on docs branch `docs/doctrine-port-audit-2026-09-28`, based on freshly fetched main `7ff9ca5f35fc49216494e2809aea2c189480cde4`, matching the reviewed revision. No pull request requested.

## Component reproduction appendix

Run from the reviewed DeepStruggle checkout with NumPy installed. This uses deliberately fake state accessors, not native game simulation, and should become proper native regression fixtures before fixes are declared validated.

```python
"""Source-level probes using parsed native metadata and fake state accessors; NOT native-engine tests."""
import importlib.util,sys,types,re,copy,random
from pathlib import Path
from types import SimpleNamespace as NS
root=Path.cwd()
ts=types.ModuleType('ts_engine')
ts.Player=NS(US=1,USSR=-1,NONE=0)
ts.Phase=NS(SETUP=0,HEADLINE=1,ACTION_ROUND=2,GAME_OVER=6)
ts.DecisionType=NS()
ts.CardLocation=NS(**{k:i for i,k in enumerate(['UNAVAILABLE','DRAW_DECK','HAND_US_UNKNOWN','HAND_US_KNOWN','HAND_USSR_UNKNOWN','HAND_USSR_KNOWN','DISCARD_PILE','REMOVED_FROM_GAME','ONGOING_EVENT','PEEKED_TEMP','HEADLINE_COMMITTED'])})
ts.in_hand_of=lambda loc,p:loc in ((2,3) if p==1 else (4,5))
ts.known_to_opponent=lambda loc:loc in (3,5)
ts.Engine=NS(is_terminal=lambda s:s.terminal)
ts.EffectBits=NS(FORMOSAN_RESOLUTION_ACTIVE=1,SHUTTLE_DIPLOMACY_ACTIVE=2)
regions={k:i for i,k in enumerate(['EUROPE','ASIA','MIDDLE_EAST','AFRICA','CENTRAL_AMERICA','SOUTH_AMERICA'])}
pat=r'\{(\d+), "([^"]+)", (\d+), (true|false), Region::(\w+), (true|false), (true|false), (true|false), Player::(\w+), (\d+), \{([^}]+)\}'
meta={}
for m in re.finditer(pat,(root/'engine/src/map_data.cpp').read_text()):
 i,name,st,bg,reg,we,ee,sea,adj,n,nb=m.groups();meta[int(i)]={'name':name,'stability':int(st),'battleground':bg=='true','region':regions[reg],'in_southeast_asia':sea=='true','superpower_adjacent':adj,'neighbors':list(map(int,nb.split(',')))[:int(n)]}
assert len(meta)==84
ts.MapData=NS(get_country_info=lambda i:meta[i],get_country_name=lambda i:meta[i]['name'])
eras={int(a):{'era':{'EARLY':0,'MID':1,'LATE':2}[b]} for a,b in re.findall(r'\{(\d+), "[^"]+", \d+, Player::\w+, WarEra::(\w+)',(root/'engine/src/card_data.cpp').read_text())}
ts.CardData=NS(get_card_info=lambda c:eras[c]);sys.modules['ts_engine']=ts
pkg=types.ModuleType('bot.doctrine');pkg.__path__=[str(root/'bot/doctrine')];sys.modules['bot.doctrine']=pkg
for name in ['evaluator','schedule','policy']:
 spec=importlib.util.spec_from_file_location('bot.doctrine.'+name,root/f'bot/doctrine/{name}.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);setattr(pkg,name,mod)
ev=pkg.evaluator;pol=pkg.policy;sch=pkg.schedule
class State:
 def __init__(self):
  self.inf=[[0]*84,[0]*84];self.turn=9;self.terminal=False;self.victory_points=0;self.defcon=2;self.us_mil_ops=2;self.ussr_mil_ops=2;self.locs={c:6 for c in range(1,111)}
 def get_country(self,i):return NS(us_influence=self.inf[0][i],ussr_influence=self.inf[1][i])
 def get_card_location(self,c):return self.locs[c]
 def has_flag(self,f):return False
p=pol.DoctrinePolicy();s=State();t=p.t
# One-Op stale reach: state has no influence, so US only reaches home-adjacent countries.
ctx=pol._Context(me=1,urgency=(1.,)*84,vp_price=1.,rounds_left=3,root_key=(9,2,1,1))
rootpos=ev.Position.of(s,t);seen=[]
original=p._board
def track(pos,side,state,context):
 added=[i for i in range(84) if pos.inf[side][i]>s.inf[side][i]]
 seen.extend(i for i in added if not rootpos.reach[side][i])
 return original(pos,side,state,context)
p._board=track
v=p._one_op_value(s,1,ctx)
print('one_op unreachable candidates:',[t.names[i] for i in seen]);assert seen
# Terminal scale: real metadata/evaluator; constructed late-war board, no assertion of play history.
p._board=original
s.inf[1]=list(t.stability)
for i in t.home[0]:s.inf[0][i]=s.inf[1][i]+t.stability[i]
ctx.cache.clear();v=p.static(s,0,ctx);dead=copy.deepcopy(s);dead.terminal=True;dead.victory_points=-20
loss=p.static(dead,0,ctx)
print('constructed board static:',v,'certain-loss static:',loss);assert v<loss
# Schedule loses draws from recycled deck during the reshuffle deal.
s=State();s.locs[4]=1
u=sch.urgency(s,1,t,p.w);asia=u[t.region_anchor[ev.ASIA]]
final=sch.FINAL_SCORING_ODDS[8]*p.w.scoring_final
print('turn9, pile1, Asia discarded: mass',asia,'final-only',final);assert asia==final
# Revealed opponent cards should not contribute to an unseen scoring card's hand probability.
s=State();s.turn=10
for c in range(4,14):s.locs[c]=1
s.locs[1]=4
for c in range(14,22):s.locs[c]=5
u=sch.urgency(s,1,t,p.w);observed=u[t.region_anchor[ev.ASIA]]-1.0
print('hidden scoring P(opponent):',observed,'correct unknown pool probability:',1/11);assert abs(observed-9/19)<1e-10
# Use actual _prepare VP pricing, rather than the diagnostic unit price above.
s=State();s.current_phase=ts.Phase.ACTION_ROUND;s.action_round=1;s.phasing_player=1
s.inf[1]=list(t.stability)
for i in t.home[0]:s.inf[0][i]=s.inf[1][i]+t.stability[i]
ctx=p._prepare(s,1);v=p.static(s,0,ctx);dead=copy.deepcopy(s);dead.terminal=True;dead.victory_points=-20;loss=p.static(dead,0,ctx)
print('actual preparation: vp_price',ctx.vp_price,'board',v,'loss',loss,'prefers loss',v<loss)
```
