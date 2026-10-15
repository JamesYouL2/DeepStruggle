# This engine and `struggler`

[`struggler`](https://github.com/alekpinel/struggler) is an independent Python Twilight Struggle
engine, vendored as the git submodule `external/struggler` for differential testing. Both
implement the Deluxe Edition's 110 cards with fine-grained decisions (one Ops point per step).

**Status:** the differential suite (`tests/differential/`) does not currently run -- its modules
fail at import and are kept out of collection (`BUGS.md` TEST-1). Nothing below is continuously
checked.

## Design differences

| | this engine (`engine/`) | `struggler` |
|:---|:---|:---|
| purpose | RL and search: throughput, bit-exact replay | LLM agents, inspectable JSON, human referee mode |
| language | C++20, no heap allocation in the core | Python dataclasses |
| state | `ts::GameState`, trivially copyable, ≤ 4 KB (`engine/include/ts/game_state.hpp`) | `Board` / `Engine` objects, influence keyed by country name |
| persistent effects | bits in `uint64_t persistent_effects` | `turn_effects` / `game_effects` dicts |
| decisions | 4-byte `MicroAction` (`DecisionType`: `SELECT_CARD`, `SELECT_PLAY_MODE`, `CHOOSE_TIMING_BRANCH`, `SELECT_OP_MODE`, `POINT_NODE`, `CHOOSE_BRANCH`, `ROLL_DIE`) and one flat action space, 220 wide (`generate_flat_mask_212`) | typed `Decision` / `Action(kind, payload)` |
| decision stack | fixed `std::array<DecisionContext, 6>` (`ctx_stack`) | unbounded Python list |
| chance | resolved inside `step()` from SplitMix64 in `state.rng_state`; a `ROLL_DIE` action can force a die | explicit `Side.CHANCE` decisions fed from `random.Random` |
| front end | browser workbench (engine as WebAssembly, models as ONNX) | CLI runner, JSON logs, prompt serialisation |

## Data parity

Map and cards were compared once: the same 84 countries with identical stability, battleground
flags, regions and adjacency (Syria and Iraq not adjacent), and the same 110 cards (number, Ops,
side, era, scoring/war flags). Names differ in spelling only (`"East Germany"` against
`"East_Germany"`, `"United Kingdom"` against `"UK"`).

## The bridge

`tests/differential/struggler_adapter.py` (country/card name normalisation, state sync,
MicroAction ↔ `Action` translation), `native_adapter.py` and `engine_interface.py` put both
engines behind one interface; `test_unified_differential.py` and `test_differential_fuzzing.py`
drive them. The scenarios covered were setup placement, Ops reachability and costs, coup DEFCON
restrictions and the coup formula, realignment modifiers, the space race track, regional
scoring, and a set of card events (war cards, influence-moving events, persistent modifiers).
To run them once TEST-1 is fixed: `pytest --run-fuzz tests/differential` with
`external/struggler/src` on `PYTHONPATH`.
