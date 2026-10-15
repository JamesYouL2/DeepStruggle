# E4 was launched with a different network from every late E3 arm

**Conclusion (found 2026-09-19; resolved by accepting it).** E4-01-01 and E4-02-01 ran the bare
architecture defaults instead of the flags all eleven late E3 arms carried, because the launch was
copied from the `CLAUDE.md` template without diffing `metadata.json`. The owner accepted E4-01/02 as
the new baseline rather than discard 560M steps, so **no E3 number is comparable to an E4 one**;
everything internal to E4 (pooled vs unpooled, the round robin, the collapse analysis) is sound.

| flag | E3-30 … E3-37 | **E4-01-01, E4-02-01** |
|:---|:---|:---|
| `identity_dim` | 16 | **0** |
| `per_entity_heads` | 64 | **0** |
| `graph_layers` | 0 | **2** |
| `self_transform` | True | **False** |

The state dicts confirm it (weights record what was built; metadata only what was asked for). The
E3 net is ~18% more expensive per forward (3.46 against 2.94 ms at batch 512), which confounds the
E3↔E4 throughput comparison of P18 (since deleted; the throughput work is
[`../../log/training_throughput_cpu.md`](../../log/training_throughput_cpu.md)) but does not explain its
100% → 55% GPU-utilisation drop. Blunder rates and training dynamics
(`entropy`, `kl_div`, `adv_std_raw`, …) are not comparable across the two lineages at all
([`../../method/detecting_collapse.md`](../../method/detecting_collapse.md)).

**The lesson, now `CLAUDE.md` invariant 15:** diff the whole `metadata.json` against a healthy run
(`tools/scripts/launch_flags.py <ref> --diff <new>`), not the flags you happen to be thinking about.
A diff says a flag differs; only a survey says whether the difference is a mistake —
`ref_update_freq` (5,000,000 in E3-37 against E4's 200,000) is not one: 43 of 51 E3 arms used 200,000
([`ref_update_freq_open_ablation.md`](ref_update_freq_open_ablation.md)).
