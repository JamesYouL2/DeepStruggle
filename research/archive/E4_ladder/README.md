# E4 — closed plans

Closed E4 plans kept because live documents cite their diagnoses. The E4 record itself is in
[`../../log/`](../../log/README.md), chiefly [`P21_ladder_status.md`](../../log/P21_ladder_status.md),
and in [`../../findings/training/`](../../findings/training/README.md).

| plan | question | answer |
|:---|:---|:---|
| [`P19_architecture_ab.md`](plans/P19_architecture_ab.md) | is the late-E3 architecture bundle stronger than the defaults on this engine? | **yes, +447 Elo** (`E4-03-01@80M` against `E4-04-01`, both cold and pooled, pre-registered; [log](../../log/E4_architecture_ab_result.md)) |
| [`P20_positional_board_encoder.md`](plans/P20_positional_board_encoder.md) | does a positional board path beat the symmetric pool? | **yes**, answered by P21 without citing it: every rung from M1 runs `aggregation="flatten"`, and M1 alone is +109. Its diagnosis — the pre-pooling token holds ~89–94% of a country's influence against the trunk's ~14% — is still the clearest statement of why pooling was the bottleneck |
| [`P25_collapse_robustness.md`](plans/P25_collapse_robustness.md) | collapse robustness | a collapse on the fixed pool is a recoverable delay; no lever adopted ([log](../../log/P25_stress_bench.md#closing-p25-2026-09-24)). Step 5's rewind design is in [reserve](../../plans/reserve.md) |
