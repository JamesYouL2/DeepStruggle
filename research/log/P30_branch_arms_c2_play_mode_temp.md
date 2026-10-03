# P30 branch arms (2026-10-03): C2's card-event head, and exploration at play-mode decisions

**Arms** ([`../runs.md`](../runs.md)), both E7-02-44 branched at 820M (`resume_820051968steps.pt`)
and run to 1,200M with its flags:

* **E7-13-44** -- plus `--aux-card-events 1.0` (P30 C2): the card-event head added to the saved
  network, trunk, policy and value restored exactly.
* **E7-14-44** -- plus `--play-mode-temp 2.0`: the learner samples its event/space/Ops choice at
  temperature 2, storing its own log-prob.

**Control.** E7-02-44's own 820 → 1,200M. **Branch reference.** E7-03-44, a branch of the same
resume state with new seeds only: what a branch with no change looks like.

Asked for by the owner to measure strength ("Lets train these arms for now just to measure
strength"). One round robin, 1,000 games per side per pairing, temperature 0
(`data/reports/e7_13_14_rr.{md,json}`); engine 1d11c2f2.

## Owner's rule (per seat against the panel E6-03-44@80/240/550M, arm minus control)

| comparison | US | USSR | head to head |
|:---|---:|---:|---:|
| **C2 E7-13-44**, 1,140–1,200M snapshots | **+1.5 ± 0.5** | −0.4 ± 0.5 | 50.0% ± 0.3 (US 52.0, USSR 48.0) |
| **C2 E7-13-44**, 1,120–1,200M SWA | **+3.2 ± 0.8** | +0.1 ± 0.9 | 51.6% ± 1.1 |
| **play-mode temp E7-14-44**, 1,140–1,200M snapshots | −0.6 ± 0.5 | **−1.7 ± 0.5** | 49.0% ± 0.3 (US 48.0, USSR 50.0) |
| **play-mode temp E7-14-44**, 1,120–1,200M SWA | −0.5 ± 0.9 | −1.7 ± 0.9 | **46.7% ± 1.1** |
| *E7-03-44 (new seeds only)*, 1,120–1,200M SWA | +1.1 ± 0.9 | −0.4 ± 0.9 | 50.3% ± 1.1 |

Elo in the same field: SWAs E7-13-44 1594, E7-02-44 1585, E7-03-44 1583, E7-14-44 1576; late
snapshots 1525–1550, the three runs interleaved.

## Reading

* **C2 passes the rule as written, narrowly.** Above the control in the US seat on snapshots and on
  the SWA (3 and 4 standard errors) and not below it in the USSR seat. The SWA is the top of the
  field, +9 Elo over the control's. But head to head it is level on snapshots (50.0%) and only
  +1.4 SE on the SWA, and the no-change branch E7-03-44 sits +1.1 in the US seat itself; the
  standard errors cover game noise, not branch-to-branch spread. **A small positive signal, not yet
  a result:** one seed, 380M of training with the head, judged on a branch rather than from scratch.
* **The play-mode temperature hurts slightly.** Below the control in the USSR seat on snapshots
  (3.4 SE), the same on the SWA, and 46.7% head to head on the SWAs. The sampled event fraction at
  play-mode decisions rose only from about 0.29 to 0.31 over the branch, so the exploration did not
  move the policy towards events either. Not worth pursuing in this form.
* **What C2 still needs before it counts:** the plan's check that the head reaches the trunk
  (`card_board_probe.py --frozen`, within-card R² against the untrained 0.66), and a from-scratch
  run against E7-02-44 -- the branch protocol cannot say whether the head helps a run that has it
  from the start.
