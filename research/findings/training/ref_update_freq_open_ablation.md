# `ref_update_freq`: not a drift, and since measured

**Conclusion.** The 200,000 default is the ladder's baseline (43 of 51 E3 arms and every arm since);
the 5,000,000 slow anchor was a late E3 experiment (8 arms, E3-30 … E3-37), and E3-31-28 collapsed by
KL domination while carrying it ([`../../method/detecting_collapse.md`](../../method/detecting_collapse.md)).
The owner kept 200k (2026-09-19: "it didn't give clear signal, so should stay as experimental option
to ablate later"). The ablation was then run on E4 M2d:

* **From scratch, the slow anchor loses**: −168 / −68 Elo at 80M on seeds 3 / 5, still −59 at 160M
  ([`../../log/E4_dynamics_ref_lambda.md`](../../log/E4_dynamics_ref_lambda.md)).
* **Switched on after 160M it helped** (+78 to +108 at 240M on seed 3; +91 at 200M on seed 5) but
  delayed rather than prevented a collapse — on legs that all ran on a drained opponent pool, so only
  the between-arm comparison stands ([`../../log/E4_late_dynamics.md`](../../log/E4_late_dynamics.md)).

The general point stays: a diff says a flag differs; only a survey of runs says whether the
difference is a mistake (contrast the real drifts in
[`e4_architecture_discontinuity.md`](e4_architecture_discontinuity.md)).
