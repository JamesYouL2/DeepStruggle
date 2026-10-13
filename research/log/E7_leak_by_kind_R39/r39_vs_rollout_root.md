# Twilight Struggle AI: Massive Tournament Evaluation Report

- **Total Models**: 3
- **Total Games Played**: 3,000
- **Games Per Matchup Pair**: 1,000 (500 per side)
- **Sampling Temperature**: 0.0 (deterministic)
- **Total Evaluation Time**: 1872.8 seconds (1.6 games/sec)

## 1. Bradley-Terry MLE Elo Leaderboard

| Rank | Model | Elo Rating | Total Matches | Total Record (W-L-D) | Overall Win Rate |
|:---:|:---|:---:|:---:|:---:|:---:|
| **1** | **r39-rollout** | **1550.0** | 2,000 | 1,202W - 787L - 11D | **60.1%** |
| **2** | **soup-rollout** | **1529.4** | 2,000 | 1,116W - 872L - 12D | **55.8%** |
| **3** | **r39** | **1420.7** | 2,000 | 663W - 1,322L - 15D | **33.1%** |

---

## 2. Head-to-Head Total Win Rate Matrix (% Win for Row vs Column)

| Model | **r39-rollout** | **soup-rollout** | **r39** |
|:---|---:|---:|---:|
| **r39-rollout** | — | 52.3% | 67.9% |
| **soup-rollout** | 47.3% | — | 64.3% |
| **r39** | 31.4% | 34.9% | — |

---

## 3. Per-Side Win Rate Matrix (Row vs Column: as USSR / as US)

Each cell is the **row** model's win rate against the **column** model, playing USSR and playing US, over 500 games per side.

| Model | **r39-rollout** | **soup-rollout** | **r39** |
|:---|---:|---:|---:|
| **r39-rollout** | — | 46.0% / 58.6% | 61.2% / 74.6% |
| **soup-rollout** | 40.6% / 54.0% | — | 61.0% / 67.6% |
| **r39** | 24.8% / 38.0% | 31.8% / 38.0% | — |

### Side balance per arm (across all opponents)

| Model | as USSR | as US | USSR - US |
|:---|---:|---:|---:|
| **r39-rollout** | 53.6% | 66.6% | -13.0 pp |
| **soup-rollout** | 50.8% | 60.8% | -10.0 pp |
| **r39** | 28.3% | 38.0% | -9.7 pp |

---

