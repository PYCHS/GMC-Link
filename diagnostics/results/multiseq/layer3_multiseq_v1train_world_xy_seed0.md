# Multi-Sequence Eval: v1train_world_xy_seed0

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.752** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.829** ± 0.072 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.974 | — | — | 0.974 ± — | 0.974 | 295/0/0 |
| cars in front of ours | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| right cars which are parking | — | 0.949 | — | 0.949 ± — | 0.949 | 0/1056/0 |
| right vehicles which are parking | — | 0.949 | — | 0.949 ± — | 0.949 | 0/1056/0 |
| left vehicles in the counter direction of ours | 0.814 | 0.956 | — | 0.885 ± 0.071 | 0.894 | 526/1692/0 |
| counter direction cars in the left | 0.814 | 0.955 | — | 0.884 ± 0.070 | 0.893 | 526/1692/0 |
| left cars in the counter direction of ours | 0.814 | 0.954 | — | 0.884 ± 0.070 | 0.893 | 526/1692/0 |
| counter direction vehicles in the left | 0.814 | 0.954 | — | 0.884 ± 0.070 | 0.893 | 526/1692/0 |
| moving right pedestrian | — | — | 0.869 | 0.869 ± — | 0.869 | 0/0/261 |
| moving vehicles | 0.933 | 0.818 | — | 0.876 ± 0.058 | 0.857 | 569/765/0 |
| moving cars | 0.933 | 0.818 | — | 0.876 ± 0.058 | 0.857 | 569/765/0 |
| vehicles which are braking | 0.846 | — | — | 0.846 ± — | 0.846 | 295/0/0 |
| cars which are braking | 0.846 | — | — | 0.846 ± — | 0.846 | 295/0/0 |
| vehicles in the same direction of ours | 0.767 | 0.983 | — | 0.875 ± 0.108 | 0.805 | 510/371/0 |
| cars in the same direction of ours | 0.767 | 0.983 | — | 0.875 ± 0.108 | 0.805 | 510/371/0 |
| moving left pedestrian | — | — | 0.795 | 0.795 ± — | 0.795 | 0/0/253 |
| left vehicles which are parking | 0.558 | 0.909 | — | 0.734 ± 0.176 | 0.791 | 169/1375/0 |
| left cars which are parking | 0.558 | 0.909 | — | 0.734 ± 0.176 | 0.791 | 169/1375/0 |
| vehicles in the counter direction of ours | 0.799 | 0.647 | — | 0.723 ± 0.076 | 0.694 | 526/1692/0 |
| cars in the counter direction of ours | 0.799 | 0.647 | — | 0.723 ± 0.076 | 0.694 | 526/1692/0 |
| vehicles in horizon direction | — | 0.654 | — | 0.654 ± — | 0.654 | 0/178/0 |
| cars in horizon direction | — | 0.654 | — | 0.654 ± — | 0.654 | 0/178/0 |
| cars which are faster than ours | — | 0.652 | — | 0.652 ± — | 0.652 | 0/371/0 |
| vehicles which are faster than ours | — | 0.652 | — | 0.652 ± — | 0.652 | 0/371/0 |
| left cars in the same direction of ours | 0.648 | — | — | 0.648 ± — | 0.648 | 215/0/0 |
| left vehicles in the same direction of ours | 0.647 | — | — | 0.647 ± — | 0.647 | 215/0/0 |
| same direction cars in the left | 0.647 | — | — | 0.647 ± — | 0.647 | 215/0/0 |
| same direction vehicles in the left | 0.647 | — | — | 0.647 ± — | 0.647 | 215/0/0 |
| parking vehicles | — | 0.577 | — | 0.577 ± — | 0.577 | 0/2851/0 |
| parking cars | — | 0.577 | — | 0.577 ± — | 0.577 | 0/2851/0 |
| moving pedestrian | — | 0.540 | — | 0.540 ± — | 0.540 | 0/88/0 |
| turning cars | — | 0.433 | — | 0.433 ± — | 0.433 | 0/30/0 |
| turning vehicles | — | 0.431 | — | 0.431 ± — | 0.431 | 0/30/0 |
