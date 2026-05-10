# Multi-Sequence Eval: v1train_world_xy_seed2

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.756** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.818** ± 0.072 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| cars in front of ours | 0.977 | — | — | 0.977 ± — | 0.977 | 295/0/0 |
| right vehicles which are parking | — | 0.945 | — | 0.945 ± — | 0.945 | 0/1056/0 |
| right cars which are parking | — | 0.945 | — | 0.945 ± — | 0.945 | 0/1056/0 |
| cars which are braking | 0.901 | — | — | 0.901 ± — | 0.901 | 295/0/0 |
| vehicles which are braking | 0.901 | — | — | 0.901 ± — | 0.901 | 295/0/0 |
| left vehicles in the counter direction of ours | 0.799 | 0.945 | — | 0.872 ± 0.073 | 0.886 | 526/1692/0 |
| counter direction vehicles in the left | 0.799 | 0.944 | — | 0.871 ± 0.073 | 0.886 | 526/1692/0 |
| counter direction cars in the left | 0.798 | 0.944 | — | 0.871 ± 0.073 | 0.886 | 526/1692/0 |
| left cars in the counter direction of ours | 0.798 | 0.943 | — | 0.870 ± 0.072 | 0.885 | 526/1692/0 |
| moving right pedestrian | — | — | 0.877 | 0.877 ± — | 0.877 | 0/0/261 |
| moving vehicles | 0.928 | 0.799 | — | 0.864 ± 0.064 | 0.841 | 569/765/0 |
| moving cars | 0.928 | 0.799 | — | 0.864 ± 0.064 | 0.841 | 569/765/0 |
| vehicles in the same direction of ours | 0.765 | 0.974 | — | 0.869 ± 0.104 | 0.798 | 510/371/0 |
| cars in the same direction of ours | 0.765 | 0.974 | — | 0.869 ± 0.104 | 0.798 | 510/371/0 |
| left cars which are parking | 0.574 | 0.898 | — | 0.736 ± 0.162 | 0.784 | 169/1375/0 |
| left vehicles which are parking | 0.574 | 0.898 | — | 0.736 ± 0.162 | 0.784 | 169/1375/0 |
| moving left pedestrian | — | — | 0.783 | 0.783 ± — | 0.783 | 0/0/253 |
| vehicles in horizon direction | — | 0.701 | — | 0.701 ± — | 0.701 | 0/178/0 |
| cars in horizon direction | — | 0.701 | — | 0.701 ± — | 0.701 | 0/178/0 |
| vehicles in the counter direction of ours | 0.783 | 0.614 | — | 0.699 ± 0.084 | 0.668 | 526/1692/0 |
| cars in the counter direction of ours | 0.783 | 0.614 | — | 0.698 ± 0.085 | 0.668 | 526/1692/0 |
| vehicles which are faster than ours | — | 0.648 | — | 0.648 ± — | 0.648 | 0/371/0 |
| cars which are faster than ours | — | 0.648 | — | 0.648 ± — | 0.648 | 0/371/0 |
| same direction vehicles in the left | 0.621 | — | — | 0.621 ± — | 0.621 | 215/0/0 |
| same direction cars in the left | 0.621 | — | — | 0.621 ± — | 0.621 | 215/0/0 |
| left vehicles in the same direction of ours | 0.620 | — | — | 0.620 ± — | 0.620 | 215/0/0 |
| left cars in the same direction of ours | 0.620 | — | — | 0.620 ± — | 0.620 | 215/0/0 |
| parking vehicles | — | 0.576 | — | 0.576 ± — | 0.576 | 0/2851/0 |
| parking cars | — | 0.575 | — | 0.575 ± — | 0.575 | 0/2851/0 |
| turning vehicles | — | 0.538 | — | 0.538 ± — | 0.538 | 0/30/0 |
| turning cars | — | 0.537 | — | 0.537 ± — | 0.537 | 0/30/0 |
| moving pedestrian | — | 0.515 | — | 0.515 ± — | 0.515 | 0/88/0 |
