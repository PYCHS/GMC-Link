# Multi-Sequence Eval: v1train_world_xy_seed1

## What this measures
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

## Headline
- Mean AUC (micro, pooled across 3 seqs): **0.759** (over 33 expressions)
- Mean AUC (macro, per-seq averaged):     **0.829** ± 0.069 (over 12 expressions in ≥2 seqs)
- Seq-0011 only (legacy, for continuity): **—**

## Per-expression breakdown
| Expression | 0005 | 0011 | 0013 | macro μ ± σ | micro | GT counts |
|---|---|---|---|---|---|---|
| vehicles in front of ours | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| cars in front of ours | 0.973 | — | — | 0.973 ± — | 0.973 | 295/0/0 |
| right vehicles which are parking | — | 0.941 | — | 0.941 ± — | 0.941 | 0/1056/0 |
| right cars which are parking | — | 0.941 | — | 0.941 ± — | 0.941 | 0/1056/0 |
| cars which are braking | 0.914 | — | — | 0.914 ± — | 0.914 | 295/0/0 |
| vehicles which are braking | 0.914 | — | — | 0.914 ± — | 0.914 | 295/0/0 |
| left cars in the counter direction of ours | 0.815 | 0.959 | — | 0.887 ± 0.072 | 0.893 | 526/1692/0 |
| left vehicles in the counter direction of ours | 0.815 | 0.959 | — | 0.887 ± 0.072 | 0.893 | 526/1692/0 |
| counter direction cars in the left | 0.815 | 0.957 | — | 0.886 ± 0.071 | 0.892 | 526/1692/0 |
| counter direction vehicles in the left | 0.815 | 0.957 | — | 0.886 ± 0.071 | 0.892 | 526/1692/0 |
| moving vehicles | 0.934 | 0.812 | — | 0.873 ± 0.061 | 0.849 | 569/765/0 |
| moving cars | 0.934 | 0.811 | — | 0.873 ± 0.061 | 0.849 | 569/765/0 |
| moving right pedestrian | — | — | 0.829 | 0.829 ± — | 0.829 | 0/0/261 |
| left cars which are parking | 0.584 | 0.914 | — | 0.749 ± 0.165 | 0.800 | 169/1375/0 |
| left vehicles which are parking | 0.584 | 0.914 | — | 0.749 ± 0.165 | 0.800 | 169/1375/0 |
| moving left pedestrian | — | — | 0.789 | 0.789 ± — | 0.789 | 0/0/253 |
| vehicles in the same direction of ours | 0.745 | 0.979 | — | 0.862 ± 0.117 | 0.788 | 510/371/0 |
| cars in the same direction of ours | 0.745 | 0.979 | — | 0.862 ± 0.117 | 0.787 | 510/371/0 |
| cars in horizon direction | — | 0.696 | — | 0.696 ± — | 0.696 | 0/178/0 |
| vehicles in horizon direction | — | 0.696 | — | 0.696 ± — | 0.696 | 0/178/0 |
| vehicles in the counter direction of ours | 0.791 | 0.639 | — | 0.715 ± 0.076 | 0.683 | 526/1692/0 |
| cars in the counter direction of ours | 0.791 | 0.638 | — | 0.715 ± 0.076 | 0.683 | 526/1692/0 |
| turning vehicles | — | 0.637 | — | 0.637 ± — | 0.637 | 0/30/0 |
| turning cars | — | 0.636 | — | 0.636 ± — | 0.636 | 0/30/0 |
| vehicles which are faster than ours | — | 0.614 | — | 0.614 ± — | 0.614 | 0/371/0 |
| same direction cars in the left | 0.610 | — | — | 0.610 ± — | 0.610 | 215/0/0 |
| same direction vehicles in the left | 0.610 | — | — | 0.610 ± — | 0.610 | 215/0/0 |
| left cars in the same direction of ours | 0.610 | — | — | 0.610 ± — | 0.610 | 215/0/0 |
| left vehicles in the same direction of ours | 0.610 | — | — | 0.610 ± — | 0.610 | 215/0/0 |
| cars which are faster than ours | — | 0.610 | — | 0.610 ± — | 0.610 | 0/371/0 |
| parking vehicles | — | 0.581 | — | 0.581 ± — | 0.581 | 0/2851/0 |
| parking cars | — | 0.580 | — | 0.580 ± — | 0.580 | 0/2851/0 |
| moving pedestrian | — | 0.478 | — | 0.478 ± — | 0.478 | 0/88/0 |
