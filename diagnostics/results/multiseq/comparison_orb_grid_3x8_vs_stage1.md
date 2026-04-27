# Zoned-orb orb 3x8 (61D) vs production stage1 (13D)

**Held-out seqs**: 0005, 0011, 0013
**n expressions**: 33

## TL;DR

Δ = orb 3x8 (61D) − stage1 (13D).  Positive Δ AUC → orb 3x8 zoned-flow concat HELPED, negative → it HURT vs the stage1 (13D) production stage1.

Of 33 expressions, **4 were helped** by orb 3x8 zoned flow (Δ > +0.001), **29 were hurt** (Δ < −0.001), **0 stayed flat**. Overall micro AUC moved from **0.767** (stage1 (13D)) to **0.659** (orb 3x8 (61D)), Δ = -0.107.

**4 expressions had a Cohen's d sign flip** (see Inversions section).

## Summary (overall, pooled across all expressions and seqs)

| metric | stage1 (13D) | orb 3x8 (61D) | Δ (orb 3x8 (61D) − stage1 (13D)) |
|---|---|---|---|
| AUC (micro) | 0.7665 | 0.6593 | -0.1073 |
| separation | 0.2087 | 0.1209 | -0.0878 |
| GT mean | 0.2546 | 0.2281 | -0.0265 |
| non-GT mean | 0.0459 | 0.1071 | +0.0613 |
| Cohen d | 1.0320 | 0.5677 | -0.4643 |

**Sample counts (pooled)**: stage1 (13D) gt=32440, ngt=70777; orb 3x8 (61D) gt=32440, ngt=70777.

## Top 5 expressions where orb 3x8 (61D) HELPED (most positive Δ AUC)

| Expression | stage1 (13D) AUC | orb 3x8 (61D) AUC | Δ AUC | base d | new d | Δ d |
|---|---|---|---|---|---|---|
| moving pedestrian | 0.409 | 0.463 | +0.054 | -0.251 | -0.131 | +0.119 |
| moving left pedestrian | 0.789 | 0.840 | +0.051 | 1.308 | 1.552 | +0.244 |
| parking cars | 0.493 | 0.497 | +0.005 | -0.115 | -0.018 | +0.097 |
| parking vehicles | 0.493 | 0.498 | +0.005 | -0.115 | -0.017 | +0.098 |
| cars in the counter direction of ours | 0.731 | 0.669 | -0.063 | 0.854 | 0.627 | -0.227 |

## Top 5 expressions where orb 3x8 (61D) HURT (most negative Δ AUC)

| Expression | stage1 (13D) AUC | orb 3x8 (61D) AUC | Δ AUC | base d | new d | Δ d |
|---|---|---|---|---|---|---|
| turning vehicles | 0.720 | 0.372 | -0.348 | 0.522 | -0.350 | -0.872 |
| turning cars | 0.720 | 0.372 | -0.348 | 0.523 | -0.349 | -0.873 |
| cars which are braking | 0.991 | 0.698 | -0.294 | 3.857 | 0.698 | -3.159 |
| vehicles which are braking | 0.991 | 0.698 | -0.294 | 3.857 | 0.699 | -3.159 |
| cars which are faster than ours | 0.612 | 0.441 | -0.171 | 0.393 | -0.217 | -0.610 |

## Cohen's d sign flips (inversions)

| Expression | stage1 (13D) d | orb 3x8 (61D) d |
|---|---|---|
| turning vehicles | 0.522 | -0.350 |
| turning cars | 0.523 | -0.349 |
| cars which are faster than ours | 0.393 | -0.217 |
| vehicles which are faster than ours | 0.391 | -0.216 |

## Per-expression delta (sorted by |Δ AUC| descending)

| Expression | stage1 (13D) AUC | orb 3x8 (61D) AUC | Δ AUC | base sep | new sep | Δ sep | base d | new d | Δ d |
|---|---|---|---|---|---|---|---|---|---|
| turning vehicles | 0.720 | 0.372 | -0.348 | 0.076 | -0.064 | -0.140 | 0.522 | -0.350 | -0.872 |
| turning cars | 0.720 | 0.372 | -0.348 | 0.076 | -0.064 | -0.141 | 0.523 | -0.349 | -0.873 |
| cars which are braking | 0.991 | 0.698 | -0.294 | 0.460 | 0.137 | -0.323 | 3.857 | 0.698 | -3.159 |
| vehicles which are braking | 0.991 | 0.698 | -0.294 | 0.460 | 0.137 | -0.323 | 3.857 | 0.699 | -3.159 |
| cars which are faster than ours | 0.612 | 0.441 | -0.171 | 0.087 | -0.043 | -0.129 | 0.393 | -0.217 | -0.610 |
| vehicles which are faster than ours | 0.611 | 0.442 | -0.169 | 0.087 | -0.043 | -0.130 | 0.391 | -0.216 | -0.607 |
| left vehicles in the counter direction of ours | 0.912 | 0.789 | -0.123 | 0.382 | 0.232 | -0.150 | 2.032 | 1.134 | -0.898 |
| left cars in the counter direction of ours | 0.912 | 0.789 | -0.123 | 0.379 | 0.232 | -0.147 | 2.019 | 1.133 | -0.886 |
| counter direction vehicles in the left | 0.912 | 0.789 | -0.122 | 0.377 | 0.232 | -0.145 | 2.013 | 1.136 | -0.877 |
| counter direction cars in the left | 0.911 | 0.790 | -0.121 | 0.377 | 0.233 | -0.144 | 2.009 | 1.138 | -0.871 |
| same direction vehicles in the left | 0.726 | 0.606 | -0.120 | 0.121 | 0.079 | -0.041 | 0.782 | 0.373 | -0.410 |
| left vehicles in the same direction of ours | 0.725 | 0.606 | -0.120 | 0.120 | 0.080 | -0.040 | 0.779 | 0.373 | -0.405 |
| same direction cars in the left | 0.725 | 0.606 | -0.119 | 0.120 | 0.080 | -0.040 | 0.778 | 0.373 | -0.405 |
| left cars in the same direction of ours | 0.725 | 0.606 | -0.119 | 0.119 | 0.080 | -0.040 | 0.777 | 0.373 | -0.404 |
| left vehicles which are parking | 0.782 | 0.675 | -0.107 | 0.250 | 0.129 | -0.121 | 1.244 | 0.637 | -0.606 |
| left cars which are parking | 0.782 | 0.675 | -0.107 | 0.248 | 0.129 | -0.119 | 1.239 | 0.637 | -0.602 |
| moving right pedestrian | 0.846 | 0.740 | -0.107 | 0.193 | 0.114 | -0.079 | 1.457 | 0.875 | -0.582 |
| moving cars | 0.796 | 0.689 | -0.106 | 0.185 | 0.109 | -0.076 | 1.116 | 0.651 | -0.465 |
| moving vehicles | 0.796 | 0.689 | -0.106 | 0.185 | 0.109 | -0.075 | 1.115 | 0.651 | -0.464 |
| vehicles in the same direction of ours | 0.800 | 0.698 | -0.102 | 0.096 | 0.087 | -0.009 | 0.729 | 0.528 | -0.201 |
| cars in the same direction of ours | 0.800 | 0.698 | -0.101 | 0.096 | 0.087 | -0.009 | 0.727 | 0.528 | -0.200 |
| vehicles in horizon direction | 0.719 | 0.617 | -0.101 | 0.130 | 0.060 | -0.071 | 0.769 | 0.386 | -0.383 |
| cars in horizon direction | 0.718 | 0.618 | -0.100 | 0.129 | 0.060 | -0.070 | 0.765 | 0.386 | -0.379 |
| cars in front of ours | 0.982 | 0.904 | -0.077 | 0.469 | 0.358 | -0.112 | 3.631 | 2.164 | -1.467 |
| vehicles in front of ours | 0.981 | 0.904 | -0.077 | 0.469 | 0.358 | -0.111 | 3.627 | 2.164 | -1.463 |
| right cars which are parking | 0.937 | 0.866 | -0.071 | 0.411 | 0.350 | -0.061 | 2.687 | 1.701 | -0.987 |
| right vehicles which are parking | 0.937 | 0.867 | -0.071 | 0.411 | 0.350 | -0.061 | 2.687 | 1.702 | -0.985 |
| vehicles in the counter direction of ours | 0.732 | 0.669 | -0.063 | 0.155 | 0.127 | -0.028 | 0.855 | 0.627 | -0.229 |
| cars in the counter direction of ours | 0.731 | 0.669 | -0.063 | 0.155 | 0.127 | -0.028 | 0.854 | 0.627 | -0.227 |
| moving pedestrian | 0.409 | 0.463 | +0.054 | -0.043 | -0.024 | +0.019 | -0.251 | -0.131 | +0.119 |
| moving left pedestrian | 0.789 | 0.840 | +0.051 | 0.262 | 0.304 | +0.042 | 1.308 | 1.552 | +0.244 |
| parking cars | 0.493 | 0.497 | +0.005 | -0.020 | -0.004 | +0.016 | -0.115 | -0.018 | +0.097 |
| parking vehicles | 0.493 | 0.498 | +0.005 | -0.020 | -0.004 | +0.016 | -0.115 | -0.017 | +0.098 |

## Per-expression GT vs non-GT means

| Expression | base GT μ | new GT μ | Δ GT μ | base ngt μ | new ngt μ | Δ ngt μ | n_gt | n_ngt |
|---|---|---|---|---|---|---|---|---|
| turning vehicles | 0.118 | 0.064 | -0.054 | 0.042 | 0.128 | +0.086 | 30 | 3371 |
| turning cars | 0.118 | 0.064 | -0.055 | 0.042 | 0.128 | +0.086 | 30 | 3371 |
| cars which are braking | 0.398 | 0.177 | -0.221 | -0.061 | 0.040 | +0.102 | 295 | 868 |
| vehicles which are braking | 0.398 | 0.177 | -0.221 | -0.061 | 0.041 | +0.102 | 295 | 868 |
| cars which are faster than ours | 0.175 | 0.073 | -0.103 | 0.088 | 0.115 | +0.027 | 371 | 3030 |
| vehicles which are faster than ours | 0.177 | 0.077 | -0.100 | 0.090 | 0.120 | +0.029 | 371 | 3030 |
| left vehicles in the counter direction of ours | 0.222 | 0.201 | -0.021 | -0.160 | -0.031 | +0.129 | 2218 | 2346 |
| left cars in the counter direction of ours | 0.222 | 0.201 | -0.021 | -0.157 | -0.031 | +0.126 | 2218 | 2346 |
| counter direction vehicles in the left | 0.221 | 0.201 | -0.020 | -0.157 | -0.031 | +0.125 | 2218 | 2346 |
| counter direction cars in the left | 0.221 | 0.201 | -0.019 | -0.156 | -0.031 | +0.125 | 2218 | 2346 |
| same direction vehicles in the left | 0.174 | 0.193 | +0.019 | 0.053 | 0.114 | +0.060 | 215 | 948 |
| left vehicles in the same direction of ours | 0.174 | 0.194 | +0.020 | 0.054 | 0.114 | +0.060 | 215 | 948 |
| same direction cars in the left | 0.174 | 0.194 | +0.020 | 0.054 | 0.114 | +0.060 | 215 | 948 |
| left cars in the same direction of ours | 0.174 | 0.194 | +0.020 | 0.054 | 0.114 | +0.060 | 215 | 948 |
| left vehicles which are parking | 0.268 | 0.225 | -0.044 | 0.018 | 0.096 | +0.077 | 1544 | 3020 |
| left cars which are parking | 0.268 | 0.225 | -0.044 | 0.020 | 0.096 | +0.076 | 1544 | 3020 |
| moving right pedestrian | 0.007 | 0.022 | +0.015 | -0.186 | -0.092 | +0.094 | 261 | 606 |
| moving cars | 0.327 | 0.345 | +0.018 | 0.142 | 0.236 | +0.094 | 1334 | 3230 |
| moving vehicles | 0.327 | 0.345 | +0.018 | 0.142 | 0.236 | +0.094 | 1334 | 3230 |
| vehicles in the same direction of ours | 0.342 | 0.343 | +0.001 | 0.246 | 0.255 | +0.010 | 881 | 3683 |
| cars in the same direction of ours | 0.341 | 0.343 | +0.001 | 0.246 | 0.255 | +0.010 | 881 | 3683 |
| vehicles in horizon direction | 0.140 | 0.150 | +0.010 | 0.010 | 0.090 | +0.081 | 178 | 3223 |
| cars in horizon direction | 0.139 | 0.150 | +0.011 | 0.010 | 0.090 | +0.080 | 178 | 3223 |
| cars in front of ours | 0.471 | 0.451 | -0.020 | 0.001 | 0.093 | +0.092 | 295 | 868 |
| vehicles in front of ours | 0.471 | 0.451 | -0.020 | 0.002 | 0.093 | +0.091 | 295 | 868 |
| right cars which are parking | 0.392 | 0.324 | -0.067 | -0.019 | -0.025 | -0.006 | 1056 | 2345 |
| right vehicles which are parking | 0.392 | 0.324 | -0.067 | -0.020 | -0.026 | -0.006 | 1056 | 2345 |
| vehicles in the counter direction of ours | 0.216 | 0.194 | -0.022 | 0.061 | 0.066 | +0.006 | 2218 | 2346 |
| cars in the counter direction of ours | 0.217 | 0.194 | -0.023 | 0.062 | 0.066 | +0.005 | 2218 | 2346 |
| moving pedestrian | 0.134 | 0.204 | +0.069 | 0.177 | 0.228 | +0.050 | 88 | 3313 |
| moving left pedestrian | 0.177 | 0.240 | +0.063 | -0.085 | -0.064 | +0.021 | 253 | 614 |
| parking cars | 0.228 | 0.196 | -0.032 | 0.248 | 0.200 | -0.048 | 2851 | 550 |
| parking vehicles | 0.227 | 0.196 | -0.031 | 0.247 | 0.200 | -0.048 | 2851 | 550 |

