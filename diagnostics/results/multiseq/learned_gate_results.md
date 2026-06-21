# Learned posthoc state-aware gate — V1 holdout sep

Stage1 weights: `gmc_link_weights_v1train_stage1.pth` (frozen)
Gate weights: `learned_gate_v1train.pt`
Holdout seqs: 0005, 0011, 0013
Analytical baseline: alpha=0.5, sigma=4.0

## Headline

Pooled = single mean across every (track, frame, expr) row.
Macro  = mean of per-expression seps (each expression weighted equally).

| metric | raw | analytical | learned | Δ lrn vs raw | Δ lrn vs ana |
|---|---|---|---|---|---|
| pooled sep | +0.2087 | +0.3350 | +0.2011 | -0.0076 | -0.1338 |
| macro sep  | +0.2122 | +0.2297 | +0.2475 | +0.0354 | +0.0178 |

Per-expression win counts (out of 33): learned > analytical in **15**; learned > raw in **19**.

**Verdict (spec criterion = pooled sep): FAIL**

Pass criterion (spec): learned pooled sep > analytical pooled sep AND braking sep >= +0.350.

## Canary checks

| canary | n_exprs | raw_sep | analytical_sep | learned_sep | target |
|---|---|---|---|---|---|
| braking | 2 | +0.460 | +0.158 | +0.629 | >= +0.350 |
| parking_cars_or_vehicles | 2 | -0.020 | +0.162 | +0.077 | >= +0.100 |
| turning_cars_or_vehicles | 2 | +0.076 | +0.343 | +0.048 | >= +0.200 |
| moving_pedestrian | 1 | -0.043 | +0.160 | +0.131 | >= +0.050 |
| cars_in_front_of_ours | 1 | +0.469 | +0.469 | +0.569 | >= +0.400 |

## Per-expression (sorted by Δ learned vs analytical)

| Expression | class | n_gt | n_ngt | raw_sep | analytical_sep | learned_sep | Δ vs analytical | Δ vs raw |
|---|---|---|---|---|---|---|---|---|
| vehicles which are braking | motion | 295 | 868 | +0.460 | +0.158 | +0.630 | +0.472 | +0.171 |
| cars which are braking | motion | 295 | 868 | +0.460 | +0.158 | +0.628 | +0.470 | +0.168 |
| right vehicles which are parking | static | 1,056 | 2,345 | +0.411 | +0.345 | +0.519 | +0.174 | +0.108 |
| right cars which are parking | static | 1,056 | 2,345 | +0.411 | +0.345 | +0.505 | +0.160 | +0.094 |
| cars in front of ours | appearance | 295 | 868 | +0.469 | +0.469 | +0.569 | +0.099 | +0.099 |
| vehicles in front of ours | appearance | 295 | 868 | +0.469 | +0.469 | +0.554 | +0.085 | +0.085 |
| moving left pedestrian | motion | 253 | 614 | +0.262 | +0.301 | +0.363 | +0.062 | +0.101 |
| left cars in the counter direction of ours | appearance | 2,218 | 2,346 | +0.379 | +0.379 | +0.429 | +0.051 | +0.051 |
| moving right pedestrian | motion | 261 | 606 | +0.193 | +0.181 | +0.231 | +0.050 | +0.038 |
| left vehicles in the counter direction of ours | appearance | 2,218 | 2,346 | +0.382 | +0.382 | +0.426 | +0.044 | +0.044 |
| left cars which are parking | static | 1,544 | 3,020 | +0.248 | +0.367 | +0.408 | +0.040 | +0.159 |
| left vehicles which are parking | static | 1,544 | 3,020 | +0.250 | +0.369 | +0.406 | +0.037 | +0.156 |
| cars in horizon direction | appearance | 178 | 3,223 | +0.129 | +0.129 | +0.143 | +0.014 | +0.014 |
| counter direction vehicles in the left | appearance | 2,218 | 2,346 | +0.377 | +0.377 | +0.391 | +0.013 | +0.013 |
| counter direction cars in the left | appearance | 2,218 | 2,346 | +0.377 | +0.377 | +0.388 | +0.012 | +0.012 |
| vehicles in horizon direction | appearance | 178 | 3,223 | +0.130 | +0.130 | +0.125 | -0.005 | -0.005 |
| vehicles in the same direction of ours | appearance | 881 | 3,683 | +0.096 | +0.096 | +0.084 | -0.012 | -0.012 |
| cars in the same direction of ours | appearance | 881 | 3,683 | +0.096 | +0.096 | +0.080 | -0.015 | -0.015 |
| cars in the counter direction of ours | appearance | 2,218 | 2,346 | +0.155 | +0.155 | +0.137 | -0.017 | -0.017 |
| vehicles in the counter direction of ours | appearance | 2,218 | 2,346 | +0.155 | +0.155 | +0.137 | -0.019 | -0.019 |
| left cars in the same direction of ours | appearance | 215 | 948 | +0.119 | +0.119 | +0.100 | -0.020 | -0.020 |
| left vehicles in the same direction of ours | appearance | 215 | 948 | +0.120 | +0.120 | +0.100 | -0.020 | -0.020 |
| cars which are faster than ours | motion | 371 | 3,030 | +0.087 | +0.054 | +0.032 | -0.022 | -0.055 |
| same direction cars in the left | appearance | 215 | 948 | +0.120 | +0.120 | +0.091 | -0.029 | -0.029 |
| moving pedestrian | motion | 88 | 3,313 | -0.043 | +0.160 | +0.131 | -0.029 | +0.174 |
| same direction vehicles in the left | appearance | 215 | 948 | +0.121 | +0.121 | +0.088 | -0.032 | -0.032 |
| vehicles which are faster than ours | motion | 371 | 3,030 | +0.087 | +0.054 | +0.008 | -0.046 | -0.079 |
| moving cars | motion | 1,334 | 3,230 | +0.185 | +0.193 | +0.112 | -0.081 | -0.073 |
| parking cars | static | 2,851 | 550 | -0.020 | +0.162 | +0.082 | -0.081 | +0.101 |
| moving vehicles | motion | 1,334 | 3,230 | +0.185 | +0.193 | +0.106 | -0.087 | -0.079 |
| parking vehicles | static | 2,851 | 550 | -0.020 | +0.162 | +0.073 | -0.089 | +0.093 |
| turning cars | motion | 30 | 3,371 | +0.076 | +0.343 | +0.095 | -0.248 | +0.019 |
| turning vehicles | motion | 30 | 3,371 | +0.076 | +0.343 | +0.000 | -0.342 | -0.076 |
