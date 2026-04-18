# Multi-Sequence Eval: Comparison Across Weights

## What AUC means here
AUC = probability that a randomly chosen GT-matching track gets a higher cosine score than a randomly chosen non-matching track, for a given expression. 0.50 = chance, 1.00 = perfect. < 0.50 means inverted.

**Held-out sequences:** 0005, 0011, 0013

| model_tag | mean_auc_micro | mean_auc_macro ± std | best_seq | worst_seq | max_gap |
|---|---|---|---|---|---|
| v1train_stage1 | 0.779 | 0.838 ± 0.064 | 0005: 0.821 | 0011: 0.779 | 0.042 |
