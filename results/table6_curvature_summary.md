# Table VI: Kendall's W vs. curvature-sharpness parameter k (n=7 optimizers)

| k | W | 95% CI |
|---|---|---|
| 0.000 | 0.280 | [0.057, 0.432] |
| 0.588 | 0.662 | [0.170, 0.860] |
| 1.177 | 0.523 | [0.003, 0.827] |
| 1.765 | 0.587 | [0.013, 0.857] |
| 2.353 | 0.621 | [0.040, 0.862] |
| 2.941 | 0.612 | [0.040, 0.893] |
| 3.529 | 0.768 | [0.263, 0.946] |
| 4.118 | 0.782 | [0.271, 0.964] |
| 4.706 | 0.765 | [0.268, 0.946] |
| 5.294 | 0.587 | [0.062, 0.911] |
| 5.882 | 0.587 | [0.080, 0.912] |
| 6.471 | 0.587 | [0.062, 0.911] |
| 7.059 | 0.590 | [0.089, 0.924] |
| 7.647 | 0.587 | [0.080, 0.911] |
| 8.235 | 0.440 | [0.019, 0.804] |
| 8.823 | 0.453 | [0.030, 0.780] |
| 9.412 | 0.453 | [0.027, 0.780] |
| 10.000 | 0.452 | [0.027, 0.786] |

Spearman rho(k, W) = -0.361, n=18 k-values, 95% bootstrap CI = [-0.820, +0.257] (B=2000 resamples).

**This CI is wide primarily because each individual W(k) estimate is itself noisy at n_items=7 optimizers (see per-k CIs above, each ~0.35-0.5 units wide), not because the k-sweep (n=18 conditions) is too coarse** -- see SANITY_CHECKS.md Check 2 for detail. Read as underpowered-to-detect-a-relationship, not as evidence of no relationship.
