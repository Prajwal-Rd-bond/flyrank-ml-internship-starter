# Model improvement experiment

Latest Colab execution: **96% P@50, 100% P@20, 80% mean development CV P@50**. See [Colab run evidence](colab/README.md). The historical local results below use a different environment and are retained for reproducibility.

## Pinned local run

Exploratory current-state decline detection. Seed-42 holdout was used in earlier work and is not sealed. 90-day inputs overlap label window. No future or causal claim.

Highest unweighted mean P@50 across four GroupKFold splits on the 26 development clients. Six fixed candidates; ties use declared order.

## Development validation

| Model | Mean P@50 | Mean P@20 |
|---|---:|---:|
| forest_reference | 0.765 | 0.738 |
| forest_deeper | 0.785 | 0.750 |
| extra_trees | 0.740 | 0.700 |
| boosting_small | 0.805 | 0.812 |
| boosting_features | 0.780 | 0.812 |
| boosting_deeper | 0.765 | 0.812 |

Fold base rates: 0.490, 0.641, 0.470, 0.619. Individual fold metrics are in the results JSON.

Selected before holdout scoring: **boosting_small**.

## Historical holdout comparison

| Method | P@50 | P@20 | ROC-AUC | Base rate |
|---|---:|---:|---:|---:|
| rule_baseline | 0.360 | 0.400 | 0.620 | 0.391 |
| forest_reference | 0.760 | 0.800 | 0.748 | 0.391 |
| boosting_small | 0.920 | 0.950 | 0.769 | 0.391 |

The reference forest here uses training-only median imputation, missingness flags and category encoding. It is a controlled comparator, not an exact replay of the previous zero-imputed 68% run. Stable ranking resolves ties by dataset order.

Selected model has 4 false positives among its top 50.

Feature additions: activity per active day, relative staleness, characters per word and missing-position flag. No trend labels, comparison-window ingredients or IDs enter the model. Preprocessors fit within each training fold. The rule baseline also fits its distributions and tier medians on training clients only.

Do not replace the published capstone headline merely because this historical split improves. Inspect the development fold results and validate on genuinely new clients and non-overlapping future outcomes before deployment.

Reproduce with the existing requirements-capstone.lock.txt environment and run work/notebooks/model_improvement.ipynb top to bottom. Aggregate evidence is in work/outputs/tuning_results.json.

## Model inspection

Top features by post-selection permutation ROC-AUC drop: days_with_impressions (0.1842), avg_position (0.0406), ctr (0.0209), log_impressions_90d (0.0132), scroll_rate (0.0089). This diagnostic is not used for selection and does not establish causality.

Selected boosting_small uses the original feature list plus imputation missingness flags; the extra activity ratios did not improve validation performance.

Trained pipeline saved locally to work/models/tuned_capstone.joblib (excluded from Git). Feed it the feature frame returned by features(data, expanded=False); it includes fitted imputation and encoding. Only load model binaries from trusted sources.
