# Content refresh prioritisation

Prajwal Ravi Devadiga · FlyRank ML Internship

## Measured result

Random Forest Precision@50 is 68%, versus 42% for the baseline (26 percentage points higher). The seed-42 holdout contains 2,325 pages from 6 clients, with a declining-label base rate of 39.1%.

This exploratory study ranks pages by observed impression decline. It does not predict future traffic or demonstrate that refreshing a page improves its performance.

## Question and data

Can content and search signals help an editor prioritise a hypothetical budget of 50 page reviews? Precision@50 measures how many selected pages carry the observed declining label.

The bundled FlyRank sample contains 30,000 retained pages across 32 pseudonymous clients. Keep pages with positive 90-day impressions and content age of at least 90 days; deduplicate content IDs. The target is trend_direction == down.

Direct trend variables, last/previous 30-day ingredients and IDs are excluded from features. IDs are used only for deduplication and grouped splitting. Published outputs are aggregate summaries; no action queue or additional dataset is committed.

## Baseline and models

Baseline score = 0.45 × visibility percentile + 0.35 × staleness percentile + 0.20 × positive CTR-gap percentile. Expected CTR is the median for each position tier among pages with positions above zero and at most 50; unmapped tiers fall back to observed CTR. Nonpositive positions receive zero CTR-gap contribution.

Baseline ranks, CTR medians and categorical vocabulary are computed from the full unlabelled snapshot. This transductive setup is retained to reproduce the original experiment. Model fitting uses training clients only.

The Random Forest uses 200 trees, maximum depth 10, minimum leaf size 25, balanced_subsample weights and random_state=42. Comparators are scaled balanced Logistic Regression (max_iter=1000) and a balanced depth-5 Decision Tree (minimum leaf size 50). Inputs are the 18 numeric and 8 categorical features listed in scripts/ml_utils.py.

## Client-grouped holdout

Seed 42 selects 6 of 32 clients for testing, leaving 27,675 training pages. Assertions verify no client overlap and exclude forbidden features. Models share the same holdout.

These models have been compared using this holdout, so it is development evidence rather than an untouched final test. The table reports observed point estimates, without confidence intervals.

| Method | ROC-AUC | Precision@20 | Precision@50 |
| --- | --- | --- | --- |
| Baseline | 0.646 | 0.400 | 0.420 |
| Logistic Regression | 0.700 | 0.350 | 0.400 |
| Decision Tree | 0.742 | 0.650 | 0.660 |
| Random Forest | 0.747 | 0.700 | 0.680 |

## Actually executed split sensitivity

For each split seed, refit the Random Forest with its model seed fixed at 42. Recompute metrics against that split’s baseline and base rate. These three chosen splits are a sensitivity check, not a confidence interval or evidence that every unseen client will benefit.

| Split seed | Test pages | Base rate | RF P@50 | Baseline P@50 |
| --- | --- | --- | --- | --- |
| 42 | 2325 | 0.391 | 0.680 | 0.420 |
| 7 | 4280 | 0.611 | 0.840 | 0.760 |
| 123 | 3090 | 0.549 | 0.980 | 0.980 |

## Interpretation

In the canonical holdout, 34 of the top 50 pages carry the declining label; 16 do not. This is a measured label count, not a forecast of time saved or traffic recovered.

The following impurity-based feature importances describe the fitted forest. Correlated features and high-cardinality inputs can affect these values; they do not establish causes or the direction of effects.

| Feature | Importance |
| --- | --- |
| log_impressions_90d | 0.1318 |
| days_with_impressions | 0.1305 |
| avg_position | 0.1108 |
| content_age_days | 0.0908 |
| age_tier_365+ | 0.0376 |
| word_count | 0.0370 |
| log_clicks_90d | 0.0366 |
| char_count | 0.0362 |
| days_with_sessions | 0.0348 |
| ctr | 0.0342 |

## Practical use and limitations

The local demonstration queue includes training pages, and its action rules use the observed trend label. It is not validation evidence. Review the trend in the source analytics before deciding whether any content needs work.

The 90-day aggregates overlap the window used to define decline. Client grouping does not remove this temporal overlap. A future-prediction claim requires non-overlapping outcome windows.

Only six clients are held out in the canonical split; client mix can materially change results. Full-snapshot preprocessing also limits claims about completely unseen data.

Next experiment: fit preprocessing only on training clients, evaluate on future non-overlapping outcomes and reserve a new untouched set after model selection. Earlier weekly notebooks and supplied starter outputs are historical experiments, not the canonical capstone results.

## Reproduce and audit

Executed with Python 3.14.0. Exact package versions are recorded in requirements-capstone.lock.txt; input and code hashes are in work/outputs/capstone_model_metrics.json. The notebook generates this page and work/capstone_report.md from that same JSON. Numerical results can change with library versions.

```bash
python3.14 -m venv .venv
.venv/bin/pip install -r requirements-capstone.lock.txt
.venv/bin/python scripts/run_capstone.py
```

## Acknowledgments and data credit

Dataset and starter foundation courtesy of FlyRank. Original starter scripts and teaching materials remain credited to their contributors.

[FlyRank](https://flyrank.ai) · [Executed notebook](notebooks/capstone.ipynb) · [Aggregate metrics](outputs/capstone_model_metrics.json)
