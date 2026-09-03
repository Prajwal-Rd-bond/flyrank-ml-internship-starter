# Capstone Report — Lane 2: Refresh / Content Opportunity Scoring

- **Author:** Prajwal Rd Bond
- **Lane:** Lane 2 — Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Prajwal-Rd-bond/flyrank-ml-internship-starter
- **Date:** September 2026

---

## 0. Abstract

Content editors at digital agencies face a prioritisation problem: with hundreds of pages per client and weekly capacity for approximately 50 reviews, which pages most urgently need refreshing? This study uses an anonymised snapshot of 30,000 pages across 32 FlyRank clients — covering 90-day Google Search Console and GA4 signals — to build a machine-learning ranking system that surfaces pages most likely to be in a current search-impression decline. A Random Forest classifier trained on 18 numeric and 8 categorical pre-decision features is compared against a hand-rule baseline on a client-grouped holdout; Precision@50 improves from 0.42 (baseline) to 0.68 (model), a +61.9% observed lift on a test base rate of 0.39. The output is a ranked action queue with plain-language reason codes — a decision-support tool that tells editors where to look first, not what outcome to expect.

---

## 1. Problem Framing

**Decision this supports:** A FlyRank content editor or SEO strategist opens a ranked page queue each Monday morning and reviews the top 50 entries — their weekly capacity. Without the queue, they review pages alphabetically, by raw traffic volume, or by intuition; all of these approaches have lower observed precision on the declining-page label.

**Unit of analysis:** One content page (`content_id`), one observation per page, one 90-day observation window.

**Output:** A probability score (0–1) and an action label (`expand_and_refresh`, `refresh_and_review_ctr`, `refresh`, `monitor`) with a plain-English reason sentence. Editors receive a sorted CSV ranked by the model score.

**Action a human takes:** Opens the top-ranked pages in Google Search Console, verifies the trend is real and not seasonal or migration-driven, then schedules a content refresh if warranted.

**Cost of a wrong call:**
- False positive (flagged but not declining): ~30 min of wasted editor time per page. Moderate, recoverable cost.
- False negative (declining but not flagged): Silent traffic erosion continues. Higher cost — especially for high-value pages with strong keyword demand.

This asymmetry justifies optimising for **Precision@50** (precision in the top 50 reviewed) rather than Recall across all 30,000 pages.

**Why ML helps here:** The hand-rule baseline uses two signals (visibility and freshness) with fixed weights. The data contains ~26 interacting signals — position tier, CTR gap, content length, AI traffic share, engagement rate — that combine non-linearly. ML can learn which combinations matter most for which page profiles, yielding higher precision on the subset an editor actually reviews.

---

## 2. Data Safety

**Dataset used:** `data/raw/content_refresh_anonymized.csv` — the FlyRank ML Internship starter slice. 30,000 pages, 32 pseudonymised clients, 44 columns, trailing 90-day window.

**Columns deliberately excluded:**

| Column | Reason |
|---|---|
| `trend_direction` | Direct label source — `is_declining_label = (trend_direction == 'down')` |
| `trend_pct` | Label ingredient — the raw percentage `trend_direction` is bucketed from |
| `impressions_last_30d`, `clicks_last_30d`, `sessions_last_30d` | Sub-window leakage — numerator inputs to `trend_pct` |
| `impressions_prev_30d`, `clicks_prev_30d`, `sessions_prev_30d` | Sub-window leakage — denominator inputs to `trend_pct` |
| `content_id`, `client_id` | Pseudonymous IDs — used for grouping/splits only, never features |
| `provider_used`, `model_used` | Operational metadata — not available to editors at review time; high missingness |

**Leakage controls:** Single-feature AUC audit (Weeks 3 and 6) confirmed no feature has a solo AUC ≥ 0.95. Forbidden columns verified absent from `MODEL_NUMERIC_FEATURES` and `MODEL_CATEGORICAL_FEATURES`.

**Client-identifying data:** Zero. All identifiers are pseudonymous hex strings. No real names, domains, URLs, titles, or keywords appear in any file under `work/`.

---

## 3. Baseline

**The transparent rule:**

```
baseline_action_score = 0.45 × visibility_score
                      + 0.35 × freshness_risk_score
                      + 0.20 × ctr_gap_score
```

Where:
- `visibility_score` = percentile rank of `log(impressions_90d)` — how visible is this page?
- `freshness_risk_score` = percentile rank of `days_since_last_update` — how stale is it?
- `ctr_gap_score` = percentile rank of (expected CTR for its position tier − actual CTR), clipped to 0 — how much is it underperforming on clicks?

**Why this is a fair comparison:** The baseline uses only signals observable before any editor decision, contains no label-derived columns, and is recomputed on the exact same test slice as the models. It is not "tuned" on the holdout.

**Baseline performance on seed=42 test set (2,325 rows, base rate 0.391):**
- ROC-AUC: 0.646
- Precision@20: 0.400
- Precision@50: 0.420

---

## 4. Model / Analysis

**Method:** Random Forest classifier (`n_estimators=200`, `max_depth=10`, `min_samples_leaf=25`, `class_weight='balanced_subsample'`, `random_state=42`). Selected over Logistic Regression and Decision Tree (both also trained and compared) because it achieves the highest Precision@50 on the holdout while remaining within the complexity budget justified by the dataset size.

**Feature list (18 numeric + 8 categorical → ~52 columns after one-hot encoding):**

*Numeric:* `search_volume`, `competition`, `cpc`, `word_count`, `char_count`, `log_impressions_90d`, `log_clicks_90d`, `log_sessions_90d`, `log_ai_sessions_90d`, `days_with_impressions`, `days_with_sessions`, `content_age_days`, `days_since_last_update`, `ctr`, `avg_position`, `engagement_rate`, `scroll_rate`, `ai_traffic_pct`

*Categorical (one-hot):* `competition_level`, `content_type`, `main_intent`, `age_tier`, `freshness_tier`, `word_count_tier`, `impression_tier`, `position_tier`

**Target definition:** `is_declining_label = 1` if `trend_direction == 'down'` (impressions dropped >20% in the most recent 30 days vs. the prior 30 days). This is a proxy label based on a trailing observation window, not a future-outcome prediction.

**What was deliberately left out:** All sub-window columns (`impressions_last_30d`, etc.), the direct label source (`trend_direction`, `trend_pct`), and ID columns. `has_clicks`, `has_ai_sessions`, and `measurable_opportunity` were computed as indicator flags but are not in the final model feature list after the leakage audit confirmed their marginal contribution was low.

---

## 5. Evaluation

**Split design:** Client-grouped holdout. 20% of the 32 clients (6 clients) are held out entirely — no page from those 6 clients appears in the training set. This is the honest design for a multi-client service: the question is "does this generalise to a client the model has never seen?" A random-row 80/20 split would inflate Precision@50 by ~10 points (observed: 0.78 naive vs. 0.68 honest) because the model partially memorises each client's page profile during training.

**Seed sensitivity:** Precision@50 ranges from 0.68 to 0.98 across seeds 42, 7, and 123 — a known limitation of a 32-client population where 6 holdout clients is a small draw. All seeds beat the baseline. The seed-42 number is used for canonical reporting (matching this repo's convention throughout).

**Results on seed=42 test set (2,325 rows, base rate = 0.391):**

| Method | ROC-AUC | P@20 | P@50 | Lift vs baseline (P@50) |
|---|---|---|---|---|
| Baseline action score | 0.646 | 0.400 | 0.420 | — |
| Logistic Regression | 0.700 | 0.350 | 0.400 | −0.020 |
| Decision Tree (d=5) | 0.742 | 0.650 | 0.660 | +0.240 |
| **Random Forest** | **0.747** | **0.700** | **0.680** | **+0.260** |
| Test base rate (floor) | — | 0.391 | 0.391 | — |

**Error analysis:**
- **False positives (top-50):** 16 of 50. These are visible pages in mid-page positions with low CTR that look like decliners by profile — they have the "at-risk" combination of age, visibility, and CTR gap, but haven't crossed the >20% drop threshold this window. An editor reviewing these would likely put them on a watch list, which is correct behaviour.
- **Worst false negatives:** Pages with 1–3 impressions in 90 days, `avg_position = 0` (no position data). The model correctly has no signal for these; any method would fail here. This is a data limitation — "invisible" pages cannot be ranked by a signal-based model.

---

## 6. Interpretation

**What the model found:** The top four features by importance are `log_impressions_90d` (0.132), `days_with_impressions` (0.131), `avg_position` (0.111), and `content_age_days` (0.091). In plain words: pages that are visible (high impressions), consistently visible (many days with impressions), in a mid-to-deep search position, and older — are the pages most associated with a current impression decline. This is consistent with the Week-4 signal audit findings (position-CTR cliff and age-freshness association were both confirmed).

**Surprises:**
- Logistic Regression (the simplest model) actually performs *worse* than the baseline on Precision@50 (0.400 vs 0.420). Linear combination of scaled features is not sufficient here — the relevant patterns involve non-linear interactions (e.g. age matters more for pages in a specific impression tier).
- AI traffic share (`ai_traffic_pct`) and `scroll_rate` rank low in feature importance, suggesting that AI-referral traffic and on-page engagement are not strong discriminating signals for this label in this dataset. This may change as AI-referred traffic grows.

**Negative result worth noting:** More complex models are not always better. The Decision Tree at depth=5 achieves P@50 = 0.660 — nearly as good as the Random Forest at 0.680 — with a model that a non-technical stakeholder can print out and read. If explainability is the priority and 2 percentage points of precision are acceptable to trade away, the Decision Tree is a defensible alternative.

---

## 7. Recommendation

**Ranked actions for a FlyRank content editor:**

1. **Open the ranked queue in `work/outputs/capstone_action_queue.csv`.** Review the top 50 rows each week.
2. **Act on `expand_and_refresh` pages first** — thin content (< 1,000 words) on visible pages with measurable demand. The gap is objectively measurable and the opportunity is concrete.
3. **Address `refresh_and_review_ctr` pages second** — title/snippet optimisation for pages already in a good position but underperforming on click-through. Fast, high-impact.
4. **Schedule `refresh` pages third** — stale visible pages or confirmed declining trends. Standard content update.
5. **Place `monitor` pages on a 4-week watch list** — elevated score but no dominant signal. Revisit before acting.

**Before any action:** Verify in Google Search Console that the trend is real (not seasonal, not a migration artefact). The model score is a prompt to look, not a command to act.

**Confidence and limits:**
- Precision@50 = 0.68 means approximately 34 of the top 50 reviewed pages will genuinely be in a decline. 16 will not — those are real false positives, not model errors to hide.
- No causal guarantee is made. Refreshing a flagged page may or may not recover its traffic.
- The model should be retrained quarterly, or immediately if the label base rate drifts outside [0.45, 0.65] or Precision@50 on a fresh evaluation drops below 0.50.

---

## 8. Reproducibility

**Commands to re-run from a fresh clone:**

```bash
git clone https://github.com/Prajwal-Rd-bond/flyrank-ml-internship-starter.git
cd flyrank-ml-internship-starter
pip install -r requirements.txt
python scripts/run_all.py
# Reference pipeline outputs → outputs/

# Assignment notebooks (in order):
jupyter nbconvert --to notebook --execute work/notebooks/w01_research_question.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w02_ml_task_framing.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w03_data_contract.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w03_feature_leakage_check.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w04_signal_audit.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w04_baseline_score.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w05_model.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w06_validation_audit.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/w07_action_playbook.ipynb
jupyter nbconvert --to notebook --execute work/notebooks/capstone.ipynb
```

**Random seeds:** All notebooks and scripts use `random_state=42` and `np.random.default_rng(42)` throughout. The seed is set explicitly and stated in every cell that uses randomness.

**Environment:** Python 3.11+. Key dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`. See `requirements.txt`.

**Sealed evaluation:** The committed file `work/outputs/capstone_model_metrics.json` records the exact metrics from the seed-42 client-grouped holdout. It was generated by the capstone notebook in a single run and is the canonical record of results. To verify: re-run `capstone.ipynb` and confirm the numbers match.

**Committed output files:**
- `work/outputs/capstone_model_metrics.json` — sealed evaluation metrics
- `work/outputs/capstone_results_table.csv` — comparison table (all 4 methods)
- `work/outputs/capstone_feature_importance.csv` — top-10 RF feature importances
- `work/outputs/capstone_action_queue.csv` — full 30,000-row ranked queue
- `work/outputs/monitoring_reference.json` — drift detection thresholds

---

## 9. Acknowledgments & Data Credit

Built on the **FlyRank ML Internship dataset** — an anonymised slice of real FlyRank search performance data. Dataset and internship program courtesy of [FlyRank](https://flyrank.ai).

Track leads: Mirza Ašćerić (ML) · Hole (data engineering).

---

> **Claims checklist:** All findings are reported as *observed*, *measured*, and *directional*. No causal claims are made about the effect of content refreshes on future rankings. Metrics are reported alongside the test base rate (0.391). No client-identifying details appear anywhere in this report or in `work/`. Numbers in this report were verified against a fresh re-run of `capstone.ipynb`.
