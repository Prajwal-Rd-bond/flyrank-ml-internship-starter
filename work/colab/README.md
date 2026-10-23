# Verified Colab execution

In the completed Colab run, the selected gradient-boosting model reached **96% Precision@50** (48/50) and **100% Precision@20** (20/20); mean development cross-validation Precision@50 was **80%**. On the same historical holdout, the training-only forest scored 76% P@50 and the training-only rule baseline 36%. The original capstone forest scored 74% against its separate 42% baseline.

The holdout contains 2,325 pages from six clients (39.1% declining-label base rate). It has been used in earlier experiments and is not sealed; overlapping feature/label windows limit this to exploratory current-state ranking. Colab library versions differ from the pinned local environment, whose scores remain preserved as historical results.

## Model experiment

| Method | P@50 | P@20 | ROC-AUC |
|---|---:|---:|---:|
| rule_baseline | 36% | 40% | 0.620 |
| forest_reference | 76% | 75% | 0.750 |
| boosting_small | 96% | 100% | 0.766 |

## Development validation

| Candidate | Mean P@50 |
|---|---:|
| forest_reference | 77.5% |
| forest_deeper | 78.0% |
| extra_trees | 74.5% |
| boosting_small | 80.0% |
| boosting_features | 79.0% |
| boosting_deeper | 73.0% |

## Environment and evidence

Python 3.13.16; NumPy 2.1.3; pandas 2.2.3; scikit-learn 1.6.1; SciPy 1.16.3.

Both notebooks completed top to bottom in Colab. The text/JSON evidence files contain their displayed aggregate outputs. The `.ipynb` files in this folder are runnable launch copies with cleared outputs, not downloaded execution artifacts. They bundle the helper code and verify the public starter dataset SHA-256. Original local notebooks and JSON keep their earlier environment-specific results.

- [Saved capstone notebook](https://colab.research.google.com/drive/1Up6KJKxQMxdSvdDOK28foMQbpBgGI8q0) (owner access; sharing unchanged)
- [Saved model_improvement notebook](https://colab.research.google.com/drive/1UQMifppUFaNLIIgvqY_yrh0HhmjsKr-z) (owner access; sharing unchanged)
