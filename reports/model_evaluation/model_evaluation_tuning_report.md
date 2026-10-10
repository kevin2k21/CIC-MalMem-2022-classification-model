# Model Evaluation and Tuning Report

## Outcome

The **Extra Trees** pipeline was evaluated on 11,606 held-out test rows after threshold tuning on the separate validation split.

| Metric | Test score | Stratified bootstrap 95% CI |
| --- | ---: | ---: |
| Accuracy | 1.000000 | [1.000000, 1.000000] |
| Precision | 1.000000 | [1.000000, 1.000000] |
| Recall | 1.000000 | [1.000000, 1.000000] |
| F1 | 1.000000 | [1.000000, 1.000000] |
| F2 | 1.000000 | [1.000000, 1.000000] |
| ROC-AUC | 1.000000 | [1.000000, 1.000000] |

Confusion matrix: `[[5847, 0], [0, 5759]]` (`[[TN, FP], [FN, TP]]`). The tuned decision threshold is **0.50**; the original threshold was **0.50**.
Because there were no observed test classification or ranking errors, the empirical bootstrap intervals collapse to the observed value. They describe resampling uncertainty within this test set, not uncertainty under dataset shift.

## Tuning method

Step 7 searched hyperparameters with stratified cross-validation and selected: `{"classifier__class_weight": null, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 1}`. Step 8 cloned that configured pipeline, fitted it only on training data, and searched thresholds from 0.01 through 0.99 on validation F2. Recall and precision break F2 ties, followed by proximity to 0.50. The chosen pipeline was then refitted on train + validation before the one-time test evaluation.

## Strengths

- Clean test accuracy is **100.0000%**, precision is **100.0000%**, recall is **100.0000%**, and F1 is **100.0000%**.
- Malware misses: **0**; benign false alarms: **0**.
- The estimator keeps median imputation inside its pipeline, so missing inputs are handled consistently at inference.

## Weaknesses and robustness

The weakest synthetic stress case was **gaussian_noise at 10%**, with F2 **0.999687** and recall **0.999653**. These perturbations are sensitivity checks, not substitutes for naturally corrupted or shifted capture data.

The near-perfect random-split result is a warning as well as a strength: related captures, repeated behavioral patterns, or malware-family overlap may occur across partitions. Before deployment, repeat evaluation on family-separated, time-separated, and independently collected data; measure latency and monitor feature drift and class prevalence.

## Reproducible artifacts

- `test_metrics.csv`: final scalar metrics and confidence bounds
- `classification_report.csv`: per-class precision, recall, F1, and support
- `threshold_search.csv`: validation scores for every candidate threshold
- `robustness_results.csv`: clean, noise, and missing-value stress results
- `figures/`: confusion matrix, ROC/PR, threshold, and robustness plots
- `models/malware_detector_tuned.joblib`: final pipeline plus tuned threshold and evaluation metadata

## Reproduce

```bash
.venv/bin/python scripts/model_evaluation_tuning.py
```
