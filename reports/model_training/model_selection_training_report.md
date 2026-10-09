# Model Selection and Training Report

## Outcome

**Extra Trees** was selected after training-only stratified 3-fold cross-validation and validation-set comparison. It was then refitted on the combined train + validation data and evaluated once on the untouched test set.

- Test accuracy: **100.0000%**
- Test malware recall: **100.0000%**
- Test precision: **100.0000%**
- Test F2 score: **100.0000%**
- Test ROC-AUC: **1.000000**
- Test confusion matrix: `[[5847, 0], [0, 5759]]` (`[[TN, FP], [FN, TP]]`)

## Model-selection decision

F2 is the primary metric because a missed malware sample (false negative) is more costly than an additional investigation, while F2 still penalizes excessive false alarms. ROC-AUC, precision, runtime, interpretability, and scalability are retained as decision evidence. The final winner is the candidate with the highest validation F2; validation ROC-AUC and lower fit time break ties.

| Model | CV F2 | CV recall | Validation F2 | Validation recall | Validation precision | Validation ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Extra Trees | 0.99999 | 1.00000 | 0.99996 | 1.00000 | 0.99978 | 1.00000 |
| Histogram Gradient Boosting | 0.99967 | 0.99962 | 0.99996 | 1.00000 | 0.99978 | 1.00000 |
| Random Forest | 0.99985 | 0.99984 | 0.99996 | 1.00000 | 0.99978 | 1.00000 |
| Logistic Regression | 0.99900 | 0.99897 | 0.99905 | 0.99913 | 0.99870 | 0.99974 |

The candidates cover an interpretable linear baseline, two robust nonlinear tree ensembles, and gradient boosting. Each candidate received a compact hyperparameter search rather than a single default fit.

## Final training

- Training rows used for model search: 37,136
- Validation rows used for final selection: 9,285
- Rows used in final refit: 46,421
- Untouched test rows: 11,606
- Selected predictors: 25
- Decision threshold: 0.5
- Best parameters: `{"classifier__class_weight": null, "classifier__max_features": "sqrt", "classifier__min_samples_leaf": 1}`
- Serialized model bundle: `models/malware_detector.joblib`

The saved bundle contains the fitted pipeline, exact ordered feature list, threshold, class meaning, training metadata, and input hashes. Median imputation (and scaling where required) remains inside the pipeline, so preprocessing learned during training is applied consistently at inference.

## Most influential features

Importance method: impurity-based feature importance.

1. `derived.dlls_per_process_recomputed` (0.112860)
2. `handles.nmutant` (0.104851)
3. `derived.mutants_per_process` (0.102634)
4. `handles.nevent` (0.088477)
5. `derived.process_service_rate` (0.084546)
6. `pslist.avg_threads` (0.077058)
7. `ldrmodules.not_in_load` (0.056129)
8. `derived.driver_service_rate` (0.055737)
9. `handles.nkey` (0.047457)
10. `handles.ntimer` (0.042930)

## Leakage and generalization controls

- Hyperparameters were tuned only with folds inside the training partition.
- Candidate comparison used validation data, never test data.
- The test set was scored only after model choice and final refit.
- Every preprocessing transform is fitted inside its model pipeline.
- Random seeds and input SHA-256 hashes are recorded for reproducibility.

The source EDA found near-perfect single-feature separation. Therefore, even this strong held-out result may be optimistic if related captures or malware families occur in multiple random splits. A family-separated, time-separated, or independently collected external test set is required before deployment.

## Reproduce

From the project root:

```bash
.venv/bin/python scripts/model_selection_training.py
```
